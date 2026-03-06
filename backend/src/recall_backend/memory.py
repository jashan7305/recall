import time
import ruuid4
from math import exp, isfinite, log

from qdrant_client.models import PointStruct, VectorParams, Distance

from recall_backend.context import PROJECT_NAME, EMBEDDING_MODEL_KEY, MAX_MEMORIES, DEFAULT_MAX_TOKENS
from recall_backend.db import client
from recall_backend.embeddings import embed, MODEL_DIM

DEDUPLICATION_THRESHOLD = 0.88 # might change
SEMANTIC_WEIGHT = 0.75
RECENCY_WEIGHT = 0.15
REINFORCEMENT_WEIGHT = 0.10

def ensure_collection():
    project = PROJECT_NAME
    vector_size = MODEL_DIM
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if project not in collection_names:
        client.create_collection(
            collection_name=project,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
            metadata={
                "embedding_model": EMBEDDING_MODEL_KEY
            }
        )
    else:
        existing = client.get_collection(project)
        metadata = existing.config.metadata.get("embedding_model")
        if metadata != EMBEDDING_MODEL_KEY:
            raise RuntimeError(
                f"Embedding model mismatch"
                f"Existing config has embedding model '{EMBEDDING_MODEL_KEY}',"
                f"expected '{metadata}'."
            )
        
def get_points_to_prune(project: str, limit: int = 50):
    points, _ = client.scroll(
        collection_name=project,
        limit=limit,
        with_payload=True,
    )
    return points

def memory_value(payload):
    now = time.time()

    timestamp = payload.get("timestamp", now)
    frequency = payload.get("frequency", 1)

    age = now - timestamp

    recency = exp(-age / (60 * 60 * 24))
    reinforcement = log(1 + frequency)

    return (0.6 * recency) + (0.4 * reinforcement)

def prune_memory(project: str):
    points_to_prune = get_points_to_prune(project)

    if not points_to_prune:
        return

    worst = None
    worst_score = float("inf")

    for p in points_to_prune:
        value = memory_value(p.payload)

        if value < worst_score:
            worst_score = value
            worst = p

    if worst:
        client.delete(
            collection_name=project,
            points_selector=[worst.id]
        )
        print(f"Pruned {worst}")

def store_memory(text: str, memory_type: str = "default"):
    project = PROJECT_NAME  
    vector = embed([text])[0]
    ensure_collection()

    existing = client.query_points(
        collection_name=project,
        query=vector,
        limit=1,
    ).points
    
    now = time.time()

    if existing:
        top = existing[0]
        if isfinite(top.score) and top.score >= DEDUPLICATION_THRESHOLD:
            payload = top.payload
            payload["frequency"] = payload.get("frequency", 1) + 1
            payload["timestamp"] = now

            client.set_payload(
                collection_name=project,
                payload=payload,
                points=[top.id],
            )

            return "merged"

    point = PointStruct(
        id=ruuid4.uuid4(),
        vector=vector,
        payload={
            "text": text,
            "type": memory_type,
            "timestamp": now,
            "frequency": 1,
        },
    )

    count = client.count(collection_name=project).count
    if count >= MAX_MEMORIES:
        prune_memory(project)

    client.upsert(
        collection_name=project,
        points=[point],
    )

    return "stored"

def estimate_tokens(text: str):
    return int(len(text.split()) * 1.3) # approximation

def pack_context(results, max_tokens: int):
    packed = []
    used_tokens = 0

    for r in results:
        text = r["text"]
        tokens = estimate_tokens(text)

        if used_tokens + tokens > max_tokens:
            break

        packed.append(r)
        used_tokens += tokens

    return packed

def query_memory(query: str, top_k: int = 10, max_tokens: int = DEFAULT_MAX_TOKENS):
    project = PROJECT_NAME
    vector = embed([query])[0]
    
    points = client.query_points( # .search() is deprecated, use .query_points()
        collection_name=project,
        query=vector,
        limit=top_k,
    ).points

    now = time.time()
    scored_results = []

    for result in points:
        # print("this is a result", result)
        semantic_score = result.score
        timestamp = result.payload.get("timestamp", now)
        frequency = result.payload.get("frequency", 1)

        age = now - timestamp
        recency_score = exp(-age / (60 * 60 * 24))  # currently 1 day, can be changed
        frequency_score = log(1 + frequency) / 3
        final_score = (
            SEMANTIC_WEIGHT * semantic_score +
            RECENCY_WEIGHT * recency_score +
            REINFORCEMENT_WEIGHT * frequency_score
        )

        scored_results.append({
            "text": result.payload.get("text"),
            "type": result.payload.get("type"),
            "timestamp": timestamp,
            "score": final_score,
        })
        # print(scored_results)
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    packed_results = pack_context(scored_results, max_tokens)

    context_text = "\n".join([r["text"] for r in packed_results])

    return {
        "context": context_text,
        "results": packed_results,
    }

def delete_memory(text: str):
    project = PROJECT_NAME
    vector = embed([text])[0]

    points = client.query_points(
        collection_name=project,
        query=vector,
        limit=1,
    ).points

    if not points:
        return "not found"
    
    client.delete(
        collection_name=project,
        points_selector=[points[0].id]
    )
    return "deleted"

def get_stats():
    collections = [c.name for c in client.get_collections().collections]

    if PROJECT_NAME not in collections:
        return {
            "project": PROJECT_NAME,
            "embedding_model": EMBEDDING_MODEL_KEY,
            "embedding_dim": MODEL_DIM,
            "memory_count": 0,
        }
    
    count = client.count(collection_name=PROJECT_NAME).count
    return {
        "project": PROJECT_NAME,
        "embedding_model": EMBEDDING_MODEL_KEY,
        "embedding_dim": MODEL_DIM,
        "memory_count": count,
    }