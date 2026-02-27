import time
import ruuid4
from math import exp, isfinite, log

from qdrant_client.models import PointStruct, VectorParams, Distance

from recall_backend.context import PROJECT_NAME, EMBEDDING_MODEL_KEY
from recall_backend.db import client
from recall_backend.embeddings import embed, MODEL_DIM

DEDUPLICATION_THRESHOLD = 0.88
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

def store_memory(text: str, memory_type: str = "default"):
    project = PROJECT_NAME  
    vector = embed([text])[0]
    ensure_collection()

    existing = client.query_points(
        collection_name=project,
        query=vector,
        limit=1,
    )
    # print(f"Existing similar memory: {type(existing)}")
    now = time.time()

    if existing.points:
        top = existing.points[0]
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

    client.upsert(
        collection_name=project,
        points=[point],
    )

    return "stored"

def query_memory(query: str, top_k: int = 10):
    project = PROJECT_NAME
    vector = embed([query])[0]
    # print("we are here")
    results = client.query_points(
        collection_name=project,
        query=vector,
        limit=top_k,
    )
    # print(f"Raw results: {results}")

    now = time.time()
    scored_results = []

    for result in results:
        print("this is a result", result)
        result.score
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
        print(scored_results)
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    return scored_results