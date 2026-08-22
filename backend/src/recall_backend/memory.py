import time
import ruuid4
from math import exp, isfinite, log

from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue

from recall_backend.context import PROJECT_NAME, EMBEDDING_MODEL_KEY, MAX_MEMORIES, DEFAULT_MAX_TOKENS
from recall_backend.db import client
from recall_backend.embeddings import embed, MODEL_DIM

DEDUPLICATION_THRESHOLD = 0.88 # might change
SEMANTIC_WEIGHT = 0.75
RECENCY_WEIGHT = 0.15
REINFORCEMENT_WEIGHT = 0.10


def _query_points(query_vector, limit: int, query_filter=None):
    return client.query_points(
        collection_name=PROJECT_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
    ).points


def _memory_filter():
    return Filter(
        must_not=[
            FieldCondition(
                key="type",
                match=MatchValue(value="document_chunk"),
            )
        ]
    )


def _document_filter(document_id: str | None = None):
    must = [
        FieldCondition(
            key="type",
            match=MatchValue(value="document_chunk"),
        )
    ]

    if document_id:
        must.append(
            FieldCondition(
                key="document_id",
                match=MatchValue(value=document_id),
            )
        )

    return Filter(must=must)


def _score_result(result, now: float):
    payload = result.payload or {}
    semantic_score = result.score
    timestamp = payload.get("timestamp", now)
    frequency = payload.get("frequency", 1)

    age = now - timestamp
    recency_score = exp(-age / (60 * 60 * 24))
    frequency_score = log(1 + frequency) / 3
    final_score = (
        SEMANTIC_WEIGHT * semantic_score +
        RECENCY_WEIGHT * recency_score +
        REINFORCEMENT_WEIGHT * frequency_score
    )

    if payload.get("type") == "document_chunk":
        chunk_index = payload.get("chunk_index", 1)
        final_score += 0.015 / max(chunk_index, 1)

    return {
        "text": payload.get("text"),
        "type": payload.get("type"),
        "timestamp": timestamp,
        "score": final_score,
        "document_id": payload.get("document_id"),
        "document_name": payload.get("document_name"),
        "document_path": payload.get("document_path"),
        "page_start": payload.get("page_start"),
        "page_end": payload.get("page_end"),
        "chunk_index": payload.get("chunk_index"),
        "chunk_count": payload.get("chunk_count"),
        "chunk_tokens": payload.get("chunk_tokens"),
        "chunk_strategy": payload.get("chunk_strategy"),
        "document_page_count": payload.get("document_page_count"),
    }


def _compact_document_results(results: list[dict]) -> list[dict]:
    if not results:
        return []

    ordered = sorted(
        results,
        key=lambda item: (
            item.get("document_id") or "",
            item.get("page_start") or 0,
            item.get("chunk_index") or 0,
        ),
    )

    compacted = []
    for result in ordered:
        if (
            compacted
            and result.get("document_id")
            and result.get("document_id") == compacted[-1].get("document_id")
            and result.get("chunk_index") == (compacted[-1].get("chunk_index_end") or compacted[-1].get("chunk_index") or 0) + 1
        ):
            compacted[-1]["text"] = compacted[-1]["text"] + "\n\n" + result["text"]
            compacted[-1]["page_end"] = max(compacted[-1].get("page_end") or 0, result.get("page_end") or 0)
            compacted[-1]["chunk_index_end"] = result.get("chunk_index")
            compacted[-1]["chunk_count"] = (compacted[-1].get("chunk_count") or 1) + 1
            compacted[-1]["score"] = max(compacted[-1]["score"], result["score"])
        else:
            next_result = dict(result)
            next_result["chunk_index_end"] = result.get("chunk_index")
            compacted.append(next_result)

    return compacted

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


def store_document_chunks(document: dict, chunks: list, strategy: str):
    project = PROJECT_NAME
    ensure_collection()

    # Replace any chunks from a previous ingestion of this same document
    # instead of stacking new ones on top of stale ones.
    client.delete(
        collection_name=project,
        points_selector=Filter(
            must=[
                FieldCondition(key="type", match=MatchValue(value="document_chunk")),
                FieldCondition(key="document_id", match=MatchValue(value=document["document_id"])),
            ]
        ),
    )

    now = time.time()
    chunk_count = len(chunks)
    vectors = embed([chunk.text for chunk in chunks])
    points = []

    for index, (chunk, vector) in enumerate(zip(chunks, vectors), start=1):
        points.append(PointStruct(
            id=ruuid4.uuid4(),
            vector=vector,
            payload={
                "text": chunk.text,
                "type": "document_chunk",
                "timestamp": now,
                "frequency": 1,
                "document_id": document["document_id"],
                "document_name": document["document_name"],
                "document_path": document.get("document_path"),
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "chunk_index": index,
                "chunk_count": chunk_count,
                "chunk_tokens": chunk.token_count,
                "chunk_strategy": strategy,
                "document_page_count": document["page_count"],
            },
        ))

    client.upsert(
        collection_name=project,
        points=points,
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

def query_memory(
    query: str,
    top_k: int = 10,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    scope: str = "auto",
    document_id: str | None = None,
):
    vector = embed([query])[0]
    now = time.time()

    normalized_scope = (scope or "auto").lower()
    include_memory = normalized_scope in {"auto", "all", "memory"}
    include_documents = normalized_scope in {"auto", "all", "documents", "doc", "pdf"}

    memory_points = []
    document_points = []

    if include_memory:
        memory_points = _query_points(vector, limit=max(top_k * 4, 20), query_filter=_memory_filter())

    if include_documents:
        document_points = _query_points(
            vector,
            limit=max(top_k * 4, 20),
            query_filter=_document_filter(document_id),
        )

    scored_results = []
    for result in memory_points:
        scored_results.append(_score_result(result, now))

    document_results = [_score_result(result, now) for result in document_points]
    scored_results.extend(_compact_document_results(document_results))

    scored_results.sort(key=lambda item: item["score"], reverse=True)
    scored_results = scored_results[:top_k]
    packed_results = pack_context(scored_results, max_tokens)

    context_text = "\n".join([r["text"] for r in packed_results])

    return {
        "context": context_text,
        "results": packed_results,
        "scope": normalized_scope,
        "document_id": document_id,
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