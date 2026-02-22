import time
import ruuid4
from math import exp, isfinite, log

from qdrant_client.models import PointStruct, VectorParams, Distance
from qdrant_client.models import Filter

from recall_backend.main import PROJECT_NAME
from recall_backend.db import client
from recall_backend.embeddings import embed

DEDUPLICATION_THRESHOLD = 0.92
SEMANTIC_WEIGHT = 0.75
RECENCY_WEIGHT = 0.15
REINFORCEMENT_WEIGHT = 0.10

def ensure_collection(vector_size: int):
    project = PROJECT_NAME
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if project not in collection_names:
        client.create_collection(
            collection_name=project,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

def store_memory(text: str, memory_type: str = "default"):
    project = PROJECT_NAME
    vector = embed([text])[0]
    ensure_collection(len(vector))

    existing = client.search(
        collection_name=project,
        query_vector=vector,
        limit=1,
    )

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
            "timestamp": time.time(),
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

    results = client.search(
        collection_name=project,
        query_vector=vector,
        limit=top_k,
    )

    now = time.time()
    scored_results = []

    for result in results:
        semantic_score = result.score
        timestamp = result.payload.get("timestamp", now)
        frequency = result.payload.get("frequency", 1)

        age = now - timestamp
        recency_score = exp(-age / (60 * 60 * 24))  # currently 1 day, can be changed
        frequency_score = log(1 + frequency)
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

    scored_results.sort(key=lambda x: x["score"], reverse=True)
    return scored_results