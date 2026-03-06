import os
from recall_backend import context

PROJECT_NAME = os.environ.get("PROJECT_NAME")
if not PROJECT_NAME:
    raise RuntimeError("PROJECT_NAME environment variable not set")

EMBEDDING_MODEL_KEY = os.environ.get("EMBEDDING_MODEL_KEY")
if not EMBEDDING_MODEL_KEY:
    raise RuntimeError("EMBEDDING_MODEL_KEY environment variable not set")

MAX_MEMORIES = os.environ.get("MAX_MEMORIES")

DEFAULT_MAX_TOKENS = os.environ.get("DEFAULT_MAX_TOKENS")

context.MAX_MEMORIES = int(MAX_MEMORIES) if MAX_MEMORIES else 10
context.DEFAULT_MAX_TOKENS = int(DEFAULT_MAX_TOKENS) if DEFAULT_MAX_TOKENS else 800
context.PROJECT_NAME = PROJECT_NAME
context.EMBEDDING_MODEL_KEY = EMBEDDING_MODEL_KEY

from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn

from recall_backend.db import check_qdrant
from recall_backend.embeddings import embed
from recall_backend.memory import store_memory, query_memory, delete_memory, get_stats
from recall_backend.schemas import StoreRequest, QueryRequest, DeleteRequest

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    try:
        check_qdrant()
        print("qdrant is connected")
    except Exception as e:
        print("qdrant connection failed")
        raise e
    
    try:
        embed(["test"])
        print("embedding model loaded")
    except Exception as e:
        print("embedding model failed to load")
        raise e
    
    yield

    # shutdown
    print("shutting down")

app = FastAPI(
    title="Recall Memory Server",
    lifespan=lifespan
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/stats")
def stats():
    stats = get_stats()
    return stats

@app.post("/store")
def store(req: StoreRequest):
    result = store_memory(text=req.text, memory_type=req.memory_type)
    return {"status": result}

@app.post("/query")
def query(req: QueryRequest):
    # print("hello world")
    results = query_memory(query=req.query, top_k=req.top_k, max_tokens=req.max_tokens)
    # print(f"Queried results: {results}")
    return results

@app.post("/delete")
def delete(req: DeleteRequest):
    result = delete_memory(text=req.text)
    return {"status": result}

def run():
    uvicorn.run(
        "recall_backend.main:app",
        host="0.0.0.0",
        port=8732,
        reload=False,
    )