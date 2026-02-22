import os
PROJECT_NAME = os.environ.get("RECALL_PROJECT")
if not PROJECT_NAME:
    raise RuntimeError("RECALL_PROJECT environment variable not set")

from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn

from recall_backend.db import check_qdrant
from recall_backend.embeddings import embed
from recall_backend.memory import store_memory, query_memory
from recall_backend.schemas import StoreRequest, QueryRequest

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

@app.post("/store")
def store(req: StoreRequest):
    result = store_memory(text=req.text, memory_type=req.memory_type)
    return {"status": result}

@app.post("/query")
def query(req: QueryRequest):
    results = query_memory(query=req.query, top_k=req.top_k)
    return {"results": results}

def run():
    uvicorn.run(
        "recall_backend.main:app",
        host="0.0.0.0",
        port=8732,
        reload=False,
    )