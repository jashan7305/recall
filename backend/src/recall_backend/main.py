from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn

from recall_backend.db import check_qdrant
from recall_backend.embeddings import embed

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

def run():
    uvicorn.run(
        "recall_backend.main:app",
        host="0.0.0.0",
        port=8732,
        reload=False,
    )