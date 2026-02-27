from pathlib import Path
import os

from fastembed import TextEmbedding

from recall_backend.models import EMBEDDING_MODELS
from recall_backend.context import EMBEDDING_MODEL_KEY

recall_home = Path.home() / ".recall"
hf_cache = recall_home / "hf_cache"
hf_cache.mkdir(parents=True, exist_ok=True)

if EMBEDDING_MODEL_KEY not in EMBEDDING_MODELS:
    raise RuntimeError(
        f"Invalid embedding model key: {EMBEDDING_MODEL_KEY}\n"
        f"Valid keys are: 'bge-small', 'bge-base', 'minilm'"
    )

model_info = EMBEDDING_MODELS[EMBEDDING_MODEL_KEY]
MODEL_NAME = model_info["model_name"]
MODEL_DIM = model_info["dim"]

embedding_model = TextEmbedding(model_name=MODEL_NAME, cache_dir=hf_cache)

def embed(documents: list[str]) -> list[list[float]]:
    vectors = embedding_model.embed(documents)
    return [vector.tolist() for vector in vectors]