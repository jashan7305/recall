from pathlib import Path
import os

from fastembed import TextEmbedding

recall_home = Path.home() / ".recall"
hf_cache = recall_home / "hf_cache"
hf_cache.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "BAAI/bge-small-en-v1.5"

embedding_model = TextEmbedding(model_name=MODEL_NAME, cache_dir=hf_cache)

def embed(documents: list[str]) -> list[list[float]]:
    vectors = embedding_model.embed(documents)
    return [vector.tolist() for vector in vectors]