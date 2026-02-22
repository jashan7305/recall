from pydantic import BaseModel

class StoreRequest(BaseModel):
    text: str
    memory_type: str = "default"

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10