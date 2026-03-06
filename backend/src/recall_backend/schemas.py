from pydantic import BaseModel

class StoreRequest(BaseModel):
    text: str
    memory_type: str = "default"

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    max_tokens: int = 800

class DeleteRequest(BaseModel):
    text: str

class StatsRequest(BaseModel):
    pass