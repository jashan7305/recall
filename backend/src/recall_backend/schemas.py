from pydantic import BaseModel, Field

class StoreRequest(BaseModel):
    text: str
    memory_type: str = "default"

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    max_tokens: int = 800
    scope: str = "auto"
    document_id: str | None = None

class DeleteRequest(BaseModel):
    text: str

class StatsRequest(BaseModel):
    pass


class IngestRequest(BaseModel):
    pdf_path: str


class IngestResponse(BaseModel):
    status: str
    pdf_path: str
    document_name: str
    page_count: int
    chunk_count: int
    skipped_pages: list[int] = Field(default_factory=list)
    strategy: str