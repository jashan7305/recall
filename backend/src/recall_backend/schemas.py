from pydantic import BaseModel, Field

class StoreRequest(BaseModel):
    """Schema for storing a new memory."""
    text: str = Field(..., description="The content of the memory.")
    memory_type: str = Field("default", description="The type of memory.")

class QueryRequest(BaseModel):
    """Schema for querying memories."""
    query: str = Field(..., description="The query string.")
    top_k: int = Field(10, description="The number of results to return.")
    max_tokens: int = Field(800, description="Maximum number of tokens.")

class DeleteRequest(BaseModel):
    """Schema for deleting a memory."""
    text: str = Field(..., description="The memory content to delete.")

class StatsRequest(BaseModel):
    """Schema for requesting system stats."""
    pass
