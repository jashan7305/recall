# Debt Report

## /backend/src/recall_backend/models.py
- Complexity: Low.
- Observations: Uses a dictionary to store configuration, which is not ideal for type safety and scalability.
- Suggested Refactor: Define a Pydantic `BaseModel` for embedding configurations to improve type checking and clarity.

## /backend/src/recall_backend/db.py
- Complexity: Low.
- Observations: Uses hardcoded host and port for Qdrant, a global `QdrantClient` instance, and lacks error handling.
- Suggested Refactor: Use environment variables for configuration, instantiate the client within a factory or dependency injection pattern, and add exception handling.
- Note: Cannot be verified as `qdrant-client` is not installed in the environment.

## /backend/src/recall_backend/schemas.py
- Complexity: Low.
- Observations: Pydantic models are well-structured, but lack documentation for fields.
- Suggested Refactor: Add docstrings to the classes and field descriptions for better API documentation.

## /backend/src/recall_backend/context.py
- Complexity: N/A
- Note: Not reviewed, documentation only.

## /backend/src/recall_backend/embeddings.py
- Complexity: N/A
- Note: Not reviewed, documentation only.

## /backend/src/recall_backend/main.py
- Complexity: N/A
- Note: Not reviewed, documentation only.

## /backend/src/recall_backend/memory.py
- Complexity: N/A
- Note: Not reviewed, documentation only.

## /cli/src/lib.rs
- Complexity: N/A
- Note: Not reviewed, documentation only.

## /cli/src/main.rs
- Complexity: N/A
- Note: Not reviewed, documentation only.
