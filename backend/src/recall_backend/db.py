from qdrant_client import QdrantClient

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def check_qdrant():
    return client.get_collections()