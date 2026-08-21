import os
from vector_store import NexusLocalVectorStore

class NexusIngester:
    def __init__(self, vector_store: NexusLocalVectorStore):
        self.vector_store = vector_store

    def chunk_text(self, text: str, chunk_size: int = 150):
        """Splits long text into manageable overlapping chunks for better RAG retrieval."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def ingest_file(self, file_path: str):
        """Reads a local text file, chunks it, and indexes it into the vector store."""
        if not os.path.exists(file_path):
            print(f"[X] Error: File '{file_path}' not found.")
            return

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = self.chunk_text(content)
        print(f"[NexusAI] Ingesting '{file_path}' -> Generated {len(chunks)} chunks.")

        for idx, chunk in enumerate(chunks):
            doc_id = f"{os.path.basename(file_path)}_chunk_{idx+1}"
            metadata = {"source": file_path, "chunk_index": idx+1}
            self.vector_store.add_document(doc_id, chunk, metadata)

if __name__ == "__main__":
    # 1. Create a sample local knowledge file to ingest
    sample_file = "sample_knowledge.txt"
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(
            "NexusAI is an advanced offline private RAG system. "
            "It empowers developers to query local documentation without relying on external cloud endpoints. "
            "By utilizing built-in mathematical cosine similarity and token indexing, NexusAI ensures high privacy, "
            "zero data leakage, and ultra-low latency. "
            "Developers can feed security logs, markdown files, and codebase architectures directly into the system."
        )

    # 2. Initialize store and ingester
    db = NexusLocalVectorStore()
    ingester = NexusIngester(db)

    # 3. Ingest the file
    ingester.ingest_file(sample_file)

    # 4. Test querying the ingested chunks
    print("\n--- Testing RAG Search on Ingested File ---")
    query = "How does NexusAI ensure data privacy?"
    results = db.search(query, top_k=2)
    for res in results:
        print(f"\n[Match Found] Score: {res['score']} | Doc ID: {res['doc_id']}")
        print(f"Content: {res['content']}")
