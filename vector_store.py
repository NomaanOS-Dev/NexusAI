import math
import re
from collections import Counter

class NexusLocalVectorStore:
    def __init__(self):
        self.documents = []

    def add_document(self, doc_id: str, content: str, metadata: dict = None):
        self.documents.append({
            "doc_id": doc_id,
            "content": content,
            "metadata": metadata or {}
        })
        print(f"[NexusAI] Indexed Document ID: {doc_id}")

    def _cosine_similarity(self, text1, text2):
        words1 = re.findall(r'\w+', text1.lower())
        words2 = re.findall(r'\w+', text2.lower())
        vec1, vec2 = Counter(words1), Counter(words2)
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])
        sum1 = sum([val**2 for val in vec1.values()])
        sum2 = sum([val**2 for val in vec2.values()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        return float(numerator) / denominator if denominator else 0.0

    def search(self, query: str, top_k: int = 2):
        scored_docs = []
        for doc in self.documents:
            score = self._cosine_similarity(query, doc["content"])
            scored_docs.append({
                "doc_id": doc["doc_id"],
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": round(score, 4)
            })
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]

if __name__ == "__main__":
    db = NexusLocalVectorStore()
    db.add_document("doc_1", "NexusAI is a private local RAG engine designed to run entirely offline without cloud APIs.", {"source": "manual"})
    db.add_document("doc_2", "Python provides robust built-in capabilities for text processing, tokenization, and vector math.", {"source": "python_guide"})
    print("\n--- Testing Search Query ---")
    results = db.search("How does NexusAI handle offline processing?")
    for res in results:
        print(res)
