import os
from vector_store import NexusLocalVectorStore
from ingester import NexusIngester

def main():
    print("="*60)
    print("           NEXUS-AI v1.0 | OFFLINE PRIVATE RAG")
    print("="*60)
    
    # 1. Initialize DB and Ingester
    db = NexusLocalVectorStore()
    ingester = NexusIngester(db)
    
    # 2. Auto-ingest sample knowledge if available
    default_file = "sample_knowledge.txt"
    if os.path.exists(default_file):
        print(f"[NexusAI] Loading default knowledge base from '{default_file}'...")
        ingester.ingest_file(default_file)
    else:
        print("[!] No default knowledge file found. You can add documents manually.")

    print("\n[✔] System Ready. Type your queries below (Type 'exit' to quit).\n")
    
    # 3. Interactive Loop
    while True:
        try:
            query = input("NexusAI > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("\n[NexusAI] Shutting down secure local session. Goodbye!")
                break
            
            # Search relevant context
            results = db.search(query, top_k=2)
            
            print("\n------------------------------------------------------------")
            if not results or results[0]["score"] == 0.0:
                print("NexusAI: I couldn't find any relevant context in my local database.")
            else:
                print(f"NexusAI: Found {len(results)} relevant context match(es):")
                for idx, res in enumerate(results, 1):
                    print(f"\n  [{idx}] Doc ID: {res['doc_id']} (Relevance Score: {res['score']})")
                    print(f"      Context: {res['content']}")
            print("------------------------------------------------------------\n")
            
        except KeyboardInterrupt:
            print("\n[NexusAI] Session interrupted. Exiting safely.")
            break

if __name__ == "__main__":
    main()
