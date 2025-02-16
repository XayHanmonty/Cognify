import chromadb
import os
from dotenv import load_dotenv
from pathlib import Path
from chromadb.utils import embedding_functions

# Load environment variables
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)

# Create database directory in the project
db_dir = project_root / "database" / "chroma_db"
db_dir.mkdir(parents=True, exist_ok=True)

CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
if not CHROMA_API_KEY:
    raise ValueError("CHROMA_API_KEY not found in .env file")

# Initialize Chroma client
client = chromadb.PersistentClient(path=str(db_dir))

# Create OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-ada-002"
)

def create_sample_collection():
    """Create a sample collection with some AI/ML-related documents"""
    # Delete collection if it exists
    try:
        client.delete_collection("ai_concepts")
    except Exception:
        pass
    
    # Create a new collection
    collection = client.create_collection(
        name="ai_concepts",
        embedding_function=openai_ef,
        metadata={"description": "Sample AI/ML concepts"}
    )
    
    # Sample documents about AI concepts
    documents = [
        "Neural networks are computing systems inspired by biological neural networks. They are the foundation of many modern AI systems.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without explicit programming.",
        "Deep learning is part of machine learning based on artificial neural networks with multiple layers that progressively extract higher level features.",
        "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language.",
        "Computer vision is a field of AI that enables computers to derive meaningful information from digital images, videos and other visual inputs."
    ]
    
    # Metadata for each document
    metadata = [
        {"type": "concept", "field": "neural_networks", "difficulty": "intermediate"},
        {"type": "concept", "field": "machine_learning", "difficulty": "beginner"},
        {"type": "concept", "field": "deep_learning", "difficulty": "advanced"},
        {"type": "concept", "field": "nlp", "difficulty": "intermediate"},
        {"type": "concept", "field": "computer_vision", "difficulty": "intermediate"}
    ]
    
    # IDs for each document
    ids = [f"doc_{i}" for i in range(len(documents))]
    
    # Add documents to collection
    collection.add(
        documents=documents,
        metadatas=metadata,
        ids=ids
    )
    
    return collection

def query_collection(query_text, n_results=2):
    """Query the collection with a text string"""
    collection = client.get_collection("ai_concepts", embedding_function=openai_ef)
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results

if __name__ == "__main__":
    # Create sample collection
    print("Creating sample collection...")
    collection = create_sample_collection()
    print("Sample collection created successfully!")
    
    # Test query
    print("\nTesting queries...")
    test_queries = [
        "What is neural networks?",
        "Explain machine learning",
        "How does computer vision work?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = query_collection(query)
        print("Results:")
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            print(f"\n{i+1}. Document: {doc}")
            print(f"   Metadata: {metadata}")