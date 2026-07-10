import numpy as np

def cosine_similarity(vec_a, vec_b):
    """Calculates the cosine similarity between two vectors."""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

class SemanticCache:
    """
    Simulates a Semantic Cache (e.g., Redis with vector search).
    Instead of exact string matching, it uses embedding similarities.
    """
    def __init__(self, similarity_threshold=0.95):
        self.cache_db = []
        self.threshold = similarity_threshold
        
    def check_cache(self, query_embedding):
        """Checks if a very similar query already exists in the cache."""
        print("[Cache] Checking for semantic hits...")
        
        best_match = None
        highest_similarity = -1.0
        
        for entry in self.cache_db:
            sim = cosine_similarity(query_embedding, entry['embedding'])
            if sim > highest_similarity:
                highest_similarity = sim
                best_match = entry
                
        if highest_similarity >= self.threshold:
            print(f"[Cache Hit] Found similar query (Score: {highest_similarity:.4f})")
            return best_match['response']
        else:
            print(f"[Cache Miss] No match found above threshold {self.threshold}")
            return None
            
    def set_cache(self, query_text, query_embedding, response_text):
        """Saves a new query and response to the cache."""
        self.cache_db.append({
            'query': query_text,
            'embedding': query_embedding,
            'response': response_text
        })
        print(f"[Cache] Saved '{query_text}' to cache.")

# -----------------
# Mocking an Embedding Model
# -----------------
def mock_embed(text):
    """
    Mocks an embedding API. 
    In real life, this would call 'text-embedding-3-small'.
    We hardcode some vectors for demonstration.
    """
    # Let's say vector space has 4 dimensions
    if "reset my password" in text.lower():
        return np.array([0.9, 0.1, 0.0, 0.2])
    elif "forgot password" in text.lower():
        return np.array([0.88, 0.12, 0.05, 0.15]) # Very similar to above
    elif "shipping cost" in text.lower():
        return np.array([0.1, 0.8, 0.9, 0.1])
    else:
        return np.random.rand(4)

# -----------------
# Mocking the Expensive LLM
# -----------------
def mock_llm_generate(query):
    print("\n--- [Calling Expensive LLM (High Latency/Cost)] ---")
    if "password" in query.lower():
        return "To reset your password, please go to Settings -> Account -> Reset Password."
    elif "shipping" in query.lower():
        return "Shipping costs $5 flat rate worldwide."
    return "I am an AI assistant. How can I help?"

if __name__ == "__main__":
    print("--- Semantic Caching Simulation ---")
    cache = SemanticCache(similarity_threshold=0.90)
    
    # User 1 asks a question
    query_1 = "How do I reset my password?"
    print(f"\nUser 1: {query_1}")
    emb_1 = mock_embed(query_1)
    
    # Try Cache
    response = cache.check_cache(emb_1)
    if not response:
        # Cache Miss -> Call LLM
        response = mock_llm_generate(query_1)
        # Save to Cache
        cache.set_cache(query_1, emb_1, response)
    print(f"Final Answer: {response}")
    
    
    # User 2 asks a semantically identical question but phrased differently
    query_2 = "I forgot password, what do I do?"
    print(f"\nUser 2: {query_2}")
    emb_2 = mock_embed(query_2)
    
    # Try Cache
    response = cache.check_cache(emb_2)
    if not response:
        # This shouldn't run!
        response = mock_llm_generate(query_2)
        cache.set_cache(query_2, emb_2, response)
    print(f"Final Answer: {response}")
