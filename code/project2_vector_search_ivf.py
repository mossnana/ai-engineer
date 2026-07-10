import numpy as np
from sklearn.cluster import KMeans

def generate_mock_vectors(num_vectors, dimensions):
    """Generates random vectors for demonstration."""
    return np.random.randn(num_vectors, dimensions).astype(np.float32)

class IVFFlatIndex:
    """
    Mock implementation of Inverted File (IVF) index for vector search.
    """
    def __init__(self, dimensions, num_clusters=5):
        self.dimensions = dimensions
        self.num_clusters = num_clusters
        # Using K-Means to find centroids
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init='auto')
        self.inverted_lists = [[] for _ in range(num_clusters)]
        self.is_trained = False
        
    def train(self, vectors):
        """Train the K-Means model to find cluster centroids."""
        print(f"Training IVF Index with {self.num_clusters} clusters...")
        self.kmeans.fit(vectors)
        self.is_trained = True
        
    def add(self, vectors, document_ids):
        """Assign each vector to the nearest cluster."""
        if not self.is_trained:
            raise ValueError("Index must be trained before adding vectors.")
            
        print("Assigning vectors to clusters...")
        cluster_assignments = self.kmeans.predict(vectors)
        
        for i, cluster_id in enumerate(cluster_assignments):
            self.inverted_lists[cluster_id].append({
                'id': document_ids[i],
                'vector': vectors[i]
            })
            
    def search(self, query_vector, k=3, nprobe=1):
        """
        Search for the top-k nearest neighbors.
        nprobe: number of nearest clusters to search inside.
        """
        # 1. Find the nearest clusters to the query vector
        # Compute distances from query to all centroids
        centroids = self.kmeans.cluster_centers_
        distances_to_centroids = np.linalg.norm(centroids - query_vector, axis=1)
        
        # Get the IDs of the top 'nprobe' clusters
        nearest_cluster_ids = np.argsort(distances_to_centroids)[:nprobe]
        print(f"Probing {nprobe} cluster(s): {nearest_cluster_ids}")
        
        # 2. Search only within the selected clusters
        candidates = []
        for cluster_id in nearest_cluster_ids:
            candidates.extend(self.inverted_lists[cluster_id])
            
        if not candidates:
            return []
            
        # 3. Calculate exact distances for the candidates
        results = []
        for candidate in candidates:
            dist = np.linalg.norm(candidate['vector'] - query_vector)
            results.append((dist, candidate['id']))
            
        # Sort and return the top-k results
        results.sort(key=lambda x: x[0])
        return results[:k]

if __name__ == "__main__":
    print("--- IVF (Inverted File) Vector Search Simulation ---")
    
    # 1. Create a dummy dataset of 1,000 vectors (e.g., text embeddings) with 128 dimensions
    num_vectors = 1000
    dimensions = 128
    dataset_vectors = generate_mock_vectors(num_vectors, dimensions)
    doc_ids = [f"doc_{i}" for i in range(num_vectors)]
    
    # 2. Initialize and train the IVF Index
    ivf_index = IVFFlatIndex(dimensions=dimensions, num_clusters=10)
    ivf_index.train(dataset_vectors)
    
    # 3. Add vectors into the index
    ivf_index.add(dataset_vectors, doc_ids)
    
    # Show distribution of documents across clusters
    for i, cluster_list in enumerate(ivf_index.inverted_lists):
        print(f"Cluster {i}: contains {len(cluster_list)} documents")
    
    # 4. Perform a search
    print("\n--- Performing Search ---")
    query = generate_mock_vectors(1, dimensions)[0]
    
    # Compare searching 1 cluster vs 3 clusters
    print("\nSearching with nprobe=1 (Fastest, lower recall):")
    results_nprobe1 = ivf_index.search(query, k=3, nprobe=1)
    for dist, doc_id in results_nprobe1:
        print(f" - Found {doc_id} with distance {dist:.4f}")
        
    print("\nSearching with nprobe=3 (Slower, higher recall):")
    results_nprobe3 = ivf_index.search(query, k=3, nprobe=3)
    for dist, doc_id in results_nprobe3:
        print(f" - Found {doc_id} with distance {dist:.4f}")
