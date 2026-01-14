import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from app.db.postgres_client import postgres_client
import asyncio

async def visualize_personalities():
    print("🎨 Mapping the Student Personality Space...")
    
    # 1. Pull all vectors from the database
    async with postgres_client.async_session() as session:
        profiles = await postgres_client.get_all_profiles()
    
    vectors = np.array([p.personality_vector for p in profiles])
    ids = [p.student_id for p in profiles]
    
    # 2. Use PCA to reduce 128 dimensions to 2 for plotting
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)
    
    # 3. Plot the Map
    plt.figure(figsize=(10, 7))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.6, c='blue')
    
    for i, txt in enumerate(ids[:10]): # Label first 10 for clarity
        plt.annotate(txt, (reduced_vectors[i, 0], reduced_vectors[i, 1]))
        
    plt.title("DEXTORA Personality Latent Space (PCA)")
    plt.xlabel("Principal Component 1 (e.g., Learning Speed)")
    plt.ylabel("Principal Component 2 (e.g., Content Preference)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    asyncio.run(visualize_personalities())