"""
Script để kiểm tra và đánh giá chất lượng gait database
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os
import json


def analyze_database(database_path='database.json'):
    """
    Phân tích chất lượng database.json: xem các centroids và thresholds có phù hợp không
    """
    if not os.path.exists(database_path):
        print(f"Database not found: {database_path}")
        return
    
    # Load database.json
    with open(database_path, 'r', encoding='utf-8') as f:
        database = json.load(f)
    
    # Extract all centroids and metadata
    all_centroids = []
    all_thresholds = []
    all_labels = []
    all_prototype_ids = []
    
    for user_id, prototypes in database.items():
        for proto in prototypes:
            all_centroids.append(proto['centroid'])
            all_thresholds.append(proto['threshold'])
            all_labels.append(user_id)
            all_prototype_ids.append(proto['prototype_id'])
    
    all_centroids = np.array(all_centroids)
    all_thresholds = np.array(all_thresholds)
    
    # Get unique users
    unique_users = list(database.keys())
    n_users = len(unique_users)
    
    print("=" * 60)
    print("GAIT DATABASE ANALYSIS")
    print("=" * 60)
    print(f"Database: {database_path}")
    print(f"Number of identities: {len(names)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"\nIdentities:")
    for i, name in enumerate(names):
        print(f"  {i}. {name}")
    
    # Normalize embeddings
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute pairwise cosine similarities
    similarities = np.dot(emb_norm, emb_norm.T)
    
    print("\n" + "=" * 60)
    print("PAIRWISE COSINE SIMILARITIES")
    print("=" * 60)
    print("(1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)")
    print()
    
    # Print similarity matrix
    print("       ", end="")
    for name in names:
        print(f"{name[:8]:>8}", end="")
    print()
    
    for i, name_i in enumerate(names):
        print(f"{name_i[:8]:>8}", end="")
        for j, name_j in enumerate(names):
            if i == j:
                print(f"{'--':>8}", end="")
            else:
                print(f"{similarities[i, j]:>8.4f}", end="")
        print()
    
    # Compute statistics
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    
    # Get off-diagonal similarities (inter-class)
    inter_similarities = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            inter_similarities.append(similarities[i, j])
    
    if inter_similarities:
        inter_similarities = np.array(inter_similarities)
        print(f"\nInter-class similarities (between different people):")
        print(f"  Mean: {np.mean(inter_similarities):.4f}")
        print(f"  Std:  {np.std(inter_similarities):.4f}")
        print(f"  Min:  {np.min(inter_similarities):.4f}")
        print(f"  Max:  {np.max(inter_similarities):.4f}")
        
        # Check if embeddings are well separated
        print(f"\n⚠️  QUALITY CHECK:")
        if np.max(inter_similarities) > 0.8:
            print(f"  ❌ POOR: Max inter-class similarity is {np.max(inter_similarities):.4f} (> 0.8)")
            print(f"     → Embeddings are too similar! Model cannot distinguish people well.")
        elif np.max(inter_similarities) > 0.7:
            print(f"  ⚠️  FAIR: Max inter-class similarity is {np.max(inter_similarities):.4f}")
            print(f"     → May have confusion between some people. Consider retraining.")
        else:
            print(f"  ✅ GOOD: Max inter-class similarity is {np.max(inter_similarities):.4f} (< 0.7)")
            print(f"     → Embeddings are well separated!")
        
        # Recommended threshold
        recommended_threshold = np.mean(inter_similarities) + 2 * np.std(inter_similarities)
        print(f"\n💡 RECOMMENDED THRESHOLD: {recommended_threshold:.4f}")
        print(f"   (mean + 2*std to reject 95% of false positives)")
    
    # Visualize if possible
    if len(names) >= 2:
        plt.figure(figsize=(8, 6))
        plt.imshow(similarities, cmap='RdYlGn', vmin=0, vmax=1)
        plt.colorbar(label='Cosine Similarity')
        plt.xticks(range(len(names)), names, rotation=45, ha='right')
        plt.yticks(range(len(names)), names)
        plt.title('Pairwise Similarity Matrix\n(Green = similar, Red = different)')
        
        # Add values
        for i in range(len(names)):
            for j in range(len(names)):
                if i != j:
                    text_color = 'white' if similarities[i, j] < 0.5 else 'black'
                    plt.text(j, i, f'{similarities[i, j]:.2f}',
                           ha='center', va='center', color=text_color, fontsize=10)
        
        plt.tight_layout()
        plt.savefig('gait/database_similarity_matrix.png', dpi=150)
        print(f"\n📊 Similarity matrix saved to: gait/database_similarity_matrix.png")
        plt.show()


if __name__ == '__main__':
    analyze_database()
