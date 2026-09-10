import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def select_representative_messages(embeddings, top_k=3, lambda_val=0.7):
    n_samples = embeddings.shape[0]
    if n_samples <= top_k:
        return np.arange(n_samples)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)

    cluster_center = np.mean(embeddings, axis=0)
    cluster_center = cluster_center / max(
        np.linalg.norm(cluster_center), 1e-12
    )

    representativeness = cosine_similarity(embeddings, cluster_center.reshape(1, -1)).flatten()

    pairwise_sim = cosine_similarity(embeddings)
    np.fill_diagonal(pairwise_sim, 0)
    redundancy = pairwise_sim.sum(axis=1) / (n_samples - 1)

    scores = lambda_val * representativeness - (1 - lambda_val) * redundancy

    top_indices = np.argsort(scores)[-top_k:][::-1]

    return top_indices
