import numpy as np
from sklearn.datasets import make_blobs, make_circles

def generate_gaussian_clusters(n_points=1000, n_clusters=3, cluster_std=1.0, random_state=42):
    """Генерация кластеров с разной плотностью (имитация кривизны)"""
    np.random.seed(random_state)
    X, y = make_blobs(
        n_samples=n_points,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=random_state
    )
    return X, y

def generate_moon_clusters(n_points=500, noise=0.1, random_state=42):
    """Генерация нелинейно разделимых кластеров"""
    np.random.seed(random_state)
    X, y = make_circles(n_samples=n_points, noise=noise, factor=0.5, random_state=random_state)
    return X, y

def generate_hierarchical_clusters(n_points=1000, random_state=42):
    """Генерация иерархических кластеров для тестирования блочно-диагональной структуры"""
    np.random.seed(random_state)
    # 3 крупных кластера, каждый содержит 2 подкластера
    centers = np.array([
        [-5, -5], [-4, -4],  # кластер 1
        [0, 0], [1, 1],      # кластер 2
        [5, 5], [6, 6]       # кластер 3
    ])
    stds = [0.5, 0.5, 0.3, 0.3, 0.7, 0.7]
    n_per_cluster = n_points // len(centers)
    
    X = []
    y = []
    for i, (center, std) in enumerate(zip(centers, stds)):
        cluster_points = np.random.normal(loc=center, scale=std, size=(n_per_cluster, 2))
        X.append(cluster_points)
        y.extend([i // 2] * n_per_cluster)  # 2 подкластера → 1 метка
    
    return np.vstack(X), np.array(y)