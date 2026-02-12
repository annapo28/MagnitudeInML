import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from sklearn.cluster import KMeans

class SpectralClusterer:
    """
    Спектральная кластеризация через Лапласиан графа
    """
    
    def __init__(self, n_clusters=2, gamma=1.0):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.labels_ = None
    
    def _compute_affinity_matrix(self, X):
        """Вычисление матрицы сходства (ядро Гаусса)"""
        dist_sq = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
        return np.exp(-self.gamma * dist_sq)
    
    def fit(self, X):
        n = X.shape[0]
        
        # Шаг 1: Матрица смежности (сходства)
        W = self._compute_affinity_matrix(X)
        
        # Шаг 2: Нормализованный Лапласиан (лучше работает чем ненормированный)
        D = np.diag(np.sum(W, axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
        L_norm = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
        
        # Шаг 3: Находим собственные векторы для наименьших собственных значений
        if hasattr(eigh, 'subset_by_index'):
            eigenvalues, eigenvectors = eigh(
                L_norm, 
                subset_by_index=[1, self.n_clusters]  # пропускаем первое (0)
            )
        else:
            # Старые версии
            eigenvalues, eigenvectors = eigh(L_norm)
            eigenvectors = eigenvectors[:, 1:self.n_clusters+1]
        
        # Шаг 4: Нормализуем строки собственных векторов
        rows_norm = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
        rows_norm[rows_norm == 0] = 1  # избегаем деления на ноль
        U = eigenvectors / rows_norm
        
        # Шаг 5: Применяем k-means к строкам матрицы собственных векторов
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.labels_ = kmeans.fit_predict(U)
        
        return self
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_