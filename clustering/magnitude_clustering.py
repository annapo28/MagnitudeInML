import numpy as np
from sklearn.cluster import KMeans
from core.hybrid_solver import HybridMagnitudeSolver

class MagnitudeClusterer:
    """
    Кластеризация через магнитуду (адаптированный алгоритм из диссертации Андреевой)
    """
    def __init__(self, t=1.0, threshold=None):
        self.t = t
        self.threshold = threshold
        self.labels_ = None
    
    def fit(self, X):
        n = X.shape[0]
        R = set(range(n))  
        C = [{0}]          
        R.remove(0)
        
        if self.threshold is None:
            samples = min(20, n)
            changes = []
            for _ in range(samples):
                i = np.random.choice(list(R)) if R else 0
                j = np.random.choice(list(range(n)))
                Z = np.exp(-self.t * np.linalg.norm(X[i] - X[j]))
                changes.append(1.0 - Z) 
            self.threshold = np.median(changes) if changes else 0.1
        
        while R:
            best_increase = float('inf')
            best_point = None
            best_cluster = None
            
            for b in R:
                for c_idx, c in enumerate(C):
                    points_in_cluster = np.array(list(c))
                    if len(points_in_cluster) > 0:
                        avg_dist = np.mean(np.linalg.norm(X[b] - X[points_in_cluster], axis=1))
                        increase = 1.0 - np.exp(-self.t * avg_dist)
                    else:
                        increase = 1.0
                    
                    if increase < best_increase:
                        best_increase = increase
                        best_point = b
                        best_cluster = c_idx
            
            if best_increase < self.threshold and best_cluster is not None:
                C[best_cluster].add(best_point)
            else:
                C.append({best_point})
            
            R.remove(best_point)
        
        self.labels_ = np.zeros(n, dtype=int)
        for label, cluster in enumerate(C):
            for idx in cluster:
                self.labels_[idx] = label
        
        self.n_clusters_ = len(C)
        return self
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_