import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import time

def clustering_metrics(true_labels, predicted_labels, X=None):
    """Вычисление метрик качества кластеризации"""
    metrics = {
        'ARI': adjusted_rand_score(true_labels, predicted_labels),
        'NMI': normalized_mutual_info_score(true_labels, predicted_labels)
    }
    
    if X is not None:
        metrics['silhouette'] = silhouette_score(X, predicted_labels)
    
    return metrics

def measure_time(func, *args, **kwargs):
    """Измерение времени выполнения функции"""
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start

def block_diagonality_score(Z, threshold=0.1):
    """
    Оценка степени блочно-диагональной структуры матрицы.
    Возвращает долю веса в "диагональных блоках" относительно полной матрицы.
    """
    n = Z.shape[0]
    sorted_idx = np.argsort(np.sum(Z, axis=1))
    Z_sorted = Z[sorted_idx][:, sorted_idx]
    
    block_size = max(10, n // 5)
    diagonal_weight = 0.0
    total_weight = np.sum(Z_sorted)
    
    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            if abs(i - j) < block_size * 1.5: 
                block = Z_sorted[i:min(i+block_size, n), j:min(j+block_size, n)]
                diagonal_weight += np.sum(block)
    
    return diagonal_weight / total_weight if total_weight > 0 else 0.0