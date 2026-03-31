import numpy as np
from core.magnitude import compute_zeta

def iterative_normalization(X, t=1.0, max_iter=50, tol=1e-6):
    """
    Итеративный алгоритм нормализации (Андреева)
    """
    Z = compute_zeta(X, t)
    n = Z.shape[0]
    w = np.ones(n)  
    for _ in range(max_iter):
        G = Z @ w  
        w_new = w / G 
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    
    return np.sum(w)  

#Greedy 

def greedy_algorithm(X, t=1.0, max_points=100):
    """
    Смотрим на точки с максимальным расстоянием от уже выбранных и добавляем
    """
    n = X.shape[0]
    if n <= max_points:
        return X
    
    # Выбор точек с максимальным расстоянием до уже выбранных
    selected = [0] 
    dists = np.linalg.norm(X - X[0], axis=1)
    
    for _ in range(1, max_points):
        next_idx = np.argmax(dists)
        selected.append(next_idx)
        new_dists = np.linalg.norm(X - X[next_idx], axis=1)
        dists = np.minimum(dists, new_dists)
    
    return X[selected]