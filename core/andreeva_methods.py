import numpy as np
from scipy.spatial.distance import cdist
from core.magnitude import compute_zeta

def iterative_normalization(X, t=1.0, max_iter=50, tol=1e-6):
    """
    Итеративный алгоритм нормализации (Андреева)
    Правильная реализация: решаем ζw = 1 итеративно через нормализацию
    """
    Z = compute_zeta(X, t)
    n = Z.shape[0]
    w = np.ones(n) / n  # начальные веса нормализованы
    
    for _ in range(max_iter):
        # Вычисляем ζw
        Zw = Z @ w
        # Обновляем веса: w_new = w / (ζw) с нормализацией
        w_new = w / Zw
        # Нормализуем для численной стабильности
        w_new = w_new / np.sum(w_new)
        
        # Проверка сходимости
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    
    # Магнитуда = сумма весов (должна быть ~1 при нормализации, но мы возвращаем истинную)
    # Для получения истинной магнитуды: решаем ζw = 1 без нормализации суммы
    # Восстанавливаем масштаб:
    scale = 1.0 / np.mean(Z @ w)  # масштабируем чтобы ζw ≈ 1
    w_scaled = w * scale * n
    
    return np.sum(w_scaled)

def discrete_centers(X, t=1.0, max_points=100):
    """
    Алгоритм иерархии дискретных центров (Андреева)
    Выбирает репрезентативные точки для приближённого вычисления магнитуды
    """
    n = X.shape[0]
    if n <= max_points:
        return X
    
    # Простая реализация: выбор точек с максимальным расстоянием до уже выбранных
    selected = [0]  # начинаем с первой точки
    dists = cdist(X, X[[0]], metric='euclidean').flatten()
    
    for _ in range(1, max_points):
        next_idx = np.argmax(dists)
        selected.append(next_idx)
        new_dists = cdist(X, X[[next_idx]], metric='euclidean').flatten()
        dists = np.minimum(dists, new_dists)
    
    return X[selected]