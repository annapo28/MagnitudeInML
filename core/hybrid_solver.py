import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from core.magnitude import compute_zeta, magnitude_weights

class HybridMagnitudeSolver:
    """
    Гибридный подход с корректным комбинированием весов
    """
    
    def __init__(self, t=2.0, threshold=0.1):
        self.t = t
        self.threshold = threshold
        self.local_components = None
    
    def _find_local_components(self, Z):
        """Поиск локальных компонент через разреженный граф"""
        # Более умный порог: адаптивный на основе распределения весов
        if self.threshold is None:
            # Используем 10-й перцентиль как порог
            weights = Z[np.triu_indices(Z.shape[0], k=1)]
            self.threshold = np.percentile(weights, 10)
        
        adj_matrix = (Z > self.threshold).astype(int)
        n_components, labels = connected_components(
            csgraph=adj_matrix, 
            directed=False, 
            return_labels=True
        )
        
        components = [[] for _ in range(n_components)]
        for idx, label in enumerate(labels):
            components[label].append(idx)
        
        return components, labels
    
    def fit(self, X):
        # Шаг 1: Вычисляем матрицу ζ
        Z = compute_zeta(X, t=self.t)
        # После вычисления Z:
        self.block_score_ = self.compute_block_score(Z)
        
        # Шаг 2: Находим локальные компоненты
        self.local_components, _ = self._find_local_components(Z)
        
        n = X.shape[0]
        w_final = np.zeros(n)
        
        # Шаг 3: Для каждой компоненты вычисляем локальные веса
        # ВАЖНО: решаем ζw = 1 для подматрицы компоненты БЕЗ нормализации суммы
        for component in self.local_components:
            if len(component) == 0:
                continue
            
            # Извлекаем подматрицу для компоненты
            Z_local = Z[np.ix_(component, component)]
            
            # Решаем ζ_local * w_local = 1 (локальная система)
            try:
                w_local = np.linalg.solve(Z_local, np.ones(len(component)))
            except np.linalg.LinAlgError:
                w_local = np.linalg.lstsq(Z_local, np.ones(len(component)), rcond=None)[0]
            
            # Присваиваем веса точкам компоненты
            for idx, weight in zip(component, w_local):
                w_final[idx] = weight
        
        # Шаг 4: Нормализация глобальной системы (опционально для улучшения точности)
        # Проверяем невязку ζw - 1 и корректируем
        residual = Z @ w_final - np.ones(n)
        if np.linalg.norm(residual) > 1e-3:
            # Простая коррекция: масштабируем веса
            scale = 1.0 / np.mean(Z @ w_final)
            w_final *= scale
        
        self.weights_ = w_final
        self.magnitude_ = np.sum(w_final)
        return self
    
    def magnitude(self):
        return self.magnitude_
    
    def get_block_structure_info(self):
        component_sizes = [len(c) for c in self.local_components]
        return {
            'n_components': len(self.local_components),
            'component_sizes': component_sizes,
            'threshold_used': self.threshold
        }
    
    # Добавьте этот метод в класс HybridMagnitudeSolver
    def compute_block_score(self, Z):
        """Вычисление степени блочно-диагональной структуры матрицы"""
        n = Z.shape[0]
        if n == 0:
            return 0.0
        
        # Сортируем точки по первой координате для приближённого выявления блоков
        sorted_idx = np.argsort(np.sum(Z, axis=1))
        Z_sorted = Z[np.ix_(sorted_idx, sorted_idx)]
        
        # Делим матрицу на блоки размером ~20% от диагонали
        block_size = max(10, n // 5)
        diagonal_weight = 0.0
        total_weight = np.sum(Z_sorted)
        
        for i in range(0, n, block_size):
            for j in range(0, n, block_size):
                if abs(i - j) < block_size * 1.5:  # учитываем соседние блоки
                    block = Z_sorted[i:min(i+block_size, n), j:min(j+block_size, n)]
                    diagonal_weight += np.sum(block)
        
        return diagonal_weight / total_weight if total_weight > 0 else 0.0

    # Измените метод get_block_structure_info():
    def get_block_structure_info(self):
        component_sizes = [len(c) for c in self.local_components]
        return {
            'n_components': len(self.local_components),
            'component_sizes': component_sizes,
            'threshold_used': self.threshold,
            'block_score': self.block_score_  # Теперь доступен!
        }