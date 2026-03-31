import sys
import os
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from core.magnitude import compute_zeta, magnitude_exact
from core.hybrid_solver import HybridMagnitudeSolver
from core.andreeva_methods import iterative_normalization
from utils.data_generation import generate_hierarchical_clusters
from utils.metrics import measure_time, block_diagonality_score

def quick_test():
    
    print("\n1. Генерация иерархических кластеров (500 точек)")
    X, _ = generate_hierarchical_clusters(n_points=500, random_state=42)
    print(f"{X.shape[0]} точек в {X.shape[1]}D пространстве")

    t = 2.0
    print(f"\n2. Параметры: t = {t}")
    print("\n3. Анализ блочно-диагональной структуры матрицы схожест")
    Z = compute_zeta(X, t=t)
    block_score = block_diagonality_score(Z, threshold=0.1)
    print(f"Block Diagonality Score: {block_score:.3f}")
    print(f"Интерпретация: {'Высокая блочность' if block_score > 0.6 else 'Низкая блочность'}")
    print("\n4. Сравнение методов вычисления магнитуды")
    
    mag_exact_val, time_exact = measure_time(lambda: magnitude_exact(X, t=t))
    print(f"Точный метод (инверсия): {time_exact:.4f} с, магнитуда = {mag_exact_val:.4f}")
    
    mag_iter_val, time_iter = measure_time(lambda: iterative_normalization(X, t=t))
    err_iter = abs(mag_iter_val - mag_exact_val) / mag_exact_val * 100
    print(f"Итеративная норм. (Андреева): {time_iter:.4f} с, ошибка = {err_iter:.2f}%")

    solver = HybridMagnitudeSolver(t=t, threshold=0.1)
    _, time_hybrid = measure_time(lambda: solver.fit(X))
    mag_hybrid = solver.magnitude()
    err_hybrid = abs(mag_hybrid - mag_exact_val) / mag_exact_val * 100
    info = solver.get_block_structure_info()
    print(f"Гибридный подход: {time_hybrid:.4f} с, ошибка = {err_hybrid:.2f}%, компоненты = {info['n_components']}")
    
    speedup_vs_exact = time_exact / time_hybrid if time_hybrid > 0 else 0
    speedup_vs_iter = time_iter / time_hybrid if time_hybrid > 0 else 0
    
    print(f"Ускорение гибридного подхода против точного метода: {speedup_vs_exact:.1f}x")
    print(f"против итеративной нормализации: {speedup_vs_iter:.1f}x")
    print(f"Точность гибридного подхода: {100 - err_hybrid:.2f}%")
    
    if speedup_vs_iter > 1.2 and err_hybrid < 5.0:
        print("\nгибридный подход быстрее")
    else:
        print("\nТребуется настройка параметров (threshold, t)")

if __name__ == "__main__":
    quick_test()