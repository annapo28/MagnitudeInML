import numpy as np
import matplotlib.pyplot as plt
from utils.data_generation import generate_gaussian_clusters, generate_hierarchical_clusters
from utils.metrics import measure_time, clustering_metrics
from core.hybrid_solver import HybridMagnitudeSolver
from core.andreeva_methods import iterative_normalization
from core.magnitude import magnitude_exact
from clustering.magnitude_clustering import MagnitudeClusterer
from clustering.spectral_clustering import SpectralClusterer
from sklearn.cluster import KMeans, DBSCAN

def test_block_diagonal_structure():
    print("Тест 1: Блочно-диагональная структура матрицы схожести при больших t")
    
    X, y_true = generate_hierarchical_clusters(n_points=500, random_state=42)
    t_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    for t in t_values:
        solver = HybridMagnitudeSolver(t=t, threshold=0.1, use_curvature=False)
        solver.fit(X)
        info = solver.get_block_structure_info()
        
        print(f"t = {t:4.1f} | компоненты: {info['n_components']:2d} | "
              f"размеры: {info['component_sizes'][:3]}... | "
              f"блок-скор: {info['block_score']:.3f}")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=30, alpha=0.6)
    plt.title('Исходные данные (иерархические кластеры)')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.tight_layout()
    plt.savefig('block_diagonal_analysis.png', dpi=150, bbox_inches='tight')
    print("\nГрафик сохранён: block_diagonal_analysis.png")
    plt.close()

def compare_magnitude_methods_with_curvature():
    """Сравнение методов вычисления магнитуды с коррекцией кривизны"""
    print("Тест 2: Сравнение методов вычисления магнитуды (с коррекцией кривизны)")
    
    X, _ = generate_gaussian_clusters(n_points=300, n_clusters=3, random_state=42)
    t = 2.0
    
    mag_exact, time_exact = measure_time(lambda: magnitude_exact(X, t=t))
    
    mag_iter, time_iter = measure_time(lambda: iterative_normalization(X, t=t))
    
    solver_base = HybridMagnitudeSolver(t=t, threshold=0.1, use_curvature=False)
    _, time_hybrid_base = measure_time(lambda: solver_base.fit(X))
    mag_hybrid_base = solver_base.magnitude()
    
    solver_curv = HybridMagnitudeSolver(t=t, threshold=0.1, use_curvature=True, alpha=0.1)
    _, time_hybrid_curv = measure_time(lambda: solver_curv.fit(X))
    mag_hybrid_curv = solver_curv.magnitude()
    
    print(f"\nПараметры: n={X.shape[0]}, t={t}")
    print(f"{'Метод':<35} {'Магнитуда':<15} {'Время (с)':<12} {'Ошибка':<10}")
    print("-" * 70)
    print(f"{'Точный (инверсия)':<35} {mag_exact:<15.4f} {time_exact:<12.4f} {'-':<10}")
    print(f"{'Итеративная норм. (Андреева)':<35} {mag_iter:<15.4f} {time_iter:<12.4f} {abs(mag_iter-mag_exact)/mag_exact:<10.2%}")
    print(f"{'Гибридный (без кривизны)':<35} {mag_hybrid_base:<15.4f} {time_hybrid_base:<12.4f} {abs(mag_hybrid_base-mag_exact)/mag_exact:<10.2%}")
    print(f"{'Гибридный (с кривизной)':<35} {mag_hybrid_curv:<15.4f} {time_hybrid_curv:<12.4f} {abs(mag_hybrid_curv-mag_exact)/mag_exact:<10.2%}")
    
    info_curv = solver_curv.get_block_structure_info()
    print(f"\nСтруктура при коррекции кривизны:")
    print(f"Число компонент: {info_curv['n_components']}")
    print(f"Блочная структура: {info_curv['block_score']:.3f}")
    if 'curvatures' in info_curv:
        print(f"Средняя кривизна: {np.mean(info_curv['curvatures']):.3f}")
        print(f"Макс. кривизна: {np.max(info_curv['curvatures']):.3f}")

def compare_clustering_methods():
    """Сравнение методов кластеризации"""
    print("Тест 3: Сравнение методов кластеризации")
    
    X, y_true = generate_gaussian_clusters(n_points=500, n_clusters=3, random_state=42)
    n_clusters = 3
    
    methods = {
        'Magnitude': MagnitudeClusterer(t=1.0),
        'Spectral': SpectralClusterer(n_clusters=n_clusters, gamma=1.0),
        'KMeans': KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
        'DBSCAN': DBSCAN(eps=1.5, min_samples=5)
    }
    
    print(f"\n{'Метод':<15} {'Время (с)':<12} {'ARI':<10} {'NMI':<10} {'Silhouette':<12}")
    print("-" * 60)
    
    for name, clusterer in methods.items():
        labels, time_taken = measure_time(clusterer.fit_predict, X)
        
        if name == 'DBSCAN':
            n_found = len(set(labels)) - (1 if -1 in labels else 0)
            if n_found < 2:
                print(f"{name:<15} {time_taken:<12.4f} {'N/A':<10} {'N/A':<10} {'N/A':<12}")
                continue
        
        metrics = clustering_metrics(y_true, labels, X)
        print(f"{name:<15} {time_taken:<12.4f} {metrics['ARI']:<10.3f} "
              f"{metrics['NMI']:<10.3f} {metrics['silhouette']:<12.3f}")

if __name__ == "__main__":
    test_block_diagonal_structure()
    compare_magnitude_methods_with_curvature()
    compare_clustering_methods()
    print("\nВсе тесты выполнены успешно")