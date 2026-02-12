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
    """Тестирование гипотезы о блочно-диагональной структуре при больших t"""
    print("=" * 60)
    print("Тест 1: Блочно-диагональная структура матрицы ζ при больших t")
    print("=" * 60)
    
    # Генерируем иерархические кластеры
    X, y_true = generate_hierarchical_clusters(n_points=500, random_state=42)
    
    t_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    for t in t_values:
        solver = HybridMagnitudeSolver(t=t, threshold=0.1)
        solver.fit(X)
        info = solver.get_block_structure_info()
        
        print(f"t = {t:4.1f} | компоненты: {info['n_components']:2d} | "
              f"размеры: {info['component_sizes'][:3]}... | "
              f"порог: {info['threshold_used']:.2f}")
    
    # Визуализация (без блок-скор)
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=30, alpha=0.6)
    plt.title('Исходные данные (иерархические кластеры)')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.tight_layout()
    plt.savefig('block_diagonal_analysis.png', dpi=150, bbox_inches='tight')
    print("\nГрафик сохранён: block_diagonal_analysis.png")
    plt.close()

def compare_magnitude_methods():
    """Сравнение методов вычисления магнитуды"""
    print("\n" + "=" * 60)
    print("Тест 2: Сравнение методов вычисления магнитуды")
    print("=" * 60)
    
    X, _ = generate_gaussian_clusters(n_points=300, n_clusters=3, random_state=42)
    t = 2.0
    
    # Точный метод (базовый)
    mag_exact, time_exact = measure_time(magnitude_exact, X, t=t)
    
    # Итеративная нормализация (Андреева)
    mag_iter, time_iter = measure_time(iterative_normalization, X, t=t)
    
    # Гибридный подход
    solver = HybridMagnitudeSolver(t=t, threshold=0.1)
    _, time_hybrid = measure_time(solver.fit, X)
    mag_hybrid = solver.magnitude()
    
    # Вывод результатов
    print(f"\nПараметры: n={X.shape[0]}, t={t}")
    print(f"{'Метод':<25} {'Магнитуда':<15} {'Время (с)':<12} {'Ошибка':<10}")
    print("-" * 60)
    print(f"{'Точный (инверсия)':<25} {mag_exact:<15.4f} {time_exact:<12.4f} {'-':<10}")
    print(f"{'Итеративная норм.':<25} {mag_iter:<15.4f} {time_iter:<12.4f} {abs(mag_iter-mag_exact)/mag_exact:<10.2%}")
    print(f"{'Гибридный подход':<25} {mag_hybrid:<15.4f} {time_hybrid:<12.4f} {abs(mag_hybrid-mag_exact)/mag_exact:<10.2%}")

def compare_clustering_methods():
    """Сравнение методов кластеризации"""
    print("\n" + "=" * 60)
    print("Тест 3: Сравнение методов кластеризации")
    print("=" * 60)
    
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
        
        # Обработка случая с неизвестным числом кластеров (DBSCAN)
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
    compare_magnitude_methods()
    compare_clustering_methods()
    print("\n✅ Все тесты выполнены успешно!")