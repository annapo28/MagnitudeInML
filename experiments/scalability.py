import numpy as np
import matplotlib.pyplot as plt
from utils.data_generation import generate_gaussian_clusters
from utils.metrics import measure_time
from core.hybrid_solver import HybridMagnitudeSolver
from core.andreeva_methods import iterative_normalization
from core.magnitude import magnitude_exact

def scalability_test():
    sizes = [100, 200, 500, 1000, 2000]  
    t = 2.0
    
    results = {
        'exact': {'times': [], 'magnitudes': []},
        'iterative': {'times': [], 'magnitudes': []},
        'hybrid': {'times': [], 'magnitudes': [], 'n_components': []}
    }
    
    for n in sizes:
        X, _ = generate_gaussian_clusters(n_points=n, n_clusters=5, random_state=42)
        
        if n <= 1000:
            mag_exact, time_exact = measure_time(magnitude_exact, X, t=t)
            results['exact']['times'].append(time_exact)
            results['exact']['magnitudes'].append(mag_exact)
            print(f"Точный метод: {time_exact:.4f} с, магнитуда = {mag_exact:.4f}")
        else:
            results['exact']['times'].append(np.nan)
            results['exact']['magnitudes'].append(np.nan)
            print(f"Точный метод: пропущен (слишком медленно)")

        mag_iter, time_iter = measure_time(iterative_normalization, X, t=t)
        results['iterative']['times'].append(time_iter)
        results['iterative']['magnitudes'].append(mag_iter)
        print(f"Итеративная норм.: {time_iter:.4f} с, магнитуда = {mag_iter:.4f}")
        
        solver = HybridMagnitudeSolver(t=t, threshold=0.1)
        _, time_hybrid = measure_time(solver.fit, X)
        mag_hybrid = solver.magnitude()
        info = solver.get_block_structure_info()
        
        results['hybrid']['times'].append(time_hybrid)
        results['hybrid']['magnitudes'].append(mag_hybrid)
        results['hybrid']['n_components'].append(info['n_components'])
        print(f"Гибридный подход: {time_hybrid:.4f} с, магнитуда = {mag_hybrid:.4f}, "
              f"компоненты = {info['n_components']}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    valid_sizes_exact = [s for s, t in zip(sizes, results['exact']['times']) if not np.isnan(t)]
    valid_times_exact = [t for t in results['exact']['times'] if not np.isnan(t)]
    
    if valid_times_exact:
        plt.plot(valid_sizes_exact, valid_times_exact, 'o-', label='Точный метод', linewidth=2)
    plt.plot(sizes, results['iterative']['times'], 's-', label='Итеративная норм.', linewidth=2)
    plt.plot(sizes, results['hybrid']['times'], '^-', label='Гибридный подход', linewidth=2)
    
    plt.xlabel('Размер данных (n)')
    plt.ylabel('Время выполнения (с)')
    plt.title('Масштабируемость методов')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  

    plt.subplot(1, 2, 2)
    plt.plot(sizes, results['hybrid']['n_components'], 'o-', linewidth=2, markersize=8)
    plt.xlabel('Размер данных (n)')
    plt.ylabel('Число локальных компонент')
    plt.title('Структура разложения (гибридный подход)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scalability_comparison.png', dpi=150, bbox_inches='tight')
    print("\nГрафик сохранён: scalability_comparison.png")
    plt.close()
    
    print(f"{'n':<8} {'Точный (с)':<15} {'Итеративный (с)':<18} {'Гибридный (с)':<18} {'Компоненты':<12}")
    print("=" * 70)
    for i, n in enumerate(sizes):
        exact_t = f"{results['exact']['times'][i]:.4f}" if not np.isnan(results['exact']['times'][i]) else "N/A"
        iter_t = f"{results['iterative']['times'][i]:.4f}"
        hybrid_t = f"{results['hybrid']['times'][i]:.4f}"
        comps = results['hybrid']['n_components'][i]
        print(f"{n:<8} {exact_t:<15} {iter_t:<18} {hybrid_t:<18} {comps:<12}")

if __name__ == "__main__":
    scalability_test()