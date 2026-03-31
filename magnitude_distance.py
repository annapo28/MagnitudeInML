"""
Torkamani, Gouk, Sarkar (2026). "Magnitude Distance: A Geometric Measure of Dataset Similarity"
arXiv:2602.08859
"""
import numpy as np, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

def similarity_matrix(X, t=1.0):
    D = cdist(X, X, metric='euclidean')
    return np.exp(-t * D)

def magnitude(X, t=1.0):
    X_unique = np.unique(X, axis=0)
    zeta = similarity_matrix(X_unique, t)
    try:
        w = np.linalg.solve(zeta, np.ones(len(X_unique)))
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(zeta, np.ones(len(X_unique)), rcond=None)[0]
    return float(np.sum(w))

def magnitude_distance(X, Y, t=1.0):
    XY = np.vstack([X, Y])
    return 2 * magnitude(XY, t) - magnitude(X, t) - magnitude(Y, t)

def normalized_magnitude_distance(X, Y, t=1.0):
    XY = np.vstack([X, Y])
    mag_union = magnitude(XY, t)
    if mag_union == 0: return 0.0
    return magnitude_distance(X, Y, t) / mag_union

def mmd_squared(X, Y, t=1.0):
    n, m = len(X), len(Y)
    K_XX = similarity_matrix(X, t)
    K_YY = similarity_matrix(Y, t)
    K_XY = np.exp(-t * cdist(X, Y, metric='euclidean'))
    return (K_XX.sum() / n**2 + K_YY.sum() / m**2 - 2 * K_XY.sum() / (n * m))

def sliced_wasserstein(X, Y, n_projections=50, seed=42):
    rng = np.random.default_rng(seed)
    D = X.shape[1]
    directions = rng.standard_normal((n_projections, D))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    distances = []
    for d in directions:
        px = X @ d
        py = Y @ d
        distances.append(wasserstein_distance(px, py))
    return float(np.mean(distances))

def experiment_figure1(n_samples=200, D=100, n_trials=3, seed=0):
    rng = np.random.default_rng(seed)
    t_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    colors = ['blue', 'purple', 'gold', 'red', 'green', 'darkviolet']
    mu_range = np.linspace(0, 10, 15)
    results_norm = {t: [] for t in t_values}
    results_std  = {t: [] for t in t_values}
    for mu in mu_range:
        for t in t_values:
            vals_n, vals_s = [], []
            for _ in range(n_trials):
                X = rng.standard_normal((n_samples, D))
                Y = rng.standard_normal((n_samples, D)) + mu
                vals_n.append(normalized_magnitude_distance(X, Y, t))
                vals_s.append(magnitude_distance(X, Y, t))
            results_norm[t].append(np.mean(vals_n))
            results_std[t].append(np.mean(vals_s))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [f"t = {t}" for t in t_values]
    for t, color, label in zip(t_values, colors, labels):
        axes[0].plot(mu_range, results_norm[t], color=color, label=label)
        axes[1].plot(mu_range, results_std[t],  color=color, label=label)
    for ax, title, ylabel in zip(axes, ["(a) Normalized", "(b) Standard"],
                                  ["Normalized Magnitude Distance", "Magnitude Distance"]):
        ax.set_xlabel("Mean Value of Second Gaussian Distribution")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Figure 1 (reproduction): Impact of scaling parameter t\nN(0,1) vs N(μ,1) in {D}D, {n_samples} samples each", fontsize=11)
    plt.tight_layout()
    plt.savefig("figure1_mean_shift.png", dpi=150)
    plt.close()
    print("Figure 1 сохранён.")

def experiment_figure2(n_samples=500, n_trials=5, seed=1):
    rng = np.random.default_rng(seed)
    dimensions = [2, 5, 10, 20, 50, 100, 200, 500]
    configs = {
        "MMD sigma=1": lambda X, Y, D: np.sqrt(max(mmd_squared(X, Y, t=1.0), 0)),
        "MMD sigma=1/sqrt(D)": lambda X, Y, D: np.sqrt(max(mmd_squared(X, Y, t=1/np.sqrt(D)), 0)),
        "Wasserstein": lambda X, Y, D: sliced_wasserstein(X, Y),
        "Mag t=1/D": lambda X, Y, D: normalized_magnitude_distance(X, Y, t=1/D),
        "Mag t=1/√D": lambda X, Y, D: normalized_magnitude_distance(X, Y, t=1/np.sqrt(D)),
        "Mag t=0.01":  lambda X, Y, D: normalized_magnitude_distance(X, Y, t=0.01),
        "Mag t=0.1": lambda X, Y, D: normalized_magnitude_distance(X, Y, t=0.1),
    }
    results = {k: [] for k in configs}
    for D in dimensions:
        print(f"  D={D}...")
        for name, fn in configs.items():
            vals = []
            for _ in range(n_trials):
                X = rng.standard_normal((n_samples, D))
                Y = rng.standard_normal((n_samples, D)) + 2.0
                vals.append(fn(X, Y, D))
            results[name].append(np.mean(vals))
    fig, ax = plt.subplots(figsize=(9, 5))
    styles = {
        "MMD sigma=1": ("C0", "--"),
        "MMD sigma=1/sqrt(D)": ("C1", "--"),
        "Wasserstein": ("C2", "-"),
        "Mag t=1/D": ("C3", "-"),
        "Mag t=1/√D": ("C4", "-"),
        "Mag t=0.01": ("C5", ":"),
        "Mag t=0.1": ("C6", ":"),
    }
    for name, vals in results.items():
        v = np.array(vals, dtype=float)
        v0 = v[0] if v[0] != 0 else 1.0
        color, ls = styles[name]
        ax.plot(dimensions, v / v0, label=name, color=color, linestyle=ls, marker='o', markersize=4)
    ax.set_xlabel("Dimension D")
    ax.set_ylabel("Distance (normalized to D=2)")
    ax.set_title("Figure 2 (reproduction): Mean Distance vs Dimension\nN(0,1) vs N(2,1), 500 samples, mean shift=2")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figure2_high_dim.png", dpi=150)
    plt.close()
    print("Figure 2 сохранён.")

def experiment_outlier_robustness(n_samples=300, D=2, n_trials=5, seed=2):
    rng = np.random.default_rng(seed)
    epsilons = [0.01, 0.05, 0.1]
    radii = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    t_mag = 0.001
    fig, ax = plt.subplots(figsize=(9, 5))
    colors_wass = ['#d62728', '#ff7f0e', '#8c564b']
    colors_mag  = ['#1f77b4', '#2ca02c', '#9467bd']
    for eps, cw, cm in zip(epsilons, colors_wass, colors_mag):
        wass_vals, mag_vals = [], []
        n_outliers = max(1, int(n_samples * eps))
        n_clean = n_samples - n_outliers
        for R in radii:
            wass_r, mag_r = [], []
            for _ in range(n_trials):
                P = rng.standard_normal((n_samples, D))
                clean_part   = rng.standard_normal((n_clean, D))
                outlier_dirs = rng.standard_normal((n_outliers, D))
                outlier_dirs /= np.linalg.norm(outlier_dirs, axis=1, keepdims=True)
                outliers = outlier_dirs * R
                Q = np.vstack([clean_part, outliers])
                wass_r.append(sliced_wasserstein(P, Q))
                mag_r.append(normalized_magnitude_distance(P, Q, t=t_mag))
            wass_vals.append(np.mean(wass_r))
            mag_vals.append(np.mean(mag_r))
        ax.plot(radii, wass_vals, color=cw, linestyle='--', label=f"Wass eps={eps}", marker='s', markersize=4)
        ax.plot(radii, mag_vals,  color=cm, linestyle='-', label=f"Mag eps={eps}",  marker='o', markersize=4)
    ax.set_xscale('log')
    ax.set_xlabel("Outlier Radius R (log scale)")
    ax.set_ylabel("Distance")
    ax.set_title(f"Figure 5 (reproduction): Outlier Robustness\nHuber contamination, t={t_mag}, D={D}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig("figure5_outlier_robustness.png", dpi=150)
    plt.close()
    print("Figure 5 сохранён.")

def experiment_limiting_behavior(seed=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((10, 2))
    Y = rng.standard_normal((10, 2)) + 3.0
    sym_diff_size = len(X) + len(Y)
    t_values = np.logspace(-2, 2, 60)
    distances = [magnitude_distance(X, Y, t) for t in t_values]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(t_values, distances, 'b-', linewidth=2, label="$d^t_{Mag}(X,Y)$")
    ax.axhline(sym_diff_size, color='red', linestyle='--', label=f"|X△Y| = {sym_diff_size}")
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Scale parameter t (log scale)")
    ax.set_ylabel("Magnitude Distance")
    ax.set_title("Theorem 5.3 (reproduction): Limiting behavior of magnitude distance\n→ 0 as t→0,  → |X△Y| as t→∞")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("theorem53_limiting.png", dpi=150)
    plt.close()
    print("Theorem 5.3 сохранён.")

def magnitude_as_evaluation_metric(X_test, y_true, y_pred, task='regression'):
    if task == 'regression':
        errors = y_pred - y_true
        errors_norm = errors / (np.std(errors) + 1e-8)
        zero_errors = np.zeros_like(errors_norm)
        return normalized_magnitude_distance(errors_norm.reshape(-1, 1), zero_errors.reshape(-1, 1), t=1.0)
    elif task == 'classification':
        return normalized_magnitude_distance(y_pred, y_true, t=1.0)

def experiment_mag_distance_as_metric(seed=42):
    rng = np.random.default_rng(seed)
    print("\n" + "="*55)
    print("Эксперимент: Magnitude Distance как метрика модели")
    print("="*55)
    print("\n[A] Регрессия: корреляция dMag с MSE/MAE")
    n_samples = 500
    n_models = 20
    mse_values, mae_values, mag_values = [], [], []
    for i in range(n_models):
        noise_scale = 0.1 + i * 0.3
        y_true = rng.standard_normal(n_samples)
        y_pred = y_true + rng.standard_normal(n_samples) * noise_scale
        mse = np.mean((y_pred - y_true)**2)
        mae = np.mean(np.abs(y_pred - y_true))
        y_pred_norm = (y_pred - np.mean(y_pred)) / (np.std(y_pred) + 1e-8)
        y_true_norm = (y_true - np.mean(y_true)) / (np.std(y_true) + 1e-8)
        mag_dist = normalized_magnitude_distance(y_pred_norm.reshape(-1, 1), y_true_norm.reshape(-1, 1), t=1.0)
        mse_values.append(mse)
        mae_values.append(mae)
        mag_values.append(mag_dist)
    corr_mag_mse = np.corrcoef(mag_values, mse_values)[0, 1]
    corr_mag_mae = np.corrcoef(mag_values, mae_values)[0, 1]
    print(f"Корреляция dMag - MSE: {corr_mag_mse:.3f}")
    print(f"Корреляция dMag - MAE: {corr_mag_mae:.3f}")
    print("\n[B] Систематические vs случайные ошибки")
    y_true = rng.standard_normal(n_samples)
    y_pred_random = y_true + rng.standard_normal(n_samples) * 2.0
    y_pred_systematic = y_true.copy()
    mask = rng.random(n_samples) < 0.3
    y_pred_systematic[mask] += 5.0
    mse_random = np.mean((y_pred_random - y_true)**2)
    mse_systematic = np.mean((y_pred_systematic - y_true)**2)
    mag_random = normalized_magnitude_distance(y_pred_random.reshape(-1, 1), y_true.reshape(-1, 1), t=1.0)
    mag_systematic = normalized_magnitude_distance(y_pred_systematic.reshape(-1, 1), y_true.reshape(-1, 1), t=1.0)
    print(f"Случайные ошибки: MSE={mse_random:.3f}, dMag={mag_random:.3f}")
    print(f"Систематические ошибки: MSE={mse_systematic:.3f}, dMag={mag_systematic:.3f}")
    print(f"dMag выше для систематических: {mag_systematic > mag_random}")
    print("\n[C] Устойчивость к выбросам в предсказаниях")
    y_true = rng.standard_normal(n_samples)
    y_pred_clean = y_true + rng.standard_normal(n_samples) * 0.5
    y_pred_outliers = y_pred_clean.copy()
    outlier_idx = rng.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    y_pred_outliers[outlier_idx] += rng.choice([-1, 1], size=len(outlier_idx)) * 20
    mse_clean = np.mean((y_pred_clean - y_true)**2)
    mse_outliers = np.mean((y_pred_outliers - y_true)**2)
    mag_clean = normalized_magnitude_distance(y_pred_clean.reshape(-1, 1), y_true.reshape(-1, 1), t=1.0)
    mag_outliers = normalized_magnitude_distance(y_pred_outliers.reshape(-1, 1), y_true.reshape(-1, 1), t=1.0)
    mse_change = (mse_outliers - mse_clean) / mse_clean * 100
    mag_change = (mag_outliers - mag_clean) / mag_clean * 100
    print(f"Без выбросов: MSE={mse_clean:.3f}, dMag={mag_clean:.3f}")
    print(f"С выбросами: MSE={mse_outliers:.3f}, dMag={mag_outliers:.3f}")
    print(f"Изменение MSE: {mse_change:+.1f}%")
    print(f"Изменение dMag: {mag_change:+.1f}%")
    print(f"dMag {'более' if abs(mag_change) < abs(mse_change) else 'менее'} устойчив к выбросам")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(mse_values, mag_values, alpha=0.7, edgecolors='black')
    axes[0].plot(np.unique(mse_values), np.poly1d(np.polyfit(mse_values, mag_values, 1))(np.unique(mse_values)), 'r--', linewidth=2)
    axes[0].set_xlabel("MSE")
    axes[0].set_ylabel("Magnitude Distance")
    axes[0].set_title(f"(A) Корреляция dMag ↔ MSE\n(r={corr_mag_mse:.2f})")
    axes[0].grid(alpha=0.3)
    axes[1].bar(['Random', 'Systematic'], [mag_random, mag_systematic], color=['steelblue', 'coral'], edgecolor='black')
    axes[1].set_ylabel("Magnitude Distance")
    axes[1].set_title("(B) Чувствительность к типу ошибок")
    axes[1].grid(alpha=0.3, axis='y')
    axes[2].bar(['Clean', 'Outliers'], [mag_clean, mag_outliers], color=['steelblue', 'coral'], edgecolor='black')
    axes[2].set_ylabel("Magnitude Distance")
    axes[2].set_title(f"(C) Устойчивость к выбросам\n(Δ={mag_change:+.1f}%)")
    axes[2].grid(alpha=0.3, axis='y')
    fig.suptitle("Magnitude Distance как метрика качества модели", fontsize=12)
    plt.tight_layout()
    plt.savefig("mag_distance_as_metric.png", dpi=150)
    plt.close()
    print("\nГрафик сохранён: mag_distance_as_metric.png")
    return {'corr_mse': corr_mag_mse, 'corr_mae': corr_mag_mae, 'mag_systematic': mag_systematic, 'mag_random': mag_random, 'mag_change_outliers': mag_change}

if __name__ == "__main__":
    print("=" * 55)
    print("Воспроизведение Torkamani et al. (2026)")
    print("=" * 55)
    print("\n[1/4] Theorem 5.3 - limiting behavior")
    experiment_limiting_behavior()
    print("\n[2/4] Figure 1 - distance vs mean shift (100D)")
    experiment_figure1()
    print("\n[3/4] Figure 2 - high-dim comparison")
    experiment_figure2()
    print("\n[4/4] Figure 5 - outlier robustness")
    experiment_outlier_robustness()
    print("\n[5/5] Magnitude distance как метрика модели")
    results = experiment_mag_distance_as_metric()
    print("\nГотово. Все графики сохранены в /mnt/user-data/outputs/")