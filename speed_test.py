import sys, os, time, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
    
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from core.hybrid_solver import HybridMagnitudeSolver
    from core.magnitude import compute_zeta
    print("Импорт успешен")
except ImportError as e:
    print(f"[!] {e}"); sys.exit(1)


def magnitude_direct(X, t):
    Z = compute_zeta(X, t=t)
    try:
        return float(np.sum(np.linalg.solve(Z, np.ones(len(X)))))
    except np.linalg.LinAlgError:
        return float("nan")

def bench(fn, n_rep=7):
    times = []
    for _ in range(n_rep):
        t0 = time.perf_counter(); fn(); times.append(time.perf_counter()-t0)
    return float(np.median(times))  

def run_hybrid(X, t, n_rep=7):
    solver = HybridMagnitudeSolver(t=t, use_curvature=False)
    tm = bench(lambda: solver.fit(X), n_rep)
    solver.fit(X)
    info = solver.get_block_structure_info()
    mag  = solver.magnitude()
    return tm, mag, info

def rel_error(val, ref):
    return abs(val - ref) / abs(ref) if abs(ref) > 1e-12 else float("nan")

def grid_experiment(sizes, t_values, n_rep=7, seed=42):
    rng = np.random.default_rng(seed)
    records = []

    print(f"\n{'n':>6}", end="")
    for t in t_values:
        print(f"  t={t:4.1f}", end="")
    print()
    print("-" * (6 + 9 * len(t_values)))

    for n in sizes:
        X = rng.standard_normal((n, 2))
        print(f"{n:>6}", end="", flush=True)

        for t in t_values:
            tm_dir = bench(lambda: magnitude_direct(X, t), n_rep)
            mag_dir = magnitude_direct(X, t)

            tm_hyb, mag_hyb, info = run_hybrid(X, t, n_rep)

            sp = tm_dir / tm_hyb if tm_hyb > 0 else 0.0
            err = rel_error(mag_hyb, mag_dir)
            n_comp = info["n_components"]
            max_sz = max(info["component_sizes"])

            marker = "✓" if sp >= 1.0 else " "
            print(f"{sp:4.2f}х{marker}", end="", flush=True)

            records.append({
                "n": n, "t": t,
                "Direct": tm_dir, "Hybrid": tm_hyb,
                "Speedup": sp, "Err": err,
                "n_comp": n_comp, "max_comp": max_sz,
            })
        print()

    return pd.DataFrame(records)

def find_optimal_t(df):
    print(f"\n{'='*55}")
    print("Оптимальный t (макс. ускорение) для каждого n:")
    print(f"{'='*55}")
    print(f"{'n':>6} | {'t_opt':>6} | {'Speedup':>8} | "
          f"{'Err':>8} | {'N_comp':>7}")
    print("-"*55)
    for n, grp in df.groupby("n"):
        best = grp.loc[grp["Speedup"].idxmax()]
        marker = " ✓" if best["Speedup"] >= 1.0 else " ✗"
        print(f"{n:>6} | {best['t']:>6.1f} | {best['Speedup']:>7.2f}×{marker} | "
              f"{best['Err']:>8.2e} | {int(best['n_comp']):>7d}")

def plot_heatmap(df):
    pivot_sp  = df.pivot(index="n", columns="t", values="Speedup")
    pivot_err = df.pivot(index="n", columns="t", values="Err")
    pivot_nc  = df.pivot(index="n", columns="t", values="n_comp")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#0d0d14")
    fig.suptitle("Hybrid vs Direct",
                 color="white", fontsize=13, fontweight="bold")

    def plot_panel(ax, data, title, fmt, cmap, vmin=None, vmax=None):
        ax.set_facecolor("#1a1a26")
        im = ax.imshow(data.values, aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels([f"{t}" for t in data.columns],
                           fontsize=8, color="#bbbbcc")
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels([str(n) for n in data.index],
                           fontsize=8, color="#bbbbcc")
        ax.set_xlabel("t", color="#bbbbcc", fontsize=9)
        ax.set_ylabel("n", color="#bbbbcc", fontsize=9)
        ax.set_title(title, color="white", fontsize=10, fontweight="bold")
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a3a")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data.values[i, j]
                if not np.isnan(val):
                    txt = fmt.format(val)
                    color = "black" if val > (data.values.max() * 0.6) else "white"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=7, color=color)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(
            colors="#bbbbcc", labelsize=7)

    norm_sp = TwoSlopeNorm(vmin=pivot_sp.values.min(),
                           vcenter=1.0,
                           vmax=max(pivot_sp.values.max(), 1.01))
    ax0 = axes[0]; ax0.set_facecolor("#1a1a26")
    im0 = ax0.imshow(pivot_sp.values, aspect="auto",
                     cmap="RdYlGn", norm=norm_sp)
    ax0.set_xticks(range(len(pivot_sp.columns)))
    ax0.set_xticklabels([f"{t}" for t in pivot_sp.columns],
                        fontsize=8, color="#bbbbcc")
    ax0.set_yticks(range(len(pivot_sp.index)))
    ax0.set_yticklabels([str(n) for n in pivot_sp.index],
                        fontsize=8, color="#bbbbcc")
    ax0.set_xlabel("t", color="#bbbbcc", fontsize=9)
    ax0.set_ylabel("n", color="#bbbbcc", fontsize=9)
    ax0.set_title("Ускорение Hybrid/Direct (зелёный > 1х)",
                  color="white", fontsize=10, fontweight="bold")
    for sp in ax0.spines.values(): sp.set_edgecolor("#2a2a3a")
    for i in range(pivot_sp.shape[0]):
        for j in range(pivot_sp.shape[1]):
            val = pivot_sp.values[i, j]
            if not np.isnan(val):
                marker = "ok" if val >= 1.0 else ""
                ax0.text(j, i, f"{val:.2f}×{marker}",
                         ha="center", va="center", fontsize=7, color="black")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04).ax.tick_params(
        colors="#bbbbcc", labelsize=7)

    plot_panel(axes[1], pivot_err, "Погрешность Hybrid",
               "{:.2f}", "YlOrRd")
    plot_panel(axes[2], pivot_nc,  "Число компонент",
               "{:.0f}", "YlGnBu")

    plt.tight_layout()
    plt.savefig("hybrid_useful_heatmap.png", dpi=150,
                bbox_inches="tight", facecolor="#0d0d14")
    print("\nГрафик: hybrid_useful_heatmap.png")

if __name__ == "__main__":
    SIZES   = [200, 300, 500, 700, 1000, 1500, 2000]
    T_VALS  = [2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    N_REP   = 7

    print("Сетка n x t: показано ускорение Hybrid/Direct")
    print("ok = Hybrid быстрее Direct")

    df = grid_experiment(SIZES, T_VALS, n_rep=N_REP)
    df.to_csv("hybrid_grid.csv", index=False)

    find_optimal_t(df)
    plot_heatmap(df)

    green = df[df["Speedup"] >= 1.0]
    if green.empty:
        print("\n Hybrid не быстрее Direct ни в одной точке сетки.")
        print("Рекомендация: увеличить n > 2000 или оптимизировать")
        print("реализацию (батчинг мелких компонент).")
    else:
        print(f"\nHybrid быстрее Direct в {len(green)}/{len(df)} точках сетки.")
        best = df.loc[df["Speedup"].idxmax()]
        print(f"Лучший результат: n={int(best['n'])}, t={best['t']:.1f}, "
              f"speedup={best['Speedup']:.2f}×, err={best['Err']:.2e}")