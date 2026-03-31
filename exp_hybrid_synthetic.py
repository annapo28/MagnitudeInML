import sys, os, time, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.hybrid_solver import HybridMagnitudeSolver
from core.magnitude import compute_zeta, magnitude_exact
print("Ок")

def magnitude_direct(X, t):
    """Baseline: прямое решение через LU."""
    Z = compute_zeta(X, t=t)
    try:
        w = np.linalg.solve(Z, np.ones(len(X)))
        return float(np.sum(w))
    except np.linalg.LinAlgError:
        return float("nan")


def bench(fn, n_rep=5):
    """Замер времени: среднее и std по n_rep запускам."""
    times = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def run_hybrid(X, t, use_curvature, n_rep=5):
    """
    Запускает HybridMagnitudeSolver n_rep раз для замера времени,
    затем один раз явно для получения результатов.
    Возвращает: (время_среднее, std, magnitude, info)
    """
    solver = HybridMagnitudeSolver(t=t, use_curvature=use_curvature)
    tm, std = bench(lambda: solver.fit(X), n_rep)
    solver.fit(X)
    mag = solver.magnitude()
    info = solver.get_block_structure_info()
    return tm, std, mag, info


def rel_error(val, ref):
    if ref is None or np.isnan(ref) or abs(ref) < 1e-12:
        return float("nan")
    return abs(val - ref) / abs(ref)

def make_gaussian(n, d=2, seed=42):
    return np.random.default_rng(seed).standard_normal((n, d))

def make_clusters(n, n_clusters=4, d=2, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-5, 5, (n_clusters, d))
    labels = rng.integers(0, n_clusters, n)
    return centers[labels] + rng.standard_normal((n, d)) * 0.5

def make_uniform(n, d=2, seed=42):
    return np.random.default_rng(seed).uniform(-3, 3, (n, d))

def make_ring(n, d=2, seed=42):
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, n)
    r = 3.0 + rng.standard_normal(n) * 0.2
    X = np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)
    if d > 2:
        X = np.hstack([X, rng.standard_normal((n, d - 2)) * 0.1])
    return X

DATASETS = {
    "Gaussian": make_gaussian,
    "Clusters": make_clusters,
    "Uniform": make_uniform,
    "Ring": make_ring,
}

def exp1_speed_vs_n(sizes, t=5.0, n_rep=5, seed=42):
    print(f"\n{'='*72}")
    print(f"Эксперимент 1: скорость и точность vs n  (t={t})")
    print(f"{'='*72}")
    print(f"{'n':>5} | {'Direct':>8} | {'Hybrid':>8} | {'Hyb+C':>8} | "
          f"{'Sp_H':>6} | {'Err_H':>9} | {'Err_HC':>9} | {'Comp':>5}")
    print("-"*72)

    records = []
    for n in sizes:
        X = make_gaussian(n, seed=seed)

        tm_dir, _ = bench(lambda: magnitude_direct(X, t), n_rep)
        mag_dir = magnitude_direct(X, t)

        tm_hyb, _, mag_hyb, info = run_hybrid(X, t, use_curvature=False, n_rep=n_rep)
        tm_hybc, _, mag_hybc, _ = run_hybrid(X, t, use_curvature=True,  n_rep=n_rep)

        sp_h = tm_dir / tm_hyb if tm_hyb > 0 else 0.0
        err_h = rel_error(mag_hyb, mag_dir)
        err_hc = rel_error(mag_hybc, mag_dir)
        n_comp = info["n_components"]

        records.append({
            "n": n, "t": t,
            "Direct": tm_dir, "Hybrid": tm_hyb, "Hybrid+C": tm_hybc,
            "Speedup_H": sp_h,
            "Err_H": err_h, "Err_HC": err_hc,
            "n_components": n_comp,
            "block_score": info["block_score"],
            "mag_direct": mag_dir, "mag_hybrid": mag_hyb,
        })
        print(f"{n:>5} | {tm_dir:>8.4f} | {tm_hyb:>8.4f} | {tm_hybc:>8.4f} | "
              f"{sp_h:>5.2f}× | {err_h:>9.2e} | {err_hc:>9.2e} | {n_comp:>5d}")

    return pd.DataFrame(records)

def exp2_speed_vs_t(t_values, n=200, n_rep=5, seed=42):
    print(f"\n{'='*65}")
    print(f"Эксперимент 2: скорость и точность vs t  (n={n})")
    print(f"{'='*65}")
    print(f"{'t':>6} | {'Direct':>8} | {'Hybrid':>8} | "
          f"{'Speedup':>8} | {'Err_H':>9} | {'N_comp':>7} | {'BlockSc':>8}")
    print("-"*65)

    X = make_gaussian(n, seed=seed)
    records = []

    for t in t_values:
        tm_dir, _ = bench(lambda: magnitude_direct(X, t), n_rep)
        mag_dir = magnitude_direct(X, t)

        tm_hyb, _, mag_hyb, info = run_hybrid(X, t, use_curvature=False, n_rep=n_rep)

        sp_h = tm_dir / tm_hyb if tm_hyb > 0 else 0.0
        err_h = rel_error(mag_hyb, mag_dir)
        n_comp = info["n_components"]
        bscore = info["block_score"]

        records.append({
            "t": t, "n": n,
            "Direct": tm_dir, "Hybrid": tm_hyb,
            "Speedup": sp_h, "Err_H": err_h,
            "n_components": n_comp,
            "block_score": bscore,
        })
        print(f"{t:>6.1f} | {tm_dir:>8.4f} | {tm_hyb:>8.4f} | "
              f"{sp_h:>7.2f}× | {err_h:>9.2e} | {n_comp:>7d} | {bscore:>8.3f}")

    return pd.DataFrame(records)

def exp3_components_vs_t(t_values, n=300, seed=42):
    print(f"\n{'='*60}")
    print(f"Эксперимент 3: компоненты связности vs t  (n={n})")
    print(f"{'='*60}")
    print(f"{'t':>6} | {'N_comp':>7} | {'Max_sz':>7} | "
          f"{'Mean_sz':>8} | {'BlockSc':>8}")
    print("-"*60)

    X = make_gaussian(n, seed=seed)
    records = []

    for t in t_values:
        _, _, _, info = run_hybrid(X, t, use_curvature=False, n_rep=1)

        sizes = info["component_sizes"]
        n_comp = info["n_components"]
        max_sz = max(sizes)
        mean_sz = float(np.mean(sizes))
        bscore = info["block_score"]

        records.append({
            "t": t, "n_components": n_comp,
            "max_comp_size": max_sz,
            "mean_comp_size": mean_sz,
            "block_score": bscore,
        })
        print(f"{t:>6.1f} | {n_comp:>7d} | {max_sz:>7d} | "
              f"{mean_sz:>8.1f} | {bscore:>8.3f}")

    return pd.DataFrame(records)

def exp4_dataset_types(n=150, t=5.0, n_rep=5, seed=42):
    print(f"\n{'='*72}")
    print(f"Эксперимент 4: разные типы данных  (n={n}, t={t})")
    print(f"{'='*72}")
    print(f"{'Dataset':>10} | {'Direct':>8} | {'Hybrid':>8} | "
          f"{'Speedup':>8} | {'Err_H':>9} | {'N_comp':>7} | {'Mag':>8}")
    print("-"*72)

    records = []
    for name, make_fn in DATASETS.items():
        X = make_fn(n, seed=seed)

        tm_dir, _ = bench(lambda: magnitude_direct(X, t), n_rep)
        mag_dir = magnitude_direct(X, t)

        tm_hyb, _, mag_hyb, info = run_hybrid(X, t, use_curvature=False, n_rep=n_rep)

        sp_h = tm_dir / tm_hyb if tm_hyb > 0 else 0.0
        err_h = rel_error(mag_hyb, mag_dir)
        n_comp = info["n_components"]

        records.append({
            "dataset": name,
            "Direct": tm_dir, "Hybrid": tm_hyb,
            "Speedup": sp_h, "Err_H": err_h,
            "n_components": n_comp,
            "mag_direct": mag_dir,
        })
        print(f"{name:>10} | {tm_dir:>8.4f} | {tm_hyb:>8.4f} | "
              f"{sp_h:>7.2f}× | {err_h:>9.2e} | {n_comp:>7d} | {mag_dir:>8.2f}")

    return pd.DataFrame(records)

def plot_all(df1, df2, df3, df4):
    fig = plt.figure(figsize=(18, 13), facecolor="#0d0d14")
    fig.suptitle(
        "HybridMagnitudeSolver: скорость, точность, структура",
        color="white", fontsize=14, fontweight="bold", y=0.97)
    gs = fig.add_gridspec(3, 3, hspace=0.46, wspace=0.38,
                          left=0.07, right=0.97, top=0.92, bottom=0.06)

    C = {"dir":"#e63946", "hyb":"#2a9d8f", "hybc":"#e9c46a",
         "comp":"#a8dadc", "err":"#f4a261"}

    def sa(ax, title, xl, yl):
        ax.set_facecolor("#1a1a26")
        ax.tick_params(colors="#bbbbcc", labelsize=8)
        ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=5)
        ax.set_xlabel(xl, color="#bbbbcc", fontsize=8)
        ax.set_ylabel(yl, color="#bbbbcc", fontsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a3a")
        ax.grid(True, color="#2a2a3a", lw=0.5, alpha=0.7)

    ax1 = fig.add_subplot(gs[0, 0])
    sa(ax1, "Время vs n (t=5.0)", "n", "Время (с)")
    ax1.plot(df1["n"], df1["Direct"],   "o-", color=C["dir"],  lw=1.8, ms=5, label="Direct (LU)")
    ax1.plot(df1["n"], df1["Hybrid"],   "s-", color=C["hyb"],  lw=1.8, ms=5, label="Hybrid")
    ax1.plot(df1["n"], df1["Hybrid+C"], "^-", color=C["hybc"], lw=1.8, ms=5, label="Hybrid+Curv")
    ax1.legend(fontsize=7, facecolor="#1a1a26", edgecolor="#2a2a3a", labelcolor="white")

    ax2 = fig.add_subplot(gs[0, 1])
    sa(ax2, "Ускорение Hybrid vs Direct", "n", "Speedup (×)")
    ax2.plot(df1["n"], df1["Speedup_H"], "s-", color=C["hyb"], lw=1.8, ms=5)
    ax2.axhline(1.0, color="#555577", ls="--", lw=1, label="baseline (1×)")
    ax2.legend(fontsize=7, facecolor="#1a1a26", edgecolor="#2a2a3a", labelcolor="white")

    ax3 = fig.add_subplot(gs[0, 2])
    sa(ax3, "Погрешность vs n", "n", "Rel. error")
    ax3.semilogy(df1["n"], df1["Err_H"].clip(lower=1e-16),  "s-",
                 color=C["hyb"],  lw=1.8, ms=5, label="Hybrid")
    ax3.semilogy(df1["n"], df1["Err_HC"].clip(lower=1e-16), "^-",
                 color=C["hybc"], lw=1.8, ms=5, label="Hybrid+Curv")
    ax3.legend(fontsize=7, facecolor="#1a1a26", edgecolor="#2a2a3a", labelcolor="white")

    ax4 = fig.add_subplot(gs[1, 0])
    sa(ax4, "Время vs t (n=200)", "t", "Время (с)")
    ax4.plot(df2["t"], df2["Direct"], "o-", color=C["dir"], lw=1.8, ms=5, label="Direct")
    ax4.plot(df2["t"], df2["Hybrid"], "s-", color=C["hyb"], lw=1.8, ms=5, label="Hybrid")
    ax4.legend(fontsize=7, facecolor="#1a1a26", edgecolor="#2a2a3a", labelcolor="white")

    ax5 = fig.add_subplot(gs[1, 1])
    sa(ax5, "Ускорение Hybrid vs Direct vs t", "t", "Speedup (×)")
    ax5.plot(df2["t"], df2["Speedup"], "s-", color=C["hyb"], lw=2, ms=5)
    ax5.axhline(1.0, color="#555577", ls="--", lw=1)
    t0_rows = df2[df2["Speedup"] >= 1.0]
    if not t0_rows.empty:
        t0 = float(t0_rows["t"].min())
        ax5.axvline(x=t0, color=C["err"], ls=":", lw=1.5, label=f"t₀≈{t0}")
        ax5.legend(fontsize=7, facecolor="#1a1a26", edgecolor="#2a2a3a", labelcolor="white")

    ax6 = fig.add_subplot(gs[1, 2])
    sa(ax6, "Компоненты связности vs t (n=300)", "t", "N компонент")
    ax6.plot(df3["t"], df3["n_components"], "o-", color=C["comp"], lw=2, ms=5,
             label="N компонент")
    ax6r = ax6.twinx()
    ax6r.plot(df3["t"], df3["max_comp_size"], "s--", color=C["dir"], lw=1.5, ms=4,
              label="max размер")
    ax6r.set_ylabel("Макс. размер компоненты", color=C["dir"], fontsize=7)
    ax6r.tick_params(colors="#bbbbcc", labelsize=7)
    ax6.legend(fontsize=6, facecolor="#1a1a26", edgecolor="#2a2a3a",
               labelcolor="white", loc="upper left")

    ax7 = fig.add_subplot(gs[2, 0])
    sa(ax7, "Время по типам данных", "Датасет", "Время (с)")
    xs = list(range(len(df4)))
    w  = 0.35
    ax7.bar([x - w/2 for x in xs], df4["Direct"], width=w,
            color=C["dir"], label="Direct")
    ax7.bar([x + w/2 for x in xs], df4["Hybrid"], width=w,
            color=C["hyb"], label="Hybrid")
    ax7.set_xticks(xs)
    ax7.set_xticklabels(df4["dataset"].tolist(), fontsize=7, color="#bbbbcc")
    ax7.legend(fontsize=7, facecolor="#1a1a26", edgecolor="#2a2a3a", labelcolor="white")

    ax8 = fig.add_subplot(gs[2, 1])
    sa(ax8, "Ускорение по типам данных", "Датасет", "Speedup (×)")
    bar_colors = [C["hyb"] if s >= 1 else C["dir"] for s in df4["Speedup"]]
    ax8.bar(xs, df4["Speedup"].tolist(), color=bar_colors)
    ax8.axhline(1.0, color="#555577", ls="--", lw=1)
    ax8.set_xticks(xs)
    ax8.set_xticklabels(df4["dataset"].tolist(), fontsize=7, color="#bbbbcc")
    for x, sp in zip(xs, df4["Speedup"]):
        ax8.text(x, sp + 0.02, f"{sp:.2f}×", ha="center", va="bottom",
                 color="white", fontsize=7)

    ax9 = fig.add_subplot(gs[2, 2])
    sa(ax9, "Погрешность по типам данных", "Датасет", "Rel. error")
    errs = df4["Err_H"].clip(lower=1e-16).tolist()
    ax9.bar(xs, errs, color=C["err"])
    ax9.set_yscale("log")
    ax9.set_xticks(xs)
    ax9.set_xticklabels(df4["dataset"].tolist(), fontsize=7, color="#bbbbcc")

    plt.savefig("hybrid_experiment_results_v2.png", dpi=150,
                bbox_inches="tight", facecolor="#0d0d14")
    print("\nГрафик: hybrid_experiment_results_v2.png")

if __name__ == "__main__":
    SIZES  = [50, 100, 150, 200, 300, 400, 500]
    T_VALS = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    T_COMP = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
    N_REP  = 5

    df1 = exp1_speed_vs_n(SIZES,  t=5.0, n_rep=N_REP)
    df2 = exp2_speed_vs_t(T_VALS, n=200, n_rep=N_REP)
    df3 = exp3_components_vs_t(T_COMP, n=300)
    df4 = exp4_dataset_types(n=150, t=5.0, n_rep=N_REP)

    print("\nГрафики")
    plot_all(df1, df2, df3, df4)

    df1.to_csv("hybrid_speed.csv", index=False)
    df2.to_csv("hybrid_accuracy.csv", index=False)
    df3.to_csv("hybrid_geometry.csv", index=False)
    df4.to_csv("hybrid_datasets.csv", index=False)

    print("\nГотово. Файлы:")
    print("hybrid_experiment_results_v2.png")
    print("hybrid_speed.csv")
    print("hybrid_accuracy.csv")
    print("hybrid_geometry.csv")
    print("hybrid_datasets.csv")