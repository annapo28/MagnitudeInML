import time, warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import cg as sp_cg
from scipy.sparse import diags as sp_diags
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

try:
    import network_distance as nd
    import networkx as nx
    NVD_AVAILABLE = True
except ImportError:
    NVD_AVAILABLE = False

def zeta(X, t):
    return np.exp(-t * cdist(X, X))

def solve_direct(Z):
    return np.linalg.solve(Z, np.ones(len(Z)))

def solve_cg(Z, tol=1e-10):
    w, info = sp_cg(Z, np.ones(len(Z)), rtol=tol, maxiter=5000)
    return w, info

def solve_pcg(Z, tol=1e-10):
    """Preconditioned CG: M = diag(Z) - диагональный preconditioner.
    Это первый шаг Laplacian paradigm: нормируем систему на диагональ,
    что эквивалентно переходу к системе с лапласианоподобной структурой."""
    d = np.diag(Z)
    M_inv = sp_diags(1.0 / d)
    w, info = sp_cg(Z, np.ones(len(Z)), M=M_inv, rtol=tol, maxiter=5000)
    return w, info

def condition_number(Z):
    eigvals = np.linalg.eigvalsh(Z)
    eigvals = eigvals[eigvals > 1e-14]
    return float(eigvals.max() / eigvals.min())

def count_cg_iterations(Z, tol=1e-10):
    """Считаем итерации CG через callback."""
    iters = [0]
    def cb(_): iters[0] += 1
    sp_cg(Z, np.ones(len(Z)), rtol=tol, maxiter=5000, callback=cb)
    return iters[0]

def count_pcg_iterations(Z, tol=1e-10):
    d = np.diag(Z); M_inv = sp_diags(1.0 / d)
    iters = [0]
    def cb(_): iters[0] += 1
    sp_cg(Z, np.ones(len(Z)), M=M_inv, rtol=tol, maxiter=5000, callback=cb)
    return iters[0]


# Эксперимент 1: число обусловленности и число итераций cg как функция t
def exp_kappa_vs_t(n=100, t_values=None, seed=42):
    if t_values is None:
        t_values = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    records = []
    for t in t_values:
        Z = zeta(X, t)
        kappa = condition_number(Z)
        n_cg  = count_cg_iterations(Z)
        n_pcg = count_pcg_iterations(Z)
        # диагональное доминирование: min(Z_ii - sum|Z_ij|, j!=i)
        dd = np.min(np.diag(Z) - (np.sum(np.abs(Z), axis=1) - np.diag(Z)))
        records.append({"t": t, "kappa": kappa, "iter_cg": n_cg,
                        "iter_pcg": n_pcg, "diag_dom": dd})
        print(f"  t={t:5.1f} | κ={kappa:10.2f} | iter_CG={n_cg:4d} | "
              f"iter_PCG={n_pcg:4d} | dd={dd:.4f}")
    return pd.DataFrame(records)


# Экперимент 2: время Direct vs CG vs PCG vs NVD-solve при разных t и n
def exp_speed_vs_t_n(sizes, t_values, n_rep=5, seed=42):
    rng = np.random.default_rng(seed)
    records = []

    for n in sizes:
        X = rng.standard_normal((n, 2))

        # Строим граф для nd.ge
        if NVD_AVAILABLE:
            # граф на тех же точках: рёбра = расстояния в R^2
            # используем \epsilon-граф для имитации структуры
            from scipy.spatial import cKDTree
            tree = cKDTree(X)
            G_nvd = nx.Graph()
            G_nvd.add_nodes_from(range(n))
            pairs = tree.query_pairs(r=1.5)
            for i, j in pairs:
                d = float(np.linalg.norm(X[i] - X[j]))
                G_nvd.add_edge(i, j, weight=d)
            if not nx.is_connected(G_nvd):
                # добавляем MST чтобы обеспечить связность
                from scipy.sparse.csgraph import minimum_spanning_tree
                from scipy.sparse import csr_matrix
                D_full = cdist(X, X)
                mst = minimum_spanning_tree(csr_matrix(D_full))
                cx = mst.tocoo()
                for i, j, v in zip(cx.row, cx.col, cx.data):
                    G_nvd.add_edge(int(i), int(j), weight=float(v))
            nodes_nvd = list(G_nvd.nodes())

        for t in t_values:
            Z = zeta(X, t)

            def bench(fn):
                ts = []
                for _ in range(n_rep):
                    t0 = time.perf_counter(); fn(); ts.append(time.perf_counter()-t0)
                return np.mean(ts)

            tm_dir = bench(lambda: solve_direct(Z))
            tm_cg  = bench(lambda: solve_cg(Z))
            tm_pcg = bench(lambda: solve_pcg(Z))

            # NVD: решаем систему с L вместо Z - это и есть лапласиановый решатель
            if NVD_AVAILABLE:
                u_arr = rng.dirichlet(np.ones(n))
                v_arr = rng.dirichlet(np.ones(n))
                src = {nodes_nvd[i]: u_arr[i] for i in range(n)}
                trg = {nodes_nvd[i]: v_arr[i] for i in range(n)}
                tm_nvd = bench(lambda: nd.ge(src, trg, G_nvd))
            else:
                tm_nvd = float("nan")

            kappa = condition_number(Z)
            n_iter_cg  = count_cg_iterations(Z)
            n_iter_pcg = count_pcg_iterations(Z)

            records.append({
                "n": n, "t": t,
                "Direct": tm_dir, "CG": tm_cg, "PCG": tm_pcg, "NVD": tm_nvd,
                "kappa": kappa, "iter_cg": n_iter_cg, "iter_pcg": n_iter_pcg,
                "speedup_cg":  tm_dir / tm_cg  if tm_cg  > 0 else 0,
                "speedup_pcg": tm_dir / tm_pcg if tm_pcg > 0 else 0,
            })

        print(f"  n={n:4d} done")

    return pd.DataFrame(records)

def plot_all(df_kappa, df_speed):
    fig = plt.figure(figsize=(18, 12), facecolor="#0d0d14")
    fig.suptitle(
        "Laplacian Paradigm для вычисления магнитуды: число обусловленности, итерации CG и ускорение",
        color="white", fontsize=13, fontweight="bold", y=0.97)
    gs = fig.add_gridspec(2, 3, hspace=0.44, wspace=0.38,
                          left=0.07, right=0.97, top=0.91, bottom=0.08)

    C = {"dir":"#e63946","cg":"#e9c46a","pcg":"#2a9d8f","nvd":"#f4a261","dd":"#a8dadc"}

    def sa(ax, title, xl, yl):
        ax.set_facecolor("#1a1a26")
        ax.tick_params(colors="#bbbbcc", labelsize=8)
        ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=5)
        ax.set_xlabel(xl, color="#bbbbcc", fontsize=8)
        ax.set_ylabel(yl, color="#bbbbcc", fontsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#2a2a3a")
        ax.grid(True, color="#2a2a3a", lw=0.5, alpha=0.7)

    # 1. число обусловленности vs t
    ax1 = fig.add_subplot(gs[0,0]); sa(ax1,"Число обусловленности κ(ζ) vs t","t","κ (log)")
    ax1.semilogy(df_kappa["t"], df_kappa["kappa"], "o-", color=C["dir"], lw=2, ms=6)
    ax1.axvline(x=df_kappa[df_kappa["kappa"]<100]["t"].min() if (df_kappa["kappa"]<100).any() else 999,
                color="#555577", ls="--", lw=1.2, label="κ < 100")
    ax1.legend(fontsize=7, facecolor="#1a1a26", edgecolor="#2a2a3a", labelcolor="white")

    # 2. итерации CG и PCG vs t
    ax2 = fig.add_subplot(gs[0,1]); sa(ax2,"Итерации до сходимости vs t","t","Итерации")
    ax2.plot(df_kappa["t"], df_kappa["iter_cg"],  "o-", color=C["cg"],  lw=2, ms=6, label="CG (без precon.)")
    ax2.plot(df_kappa["t"], df_kappa["iter_pcg"], "s-", color=C["pcg"], lw=2, ms=6, label="PCG (diag precon.)")
    ax2.legend(fontsize=7, facecolor="#1a1a26", edgecolor="#2a2a3a", labelcolor="white")

    # 3. диаг. доминирование vs t
    ax3 = fig.add_subplot(gs[0,2]); sa(ax3,"Диагональное доминирование vs t","t","min(Z_ii - Σ|Z_ij|)")
    ax3.plot(df_kappa["t"], df_kappa["diag_dom"], "o-", color=C["dd"], lw=2, ms=6)
    ax3.axhline(0, color="#e63946", ls="--", lw=1.2, label="порог dd=0")
    t0_idx = df_kappa[df_kappa["diag_dom"] > 0]["t"].min() if (df_kappa["diag_dom"]>0).any() else None
    if t0_idx:
        ax3.axvline(x=t0_idx, color="#f4a261", ls="--", lw=1.5, label=f"t₀ ≈ {t0_idx}")
    ax3.legend(fontsize=7, facecolor="#1a1a26", edgecolor="#2a2a3a", labelcolor="white")

    # 4. время vs t при фикс. n
    n_fix = df_speed["n"].median()
    n_fix = df_speed["n"].unique()[len(df_speed["n"].unique())//2]
    sub = df_speed[df_speed["n"]==n_fix].sort_values("t")
    ax4 = fig.add_subplot(gs[1,0]); sa(ax4,f"Время vs t (n={n_fix})","t","Время (с)")
    ax4.plot(sub["t"], sub["Direct"], "o-", color=C["dir"], lw=2, ms=5, label="Direct (LU)")
    ax4.plot(sub["t"], sub["CG"],     "o-", color=C["cg"],  lw=2, ms=5, label="CG")
    ax4.plot(sub["t"], sub["PCG"],    "s-", color=C["pcg"], lw=2, ms=5, label="PCG (diag)")
    if NVD_AVAILABLE:
        ax4.plot(sub["t"], sub["NVD"], "^-", color=C["nvd"], lw=2, ms=5, label="NVD (GE)")
    ax4.legend(fontsize=7, facecolor="#1a1a26", edgecolor="#2a2a3a", labelcolor="white")

    # 5. ускорение vs t
    ax5 = fig.add_subplot(gs[1,1]); sa(ax5,f"Ускорение vgit add .
s t (n={n_fix})","t","Speedup (×)")
    ax5.plot(sub["t"], sub["speedup_cg"],  "o-", color=C["cg"],  lw=2, ms=5, label="CG vs Direct")
    ax5.plot(sub["t"], sub["speedup_pcg"], "s-", color=C["pcg"], lw=2, ms=5, label="PCG vs Direct")
    ax5.axhline(1.0, color="#555577", ls="--", lw=1)
    # отмечаем порог t_0
    t0_cg = sub[sub["speedup_cg"]>1.0]["t"].min() if (sub["speedup_cg"]>1.0).any() else None
    if t0_cg:
        ax5.axvline(x=t0_cg, color=C["cg"], ls=":", lw=1.5, alpha=0.7, label=f"t₀(CG)≈{t0_cg}")
    ax5.legend(fontsize=7, facecolor="#1a1a26", edgecolor="#2a2a3a", labelcolor="white")

    # 6. ускорение vs n при большом t
    t_big = df_speed["t"].max()
    sub2 = df_speed[df_speed["t"]==t_big].sort_values("n")
    ax6 = fig.add_subplot(gs[1,2]); sa(ax6,f"Ускорение vs n (t={t_big})","n","Speedup (×)")
    ax6.plot(sub2["n"], sub2["speedup_cg"],  "o-", color=C["cg"],  lw=2, ms=5, label="CG vs Direct")
    ax6.plot(sub2["n"], sub2["speedup_pcg"], "s-", color=C["pcg"], lw=2, ms=5, label="PCG vs Direct")
    ax6.axhline(1.0, color="#555577", ls="--", lw=1)
    ax6.legend(fontsize=7, facecolor="#1a1a26", edgecolor="#2a2a3a", labelcolor="white")

    plt.savefig("mag_laplacian_speedup_results.png", dpi=150,
                bbox_inches="tight", facecolor="#0d0d14")
    print("График сохранён")

if __name__ == "__main__":
    T_VALUES = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    SIZES    = [50, 100, 150, 200, 300]

    print("Эксперимент 1: число обусловенности и итерации CG vs t (n=100)...")
    df_kappa = exp_kappa_vs_t(n=100, t_values=T_VALUES)
    df_kappa.to_csv("mag_kappa_data.csv", index=False)

    print("Эксперимент 2: скорость Direct vs CG vs PCG vs NVD...")
    df_speed = exp_speed_vs_t_n(SIZES, T_VALUES, n_rep=5)
    df_speed.to_csv("mag_speed_data.csv", index=False)

    print("Графики")
    plot_all(df_kappa, df_speed)

    print("\n" + "="*65)
    print("Таблица 1: число обуслвоенности, итерации и диагональное доминирование (n=100)")
    print("="*65)
    print(f"{'t':>6} | {'κ(ζ)':>12} | {'iter CG':>8} | {'iter PCG':>9} | {'dd':>8}")
    print("-"*65)
    for _, r in df_kappa.iterrows():
        dd_str = f"{r['diag_dom']:+.4f}"
        print(f"{r['t']:>6.1f} | {r['kappa']:>12.1f} | {int(r['iter_cg']):>8d} | "
              f"{int(r['iter_pcg']):>9d} | {dd_str:>8}")

    print("\n" + "="*65)
    print(f"Таблица 2: ускорение при t=10.0 (порог пройден)")
    print("="*65)
    sub = df_speed[df_speed["t"]==10.0].sort_values("n")
    print(f"{'n':>5} | {'Direct':>8} | {'CG':>8} | {'PCG':>8} | "
          f"{'sp_CG':>7} | {'sp_PCG':>8}")
    print("-"*65)
    for _, r in sub.iterrows():
        print(f"{int(r['n']):>5} | {r['Direct']:>8.4f} | {r['CG']:>8.4f} | "
              f"{r['PCG']:>8.4f} | {r['speedup_cg']:>6.2f}× | {r['speedup_pcg']:>7.2f}×")
    print("="*65)