"""绘图：相同制造解下 helmholtz1D（谱 Galerkin）vs Helmholtz_vpinn（VPINN）对比
制造解统一：ue = (1-x²) * exp(cos(2πx))，零 Dirichlet BC
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKDIR = "/Users/gyh/WorkBuddy/20260415163831"
sp_data    = np.load(f"{WORKDIR}/helmholtz_spectral_results.npz")
vpinn_data = np.load(f"{WORKDIR}/h1d_vpinn.npz")

# 统一键名
sp_xj   = sp_data["xj"];    sp_uj   = sp_data["uj"];   sp_uej   = sp_data["uej"]
vp_xj   = vpinn_data["xj"]; vp_uj   = vpinn_data["uj"]; vp_uej  = vpinn_data["uej"]
sp_l2   = float(sp_data["error"])
vp_l2   = float(vpinn_data["err_l2"])
vp_linf = float(vpinn_data["err_max"])
sp_t    = float(sp_data["time"])
vp_t    = float(vpinn_data["elapsed"])

# 谱方法的 L∞
sp_linf = float(np.max(np.abs(sp_uj - sp_uej)))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
title_str = r"Helmholtz 1D  $u_e=(1-x^2)e^{\cos(2\pi x)}$，零 Dirichlet BC"
fig.suptitle(title_str, fontsize=13, fontweight="bold")

# ── 左上：谱方法解 ────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(sp_xj, sp_uej, "k-",  lw=2,   label=r"Exact $u_e$")
ax.plot(sp_xj, sp_uj,  "r--", lw=1.5, label=f"Spectral Galerkin (N=80)\nL2={sp_l2:.2e}")
ax.set_title("Spectral Galerkin Solution", fontsize=12)
ax.set_xlabel("x"); ax.set_ylabel("u(x)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# ── 右上：VPINN 解 ─────────────────────────────────────────────────
ax = axes[0, 1]
ax.plot(vp_xj, vp_uej, "k-",  lw=2,   label=r"Exact $u_e$")
ax.plot(vp_xj, vp_uj,  "b--", lw=1.5, label=f"VPINN (MLP64 + Legendre32)\nL2={vp_l2:.2e}")
ax.set_title("VPINN Solution", fontsize=12)
ax.set_xlabel("x"); ax.set_ylabel("u(x)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# ── 左下：逐点误差曲线（对数坐标）─────────────────────────────────
ax = axes[1, 0]
ax.semilogy(sp_xj, np.abs(sp_uj - sp_uej), "r-", lw=1.5,
            label=f"Spectral  L2={sp_l2:.2e}  L∞={sp_linf:.2e}")
ax.semilogy(vp_xj, np.abs(vp_uj - vp_uej), "b-", lw=1.5,
            label=f"VPINN     L2={vp_l2:.2e}  L∞={vp_linf:.2e}")
ax.set_title("Pointwise Error  |u_h - u_e|  (log scale)", fontsize=12)
ax.set_xlabel("x"); ax.set_ylabel("Error")
ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

# ── 右下：精度 & 时间双柱状图 ─────────────────────────────────────
ax = axes[1, 1]
methods = ["Spectral\nGalerkin\n(N=80)", "VPINN\n(MLP64\n+Leg32)"]
l2s     = [sp_l2,  vp_l2]
linfs   = [sp_linf, vp_linf]
times   = [sp_t,   vp_t]
colors  = ["#e74c3c", "#3498db"]
x_pos   = np.arange(len(methods))
bars = ax.bar(x_pos, l2s, color=colors, width=0.4, edgecolor="k", linewidth=0.8)
for bar, l2, linf, t in zip(bars, l2s, linfs, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5,
            f"L2={l2:.1e}\nL∞={linf:.1e}\nt={t:.0f}s",
            ha="center", va="bottom", fontsize=9.5)
ax.set_xticks(x_pos); ax.set_xticklabels(methods)
ax.set_yscale("log")
ax.set_ylabel("L2 Error")
ax.set_title("Accuracy Comparison  (lower is better)", fontsize=12)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out = f"{WORKDIR}/helmholtz1D_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")

# ── 汇总表 ──────────────────────────────────────────────────────────
print()
print("="*72)
print(f"  制造解：ue = (1-x^2) * exp(cos(2*pi*x))，零 Dirichlet BC")
print(f"  {'Method':<35} {'L2 Error':>12} {'Linf Error':>12} {'Time':>8}")
print("  " + "-"*68)
rows = [
    ("Spectral Galerkin (N=80)",         sp_l2,  sp_linf, sp_t),
    ("VPINN (MLP64 + Legendre32)",        vp_l2,  vp_linf, vp_t),
]
for name, l2, linf, t in rows:
    print(f"  {name:<35} {l2:>12.3e} {linf:>12.3e} {t:>6.1f}s")
print("="*72)
