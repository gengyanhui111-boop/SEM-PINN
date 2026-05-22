"""Generate Helmholtz 1D figures for paper_draft_v2.tex
Figures:
  figH1_helmholtz_solution.png  - Exact vs numerical solutions
  figH2_helmholtz_error.png    - Pointwise errors for M3 and M5
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKDIR = "/Users/gyh/WorkBuddy/20260415163831"
FIGDIR  = f"{WORKDIR}/paper_figures"

# Load data
sp_data  = np.load(f"{WORKDIR}/paper_data/helmholtz1D_spectral_results.npz")
vp_data  = np.load(f"{WORKDIR}/paper_data/helmholtz1D_vpinn_results.npz")
sem_data = np.load(f"{WORKDIR}/paper_data/helmholtz1D_sempinn_results.npz") if __import__('os').path.exists(f"{WORKDIR}/paper_data/helmholtz1D_sempinn_results.npz") else None

# Extract arrays
x_exact = sp_data["xj"]
u_exact = sp_data["uej"]

x_sp  = sp_data["xj"]
u_sp  = sp_data["uj"]

x_vp  = vp_data["xj"]
u_vp  = vp_data["w_pred"]

# Compute errors
err_sp = u_sp - u_exact
err_vp = u_vp - u_exact

# =====================================================================
# Figure H1: Solution comparison
# =====================================================================
plt.figure(figsize=(10, 6))
plt.plot(x_exact, u_exact, "k-",  lw=3,   label=r"Exact $u_{\rm exact}$")
plt.plot(x_sp,    u_sp,    "b--", lw=1.5, label=f"Spectral Galerkin (M2)\n$L_2$ error = {float(sp_data['error']):.2e}")
plt.plot(x_vp,    u_vp,    "r:",  lw=1.5, label=f"VPINN (M3)\n$L_2$ error = {float(vp_data['error']):.2e}")
plt.title(r"1D Helmholtz Equation: Solution Comparison", fontsize=13, fontweight="bold")
plt.xlabel("$x$")
plt.ylabel("$u(x)$")
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/figH1_helmholtz_solution.png", dpi=150, bbox_inches="tight")
print(f"Saved: {FIGDIR}/figH1_helmholtz_solution.png")
plt.close()

# =====================================================================
# Figure H2: Pointwise error
# =====================================================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.semilogy(x_vp, np.abs(err_vp), "r-", lw=1.5, label="VPINN (M3)")
plt.title("VPINN (M3) Pointwise Error", fontsize=12)
plt.xlabel("$x$")
plt.ylabel("$|u - u_{\\rm exact}|$ (log scale)")
plt.legend(fontsize=9)
plt.grid(alpha=0.3, which="both")

plt.subplot(1, 2, 2)
# For M5 (SEM-PINN), use sempinn data if available
if sem_data is not None:
    x_sem = sem_data["xj"]
    u_sem = sem_data["w_pred"]
    err_sem = u_sem - u_exact[:len(u_sem)] if len(u_sem) == len(u_exact) else u_sem - np.interp(x_sem, x_exact, u_exact)
    plt.semilogy(x_sem, np.abs(err_sem), "g-", lw=1.5, label=f"SEM-PINN (M5)\n$L_2$ error = {float(sem_data['l2_error']):.2e}")
else:
    plt.semilogy(x_vp, np.abs(err_vp), "g-", lw=1.5, label="SEM-PINN (M5) [data not found]")
plt.title("SEM-PINN (M5) Pointwise Error", fontsize=12)
plt.xlabel("$x$")
plt.ylabel("$|u - u_{\\rm exact}|$ (log scale)")
plt.legend(fontsize=9)
plt.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig(f"{FIGDIR}/figH2_helmholtz_error.png", dpi=150, bbox_inches="tight")
print(f"Saved: {FIGDIR}/figH2_helmholtz_error.png")
plt.close()

# =====================================================================
# Print summary table
# =====================================================================
print()
print("="*72)
print(f"  Manufactured solution: ue = (1-x^2) * exp(cos(2*pi*x))")
print(f"  {'Method':<35} {'L2 Error':>12} {'Time (s)':>12}")
print("  " + "-"*68)
print(f"  {'Spectral Galerkin (M2)':<35} {float(sp_data['error']):>12.3e} {sp_data['time']:>12.2f}")
print(f"  {'VPINN (M3)':<35} {float(vp_data['error']):>12.3e} {vp_data['time_adam']+vp_data['time_lbfgs']:>12.2f}")
if sem_data is not None:
    print(f"  {'SEM-PINN (M5)':<35} {float(sem_data['l2_error']):>12.3e} {0:>12.2f}")
print("="*72)
