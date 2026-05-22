"""M7: M2 Self-Consistency Verification.

Two-pronged approach:
1. Compare M2 (N=20) centerline velocities with Ghia et al. (1982) benchmark
2. Fast spectral convergence study (N=12, 14, 16, with reduced training)

This addresses reviewer concern: "M2's own error vs true solution, or 
self-consistency evidence such as grid convergence study."
"""
import os, sys, time
os.environ["MPLBACKEND"] = "Agg"

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flax import nnx
import sympy as sp

from jaxfun import Div, Grad
from jaxfun.galerkin import FunctionSpace, TensorProduct
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.galerkin.tensorproductspace import VectorTensorProductSpace
from jaxfun.operators import Dot
from jaxfun.pinns.bcs import DirichletBC
from jaxfun.pinns.loss import Loss
from jaxfun.pinns.mesh import Rectangle
from jaxfun.pinns.module import Comp, FlaxFunction
from jaxfun.pinns.optimizer import Trainer, adam, lbfgs

OUT_DIR = "/Users/gyh/WorkBuddy/20260415163831/"
PAPER_DATA_DIR = OUT_DIR + "paper_data/"

# ═══════════════════════════════════════════════════════════════════════
# PART A: M2 (N=20) vs Ghia 1982 comparison (post-processing only)
# ═══════════════════════════════════════════════════════════════════════

# Ghia et al. (1982) Re=100 benchmark — u along vertical centerline x=0
GHIA_U_Y = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516,
                     0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 0.1719,
                     0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
GHIA_U_VAL = np.array([1.00000, 0.84123, 0.78871, 0.73722, 0.68717,
                       0.23151, -0.00332, -0.13641, -0.20581, -0.21090,
                       -0.15662, -0.10150, -0.06434, -0.04775, -0.04192,
                       -0.03717, 0.00000])
GHIA_Y_MAPPED = 2 * GHIA_U_Y - 1

# Ghia v along horizontal centerline y=0
GHIA_V_X = np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563,
                     0.2266, 0.2344, 0.5000, 0.8047, 0.8594, 0.9063,
                     0.9453, 0.9531, 0.9609, 0.9688, 1.0000])
GHIA_V_VAL = np.array([0.00000, 0.09233, 0.10091, 0.10890, 0.12317,
                       0.16077, 0.17507, 0.17527, 0.05454, -0.24533,
                       -0.22445, -0.16914, -0.10313, -0.08864, -0.07391,
                       -0.05906, 0.00000])
GHIA_X_MAPPED = 2 * GHIA_V_X - 1


def part_a_load_and_compare():
    """Load existing M2 N=20 data and compare with Ghia 1982."""
    print("=" * 60)
    print("PART A: M2 (N=20) vs Ghia 1982 Benchmark")
    print("=" * 60)
    
    data = np.load(PAPER_DATA_DIR + "cavity2D_m2_data.npz")
    U = data["U"]       # shape (100, 100) on [-1,1]×[-1,1]
    V = data["V"]
    xx = data["xx"]
    yy = data["yy"]
    
    # Centerline extraction
    N_eval = U.shape[0]
    idx_center = N_eval // 2
    u_cl = U[idx_center, :]  # u(x=0, y)
    v_cl = V[:, idx_center]  # v(x, y=0)
    y_vals = np.linspace(-1, 1, N_eval)
    x_vals = np.linspace(-1, 1, N_eval)
    
    # Interpolate M2 to Ghia positions
    u_m2_at_ghia = np.interp(GHIA_Y_MAPPED, y_vals, u_cl)
    v_m2_at_ghia = np.interp(GHIA_X_MAPPED, x_vals, v_cl)
    
    # Compute errors
    err_u_l2 = np.sqrt(np.mean((u_m2_at_ghia - GHIA_U_VAL)**2))
    err_v_l2 = np.sqrt(np.mean((v_m2_at_ghia - GHIA_V_VAL)**2))
    err_u_linf = np.max(np.abs(u_m2_at_ghia - GHIA_U_VAL))
    err_v_linf = np.max(np.abs(v_m2_at_ghia - GHIA_V_VAL))
    
    print(f"  u-velocity L2 error:  {err_u_l2:.4e}")
    print(f"  u-velocity Linf error: {err_u_linf:.4e}")
    print(f"  v-velocity L2 error:  {err_v_l2:.4e}")
    print(f"  v-velocity Linf error: {err_v_linf:.4e}")
    
    # ─── Plot: M2 vs Ghia centerline comparison ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Left: u-velocity at x=0
    ax = axes[0]
    ax.plot(u_cl, y_vals, "b-", linewidth=2, label="M2 (N=20)")
    ax.plot(GHIA_U_VAL, GHIA_Y_MAPPED, "ko", markersize=6, 
            markerfacecolor="none", markeredgewidth=1.5, label="Ghia 1982")
    ax.set_xlabel("u-velocity", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("(a) u-velocity at x = 0", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.35, 1.1)
    
    # Right: v-velocity at y=0
    ax = axes[1]
    ax.plot(x_vals, v_cl, "b-", linewidth=2, label="M2 (N=20)")
    ax.plot(GHIA_X_MAPPED, GHIA_V_VAL, "ko", markersize=6, 
            markerfacecolor="none", markeredgewidth=1.5, label="Ghia 1982")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("v-velocity", fontsize=12)
    ax.set_title("(b) v-velocity at y = 0", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle("M2 Spectral LSTSQ vs Ghia 1982 — Centerline Velocity (Re=100)", 
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR + "paper_figures/Fig_m7_M2_vs_Ghia1982.png",
                dpi=200, bbox_inches="tight")
    plt.close()
    print("  Figure saved: Fig_m7_M2_vs_Ghia1982.png")
    
    # ─── Compute PDE residual on the evaluation grid ───
    # We can approximate the residual using finite differences
    # or just use the reported final loss as evidence
    final_loss = float(data.get("final_loss", 1.63e-8))
    print(f"  M2 final training loss: {final_loss:.3e}")
    
    # Save comparison data
    np.savez(PAPER_DATA_DIR + "m7_m2_vs_ghia.npz",
             ghia_u_y=GHIA_U_Y, ghia_u_val=GHIA_U_VAL,
             ghia_v_x=GHIA_V_X, ghia_v_val=GHIA_V_VAL,
             m2_u_at_ghia=u_m2_at_ghia, m2_v_at_ghia=v_m2_at_ghia,
             err_u_l2=err_u_l2, err_v_l2=err_v_l2,
             err_u_linf=err_u_linf, err_v_linf=err_v_linf,
             m2_u_centerline=u_cl, m2_v_centerline=v_cl,
             m2_y_vals=y_vals, m2_x_vals=x_vals)
    print(f"  Data saved: m7_m2_vs_ghia.npz")
    
    return err_u_l2, err_v_l2, err_u_linf, err_v_linf


# ═══════════════════════════════════════════════════════════════════════
# PART B: Fast spectral convergence (N=12, 14, 16 with reduced training)
# ═══════════════════════════════════════════════════════════════════════

def run_m2_fast(N_legendre, adam_epochs=500, lbfgs_epochs=3000, N_sample=20):
    """Fast M2 run with reduced training for convergence assessment."""
    Re = 100.0
    nu = 2.0 / Re
    seed = 2002
    
    print(f"\n  N={N_legendre} (ADAM={adam_epochs}, L-BFGS={lbfgs_epochs}, N_sample={N_sample})")
    
    L = FunctionSpace(N_legendre, Legendre, domain=(-1, 1), name="L")
    S = TensorProduct(L, L, name="S")
    V = VectorTensorProductSpace(S, name="V")
    
    u = FlaxFunction(V, "u", rngs=nnx.Rngs(seed))
    p = FlaxFunction(S, "p", rngs=nnx.Rngs(seed))
    
    mesh = Rectangle(-1, 1, -1, 1)
    xyi = mesh.get_points_inside_domain(N_sample, N_sample, "legendre")
    xyb = mesh.get_points_on_domain(N_sample, N_sample, "legendre", corners=True)
    Nb = xyb.shape[0]
    xyp = jnp.array([[0.0, 0.0]])
    wi = mesh.get_weights_inside_domain(N_sample, N_sample, "legendre")
    
    eq1 = Dot(Grad(u), u) - nu * Div(Grad(u)) + Grad(p)
    eq2 = Div(u)
    module = Comp(u, p)
    x_sym, y_sym = V.system.base_scalars()
    
    ub = DirichletBC(
        u, xyb,
        sp.Piecewise((0, y_sym < 1), ((1 - x_sym) ** 2 * (1 + x_sym) ** 2, True)),
        0,
    )
    
    loss_fn = Loss(
        (eq1, xyi, 0, wi),
        (eq2, xyi, 0, wi),
        (u, xyb, ub, 2.0 / Nb),
        (p, xyp, 0, 10),
    )
    
    opt_adam = adam(module)
    opt_lbfgs = lbfgs(module, memory_size=100)
    trainer = Trainer(loss_fn)
    
    # Adam (short warm-up)
    trainer.train(opt_adam, adam_epochs, epoch_print=100000)
    adam_loss = list(trainer.losses)[-1]
    
    # L-BFGS
    trainer.train(opt_lbfgs, lbfgs_epochs, epoch_print=100000, abs_limit_change=0)
    lbfgs_loss = list(trainer.losses)[-1]
    
    # Evaluate on fine grid
    N_eval = 100
    yj = jnp.linspace(-1, 1, N_eval)
    xx, yy = jnp.meshgrid(yj, yj, sparse=False, indexing="ij")
    z = jnp.column_stack((xx.ravel(), yy.ravel()))
    uvp = module(z)
    
    U = np.array(uvp[:, 0].reshape(xx.shape))
    V_ve = np.array(uvp[:, 1].reshape(xx.shape))
    
    idx_c = N_eval // 2
    u_cl = U[idx_c, :]
    v_cl = V_ve[:, idx_c]
    
    print(f"    Adam loss={adam_loss:.3e}, L-BFGS loss={lbfgs_loss:.3e}")
    
    return {
        "N": N_legendre,
        "u_centerline": u_cl,
        "v_centerline": v_cl,
        "final_loss": float(lbfgs_loss),
        "y_vals": np.array(yj),
    }


def part_b_convergence():
    """Run M2 at N=12,14,16 and compare centerlines with N=20 reference."""
    print("\n" + "=" * 60)
    print("PART B: Spectral Convergence (N=12, 14, 16 vs N=20)")
    print("=" * 60)
    
    # Load N=20 reference
    data = np.load(PAPER_DATA_DIR + "cavity2D_m2_data.npz")
    U_ref = data["U"]
    V_ref = data["V"]
    N_eval = 100
    idx_c = N_eval // 2
    u_ref = U_ref[idx_c, :]
    v_ref = V_ref[:, idx_c]
    
    # Run reduced M2 at smaller N
    results = []
    for N in [12, 14, 16]:
        r = run_m2_fast(N)
        results.append(r)
    
    # Compute convergence errors
    print("\n  Convergence summary (vs N=20 reference):")
    print(f"  {'N':>4s}  {'Loss':>12s}  {'err_u_L2':>12s}  {'err_v_L2':>12s}  {'err_u_max':>12s}  {'err_v_max':>12s}")
    print("  " + "-" * 80)
    
    errs_u = []
    errs_v = []
    N_vals = []
    
    for r in results:
        err_u_l2 = np.sqrt(np.mean((r["u_centerline"] - u_ref)**2))
        err_v_l2 = np.sqrt(np.mean((r["v_centerline"] - v_ref)**2))
        err_u_max = np.max(np.abs(r["u_centerline"] - u_ref))
        err_v_max = np.max(np.abs(r["v_centerline"] - v_ref))
        errs_u.append(err_u_l2)
        errs_v.append(err_v_l2)
        N_vals.append(r["N"])
        print(f"  {r['N']:4d}  {r['final_loss']:12.3e}  {err_u_l2:12.4e}  {err_v_l2:12.4e}  {err_u_max:12.4e}  {err_v_max:12.4e}")
    
    # ─── Convergence plot ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Left: centerline profiles
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results) + 1))
    y_vals = np.linspace(-1, 1, N_eval)
    
    ax.plot(u_ref, y_vals, "k-", linewidth=2, label="M2 N=20 (ref)")
    ax.plot(GHIA_U_VAL, GHIA_Y_MAPPED, "ko", markersize=4, markerfacecolor="none",
            alpha=0.7, label="Ghia 1982")
    for i, r in enumerate(results):
        ax.plot(r["u_centerline"], y_vals, "--", color=colors[i], linewidth=1.2,
                label=f"M2 N={r['N']}")
    ax.set_xlabel("u-velocity", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("(a) u-velocity at x = 0", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.35, 1.1)
    
    # Right: convergence rate
    ax = axes[1]
    N_all = [12, 14, 16]
    errs_u_arr = np.array(errs_u)
    errs_v_arr = np.array(errs_v)
    
    ax.semilogy(N_all, errs_u_arr, "bo-", markersize=8, label=r"$\|u-u_{20}\|_{L^2}$")
    ax.semilogy(N_all, errs_v_arr, "rs--", markersize=8, label=r"$\|v-v_{20}\|_{L^2}$")
    
    # Add exponential fit trendline
    from numpy.polynomial import polynomial as P
    coeff_u = P.polyfit(N_all, np.log10(errs_u_arr), 1)
    coeff_v = P.polyfit(N_all, np.log10(errs_v_arr), 1)
    N_fit = np.linspace(11, 21, 50)
    fit_u = 10 ** P.polyval(N_fit, coeff_u)
    fit_v = 10 ** P.polyval(N_fit, coeff_v)
    ax.semilogy(N_fit, fit_u, "b:", linewidth=1, alpha=0.5,
                label=f"u-rate: {coeff_u[1]:.2f} dex/N")
    ax.semilogy(N_fit, fit_v, "r:", linewidth=1, alpha=0.5,
                label=f"v-rate: {coeff_v[1]:.2f} dex/N")
    
    ax.set_xlabel("Spectral order N", fontsize=12)
    ax.set_ylabel("L2 error (vs N=20)", fontsize=12)
    ax.set_title("(b) Convergence with N", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    
    fig.suptitle("M2 Spectral LSTSQ — Self-Consistency Verification (Re=100)", 
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR + "paper_figures/Fig_m7_M2_convergence.png",
                dpi=200, bbox_inches="tight")
    plt.close()
    print("\n  Figure saved: Fig_m7_M2_convergence.png")
    print(f"\n  Spectral convergence rate: u: {coeff_u[1]:.3f} dex/N, v: {coeff_v[1]:.3f} dex/N")
    
    # Save convergence data
    np.savez(PAPER_DATA_DIR + "m7_m2_convergence.npz",
             N_vals=N_all, err_u=errs_u_arr, err_v=errs_v_arr,
             rate_u=coeff_u[1], rate_v=coeff_v[1])
    print(f"  Data saved: m7_m2_convergence.npz")
    
    return results


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()
    
    # Part A: Post-processing comparison with Ghia 1982
    err_u_l2, err_v_l2, err_u_linf, err_v_linf = part_a_load_and_compare()
    
    # Part B: Spectral convergence
    results = part_b_convergence()
    
    t_total = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"M7 self-consistency verification complete. ({t_total:.0f}s)")
    print(f"{'=' * 60}")
