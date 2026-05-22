"""M7: M2 (Spectral LSTSQ) Grid Convergence Study for 2D Driven Cavity.

Purpose: Verify M2's self-consistency as a reference solution.
- Run M2 at N = 12, 14, 16, 18, 20, 22
- Extract centerline velocities (u at x=0, v at y=0)
- Compare with Ghia et al. (1982) benchmark data
- Compute Richardson-type convergence rates
- Generate convergence plot for paper

Ref: Ghia, U., Ghia, K.N., Shin, C.T. (1982). High-Re solutions for
     incompressible flow using the Navier-Stokes equations and a
     multigrid method. J. Comput. Phys., 48(3), 387-411.
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

# ─── Ghia et al. (1982) Re=100 benchmark data ──────────────────────
# u-velocity along vertical centerline x=0 (y ∈ [0,1] → mapped to [-1,1])
GHIA_U_Y = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516,
                     0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 0.1719,
                     0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
GHIA_U_VAL = np.array([1.00000, 0.84123, 0.78871, 0.73722, 0.68717,
                       0.23151, -0.00332, -0.13641, -0.20581, -0.21090,
                       -0.15662, -0.10150, -0.06434, -0.04775, -0.04192,
                       -0.03717, 0.00000])

# v-velocity along horizontal centerline y=0 (x ∈ [0,1] → mapped to [-1,1])
GHIA_V_X = np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563,
                     0.2266, 0.2344, 0.5000, 0.8047, 0.8594, 0.9063,
                     0.9453, 0.9531, 0.9609, 0.9688, 1.0000])
GHIA_V_VAL = np.array([0.00000, 0.09233, 0.10091, 0.10890, 0.12317,
                       0.16077, 0.17507, 0.17527, 0.05454, -0.24533,
                       -0.22445, -0.16914, -0.10313, -0.08864, -0.07391,
                       -0.05906, 0.00000])

# Map Ghia y ∈ [0,1] to [-1,1] for comparison
GHIA_Y_MAPPED = 2 * GHIA_U_Y - 1  # y ∈ [-1,1]
GHIA_X_MAPPED = 2 * GHIA_V_X - 1  # x ∈ [-1,1]


def run_m2_at_n(N_legendre, seed=2002, N_sample_base=32, 
                adam_epochs=5000, lbfgs_epochs=10000):
    """Run M2 (Spectral LSTSQ) at given spectral order N."""
    Re = 100.0
    nu = 2.0 / Re
    N_sample = max(N_sample_base, N_legendre + 4)  # oversampling for quadrature
    
    print(f"\n{'='*60}")
    print(f"M2 Grid Convergence: N = {N_legendre}")
    print(f"  Collocation: {N_sample} x {N_sample}")
    print(f"{'='*60}")
    
    # ─── Spectral function spaces ───
    L = FunctionSpace(N_legendre, Legendre, domain=(-1, 1), name="L")
    S = TensorProduct(L, L, name="S")
    V = VectorTensorProductSpace(S, name="V")
    
    n_dof_vel = V.dim
    n_dof_p = S.dim
    print(f"  Velocity DOF: {n_dof_vel}, Pressure DOF: {n_dof_p}, Total: {n_dof_vel + n_dof_p}")
    
    # ─── FlaxFunction wrappers ───
    u = FlaxFunction(V, "u", rngs=nnx.Rngs(seed))
    p = FlaxFunction(S, "p", rngs=nnx.Rngs(seed))
    
    # ─── Mesh & collocation ───
    mesh = Rectangle(-1, 1, -1, 1)
    xyi = mesh.get_points_inside_domain(N_sample, N_sample, "legendre")
    xyb = mesh.get_points_on_domain(N_sample, N_sample, "legendre", corners=True)
    Nb = xyb.shape[0]
    xyp = jnp.array([[0.0, 0.0]])
    wi = mesh.get_weights_inside_domain(N_sample, N_sample, "legendre")
    
    # ─── PDE residuals ───
    eq1 = Dot(Grad(u), u) - nu * Div(Grad(u)) + Grad(p)
    eq2 = Div(u)
    module = Comp(u, p)
    x_sym, y_sym = V.system.base_scalars()
    
    # ─── BC ───
    ub = DirichletBC(
        u, xyb,
        sp.Piecewise((0, y_sym < 1), ((1 - x_sym) ** 2 * (1 + x_sym) ** 2, True)),
        0,
    )
    
    # ─── Loss ───
    loss_fn = Loss(
        (eq1, xyi, 0, wi),
        (eq2, xyi, 0, wi),
        (u, xyb, ub, 2.0 / Nb),
        (p, xyp, 0, 10),
    )
    
    # ─── Optimizers ───
    opt_adam = adam(module)
    opt_lbfgs = lbfgs(module, memory_size=100)
    trainer = Trainer(loss_fn)
    
    # ─── Adam phase ───
    t0 = time.time()
    trainer.train(opt_adam, adam_epochs, epoch_print=100000)
    adam_time = time.time() - t0
    adam_loss = list(trainer.losses)[-1]
    
    # ─── L-BFGS phase ───
    t1 = time.time()
    trainer.train(opt_lbfgs, lbfgs_epochs, epoch_print=100000, abs_limit_change=0)
    lbfgs_time = time.time() - t1
    lbfgs_loss = list(trainer.losses)[-1]
    
    total_time = adam_time + lbfgs_time
    print(f"  Adam: {adam_time:.1f}s (loss={adam_loss:.3e})")
    print(f"  L-BFGS: {lbfgs_time:.1f}s (loss={lbfgs_loss:.3e})")
    print(f"  Total: {total_time:.1f}s")
    
    # ─── Evaluate on fine grid (100x100) ───
    N_eval = 100
    yj = jnp.linspace(-1, 1, N_eval)
    xx, yy = jnp.meshgrid(yj, yj, sparse=False, indexing="ij")
    z = jnp.column_stack((xx.ravel(), yy.ravel()))
    uvp = module(z)
    
    U = np.array(uvp[:, 0].reshape(xx.shape))
    V = np.array(uvp[:, 1].reshape(xx.shape))
    P = np.array(uvp[:, 2].reshape(xx.shape))
    
    # ─── Centerline extraction ───
    idx_center = N_eval // 2  # x=0 or y=0 index
    u_centerline = U[idx_center, :]  # u along vertical centerline x=0
    v_centerline = V[:, idx_center]  # v along horizontal centerline y=0
    y_vals = np.array(yj)
    x_vals = np.array(yj)
    
    result = {
        "N": N_legendre,
        "n_dof_vel": n_dof_vel,
        "n_dof_p": n_dof_p,
        "n_dof_total": n_dof_vel + n_dof_p,
        "adam_time": adam_time,
        "lbfgs_time": lbfgs_time,
        "total_time": total_time,
        "adam_loss": float(adam_loss),
        "lbfgs_loss": float(lbfgs_loss),
        "u_centerline": u_centerline,
        "v_centerline": v_centerline,
        "y_vals": y_vals,
        "x_vals": x_vals,
        "U": U,
        "V": V,
        "P": P,
        "xx": np.array(xx),
        "yy": np.array(yy),
    }
    
    # ─── Save individual result ───
    data_file = f"{PAPER_DATA_DIR}m2_gridconv_N{N_legendre:02d}.npz"
    np.savez(data_file,
             N=N_legendre,
             u_centerline=u_centerline, v_centerline=v_centerline,
             y_vals=y_vals, x_vals=x_vals,
             lbfgs_loss=float(lbfgs_loss),
             n_dof_total=n_dof_vel + n_dof_p,
             total_time=total_time)
    print(f"  Saved: {data_file}")
    
    return result


def compute_richardson_error(results, ref_idx=-1):
    """Compute L2 differences between successive N levels and ref-N."""
    ref = results[ref_idx]
    
    # Interpolate all centerlines to common grid (using ref grid)
    y_ref = ref["y_vals"]
    x_ref = ref["x_vals"]
    N_eval = len(y_ref)
    
    errors_u = {}
    errors_v = {}
    
    for r in results:
        if r["N"] == ref["N"]:
            continue
        # Use ref's own grid for comparison
        err_u = np.sqrt(np.mean((r["u_centerline"] - ref["u_centerline"])**2))
        err_v = np.sqrt(np.mean((r["v_centerline"] - ref["v_centerline"])**2))
        errors_u[r["N"]] = err_u
        errors_v[r["N"]] = err_v
    
    return errors_u, errors_v


def generate_figures(results):
    """Generate M2 convergence figures."""
    N_vals = [r["N"] for r in results]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results)))
    
    # ─── Figure 1: Centerline velocity convergence ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: u-velocity along vertical centerline
    ax = axes[0]
    # Ghia data
    ax.plot(GHIA_U_VAL, GHIA_Y_MAPPED, "ko", markersize=5, label="Ghia 1982", zorder=10)
    for i, r in enumerate(results):
        ax.plot(r["u_centerline"], r["y_vals"], "-", color=colors[i],
                linewidth=1.5, label=f"M2 N={r['N']}")
    ax.set_xlabel("u-velocity")
    ax.set_ylabel("y")
    ax.set_title("(a) u-velocity at x = 0 (vertical centerline)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    
    # Right: v-velocity along horizontal centerline
    ax = axes[1]
    ax.plot(GHIA_X_MAPPED, GHIA_V_VAL, "ko", markersize=5, label="Ghia 1982", zorder=10)
    for i, r in enumerate(results):
        ax.plot(r["x_vals"], r["v_centerline"], "-", color=colors[i],
                linewidth=1.5, label=f"M2 N={r['N']}")
    ax.set_xlabel("x")
    ax.set_ylabel("v-velocity")
    ax.set_title("(b) v-velocity at y = 0 (horizontal centerline)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle("M2 Spectral LSTSQ — Grid Convergence Study (Re=100)", 
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR + "paper_figures/m2_grid_convergence_centerline.png", 
                dpi=200, bbox_inches="tight")
    plt.close()
    print("Figure 1 saved: m2_grid_convergence_centerline.png")
    
    # ─── Figure 2: Convergence rate ───
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ref = results[-1]  # use largest N as reference
    
    N_arr = np.array([r["N"] for r in results[:-1]])
    err_u_arr = np.array([np.sqrt(np.mean((r["u_centerline"] - ref["u_centerline"])**2))
                          for r in results[:-1]])
    err_v_arr = np.array([np.sqrt(np.mean((r["v_centerline"] - ref["v_centerline"])**2))
                          for r in results[:-1]])
    
    dof_total = np.array([r["n_dof_total"] for r in results[:-1]])
    
    # Left: error vs N
    ax = axes[0]
    ax.semilogy(N_arr, err_u_arr, "bo-", label=r"$\|u - u_{ref}\|_{L^2}$", markersize=6)
    ax.semilogy(N_arr, err_v_arr, "rs--", label=r"$\|v - v_{ref}\|_{L^2}$", markersize=6)
    ax.set_xlabel("Spectral order N")
    ax.set_ylabel("L2 error (vs N=22 reference)")
    ax.set_title("(a) Convergence with spectral order")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    
    # Right: error vs sqrt(DOF)
    ax = axes[1]
    sqrt_dof = np.sqrt(dof_total)
    ax.loglog(sqrt_dof, err_u_arr, "bo-", label=r"$\|u - u_{ref}\|_{L^2}$", markersize=6)
    ax.loglog(sqrt_dof, err_v_arr, "rs--", label=r"$\|v - v_{ref}\|_{L^2}$", markersize=6)
    # Add exponential fit reference line
    ax.set_xlabel(r"$\sqrt{\mathrm{DOF}}$")
    ax.set_ylabel("L2 error (vs N=22 reference)")
    ax.set_title("(b) Convergence with DOF")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    
    fig.suptitle("M2 Spectral LSTSQ — Self-Consistency Convergence", 
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR + "paper_figures/m2_grid_convergence_rate.png", 
                dpi=200, bbox_inches="tight")
    plt.close()
    print("Figure 2 saved: m2_grid_convergence_rate.png")
    
    # ─── Save convergence data for paper ───
    conv_data = {
        "N": N_arr,
        "dof_total": dof_total,
        "err_u_L2": err_u_arr,
        "err_v_L2": err_v_arr,
        "ref_N": ref["N"],
        "ref_dof_total": ref["n_dof_total"],
    }
    np.savez(PAPER_DATA_DIR + "m2_grid_convergence.npz", **conv_data)
    print(f"Convergence data saved: m2_grid_convergence.npz")


def main():
    # N values for grid convergence study
    N_list = [12, 14, 16, 18, 20, 22]
    results = []
    
    t_total_start = time.time()
    for N in N_list:
        r = run_m2_at_n(N)
        results.append(r)
    
    t_total = time.time() - t_total_start
    print(f"\n{'='*60}")
    print(f"Grid convergence study complete. Total wall time: {t_total/60:.1f} min")
    print(f"{'='*60}")
    
    # ─── Summary table ───
    print(f"\n{'N':>4s}  {'DOF_v':>6s}  {'DOF_p':>6s}  {'DOF_tot':>7s}  "
          f"{'Loss':>12s}  {'Time(s)':>8s}")
    print("-" * 55)
    for r in results:
        print(f"{r['N']:4d}  {r['n_dof_vel']:6d}  {r['n_dof_p']:6d}  "
              f"{r['n_dof_total']:7d}  {r['lbfgs_loss']:12.3e}  {r['total_time']:8.1f}")
    
    # ─── Generate figures ───
    generate_figures(results)
    
    # ─── Compare with Ghia ───
    # Use N=22 as converged reference
    ref = results[-1]
    y_vals = ref["y_vals"]
    x_vals = ref["x_vals"]
    
    # Interpolate M2 N=22 to Ghia positions
    u_at_ghia = np.interp(GHIA_Y_MAPPED, y_vals, ref["u_centerline"])
    v_at_ghia = np.interp(GHIA_X_MAPPED, x_vals, ref["v_centerline"])
    
    err_u_ghia = np.sqrt(np.mean((u_at_ghia - GHIA_U_VAL)**2))
    err_v_ghia = np.sqrt(np.mean((v_at_ghia - GHIA_V_VAL)**2))
    
    print(f"\nM2 (N=22) vs Ghia 1982 (Re=100):")
    print(f"  L2 error in u-velocity: {err_u_ghia:.4e}")
    print(f"  L2 error in v-velocity: {err_v_ghia:.4e}")
    
    # Save Ghia comparison
    np.savez(PAPER_DATA_DIR + "m2_vs_ghia.npz",
             ghia_u_y=GHIA_U_VAL, ghia_u_val=GHIA_U_VAL,
             ghia_v_x=GHIA_V_X, ghia_v_val=GHIA_V_VAL,
             m2_u_at_ghia=u_at_ghia, m2_v_at_ghia=v_at_ghia,
             err_u=err_u_ghia, err_v=err_v_ghia)
    
    print("\nAll done!")


if __name__ == "__main__":
    main()
