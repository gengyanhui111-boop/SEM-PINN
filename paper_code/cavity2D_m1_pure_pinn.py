# ruff: noqa: E402
"""Pure PINN for 2D Driven Cavity Flow (Lid-Driven Square Cavity).

Method: Pure Physics-Informed Neural Network
- Random collocation points (domain interior + boundary)
- MLP for velocity (u,v) and pressure (p)
- Navier-Stokes residuals (momentum + continuity)
- Dirichlet BC: no-slip walls, parabolic lid on top
- Training: Adam + L-BFGS

Outputs:
- Streamline plot with velocity field
- Loss convergence history (Adam + L-BFGS phases)
- Cached data (.npz)
"""

import os
import sys
import time

os.environ["MPLBACKEND"] = "Agg"

PLOT_ONLY = "--plot-only" in sys.argv

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

OUT_DIR = "/Users/gyh/WorkBuddy/20260415163831/"
DATA_FILE = OUT_DIR + "pure_pinn_driven_cavity_data.npz"
LOSS_FILE = OUT_DIR + "pure_pinn_driven_cavity_losses.npz"

# ─── Physical parameters ───────────────────────────────────────────────
Re = 100.0
nu = 2.0 / Re

# ─── Sampling parameters ──────────────────────────────────────────────
N_interior = 40  # points per direction for interior
N_boundary = 40  # points per direction for boundary

# ─── Network architecture ─────────────────────────────────────────────
HIDDEN_V = [64, 64, 64, 64]  # velocity MLP hidden layers
HIDDEN_P = [64, 64, 64, 64]  # pressure MLP hidden layers

# ─── Training parameters ──────────────────────────────────────────────
ADAM_EPOCHS = 5000
LBFGS_EPOCHS = 10000
SEED = 2002

if not PLOT_ONLY:
    from flax import nnx
    import sympy as sp

    from jaxfun.operators import Div, Dot, Grad
    from jaxfun.pinns.bcs import DirichletBC
    from jaxfun.pinns.loss import Loss
    from jaxfun.pinns.mesh import Rectangle
    from jaxfun.pinns.module import Comp, FlaxFunction
    from jaxfun.pinns.nnspaces import MLPSpace
    from jaxfun.pinns.optimizer import Trainer, adam, lbfgs

    print("JAX running on", jax.devices()[0].platform.upper())
    print(f"Re = {Re}, nu = {nu}")
    print(f"Interior: {N_interior}×{N_interior}, Boundary: {N_boundary}×{N_boundary}")
    print(f"Velocity MLP: {HIDDEN_V}, Pressure MLP: {HIDDEN_P}")

    # ─── Mesh & collocation points ─────────────────────────────────────
    mesh = Rectangle(-1, 1, -1, 1)

    xyi = mesh.get_points_inside_domain(N_interior, N_interior, "random")
    xyb = mesh.get_points_on_domain(N_boundary, N_boundary, "random")
    Ni = xyi.shape[0]
    Nb = xyb.shape[0]
    xyp = jnp.array([[0.0, 0.0]])

    print(f"Interior collocation points: {Ni}")
    print(f"Boundary collocation points: {Nb}")

    # ─── Neural network spaces ─────────────────────────────────────────
    V = MLPSpace(HIDDEN_V, dims=2, rank=1, name="V")
    Q = MLPSpace(HIDDEN_P, dims=2, rank=0, name="Q")

    u = FlaxFunction(V, "u", rngs=nnx.Rngs(SEED))
    p = FlaxFunction(Q, "p", rngs=nnx.Rngs(SEED))

    # ─── PDE residuals ─────────────────────────────────────────────────
    eq1 = Dot(Grad(u), u) - nu * Div(Grad(u)) + Grad(p)  # momentum
    eq2 = Div(u)  # continuity

    module = Comp(u, p)
    x_sym, y_sym = V.system.base_scalars()

    # ─── Boundary condition ────────────────────────────────────────────
    # No-slip on walls (x=-1, x=1, y=-1), parabolic lid on top (y=1)
    ub = DirichletBC(
        u, xyb,
        sp.Piecewise((0, y_sym < 1), ((1 - x_sym) ** 2 * (1 + x_sym) ** 2, True)),
        0,
    )

    # ─── Loss function ─────────────────────────────────────────────────
    loss_fn = Loss(
        (eq1, xyi),  # momentum residual (mean over interior)
        (eq2, xyi),  # continuity residual (mean over interior)
        (u, xyb, ub, 2.0 / Nb),  # boundary condition
        (p, xyp, 0, 10),  # pressure pin-point
    )

    # ─── Optimizers ────────────────────────────────────────────────────
    opt_adam = adam(module)
    opt_lbfgs = lbfgs(module, memory_size=100)

    trainer = Trainer(loss_fn)

    # ─── Training: Adam phase ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 1: Adam optimizer ({ADAM_EPOCHS} epochs)")
    print(f"{'='*60}")

    t0 = time.time()
    trainer.train(opt_adam, ADAM_EPOCHS, epoch_print=1000)
    adam_time = time.time() - t0
    adam_losses = list(trainer.losses)
    print(f"Adam finished in {adam_time:.1f}s, final loss: {adam_losses[-1]:.6e}")

    # ─── Training: L-BFGS phase ───────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 2: L-BFGS optimizer ({LBFGS_EPOCHS} epochs)")
    print(f"{'='*60}")

    t1 = time.time()
    trainer.train(opt_lbfgs, LBFGS_EPOCHS, epoch_print=2000, abs_limit_change=0)
    lbfgs_time = time.time() - t1
    lbfgs_losses = list(trainer.losses)
    print(f"L-BFGS finished in {lbfgs_time:.1f}s, final loss: {lbfgs_losses[-1]:.6e}")

    total_time = adam_time + lbfgs_time
    print(f"\nTotal training time: {total_time:.1f}s")

    # ─── Post-processing: evaluate on fine grid ────────────────────────
    print("\nPost-processing...")
    N_eval = 100
    yj = jnp.linspace(-1, 1, N_eval)
    xx, yy = jnp.meshgrid(yj, yj, sparse=False, indexing="ij")
    z = jnp.column_stack((xx.ravel(), yy.ravel()))
    uvp = module(z)

    U = np.array(uvp[:, 0].reshape(xx.shape))
    V_vel = np.array(uvp[:, 1].reshape(xx.shape))
    P = np.array(uvp[:, 2].reshape(xx.shape))
    xx_np = np.array(xx)
    yy_np = np.array(yy)

    # ─── Save data ─────────────────────────────────────────────────────
    np.savez(
        DATA_FILE,
        xx=xx_np, yy=yy_np, U=U, V=V_vel, P=P,
        Re=Re, nu=nu,
        N_interior=N_interior, N_boundary=N_boundary,
        hidden_v=str(HIDDEN_V), hidden_p=str(HIDDEN_P),
        adam_epochs=ADAM_EPOCHS, lbfgs_epochs=LBFGS_EPOCHS,
        adam_time=adam_time, lbfgs_time=lbfgs_time,
        final_loss=lbfgs_losses[-1],
    )
    print(f"Data saved to {DATA_FILE}")

    np.savez(
        LOSS_FILE,
        adam_losses=np.array(adam_losses),
        lbfgs_losses=np.array(lbfgs_losses),
    )
    print(f"Loss history saved to {LOSS_FILE}")

else:
    print(f"Loading cached data from {DATA_FILE}")
    data = np.load(DATA_FILE)
    xx_np = data["xx"]
    yy_np = data["yy"]
    U = data["U"]
    V_vel = data["V"]
    P = data["P"]
    Re = float(data["Re"])
    N_eval = xx_np.shape[0]

    print(f"Loading loss history from {LOSS_FILE}")
    loss_data = np.load(LOSS_FILE)
    adam_losses = list(loss_data["adam_losses"])
    lbfgs_losses = list(loss_data["lbfgs_losses"])

# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════

speed = np.sqrt(U ** 2 + V_vel ** 2)

# ─── Figure 1: Loss convergence history ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"Pure PINN — Driven Cavity Flow (Re={Re:.0f})",
    fontsize=14, fontweight="bold",
)

# Left: full loss history (semilogy)
ax = axes[0]
adam_epochs = np.arange(1, len(adam_losses) + 1)
lbfgs_epochs = np.arange(len(adam_losses) + 1, len(adam_losses) + len(lbfgs_losses) + 1)

ax.semilogy(adam_epochs, adam_losses, "b-", linewidth=0.8, alpha=0.8, label="Adam")
ax.semilogy(lbfgs_epochs, lbfgs_losses, "r-", linewidth=0.8, alpha=0.8, label="L-BFGS")
ax.axvline(x=len(adam_losses) + 0.5, color="gray", linestyle="--", alpha=0.5, label="Phase switch")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (log scale)")
ax.set_title("Loss Convergence History")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which="both")

# Right: zoomed final convergence
ax = axes[1]
# Show last 1000 points of Adam + all L-BFGS
adam_tail = max(0, len(adam_losses) - 1000)
adam_x = np.arange(adam_tail + 1, len(adam_losses) + 1)
ax.semilogy(adam_x, adam_losses[adam_tail:], "b-", linewidth=0.8, alpha=0.8, label="Adam (tail)")
ax.semilogy(lbfgs_epochs, lbfgs_losses, "r-", linewidth=0.8, alpha=0.8, label="L-BFGS")
ax.axvline(x=len(adam_losses) + 0.5, color="gray", linestyle="--", alpha=0.5, label="Phase switch")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (log scale)")
ax.set_title("Loss Convergence (Zoomed)")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
fig.savefig(OUT_DIR + "pure_pinn_loss_convergence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Loss convergence plot saved.")

# ─── Figure 2: Streamlines + Speed contour ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    f"Pure PINN — Driven Cavity (Re={Re:.0f})",
    fontsize=14, fontweight="bold",
)

# Left: streamlines
strm = axes[0].streamplot(
    xx_np[:, 0], yy_np[0, :],
    U.T, V_vel.T,
    color=speed.T, cmap="jet",
    density=1.5, linewidth=1.2, arrowsize=1.0,
    arrowstyle="->", minlength=0.15,
)
axes[0].set_title("Streamlines")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_aspect("equal")
plt.colorbar(strm.lines, ax=axes[0], label="|U|", shrink=0.8)

# Right: speed contour
cf = axes[1].contourf(xx_np, yy_np, speed, 60, cmap="jet")
axes[1].set_title("Speed |U|")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_aspect("equal")
plt.colorbar(cf, ax=axes[1], shrink=0.8)

plt.tight_layout()
fig.savefig(OUT_DIR + "pure_pinn_streamlines.png", dpi=150, bbox_inches="tight")
plt.close()
print("Streamline plot saved.")

# ─── Figure 3: u, v, p contour plots ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f"Pure PINN — Driven Cavity (Re={Re:.0f}) Field Variables",
    fontsize=14, fontweight="bold",
)

for i, (ax, label) in enumerate(zip(axes, ["u (x-velocity)", "v (y-velocity)", "p (pressure)"])):
    field = np.array([U, V_vel, P][i])
    levels = 100
    im = ax.contourf(xx_np, yy_np, field, levels, cmap="RdBu_r" if i < 2 else "coolwarm")
    ax.set_title(label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
fig.savefig(OUT_DIR + "pure_pinn_field_contours.png", dpi=150, bbox_inches="tight")
plt.close()
print("Field contour plot saved.")

# ─── Summary ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Pure PINN Summary")
print(f"{'='*60}")
print(f"  Re = {Re}")
print(f"  Velocity MLP: {HIDDEN_V}")
print(f"  Pressure MLP: {HIDDEN_P}")
print(f"  Interior points: {N_interior}×{N_interior} (random)")
print(f"  Boundary points: {N_boundary}×{N_boundary} (random)")
print(f"  Adam: {len(adam_losses)} epochs")
print(f"  L-BFGS: {len(lbfgs_losses)} epochs")
if not PLOT_ONLY:
    print(f"  Adam time: {adam_time:.1f}s")
    print(f"  L-BFGS time: {lbfgs_time:.1f}s")
    print(f"  Total time: {total_time:.1f}s")
print(f"  Final loss: {lbfgs_losses[-1]:.6e}")
print(f"  Max |u|: {np.abs(U).max():.6f}")
print(f"  Max |v|: {np.abs(V_vel).max():.6f}")
print(f"  Max |p|: {np.abs(P).max():.6f}")
print(f"{'='*60}")

print("\nSUCCESS")
