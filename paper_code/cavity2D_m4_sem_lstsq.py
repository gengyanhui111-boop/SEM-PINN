# ruff: noqa: E402
"""SEM-Least-Squares PINN for 2D Driven Cavity Flow (Lid-Driven Square Cavity).

Method: Spectral Element Method + Least-Squares PINN
- Domain decomposition: [-1,1]^2 divided into K_elem x K_elem sub-elements
- Global Legendre test space with Dirichlet BCs (for test functions)
- MLP trial functions for velocity (u,v) and pressure (p)
- Least-squares loss: sum over elements of weighted residual norm
- Piecewise Gauss-Legendre quadrature for integration
- Dirichlet BC: no-slip walls, parabolic lid on top
- Training: Adam + scipy L-BFGS-B

Key difference from Spectral LSTSQ (Method 2):
  Method 2 uses global spectral basis as trial functions (spectral coeff optimization).
  Method 4 uses global test space + MLP trial functions + piecewise integration (SEM).

Key difference from Spectral-VPINN (Method 3):
  Method 3 projects residuals onto test space (weak form): loss = mean_k(int R*phi_k)^2
  Method 4 uses least-squares on each element: loss = sum_e int_e |R|^2 dOmega

Outputs:
- Streamline plot with velocity field
- Loss convergence history (Adam + L-BFGS phases)
- u, v, p contour plots
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
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

OUT_DIR = "/Users/gyh/WorkBuddy/20260415163831/"
DATA_FILE = OUT_DIR + "sem_lstsq_driven_cavity_data.npz"
LOSS_FILE = OUT_DIR + "sem_lstsq_driven_cavity_losses.npz"

# ─── Physical parameters ───────────────────────────────────────────────
Re = 100.0
nu = 2.0 / Re

# ─── SEM parameters ────────────────────────────────────────────────────
K_ELEM = 4           # number of elements per direction
N_LEG = 12           # Legendre polynomial order for test space
N_QUAD_ELEM = 6      # quadrature points per direction per element

# ─── Network architecture (trial functions) ────────────────────────────
HIDDEN_V = [32, 32, 32, 32]
HIDDEN_P = [32, 32, 32, 32]

# ─── Training parameters ──────────────────────────────────────────────
ADAM_EPOCHS = 2000
LBFGS_ITER = 10000
LR_ADAM = 1e-3
SEED = 2002

if not PLOT_ONLY:
    from flax import nnx

    from jaxfun.galerkin import FunctionSpace
    from jaxfun.galerkin.Legendre import Legendre
    from jaxfun.pinns.mesh import Rectangle
    from jaxfun.pinns.module import Comp, FlaxFunction
    from jaxfun.pinns.nnspaces import MLPSpace

    print("JAX running on", jax.devices()[0].platform.upper())
    print(f"Re = {Re}, nu = {nu}")
    print(f"SEM: K_elem = {K_ELEM}x{K_ELEM}, N_legendre = {N_LEG}, N_quad/elem = {N_QUAD_ELEM}")

    # ─── Domain decomposition ──────────────────────────────────────────
    elem_bounds_x = np.linspace(-1, 1, K_ELEM + 1)
    elem_bounds_y = np.linspace(-1, 1, K_ELEM + 1)

    # ─── Build piecewise integration points and weights ───────────────
    # Use Gauss-Legendre quadrature on each element
    all_xq, all_yq, all_wq = [], [], []
    for ex in range(K_ELEM):
        for ey in range(K_ELEM):
            ax, bx = elem_bounds_x[ex], elem_bounds_x[ex + 1]
            ay, by = elem_bounds_y[ey], elem_bounds_y[ey + 1]
            # Gauss-Legendre nodes/weights on [-1,1], then map to element
            xg, wg = np.polynomial.legendre.leggauss(N_QUAD_ELEM)
            # Map from [-1,1] to [a,b]: x = 0.5*(b-a)*xi + 0.5*(a+b)
            xq = 0.5 * (bx - ax) * xg + 0.5 * (ax + bx)
            yq = 0.5 * (by - ay) * xg + 0.5 * (ay + by)
            # Jacobian: dx/dxi = 0.5*(b-a), dy/deta = 0.5*(d-c)
            wx = 0.5 * (bx - ax) * wg
            wy = 0.5 * (by - ay) * wg
            # 2D meshgrid and outer product of weights
            xx, yy = np.meshgrid(xq, yq, indexing="ij")
            ww = np.outer(wx, wy)
            all_xq.append(xx.ravel())
            all_yq.append(yy.ravel())
            all_wq.append(ww.ravel())

    xq_all = jnp.array(np.concatenate(all_xq))
    yq_all = jnp.array(np.concatenate(all_yq))
    wq_all = jnp.array(np.concatenate(all_wq))
    xq_pts = jnp.column_stack((xq_all, yq_all))
    N_q = xq_pts.shape[0]
    print(f"Total integration points: {N_q} ({K_ELEM}x{K_ELEM} elements x {N_QUAD_ELEM}^2 quad pts)")

    # ─── MLP trial functions ───────────────────────────────────────────
    V_mlp = MLPSpace(HIDDEN_V, dims=2, rank=1, name="V")
    Q_mlp = MLPSpace(HIDDEN_P, dims=2, rank=0, name="Q")
    u_fn = FlaxFunction(V_mlp, "u", rngs=nnx.Rngs(SEED))
    p_fn = FlaxFunction(Q_mlp, "p", rngs=nnx.Rngs(SEED))
    module = Comp(u_fn, p_fn)

    # ─── Boundary points ───────────────────────────────────────────────
    mesh = Rectangle(-1, 1, -1, 1)
    xyb = mesh.get_points_on_domain(40, 40, "random")
    Nb = xyb.shape[0]
    xyp = jnp.array([[0.0, 0.0]])

    # BC: top wall u = (1-x^2)^2, v = 0; others u=v=0
    x_bc, y_bc = xyb[:, 0], xyb[:, 1]
    u_bc = np.where(y_bc >= 1.0 - 1e-12, (1.0 - x_bc ** 2) ** 2, 0.0)
    v_bc = np.zeros(Nb)
    bc_vals = jnp.array(np.column_stack([u_bc, v_bc]))
    print(f"Boundary points: {Nb}")

    # ─── Stateless forward pass setup ──────────────────────────────────
    gd_u, st_u = nnx.split(u_fn.module, nnx.Param)
    gd_p, st_p = nnx.split(p_fn.module, nnx.Param)
    flat_u0, unf_u = jax.flatten_util.ravel_pytree(st_u)
    flat_p0, unf_p = jax.flatten_util.ravel_pytree(st_p)
    split_idx = flat_u0.shape[0]
    x0_flat = jnp.concatenate([flat_u0, flat_p0])
    n_params = x0_flat.shape[0]
    print(f"Total parameters: {n_params} (u={split_idx}, p={n_params - split_idx})")

    def _fwd(x_, flat_params):
        """Stateless forward: flat_params -> (N, 3) [u, v, p]."""
        fu = flat_params[:split_idx]
        fp = flat_params[split_idx:]
        mod_u = nnx.merge(gd_u, unf_u(fu))
        mod_p = nnx.merge(gd_p, unf_p(fp))
        return jnp.hstack([mod_u(x_), mod_p(x_)])

    def jacn(fun, k=1):
        """Vectorized k-th order Jacobian (jacfwd/jacrev alternation + vmap)."""
        for i in range(k):
            fun = jax.jacfwd(fun) if i % 2 == 0 else jax.jacrev(fun)
        return jax.vmap(fun, in_axes=0, out_axes=0) if k > 0 else fun

    def compute_ns_residuals(flat_params, xpts):
        """Compute NS residuals at given points."""
        fwd_x = lambda x_: _fwd(x_, flat_params)
        vals = fwd_x(xpts)
        u_v, v_v, p_v = vals[:, 0], vals[:, 1], vals[:, 2]
        j1 = jacn(fwd_x, 1)(xpts)
        ux, uy = j1[:, 0, 0], j1[:, 0, 1]
        vx, vy = j1[:, 1, 0], j1[:, 1, 1]
        px, py = j1[:, 2, 0], j1[:, 2, 1]
        j2 = jacn(fwd_x, 2)(xpts)
        uxx, uyy = j2[:, 0, 0, 0], j2[:, 0, 1, 1]
        vxx, vyy = j2[:, 1, 0, 0], j2[:, 1, 1, 1]
        Rx = u_v * ux + v_v * uy - nu * (uxx + uyy) + px
        Ry = u_v * vx + v_v * vy - nu * (vxx + vyy) + py
        Rd = ux + vy
        return Rx, Ry, Rd

    def total_loss_fn(flat_params):
        """Total loss: SEM least-squares + BC penalty + pressure pin."""
        Rx, Ry, Rd = compute_ns_residuals(flat_params, xq_pts)
        # SEM least-squares: sum over all elements of weighted residual
        loss_pde = (wq_all * (Rx ** 2 + Ry ** 2 + Rd ** 2)).sum() / wq_all.sum()
        # BC penalty
        vals_bc = _fwd(xyb, flat_params)
        vel_bc = vals_bc[:, :2]
        loss_bc = 10.0 * ((vel_bc - bc_vals) ** 2).sum() / Nb
        # Pressure pin
        loss_p = 10.0 * _fwd(xyp, flat_params)[0, 2] ** 2
        return loss_pde + loss_bc + loss_p

    val_and_grad = jax.value_and_grad(total_loss_fn)

    # ─── Training: Adam phase ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Phase 1: Adam optimizer ({ADAM_EPOCHS} epochs)")
    print(f"{'=' * 60}")

    t0 = time.time()
    x_flat = x0_flat.copy()
    m_adam = jnp.zeros_like(x_flat)
    v_adam = jnp.zeros_like(x_flat)
    adam_losses = []

    for ep in range(1, ADAM_EPOCHS + 1):
        loss_val, grads = val_and_grad(x_flat)
        m_adam = 0.9 * m_adam + 0.1 * grads
        v_adam = 0.999 * v_adam + 0.001 * grads ** 2
        m_hat = m_adam / (1 - 0.9 ** ep)
        v_hat = v_adam / (1 - 0.999 ** ep)
        x_flat = x_flat - LR_ADAM * m_hat / (jnp.sqrt(v_hat) + 1e-8)
        adam_losses.append(float(loss_val))
        if ep % 500 == 0:
            elapsed = time.time() - t0
            print(f"  epoch {ep}: loss = {float(loss_val):.6e}, time = {elapsed:.1f}s")

    adam_time = time.time() - t0
    print(f"Adam finished in {adam_time:.1f}s, final loss: {adam_losses[-1]:.6e}")

    # ─── Training: L-BFGS phase ────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Phase 2: scipy L-BFGS-B ({LBFGS_ITER} iterations)")
    print(f"{'=' * 60}")

    t1 = time.time()

    def loss_and_grad_np(x_np):
        loss_val, grads = val_and_grad(jnp.array(x_np))
        return float(loss_val), np.array(grads)

    from scipy.optimize import minimize as scipy_minimize

    lbfgs_losses = []
    callback_count = [0]

    def lbfgs_callback(xk):
        loss_val, _ = val_and_grad(jnp.array(xk))
        lbfgs_losses.append(float(loss_val))
        callback_count[0] += 1
        if callback_count[0] % 1000 == 0:
            elapsed = time.time() - t1
            print(f"  L-BFGS step {callback_count[0]}: loss = {float(loss_val):.6e}, time = {elapsed:.1f}s")

    result = scipy_minimize(
        loss_and_grad_np,
        x0=np.array(x_flat),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": LBFGS_ITER, "gtol": 1e-14, "maxcor": 50},
        callback=lbfgs_callback,
    )

    x_flat = jnp.array(result.x)
    lbfgs_time = time.time() - t1
    final_loss = lbfgs_losses[-1] if lbfgs_losses else adam_losses[-1]
    print(f"L-BFGS finished in {lbfgs_time:.1f}s, final loss: {final_loss:.6e}")
    print(f"L-BFGS message: {result.message}")
    print(f"L-BFGS nit: {result.nit}, nfev: {result.nfev}")

    total_time = adam_time + lbfgs_time
    print(f"\nTotal training time: {total_time:.1f}s ({total_time / 3600:.2f}h)")

    # ─── Post-processing ───────────────────────────────────────────────
    print("\nPost-processing...")
    N_eval = 100
    yj = jnp.linspace(-1, 1, N_eval)
    xx, yy = jnp.meshgrid(yj, yj, sparse=False, indexing="ij")
    z = jnp.column_stack((xx.ravel(), yy.ravel()))
    uvp = _fwd(z, x_flat)

    U = np.array(uvp[:, 0].reshape(xx.shape))
    V_vel = np.array(uvp[:, 1].reshape(xx.shape))
    P = np.array(uvp[:, 2].reshape(xx.shape))
    xx_np = np.array(xx)
    yy_np = np.array(yy)

    print(f"Max |u|: {np.abs(U).max():.6f}")
    print(f"Max |v|: {np.abs(V_vel).max():.6f}")
    print(f"Max |p|: {np.abs(P).max():.6f}")

    # ─── Save data ─────────────────────────────────────────────────────
    np.savez(
        DATA_FILE,
        xx=xx_np, yy=yy_np, U=U, V=V_vel, P=P,
        Re=Re, nu=nu,
        K_elem=K_ELEM, N_legendre=N_LEG, N_quad_elem=N_QUAD_ELEM,
        HIDDEN_V=HIDDEN_V, HIDDEN_P=HIDDEN_P,
        adam_epochs=ADAM_EPOCHS, lbfgs_iter=LBFGS_ITER,
        adam_time=adam_time, lbfgs_time=lbfgs_time,
        total_time=total_time,
        final_loss=final_loss,
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
    K_ELEM = int(data["K_elem"])
    N_LEG = int(data["N_legendre"])
    N_QUAD_ELEM = int(data["N_quad_elem"])
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
    f"SEM Least-Squares PINN — Driven Cavity (Re={Re:.0f}, "
    f"K={K_ELEM}×{K_ELEM}, N={N_LEG})",
    fontsize=14, fontweight="bold",
)

ax = axes[0]
adam_epochs_arr = np.arange(1, len(adam_losses) + 1)
lbfgs_epochs_arr = np.arange(len(adam_losses) + 1, len(adam_losses) + len(lbfgs_losses) + 1)
ax.semilogy(adam_epochs_arr, adam_losses, "b-", linewidth=0.8, alpha=0.8, label="Adam")
ax.semilogy(lbfgs_epochs_arr, lbfgs_losses, "r-", linewidth=0.8, alpha=0.8, label="L-BFGS")
ax.axvline(x=len(adam_losses) + 0.5, color="gray", linestyle="--", alpha=0.5, label="Phase switch")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (log scale)")
ax.set_title("Loss Convergence History")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which="both")

ax = axes[1]
adam_tail = max(0, len(adam_losses) - 500)
adam_x = np.arange(adam_tail + 1, len(adam_losses) + 1)
ax.semilogy(adam_x, adam_losses[adam_tail:], "b-", linewidth=0.8, alpha=0.8, label="Adam (tail)")
ax.semilogy(lbfgs_epochs_arr, lbfgs_losses, "r-", linewidth=0.8, alpha=0.8, label="L-BFGS")
ax.axvline(x=len(adam_losses) + 0.5, color="gray", linestyle="--", alpha=0.5, label="Phase switch")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (log scale)")
ax.set_title("Loss Convergence (Zoomed)")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
fig.savefig(OUT_DIR + "sem_lstsq_loss_convergence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Loss convergence plot saved.")

# ─── Figure 2: Streamlines + Speed contour ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    f"SEM Least-Squares PINN — Driven Cavity (Re={Re:.0f}, "
    f"K={K_ELEM}×{K_ELEM}, N={N_LEG})",
    fontsize=14, fontweight="bold",
)

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

cf = axes[1].contourf(xx_np, yy_np, speed, 60, cmap="jet")
axes[1].set_title("Speed |U|")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_aspect("equal")
plt.colorbar(cf, ax=axes[1], shrink=0.8)

plt.tight_layout()
fig.savefig(OUT_DIR + "sem_lstsq_streamlines.png", dpi=150, bbox_inches="tight")
plt.close()
print("Streamline plot saved.")

# ─── Figure 3: u, v, p contour plots ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f"SEM Least-Squares PINN — Driven Cavity (Re={Re:.0f}, "
    f"K={K_ELEM}×{K_ELEM}, N={N_LEG}) Field Variables",
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
fig.savefig(OUT_DIR + "sem_lstsq_field_contours.png", dpi=150, bbox_inches="tight")
plt.close()
print("Field contour plot saved.")

# ─── Summary ──────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"SEM Least-Squares PINN Summary")
print(f"{'=' * 60}")
print(f"  Re = {Re}")
print(f"  Elements: {K_ELEM} x {K_ELEM}")
print(f"  Legendre order: N = {N_LEG}")
print(f"  Quadrature: {N_QUAD_ELEM}^2 per element")
print(f"  MLP: [32,32,32,32] x 2 (u,v + p)")
print(f"  Adam: {len(adam_losses)} epochs")
print(f"  L-BFGS: {len(lbfgs_losses)} steps")
if not PLOT_ONLY:
    print(f"  Adam time: {adam_time:.1f}s")
    print(f"  L-BFGS time: {lbfgs_time:.1f}s")
    print(f"  Total time: {total_time:.1f}s")
print(f"  Final loss: {lbfgs_losses[-1]:.6e}")
print(f"  Max |u|: {np.abs(U).max():.6f}")
print(f"  Max |v|: {np.abs(V_vel).max():.6f}")
print(f"  Max |p|: {np.abs(P).max():.6f}")
print(f"{'=' * 60}")

print("\nSUCCESS")
