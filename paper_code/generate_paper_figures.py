"""
generate_paper_figures_final.py
Generate all paper figures including M5 (SEM-VPINN) results.
5 methods: M1 Pure PINN, M2 Spectral LSTSQ, M3 Spectral VPINN, M4 SEM-LSTSQ, M5 SEM-VPINN
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import os

OUT = 'paper_figures'
os.makedirs(OUT, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
data = {}
DATA_DIR = "/Users/gyh/WorkBuddy/20260415163831/paper_data"

specs = {
    'M1': (f'{DATA_DIR}/cavity2D_m1_data.npz',        'M1: Pure PINN'),
    'M2': (f'{DATA_DIR}/cavity2D_m2_data.npz',        'M2: Spectral LSTSQ'),
    'M3': (f'{DATA_DIR}/cavity2D_m3_data.npz',        'M3: Spectral VPINN'),
    'M4': (f'{DATA_DIR}/cavity2D_m4_data.npz',        'M4: SEM-LSTSQ'),
    'M5': (f'{DATA_DIR}/cavity2D_m5_data.npz',        'M5: SEM-VPINN'),
}
for key, (fname, label) in specs.items():
    d = np.load(fname)
    data[key] = {
        'U': d['U'], 'V': d['V'], 'P': d['P'],
        'xx': d['xx'], 'yy': d['yy'],
        'label': label,
        'loss': float(d['final_loss']),
        'time_h': float(d.get('total_time', d.get('adam_time', 0) + d.get('lbfgs_time', 0))) / 3600,
        'adam_t': float(d.get('adam_time', 0)) / 3600,
        'lbfgs_t': float(d.get('lbfgs_time', 0)) / 3600,
    }

# Reference = M2
ref = data['M2']
for key in ['M1', 'M3', 'M4', 'M5']:
    d = data[key]
    d['eps_u'] = np.linalg.norm(d['U'] - ref['U']) / np.linalg.norm(ref['U'])
    d['eps_v'] = np.linalg.norm(d['V'] - ref['V']) / np.linalg.norm(ref['V'])
    d['eps_p'] = np.linalg.norm(d['P'] - ref['P']) / (np.linalg.norm(ref['P']) + 1e-14)
data['M2']['eps_u'] = 0.0
data['M2']['eps_v'] = 0.0
data['M2']['eps_p'] = 0.0

# Load loss histories
losses = {}
loss_files = {
    'M1': f'{DATA_DIR}/cavity2D_m1_losses.npz',
    'M2': f'{DATA_DIR}/cavity2D_m2_losses.npz',
    'M3': f'{DATA_DIR}/cavity2D_m3_losses.npz',
    'M4': f'{DATA_DIR}/cavity2D_m4_losses.npz',
}
for key, fname in loss_files.items():
    try:
        ld = np.load(fname)
        losses[key] = {k: ld[k] for k in ld.files}
    except:
        losses[key] = {}

COLORS = {'M1': '#e41a1c', 'M2': '#377eb8', 'M3': '#ff7f00', 'M4': '#4daf4a', 'M5': '#984ea3'}
MARKERS = {'M1': 'o', 'M2': 's', 'M3': '^', 'M4': 'D', 'M5': 'v'}

# ── Fig 1: Streamlines (2×3 layout with 5 methods + empty) ────────────────
print("Generating fig1_streamlines_5m.png ...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, key in enumerate(['M1', 'M2', 'M3', 'M4', 'M5']):
    ax = axes[idx]
    d = data[key]
    xx2d, yy2d = d['xx'], d['yy']
    # streamplot needs 1D arrays
    x1d = xx2d[:, 0] if xx2d.ndim == 2 else xx2d
    y1d = yy2d[0, :] if yy2d.ndim == 2 else yy2d
    U, V = d['U'], d['V']
    speed = np.sqrt(U**2 + V**2)
    strm = ax.streamplot(x1d, y1d, U.T, V.T, color=speed.T,
                         cmap='viridis', linewidth=0.8, density=1.5,
                         arrowsize=0.8)
    ax.set_title(d['label'], fontsize=12, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    cbar = plt.colorbar(strm.lines, ax=ax, label='|u|', fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=8)

# Turn off unused subplot
axes[5].set_visible(False)

fig.suptitle(r'Streamlines colored by $|\mathbf{u}|$, Re = 100', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig1_cavity_streamlines.png', dpi=150, bbox_inches='tight')
plt.close()
print("  done")

# ── Fig 2: u-velocity contours (2×3) ──────────────────────────────────────
print("Generating fig2_u_contours_5m.png ...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
vmin = min(data[k]['U'].min() for k in data)
vmax = max(data[k]['U'].max() for k in data)
levels = np.linspace(vmin, vmax, 21)

for idx, key in enumerate(['M1', 'M2', 'M3', 'M4', 'M5']):
    ax = axes[idx]
    d = data[key]
    x1d = d['xx'][:, 0]; y1d = d['yy'][0, :]
    cf = ax.contourf(x1d, y1d, d['U'].T, levels=levels, cmap='RdBu_r')
    ax.contour(x1d, y1d, d['U'].T, levels=levels[::4], colors='k', linewidths=0.4, alpha=0.4)
    ax.set_title(d['label'], fontsize=12, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='u', fraction=0.046)

axes[5].set_visible(False)
fig.suptitle(r'$u$-velocity field, Re = 100', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig2_cavity_u_contours.png', dpi=150, bbox_inches='tight')
plt.close()
print("  done")

# ── Fig 3: Centerline profiles ─────────────────────────────────────────────
print("Generating fig3_centerline_5m.png ...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for key in ['M1', 'M2', 'M3', 'M4', 'M5']:
    d = data[key]
    x1d = d['xx'][:, 0]; y1d = d['yy'][0, :]
    U, V = d['U'], d['V']
    # vertical centerline: u(x≈0, y)
    ix = np.argmin(np.abs(x1d))
    axes[0].plot(U[ix, :], y1d, color=COLORS[key], marker=MARKERS[key],
                 markevery=10, label=d['label'], linewidth=1.8)
    # horizontal centerline: v(x, y≈0)
    iy = np.argmin(np.abs(y1d))
    axes[1].plot(x1d, V[:, iy], color=COLORS[key], marker=MARKERS[key],
                 markevery=10, label=d['label'], linewidth=1.8)

axes[0].set_xlabel(r'$u$-velocity', fontsize=12)
axes[0].set_ylabel(r'$y$', fontsize=12)
axes[0].set_title(r'$u(x\approx 0, y)$ — vertical centerline', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel(r'$x$', fontsize=12)
axes[1].set_ylabel(r'$v$-velocity', fontsize=12)
axes[1].set_title(r'$v(x, y\approx 0)$ — horizontal centerline', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

fig.suptitle('Centerline velocity profiles, Re = 100', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig3_cavity_centerline.png', dpi=150, bbox_inches='tight')
plt.close()
print("  done")

# ── Fig 4: Loss convergence ────────────────────────────────────────────────
print("Generating fig4_loss_5m.png ...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Adam phase
for key, ld in losses.items():
    if 'adam_losses' in ld:
        al = ld['adam_losses']
        axes[0].semilogy(np.arange(len(al)) + 1, al,
                         color=COLORS[key], label=data[key]['label'], linewidth=1.5)
axes[0].set_xlabel('Adam epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Phase 1: Adam optimizer', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, which='both', alpha=0.3)

# L-BFGS phase
for key, ld in losses.items():
    if 'lbfgs_losses' in ld:
        ll = ld['lbfgs_losses']
        ll = ll[ll > 0]
        axes[1].semilogy(np.arange(len(ll)) + 1, ll,
                         color=COLORS[key], label=data[key]['label'], linewidth=1.5)
axes[1].set_xlabel('L-BFGS step', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Phase 2: L-BFGS optimizer', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, which='both', alpha=0.3)

# Mark final losses with horizontal dashed lines
for key in ['M1', 'M2', 'M3', 'M4']:
    axes[1].axhline(data[key]['loss'], color=COLORS[key], linestyle=':', alpha=0.6, linewidth=1)

# Add M5 final loss annotation (no history available)
axes[1].axhline(data['M5']['loss'], color=COLORS['M5'], linestyle=':', alpha=0.6, linewidth=1,
                label=f"M5 final: {data['M5']['loss']:.2e}")
axes[1].legend(fontsize=9)

fig.suptitle('Training loss convergence', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_cavity_loss.png', dpi=150, bbox_inches='tight')
plt.close()
print("  done")

# ── Fig 5: Pressure fields (2×3) ──────────────────────────────────────────
print("Generating fig5_pressure_5m.png ...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
pall = np.concatenate([data[k]['P'].ravel() for k in data])
plevels = np.linspace(pall.min(), pall.max(), 21)

for idx, key in enumerate(['M1', 'M2', 'M3', 'M4', 'M5']):
    ax = axes[idx]
    d = data[key]
    x1d = d['xx'][:, 0]; y1d = d['yy'][0, :]
    cf = ax.contourf(x1d, y1d, d['P'].T, levels=plevels, cmap='seismic')
    ax.set_title(d['label'], fontsize=12, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='p', fraction=0.046)

axes[5].set_visible(False)
fig.suptitle('Pressure field $p$, Re = 100', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_cavity_pressure.png', dpi=150, bbox_inches='tight')
plt.close()
print("  done")

# ── Fig 6: Relative errors bar chart ──────────────────────────────────────
print("Generating fig6_errors_5m.png ...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

keys_cmp = ['M1', 'M3', 'M4', 'M5']
labels_cmp = [data[k]['label'] for k in keys_cmp]
x = np.arange(len(keys_cmp))

for ax_idx, (field, title) in enumerate([('eps_u', r'$\epsilon_u$ (u-velocity)'),
                                          ('eps_v', r'$\epsilon_v$ (v-velocity)'),
                                          ('eps_p', r'$\epsilon_p$ (pressure)')]):
    ax = axes[ax_idx]
    vals = [data[k][field] for k in keys_cmp]
    bars = ax.bar(x, vals, color=[COLORS[k] for k in keys_cmp], alpha=0.85, width=0.6)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([k for k in keys_cmp], fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel('Relative L2 error', fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.15,
                f'{val:.2e}', ha='center', va='bottom', fontsize=9, rotation=0)

fig.suptitle('Relative $\\ell_2$ errors vs. M2 (Spectral LSTSQ) reference', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig6_cavity_errors.png', dpi=150, bbox_inches='tight')
plt.close()
print("  done")

# ── Fig 7: v-velocity contours ─────────────────────────────────────────────
print("Generating fig7_v_contours_5m.png ...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
vmin_v = min(data[k]['V'].min() for k in data)
vmax_v = max(data[k]['V'].max() for k in data)
vlevels = np.linspace(vmin_v, vmax_v, 21)

for idx, key in enumerate(['M1', 'M2', 'M3', 'M4', 'M5']):
    ax = axes[idx]
    d = data[key]
    x1d = d['xx'][:, 0]; y1d = d['yy'][0, :]
    cf = ax.contourf(x1d, y1d, d['V'].T, levels=vlevels, cmap='RdBu_r')
    ax.set_title(d['label'], fontsize=12, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='v', fraction=0.046)

axes[5].set_visible(False)
fig.suptitle(r'$v$-velocity field, Re = 100', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig7_cavity_v_contours.png', dpi=150, bbox_inches='tight')
plt.close()
print("  done")

# ── Fig 8: Efficiency chart ────────────────────────────────────────────────
print("Generating fig8_efficiency_5m.png ...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

keys_all = ['M1', 'M2', 'M3', 'M4', 'M5']
labels_all = [data[k]['label'] for k in keys_all]
x = np.arange(len(keys_all))

# Training time (stacked: Adam + L-BFGS)
adam_times = [data[k]['adam_t'] for k in keys_all]
lbfgs_times = [data[k]['lbfgs_t'] for k in keys_all]
bars1 = axes[0].bar(x, adam_times, color=[COLORS[k] for k in keys_all], alpha=0.6,
                    label='Adam', width=0.6)
bars2 = axes[0].bar(x, lbfgs_times, bottom=adam_times,
                    color=[COLORS[k] for k in keys_all], alpha=0.95, label='L-BFGS', width=0.6,
                    hatch='//')
axes[0].set_xticks(x)
axes[0].set_xticklabels([k for k in keys_all], fontsize=11)
axes[0].set_ylabel('Training time (h)', fontsize=12)
axes[0].set_title('Computational cost', fontsize=12)
axes[0].legend(['Adam phase', 'L-BFGS phase'], fontsize=10)
axes[0].grid(True, axis='y', alpha=0.3)
for xi, (at, lt) in enumerate(zip(adam_times, lbfgs_times)):
    total = at + lt
    axes[0].text(xi, total + 0.05, f'{total:.2f}h', ha='center', fontsize=9)

# Final loss
losses_all = [data[k]['loss'] for k in keys_all]
bars3 = axes[1].bar(x, losses_all, color=[COLORS[k] for k in keys_all], alpha=0.85, width=0.6)
axes[1].set_yscale('log')
axes[1].set_xticks(x)
axes[1].set_xticklabels([k for k in keys_all], fontsize=11)
axes[1].set_ylabel('Final training loss', fontsize=12)
axes[1].set_title('Solution accuracy', fontsize=12)
axes[1].grid(True, axis='y', alpha=0.3)
for bar, val in zip(bars3, losses_all):
    axes[1].text(bar.get_x() + bar.get_width()/2, val * 1.5,
                 f'{val:.2e}', ha='center', va='bottom', fontsize=8.5)

fig.suptitle('Efficiency and accuracy comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig8_cavity_efficiency.png', dpi=150, bbox_inches='tight')
plt.close()
print("  done")

# ── Fig 9: Vorticity fields ────────────────────────────────────────────────
print("Generating fig9_vorticity_5m.png ...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, key in enumerate(['M1', 'M2', 'M3', 'M4', 'M5']):
    ax = axes[idx]
    d = data[key]
    x1d = d['xx'][:, 0]; y1d = d['yy'][0, :]
    U, V = d['U'], d['V']
    dx = x1d[1] - x1d[0]
    dy = y1d[1] - y1d[0]
    # vorticity = dv/dx - du/dy  (finite difference)
    dVdx = np.gradient(V, dx, axis=0)
    dUdy = np.gradient(U, dy, axis=1)
    omega = dVdx - dUdy
    vmax_w = np.percentile(np.abs(omega), 99)
    cf = ax.contourf(x1d, y1d, omega.T, levels=np.linspace(-vmax_w, vmax_w, 21),
                     cmap='RdBu_r', extend='both')
    ax.set_title(d['label'], fontsize=12, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label=r'$\omega$', fraction=0.046)

axes[5].set_visible(False)
fig.suptitle(r'Vorticity $\omega = \partial v/\partial x - \partial u/\partial y$, Re = 100',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig9_cavity_vorticity.png', dpi=150, bbox_inches='tight')
plt.close()
print("  done")

# ── Fig 10: Pointwise error fields |u - u_ref| ────────────────────────────
print("Generating fig10_error_5m.png ...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, key in enumerate(['M1', 'M3', 'M4', 'M5']):
    ax = axes[idx]
    d = data[key]
    x1d = d['xx'][:, 0]; y1d = d['yy'][0, :]
    err_u = np.abs(d['U'] - ref['U'])
    cf = ax.contourf(x1d, y1d, err_u.T, levels=21, cmap='hot_r')
    ax.set_title(f"{d['label']}\n" + r"$|u - u_{\rm ref}|$", fontsize=11, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='|error|', fraction=0.046)

fig.suptitle(r'Pointwise $|u - u_{\rm ref}|$ error (ref = M2 Spectral LSTSQ)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig10_cavity_error_fields.png', dpi=150, bbox_inches='tight')
plt.close()
print("  done")

# ── Print summary ──────────────────────────────────────────────────────────
print("\n" + "="*70)
print("COMPLETE RESULTS SUMMARY")
print("="*70)
print(f"{'Method':<20} {'Loss':>12} {'eps_u':>10} {'eps_v':>10} {'eps_p':>10} {'Time(h)':>8}")
print("-"*70)
for key in ['M1', 'M2', 'M3', 'M4', 'M5']:
    d = data[key]
    eu = d['eps_u']; ev = d['eps_v']; ep = d['eps_p']
    print(f"{d['label']:<20} {d['loss']:>12.3e} {eu:>10.3e} {ev:>10.3e} {ep:>10.3e} {d['time_h']:>8.3f}")
print("="*70)
print("All figures saved to paper_figures/")
