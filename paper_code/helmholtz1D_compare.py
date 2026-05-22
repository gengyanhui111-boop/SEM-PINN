"""Step 3: Generate comparison plots from saved results."""
import numpy as np
import matplotlib.pyplot as plt

spec = np.load("helmholtz_spectral_results.npz", allow_pickle=True)
vpnn = np.load("helmholtz_vpinn_results.npz", allow_pickle=True)

xj = spec["xj"].flatten()
uej = spec["uej"].flatten()
uj_spec = spec["uj"].flatten()
w_vpinn = vpnn["w_pred"].flatten()

error_spec = float(spec["error"])
error_vpinn = float(vpnn["error"])
time_spec = float(spec["time"])
time_vpinn = float(vpnn["time_adam"]) + float(vpnn["time_lbfgs"])

# === Figure 1: Solution comparison ===
fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)

axes[0].plot(xj, uej, "r-", linewidth=2.0, label="Exact: $(1-x^2)\\cos(2\\pi x)$")
axes[0].plot(xj, uj_spec, "b--", linewidth=1.5, label="Spectral Galerkin (N=80)")
axes[0].fill_between(xj, uej, uj_spec, alpha=0.12, color="blue")
max_err_s = np.max(np.abs(uj_spec - uej))
axes[0].set_title(
    f"Spectral Galerkin — L2 Error = {error_spec:.2e}, Max Error = {max_err_s:.2e}, Time = {time_spec:.1f}s",
    fontsize=13,
)
axes[0].set_xlabel("x"); axes[0].set_ylabel("u(x)")
axes[0].legend(fontsize=11); axes[0].grid(True, alpha=0.3); axes[0].set_xlim(-1, 1)

axes[1].plot(xj, uej, "r-", linewidth=2.0, label="Exact: $(1-x^2)\\cos(2\\pi x)$")
axes[1].plot(xj, w_vpinn, "g--", linewidth=1.5, label="VPINN (MLP 64, Legendre test N=32)")
axes[1].fill_between(xj, uej, w_vpinn, alpha=0.12, color="green")
max_err_v = np.max(np.abs(w_vpinn - uej))
axes[1].set_title(
    f"VPINN — L2 Error = {error_vpinn:.2e}, Max Error = {max_err_v:.2e}, Time = {time_vpinn:.0f}s",
    fontsize=13,
)
axes[1].set_xlabel("x"); axes[1].set_ylabel("u(x)")
axes[1].legend(fontsize=11); axes[1].grid(True, alpha=0.3); axes[1].set_xlim(-1, 1)

plt.savefig("helmholtz_comparison.png", dpi=150, bbox_inches="tight")
print("Saved: helmholtz_comparison.png")

# === Figure 2: Error analysis ===
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

err_s = np.abs(uj_spec - uej)
err_v = np.abs(w_vpinn - uej)

axes2[0].semilogy(xj, err_s + 1e-18, "b-", linewidth=1.2, label=f"Spectral (N=80)")
axes2[0].semilogy(xj, err_v + 1e-18, "g-", linewidth=1.2, label="VPINN")
axes2[0].set_title("Pointwise Error |u − u_e|", fontsize=13)
axes2[0].set_xlabel("x"); axes2[0].set_ylabel("|error| (log)")
axes2[0].legend(fontsize=11); axes2[0].grid(True, alpha=0.3); axes2[0].set_xlim(-1, 1)

methods = ["Spectral\nGalerkin", "VPINN"]
x_bar = np.arange(2)
width = 0.25
b1 = axes2[1].bar(x_bar - width, [error_spec, error_vpinn], width, label="L2 Error", color="steelblue")
b2 = axes2[1].bar(x_bar, [max_err_s, max_err_v], width, label="Max Error", color="darkorange")
b3 = axes2[1].bar(x_bar + width, [time_spec, time_vpinn], width, label="Time (s)", color="forestgreen")
axes2[1].set_yscale("log")
axes2[1].set_xticks(x_bar); axes2[1].set_xticklabels(methods, fontsize=11)
axes2[1].set_title("Method Comparison", fontsize=13)
axes2[1].legend(fontsize=10); axes2[1].grid(True, alpha=0.3, axis="y")

for bar, val in zip(b1, [error_spec, error_vpinn]):
    axes2[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.1e}", ha="center", va="bottom", fontsize=8)

plt.savefig("helmholtz_error_comparison.png", dpi=150, bbox_inches="tight")
print("Saved: helmholtz_error_comparison.png")

# === Summary ===
print("\n" + "=" * 70)
print(f"  {'Method':<28} {'L2 Error':<14} {'Max Error':<14} {'Time':<10}")
print(f"  {'-'*66}")
print(f"  {'Spectral Galerkin (N=80)':<28} {error_spec:<14.2e} {max_err_s:<14.2e} {time_spec:<8.1f}s")
print(f"  {'VPINN (MLP64 + Leg32)':<28} {error_vpinn:<14.2e} {max_err_v:<14.2e} {time_vpinn:<8.0f}s")
print("=" * 70)
