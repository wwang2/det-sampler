"""Generate fig1-fig4 from results.json."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

with open(os.path.join(HERE, "results.json")) as f:
    R = json.load(f)


def curves_to_array(curves):
    omegas = sorted(float(o) for o in curves.keys())
    Qs = sorted(float(q) for q in curves[str(omegas[0])].keys())
    grid = np.array([[curves[str(o)][str(q)] for q in Qs] for o in omegas])
    return np.array(omegas), np.array(Qs), grid


# ---------- Fig 1 ----------
om, Q, grid_lo = curves_to_array(R["logosc_curves"])
fit_lo = R["fit_logosc"]

fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)
log_grid = np.log10(np.clip(grid_lo, 1, None))
im = ax.pcolormesh(np.log10(Q), np.log10(om), log_grid,
                   cmap="viridis", shading="auto")
plt.colorbar(im, ax=ax, label=r"$\log_{10}\,\tau_{int}(q^2)$")
# Q_opt overlay
q_opt_per_om = Q[np.argmin(grid_lo, axis=1)]
ax.plot(np.log10(q_opt_per_om), np.log10(om), "wo-", lw=1.5, mec="k",
        label=r"$Q_{opt}(\omega)$")
# Fit line
om_fit = np.linspace(np.log10(om.min()), np.log10(om.max()), 50)
q_fit = fit_lo["intercept"] / np.log(10) + fit_lo["slope"] * om_fit
ax.plot(q_fit, om_fit, "r--", lw=1.5,
        label=fr"fit slope$={fit_lo['slope']:.2f}$")
ax.set_xlabel(r"$\log_{10} Q$")
ax.set_ylabel(r"$\log_{10} \omega$")
ax.set_title("log-osc: single-mode landscape")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig1_single_mode_landscape.png"))
plt.close()

# ---------- Fig 2 ----------
fig, axes = plt.subplots(2, 1, figsize=(6, 5), dpi=150,
                         gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
ax = axes[0]
ax.loglog(om, q_opt_per_om, "o", ms=8, mec="k", mfc="C0",
          label="log-osc empirical")
om_line = np.logspace(np.log10(om.min()), np.log10(om.max()), 50)
ax.loglog(om_line, fit_lo["c"] * om_line ** fit_lo["slope"], "C0--",
          label=fr"$Q_{{opt}}={fit_lo['c']:.2f}\,\omega^{{{fit_lo['slope']:.2f}}}$, $R^2={fit_lo['r2']:.3f}$")
# NHC
om_n, Q_n, grid_nhc = curves_to_array(R["nhc_curves"])
fit_nhc = R["fit_nhc"]
q_opt_nhc = Q_n[np.argmin(grid_nhc, axis=1)]
ax.loglog(om_n, q_opt_nhc, "s", ms=8, mec="k", mfc="C3",
          label="NHC empirical")
ax.loglog(om_line, fit_nhc["c"] * om_line ** fit_nhc["slope"], "C3--",
          label=fr"$Q_{{opt}}={fit_nhc['c']:.2f}\,\omega^{{{fit_nhc['slope']:.2f}}}$, $R^2={fit_nhc['r2']:.3f}$")
ax.set_ylabel(r"$Q_{opt}$")
ax.set_title(r"Empirical $Q_{opt}(\omega)$ scaling")
ax.legend(fontsize=8, loc="best")
ax.grid(True, which="both", alpha=0.3)
# residuals
ax2 = axes[1]
res_lo = np.log(q_opt_per_om) - (np.log(fit_lo["c"]) + fit_lo["slope"] * np.log(om))
res_nhc = np.log(q_opt_nhc) - (np.log(fit_nhc["c"]) + fit_nhc["slope"] * np.log(om_n))
ax2.semilogx(om, res_lo, "o-", color="C0", label="log-osc")
ax2.semilogx(om_n, res_nhc, "s-", color="C3", label="NHC")
ax2.axhline(0, color="k", lw=0.5)
ax2.set_xlabel(r"$\omega$")
ax2.set_ylabel("log resid")
ax2.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig2_q_opt_law.png"))
plt.close()

# ---------- Fig 3 ----------
part3 = R["part3_2d"]
om2_vals = sorted(float(k) for k in part3.keys())
fig, axes = plt.subplots(1, len(om2_vals), figsize=(4.2 * len(om2_vals), 4),
                         dpi=150, sharey=True)
for ax, om2 in zip(axes, om2_vals):
    d = part3[str(om2)]
    Q1 = np.array(d["Q1_grid"])
    Q2 = np.array(d["Q2_grid"])
    tau = np.array(d["tau_grid"])
    log_tau = np.log10(np.clip(tau, 1, None))
    im = ax.pcolormesh(np.log10(Q1), np.log10(Q2), log_tau.T,
                       cmap="viridis", shading="auto")
    plt.colorbar(im, ax=ax, label=r"$\log_{10}\max\tau$")
    # mark single-mode predictions: Q_opt(om=1) and Q_opt(om=om2)
    pred_Q1 = fit_lo["c"] * 1.0 ** fit_lo["slope"]
    pred_Q2 = fit_lo["c"] * om2 ** fit_lo["slope"]
    ax.plot(np.log10(pred_Q1), np.log10(pred_Q2), "r*", ms=18, mec="k",
            label="single-mode pred")
    # actual minimum
    i, j = np.unravel_index(np.nanargmin(tau), tau.shape)
    ax.plot(np.log10(Q1[i]), np.log10(Q2[j]), "wo", ms=10, mec="k",
            label="2D opt")
    ax.set_xlabel(r"$\log_{10} Q_1$")
    ax.set_title(fr"$\omega_2={om2:.0f}$")
    ax.legend(fontsize=7, loc="upper right")
axes[0].set_ylabel(r"$\log_{10} Q_2$")
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig3_two_mode_landscape.png"))
plt.close()

# ---------- Fig 4 ----------
fig, ax = plt.subplots(figsize=(6, 4.2), dpi=150)
ax.loglog(om, q_opt_per_om, "o-", color="C0", lw=1.5, ms=8, mec="k",
          label=fr"log-osc: $Q\propto\omega^{{{fit_lo['slope']:.2f}}}$")
ax.loglog(om_n, q_opt_nhc, "s-", color="C3", lw=1.5, ms=8, mec="k",
          label=fr"NHC(M=3): $Q\propto\omega^{{{fit_nhc['slope']:.2f}}}$")
# Reference: Q ~ 1/omega^2 (kT/omega^2)
ax.loglog(om_line, 1.0 / om_line ** 2, "k:", lw=1, alpha=0.6,
          label=r"$kT/\omega^2$ (NHC textbook)")
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"$Q_{opt}$")
ax.set_title(r"NHC vs log-osc: $Q_{opt}(\omega)$")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, loc="best")
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig4_nhc_vs_logosc_q_opt.png"))
plt.close()

print("Figures saved to", FIGS)
