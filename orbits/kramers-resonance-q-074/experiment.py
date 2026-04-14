"""Orbit 074: Kramers-Resonance Multi-scale Q experiment.

Streamlined: Phase 1 uses single seed + 100k force evals for fast scan.
Phase 2 confirms best conditions with 3 seeds + 300k force evals.
"""

import sys
import time
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from research.eval.integrators import ThermostatState, VelocityVerletThermostat
from research.eval.evaluator import run_sampler
from research.eval.potentials import HarmonicOscillator1D, DoubleWell2D

import functools
print = functools.partial(print, flush=True)


# ---------------------------------------------------------------------------
# Potentials
# ---------------------------------------------------------------------------

class AnisotropicGaussian:
    def __init__(self, dim=2, kappa=100.0):
        self.dim = dim
        self.kappa = kappa
        if dim == 2:
            self.omegas = np.array([np.sqrt(kappa), 1.0])
        else:
            self.omegas = np.geomspace(np.sqrt(kappa), 1.0, dim)
        self.name = f"aniso_d{dim}_k{int(kappa)}"

    def energy(self, q):
        return 0.5 * np.sum(self.omegas**2 * q**2)

    def gradient(self, q):
        return self.omegas**2 * q


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

class NHLogOsc:
    name = "nh_logosc"
    def __init__(self, dim, kT=1.0, mass=1.0, Q=0.1):
        self.dim, self.kT, self.mass, self.Q = dim, kT, mass, Q

    def initial_state(self, q0, rng=None):
        rng = rng or np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        return ThermostatState(q0.copy(), p0, np.array([0.0]), 0)

    def dqdt(self, state, grad_U):
        return state.p / self.mass

    def dpdt(self, state, grad_U):
        xi = state.xi[0]
        g = 2.0 * xi / (1.0 + xi**2)
        return -grad_U - g * state.p

    def dxidt(self, state, grad_U):
        return np.array([(np.sum(state.p**2) / self.mass - self.dim * self.kT) / self.Q])


class NHCLogOsc:
    def __init__(self, dim, chain_length=2, kT=1.0, mass=1.0, Q=None, label="nhc_logosc"):
        self.dim, self.M, self.kT, self.mass, self.name = dim, chain_length, kT, mass, label
        if Q is None: self.Q = [1.0] * chain_length
        elif isinstance(Q, (int, float)): self.Q = [float(Q)] * chain_length
        else: self.Q = list(Q)

    def initial_state(self, q0, rng=None):
        rng = rng or np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        return ThermostatState(q0.copy(), p0, np.zeros(self.M), 0)

    def dqdt(self, state, grad_U):
        return state.p / self.mass

    def dpdt(self, state, grad_U):
        xi = state.xi[0]
        g = 2.0 * xi / (1.0 + xi**2)
        return -grad_U - g * state.p

    def dxidt(self, state, grad_U):
        xi, M = state.xi, self.M
        dxi = np.zeros(M)
        kinetic = np.sum(state.p**2) / self.mass
        dxi[0] = (kinetic - self.dim * self.kT) / self.Q[0]
        if M > 1: dxi[0] -= xi[1] * xi[0]
        for j in range(1, M - 1):
            dxi[j] = (self.Q[j-1] * xi[j-1]**2 - self.kT) / self.Q[j] - xi[j+1] * xi[j]
        if M > 1:
            dxi[M-1] = (self.Q[M-2] * xi[M-2]**2 - self.kT) / self.Q[M-1]
        return dxi


class NHCTanh:
    def __init__(self, dim, chain_length=3, kT=1.0, mass=1.0, Q=None, label="nhc_tanh"):
        self.dim, self.M, self.kT, self.mass, self.name = dim, chain_length, kT, mass, label
        if Q is None: self.Q = [1.0] * chain_length
        elif isinstance(Q, (int, float)): self.Q = [float(Q)] * chain_length
        else: self.Q = list(Q)

    def initial_state(self, q0, rng=None):
        rng = rng or np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        return ThermostatState(q0.copy(), p0, np.zeros(self.M), 0)

    def dqdt(self, state, grad_U):
        return state.p / self.mass

    def dpdt(self, state, grad_U):
        return -grad_U - np.tanh(state.xi[0]) * state.p

    def dxidt(self, state, grad_U):
        xi, M = state.xi, self.M
        dxi = np.zeros(M)
        kinetic = np.sum(state.p**2) / self.mass
        dxi[0] = (kinetic - self.dim * self.kT) / self.Q[0]
        if M > 1: dxi[0] -= xi[1] * xi[0]
        for j in range(1, M - 1):
            dxi[j] = (self.Q[j-1] * xi[j-1]**2 - self.kT) / self.Q[j] - xi[j+1] * xi[j]
        if M > 1:
            dxi[M-1] = (self.Q[M-2] * xi[M-2]**2 - self.kT) / self.Q[M-1]
        return dxi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_one(dyn, pot, dt, n_evals, kT, seed):
    rng = np.random.default_rng(seed)
    q0 = rng.normal(0, 0.5, size=pot.dim)
    r = run_sampler(dyn, pot, dt=dt, n_force_evals=n_evals, kT=kT,
                    q0=q0, rng=np.random.default_rng(seed))
    tau = r["ess_metrics"]["tau"] if r["ess_metrics"] else float("inf")
    kl = r["kl_divergence"] if r["kl_divergence"] is not None else float("inf")
    erg = r["ergodicity"]["score"] if r.get("ergodicity") and r["ergodicity"] else 0.0
    nan = r.get("nan_detected", False)
    return tau, kl, erg, nan


# ---------------------------------------------------------------------------
# PHASE 1: Fast scan (1 seed, 100k force evals)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()
    kT = 1.0
    seed1 = 42

    print("=" * 70)
    print("PHASE 1: FAST SCAN (1 seed, 100k force evals)")
    print("=" * 70)

    for kappa in [10, 100]:
        pot = AnisotropicGaussian(dim=2, kappa=kappa)
        omega_fast, omega_slow, dim = np.sqrt(kappa), 1.0, 2
        dt = 0.003 if kappa >= 100 else 0.005
        n_evals = 100_000

        print(f"\n--- Anisotropic Gaussian kappa={kappa} (omega_fast={omega_fast:.1f}) ---")
        Q_vals = [0.03, 0.1, 0.3, 1.0, 3.0]

        header = f"  {'Q':>5} | {'NH-losc':>8} | {'NHC2-uni':>8} | {'NHC2-Kr':>8} | {'NHC3-tanh':>9} | {'NHC2-KrR':>8}"
        print(header)
        print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}")

        for Q_val in Q_vals:
            row = [f"  {Q_val:>5}"]

            # NH log-osc
            dyn = NHLogOsc(dim=dim, kT=kT, Q=Q_val)
            tau, _, _, nan = run_one(dyn, pot, dt, n_evals, kT, seed1)
            row.append(f"{'NaN' if nan else f'{tau:.1f}':>8}")

            # NHC(M=2) uniform
            dyn = NHCLogOsc(dim=dim, chain_length=2, kT=kT, Q=[Q_val]*2)
            tau, _, _, nan = run_one(dyn, pot, dt, n_evals, kT, seed1)
            row.append(f"{'NaN' if nan else f'{tau:.1f}':>8}")

            # NHC(M=2) Kramers: Q_1=s*D*kT/wf^2, Q_2=s*D*kT/ws^2
            Qf = Q_val * dim * kT / omega_fast**2
            Qs = Q_val * dim * kT / omega_slow**2
            dyn = NHCLogOsc(dim=dim, chain_length=2, kT=kT, Q=[Qf, Qs])
            tau_kr, _, _, nan = run_one(dyn, pot, dt, n_evals, kT, seed1)
            row.append(f"{'NaN' if nan else f'{tau_kr:.1f}':>8}")

            # NHC(M=3) tanh uniform
            dyn = NHCTanh(dim=dim, chain_length=3, kT=kT, Q=[Q_val]*3)
            tau, _, _, nan = run_one(dyn, pot, dt, n_evals, kT, seed1)
            row.append(f"{'NaN' if nan else f'{tau:.1f}':>9}")

            # NHC(M=2) Kramers reversed (control)
            dyn = NHCLogOsc(dim=dim, chain_length=2, kT=kT, Q=[Qs, Qf])
            tau, _, _, nan = run_one(dyn, pot, dt, n_evals, kT, seed1)
            row.append(f"{'NaN' if nan else f'{tau:.1f}':>8}")

            print(" | ".join(row))

    # ===================================================================
    # Double-well
    # ===================================================================
    print(f"\n--- Double-Well 2D ---")
    pot_dw = DoubleWell2D()
    omega_x, omega_y = np.sqrt(8.0), 1.0
    Q_vals_dw = [0.1, 0.3, 1.0, 3.0]

    header = f"  {'Q':>5} | {'NH-losc KL':>10} | {'NHC2-uni KL':>11} | {'NHC2-Kr KL':>10} | {'NHC3-tanh KL':>12}"
    print(header)
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*11}-+-{'-'*10}-+-{'-'*12}")

    for Q_val in Q_vals_dw:
        row = [f"  {Q_val:>5}"]

        dyn = NHLogOsc(dim=2, kT=kT, Q=Q_val)
        _, kl, _, nan = run_one(dyn, pot_dw, 0.005, 200_000, kT, seed1)
        row.append(f"{'NaN' if nan else f'{kl:.4f}':>10}")

        dyn = NHCLogOsc(dim=2, chain_length=2, kT=kT, Q=[Q_val]*2)
        _, kl, _, nan = run_one(dyn, pot_dw, 0.005, 200_000, kT, seed1)
        row.append(f"{'NaN' if nan else f'{kl:.4f}':>11}")

        Qf = Q_val * 2 * kT / omega_x**2
        Qs = Q_val * 2 * kT / omega_y**2
        dyn = NHCLogOsc(dim=2, chain_length=2, kT=kT, Q=[Qf, Qs])
        _, kl, _, nan = run_one(dyn, pot_dw, 0.005, 200_000, kT, seed1)
        row.append(f"{'NaN' if nan else f'{kl:.4f}':>10}")

        dyn = NHCTanh(dim=2, chain_length=3, kT=kT, Q=[Q_val]*3)
        _, kl, _, nan = run_one(dyn, pot_dw, 0.005, 200_000, kT, seed1)
        row.append(f"{'NaN' if nan else f'{kl:.4f}':>12}")

        print(" | ".join(row))

    # ===================================================================
    # 1D Harmonic (ergodicity)
    # ===================================================================
    print(f"\n--- 1D Harmonic Oscillator (ergodicity) ---")
    pot_ho = HarmonicOscillator1D(omega=1.0)
    Q_vals_ho = [0.1, 0.3, 1.0, 3.0]

    header = f"  {'Q':>5} | {'NH-losc':>8} | {'NHC2-uni':>8} | {'NHC2-Kr':>8} | {'NHC3-tanh':>9}"
    print(header)
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}")

    for Q_val in Q_vals_ho:
        row = [f"  {Q_val:>5}"]

        dyn = NHLogOsc(dim=1, kT=kT, Q=Q_val)
        _, _, erg, _ = run_one(dyn, pot_ho, 0.005, 200_000, kT, seed1)
        row.append(f"{erg:>8.4f}")

        dyn = NHCLogOsc(dim=1, chain_length=2, kT=kT, Q=[Q_val]*2)
        _, _, erg, _ = run_one(dyn, pot_ho, 0.005, 200_000, kT, seed1)
        row.append(f"{erg:>8.4f}")

        dyn = NHCLogOsc(dim=1, chain_length=2, kT=kT, Q=[Q_val, Q_val])
        _, _, erg, _ = run_one(dyn, pot_ho, 0.005, 200_000, kT, seed1)
        row.append(f"{erg:>8.4f}")

        dyn = NHCTanh(dim=1, chain_length=3, kT=kT, Q=[Q_val]*3)
        _, _, erg, _ = run_one(dyn, pot_ho, 0.005, 200_000, kT, seed1)
        row.append(f"{erg:>9.4f}")

        print(" | ".join(row))

    t_phase1 = time.time() - t0
    print(f"\nPhase 1 wall time: {t_phase1:.0f}s ({t_phase1/60:.1f} min)")

    # ===================================================================
    # PHASE 2: Confirm best conditions with 3 seeds, 300k force evals
    # ===================================================================
    print(f"\n{'='*70}")
    print("PHASE 2: CONFIRMATION (3 seeds, 300k force evals)")
    print("=" * 70)

    seeds = [42, 123, 456]

    for kappa in [10, 100]:
        pot = AnisotropicGaussian(dim=2, kappa=kappa)
        omega_fast, omega_slow, dim = np.sqrt(kappa), 1.0, 2
        dt = 0.003 if kappa >= 100 else 0.005
        n_evals = 300_000

        print(f"\n--- Anisotropic Gaussian kappa={kappa} ---")

        # Test best Q values from phase 1 sweep: {0.1, 0.3, 1.0}
        Q_confirm = [0.1, 0.3, 1.0]

        methods = {}
        for Q_val in Q_confirm:
            for seed in seeds:
                # NH log-osc
                dyn = NHLogOsc(dim=dim, kT=kT, Q=Q_val)
                tau, _, _, _ = run_one(dyn, pot, dt, n_evals, kT, seed)
                methods.setdefault(f"NH_logosc_Q{Q_val}", []).append(tau)

                # NHC(M=2) uniform
                dyn = NHCLogOsc(dim=dim, chain_length=2, kT=kT, Q=[Q_val]*2)
                tau, _, _, _ = run_one(dyn, pot, dt, n_evals, kT, seed)
                methods.setdefault(f"NHC2_uni_Q{Q_val}", []).append(tau)

                # NHC(M=2) Kramers
                Qf = Q_val * dim * kT / omega_fast**2
                Qs = Q_val * dim * kT / omega_slow**2
                dyn = NHCLogOsc(dim=dim, chain_length=2, kT=kT, Q=[Qf, Qs])
                tau, _, _, _ = run_one(dyn, pot, dt, n_evals, kT, seed)
                methods.setdefault(f"NHC2_Kr_s{Q_val}", []).append(tau)

                # NHC(M=3) tanh
                dyn = NHCTanh(dim=dim, chain_length=3, kT=kT, Q=[Q_val]*3)
                tau, _, _, _ = run_one(dyn, pot, dt, n_evals, kT, seed)
                methods.setdefault(f"NHC3_tanh_Q{Q_val}", []).append(tau)

            print(f"  Q/s={Q_val}: done")

        print(f"\n  {'Method':<25} | {'mean tau':>9} | {'std':>6}")
        print(f"  {'-'*25}-+-{'-'*9}-+-{'-'*6}")
        for m in sorted(methods.keys()):
            vals = methods[m]
            print(f"  {m:<25} | {np.mean(vals):>9.1f} | {np.std(vals):>6.1f}")

    t_total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Total wall time: {t_total:.0f}s ({t_total/60:.1f} min)")
    print(f"{'='*70}")
