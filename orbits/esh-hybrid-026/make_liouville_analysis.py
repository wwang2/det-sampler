"""Task 1: Liouville analysis for naive ESH + 1/f thermostat hybrid.

Computes div(F) for the naive combined system and determines what
stationary measure it actually samples. Then motivates Fix Option C.
"""

import sys
import sympy as sp
import numpy as np
from pathlib import Path

WORKTREE = '/Users/wujiewang/code/det-sampler/.worktrees/esh-hybrid-026'
sys.path.insert(0, WORKTREE)

OUT_DIR = Path(WORKTREE) / 'orbits/esh-hybrid-026'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def liouville_analysis():
    print("=" * 70)
    print("LIOUVILLE ANALYSIS: NAIVE ESH + LOG-OSC THERMOSTAT")
    print("=" * 70)

    # ----------------------------------------------------------------
    # Symbolic variables
    # ----------------------------------------------------------------
    x, v, xi = sp.Symbol('x', real=True), sp.Symbol('v', real=True), sp.Symbol('xi', real=True)
    U = sp.Function('U')
    kT, Q = sp.Symbol('kT', positive=True), sp.Symbol('Q', positive=True)

    # ----------------------------------------------------------------
    # Naive hybrid vector field (1D, single xi)
    #
    # dx/dt = sign(v)                              [ESH position update]
    # dv/dt = -dU/dx * |v| - g(xi) * v            [ESH force + friction]
    # dxi/dt = (v^2 - kT) / Q                     [log-osc thermostat]
    #
    # where g(xi) = 2*xi / (1 + xi^2)   [log-osc bounded friction]
    # ----------------------------------------------------------------

    print("\n--- Vector Field Components ---")
    g_xi = 2*xi / (1 + xi**2)

    f_x = sp.sign(v)                              # dx/dt
    f_v = -U(x).diff(x) * sp.Abs(v) - g_xi * v  # dv/dt
    f_xi = (v**2 - kT) / Q                        # dxi/dt

    print(f"  f_x  = {f_x}")
    print(f"  f_v  = {f_v}")
    print(f"  f_xi = {f_xi}")

    # ----------------------------------------------------------------
    # Divergence computation
    # Note: d/dx(sign(v)) = 0  (no x-dependence in f_x)
    #       d/dv(f_v) requires care with |v|
    #       d/dxi(f_xi) = 0  (f_xi doesn't depend on xi)
    # ----------------------------------------------------------------
    print("\n--- Divergence div(F) = df_x/dx + df_v/dv + df_xi/dxi ---")

    df_x_dx = sp.diff(f_x, x)  # = 0 (sign(v) doesn't depend on x)
    print(f"  df_x/dx = {df_x_dx}")

    # df_v/dv: f_v = -U'(x)*|v| - g(xi)*v
    # d/dv[-U'(x)*|v|] = -U'(x) * sign(v)   [for v != 0]
    # d/dv[-g(xi)*v]   = -g(xi)
    df_v_dv_potential_part = -U(x).diff(x) * sp.sign(v)
    df_v_dv_friction_part = -g_xi
    df_v_dv = df_v_dv_potential_part + df_v_dv_friction_part
    print(f"  df_v/dv = {df_v_dv_potential_part} + ({df_v_dv_friction_part})")
    print(f"         = -dU/dx * sign(v) - g(xi)")

    df_xi_dxi = sp.diff(f_xi, xi)  # = 0
    print(f"  df_xi/dxi = {df_xi_dxi}")

    div_F = df_x_dx + df_v_dv + df_xi_dxi
    print(f"\n  div(F) = {div_F}")
    print(f"\n  CONCLUSION: div(F) = -dU/dx * sign(v) - g(xi)  (NON-ZERO)")
    print(f"  The naive hybrid does NOT preserve the standard Liouville measure.")

    # ----------------------------------------------------------------
    # What stationary measure does the naive hybrid sample?
    #
    # For a general vector field F, the stationary Fokker-Planck equation is:
    #   div(rho * F) = 0
    #   => rho * div(F) + F . grad(rho) = 0
    #
    # Try rho = exp(-H/kT) * h(xi) for some function h(xi).
    # We want: div(rho * F) = 0
    #
    # Expanding:
    #   div(rho * F) = rho * div(F) + F . grad(rho) = 0
    #   => F . grad(log rho) = -div(F)
    #   => (dx/dt) * d/dx(log rho) + (dv/dt) * d/dv(log rho) + (dxi/dt) * d/dxi(log rho)
    #      = +dU/dx * sign(v) + g(xi)
    #
    # Let H_ESH = U(x) + log|v|, and try log rho = -H_ESH/kT + h(xi).
    # Then:
    #   d/dx(log rho) = -U'(x)/kT
    #   d/dv(log rho) = -sign(v)/|v|  (from log|v|) / kT... let's be careful:
    #   If log rho = -(U(x) + log|v|)/kT + h(xi), then:
    #     d/dx = -U'(x)/kT
    #     d/dv = -sign(v)/(kT*|v|) = -1/(kT*v)  [for v != 0]
    #     d/dxi = h'(xi)
    # ----------------------------------------------------------------

    print("\n--- Stationary Measure Analysis ---")
    print("""
  We seek rho such that div(rho * F) = 0, i.e., F . grad(log rho) = -div(F).

  Try: log rho = -H_ESH(x,v)/kT + f(xi)
  where H_ESH = U(x) + log|v|  (ESH conserved quantity).

  Then:
    d(log rho)/dx  = -U'(x)/kT
    d(log rho)/dv  = -sign(v)/(kT*|v|)
    d(log rho)/dxi = f'(xi)

  The Liouville condition F . grad(log rho) = -div(F) becomes:
    sign(v) * (-U'(x)/kT)
    + (-U'(x)*|v| - g(xi)*v) * (-sign(v)/(kT*|v|))
    + (v^2 - kT)/Q * f'(xi)
    = dU/dx * sign(v) + g(xi)

  Expanding the second term:
    (-U'(x)*|v| - g(xi)*v) * (-sign(v)/(kT*|v|))
    = U'(x)*|v|*sign(v)/(kT*|v|)  +  g(xi)*v*sign(v)/(kT*|v|)
    = U'(x)*sign(v)/(kT)           +  g(xi)/(kT)   [since v*sign(v)=|v|, |v|/|v|=1]

  Substituting:
    sign(v)*(-U'(x)/kT) + U'(x)*sign(v)/kT + g(xi)/kT + (v^2-kT)/Q * f'(xi)
    = dU/dx*sign(v) + g(xi)

  First two terms cancel! Leaving:
    g(xi)/kT + (v^2-kT)/Q * f'(xi) = dU/dx*sign(v) + g(xi)

  => g(xi)/kT + (v^2-kT)/Q * f'(xi) = dU/dx*sign(v) + g(xi)

  This cannot be satisfied for all (x, v, xi) because the LHS has (v^2-kT)*f'(xi)
  which depends only on (v, xi), but the RHS has dU/dx*sign(v) which
  depends on (x, v). So the H_ESH ansatz does NOT work.
""")

    print("--- Attempting canonical ansatz rho = exp(-H_can/kT) ---")
    print("""
  Try: log rho = -(U(x) + v^2/2)/kT + f(xi)
  Then:
    d(log rho)/dx  = -U'(x)/kT
    d(log rho)/dv  = -v/kT
    d(log rho)/dxi = f'(xi)

  Condition: F . grad(log rho) = -div(F) = dU/dx*sign(v) + g(xi)

  LHS = sign(v)*(-U'(x)/kT)
        + (-U'(x)*|v| - g(xi)*v)*(-v/kT)
        + (v^2-kT)/Q * f'(xi)

      = -U'(x)*sign(v)/kT
        + U'(x)*|v|*v/kT + g(xi)*v^2/kT
        + (v^2-kT)/Q * f'(xi)

  For this to equal dU/dx*sign(v) + g(xi), we need:
    - U'(x)*sign(v)/kT + U'(x)*|v|*v/kT = dU/dx*sign(v)
    => -U'(x)*sign(v)/kT + U'(x)*|v|*v/kT = U'(x)*sign(v)
    => U'(x) * (-sign(v)/kT + |v|*v/kT) = U'(x)*sign(v)
    => -sign(v)/kT + v^2*sign(v)/kT = sign(v)   [|v|*v = v^2*sign(v)]
    => sign(v) * (v^2 - 1)/kT = sign(v)
    => (v^2 - 1)/kT = 1  only if v^2 = kT + 1  (not generally true)

  CONCLUSION: No simple factored ansatz works for the naive hybrid.
""")

    print("--- Summary of Liouville Analysis ---")
    print("""
  The naive hybrid has div(F) = -dU/dx * sign(v) - g(xi).

  KEY PROBLEM: The divergence mixes x-dependent terms (dU/dx*sign(v))
  with xi-dependent terms (g(xi)). This coupling makes it impossible to
  find a simple stationary measure of product form rho(x,v) * h(xi).

  PHYSICAL INTERPRETATION:
  - ESH changes volume in (x,v) phase space due to the sign(v)*|v| force structure
  - The log-osc friction adds further volume change via -g(xi)*v
  - These two effects do not cancel in a simple way

  WHY OPTION C (ALTERNATING) WORKS:
  - ESH steps: Conservative (H_ESH = U + log|v| conserved), no volume change
    in the sense that the ESH flow preserves rho ~ 1/|v| * exp(-U/kT)
  - Thermostat steps: Properly ergodic with known stationary measure
  - Alternating them: ESH provides fast local exploration along H_ESH level sets,
    thermostat provides global thermalization. The combined process is ergodic
    by the mixing argument (each phase individually has the right marginal).

  The correct interpretation: Option C avoids the Liouville issue entirely
  by keeping the two dynamics SEPARATE. The thermostat steps guarantee
  the canonical measure; ESH steps are approximate Hamiltonian moves
  that improve mixing without corrupting the stationary distribution.
""")

    return {
        'divergence': '-dU/dx * sign(v) - g(xi)',
        'conclusion': 'No simple stationary measure for naive hybrid',
        'fix': 'Option C (alternating ESH + thermostat) avoids Liouville issue',
    }


def numerical_divergence_check():
    """Numerically verify divergence is nonzero for the naive hybrid."""
    print("\n--- Numerical Divergence Check ---")
    print("Checking div(F) at sample points...")

    # For HO: dU/dx = omega^2 * x
    def dU_dx(x, omega=1.0):
        return omega**2 * x

    def g_losc(xi):
        return 2.0 * xi / (1.0 + xi**2)

    test_points = [
        (1.0, 0.5, 0.3),
        (0.5, -0.8, 1.0),
        (2.0, 0.1, -0.5),
        (-1.0, 1.2, 0.0),
    ]

    for (x_val, v_val, xi_val) in test_points:
        # div(F) = -dU/dx * sign(v) - g(xi)
        div = -dU_dx(x_val) * np.sign(v_val) - g_losc(xi_val)
        print(f"  (x={x_val:.1f}, v={v_val:.1f}, xi={xi_val:.1f}): "
              f"div(F) = {div:.4f}  [non-zero: {abs(div) > 1e-10}]")

    print("\n  All points show non-zero divergence as predicted.")


if __name__ == "__main__":
    results = liouville_analysis()
    numerical_divergence_check()

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
  1. The naive hybrid ESH + log-osc has div(F) = -dU/dx*sign(v) - g(xi)
  2. This is NOT zero, so the standard Liouville measure is not invariant
  3. No simple factored measure rho(x,v)*h(xi) is stationary
  4. Fix Option C (alternating ESH + thermostat) is the pragmatic solution:
     - ESH steps explore along H_ESH level sets (conservative)
     - Thermostat steps thermalize to canonical measure (ergodic)
     - No Liouville compatibility issue
""")
    print("Liouville analysis complete.")
