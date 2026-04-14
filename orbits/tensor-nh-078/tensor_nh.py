"""Tensor Nose-Hoover thermostat: symmetric friction matrix Xi_{ij}.

Instead of a single scalar xi monitoring tr(p^2), we use a symmetric
tensor Xi that monitors the FULL kinetic energy tensor p_i * p_j.

Equations of motion (2D):
    dq_i/dt = p_i / m
    dp_i/dt = -dU/dq_i - sum_j Xi_{ij} * p_j
    dXi_{ij}/dt = (p_i*p_j/m - kT*delta_{ij}) / Q

The invariant measure is:
    mu ~ exp(-beta*H(q,p)) * exp(-Q/(2kT) * sum_{ij} Xi_{ij}^2)

Proof of measure preservation (Liouville condition):
    div(flow) = -tr(Xi)
    flow . grad(log mu) = +tr(Xi)
    => div(flow) + flow . grad(log mu) = 0  QED

For 2D, Xi is symmetric => 3 independent variables: Xi_xx, Xi_xy, Xi_yy.
"""

import numpy as np
from scipy.linalg import expm
from research.eval.integrators import ThermostatState


# ---------------------------------------------------------------------------
# Packing/unpacking symmetric matrix <-> vector
# ---------------------------------------------------------------------------

def pack_symmetric_2d(mat):
    """Pack 2x2 symmetric matrix -> [xx, xy, yy]."""
    return np.array([mat[0, 0], mat[0, 1], mat[1, 1]])


def unpack_symmetric_2d(vec):
    """Unpack [xx, xy, yy] -> 2x2 symmetric matrix."""
    return np.array([[vec[0], vec[1]],
                     [vec[1], vec[2]]])


def pack_symmetric(mat, dim):
    """Pack dim x dim symmetric matrix -> vector of dim*(dim+1)/2 elements."""
    if dim == 2:
        return pack_symmetric_2d(mat)
    idx = []
    for i in range(dim):
        for j in range(i, dim):
            idx.append(mat[i, j])
    return np.array(idx)


def unpack_symmetric(vec, dim):
    """Unpack vector -> dim x dim symmetric matrix."""
    if dim == 2:
        return unpack_symmetric_2d(vec)
    mat = np.zeros((dim, dim))
    k = 0
    for i in range(dim):
        for j in range(i, dim):
            mat[i, j] = vec[k]
            mat[j, i] = vec[k]
            k += 1
    return mat


# ---------------------------------------------------------------------------
# Friction functions (applied element-wise to Xi matrix)
# ---------------------------------------------------------------------------

def friction_linear(Xi):
    """g(Xi) = Xi (standard linear friction)."""
    return Xi


def friction_log_osc(Xi):
    """g(Xi) = 2*Xi / (1 + Xi^2) — bounded, log-oscillator inspired.

    This is the derivative of log(1 + Xi^2), element-wise.
    Bounded: |g| <= 1, so friction cannot diverge.
    """
    return 2.0 * Xi / (1.0 + Xi ** 2)


# ---------------------------------------------------------------------------
# Tensor NH dynamics
# ---------------------------------------------------------------------------

class TensorNH:
    """Symmetric tensor Nose-Hoover thermostat.

    xi vector stores the upper-triangular entries of the symmetric Xi matrix.
    For dim=2: xi = [Xi_xx, Xi_xy, Xi_yy] (3 variables).
    For dim=1: xi = [Xi_xx] (1 variable, reduces to scalar NH).

    Parameters:
        dim: spatial dimension
        kT: temperature
        mass: particle mass
        Q: thermostat mass (uniform for all Xi components)
        Q_offdiag: if set, use different Q for off-diagonal components
        friction_fn: function Xi_matrix -> effective friction matrix
    """

    def __init__(self, dim: int, kT: float = 1.0, mass: float = 1.0,
                 Q: float = 1.0, Q_offdiag: float = None,
                 friction_fn=None, friction_name: str = "linear"):
        self.dim = dim
        self.kT = kT
        self.mass = mass
        self.Q_diag = Q
        self.Q_offdiag = Q_offdiag if Q_offdiag is not None else Q
        self.n_xi = dim * (dim + 1) // 2  # independent symmetric entries

        if friction_fn is None:
            self.friction_fn = friction_linear
            self.friction_name = "linear"
        else:
            self.friction_fn = friction_fn
            self.friction_name = friction_name

        self.name = f"tensor_nh_{friction_name}_d{dim}"

        # Build Q vector for each xi component
        self._Q_vec = np.zeros(self.n_xi)
        k = 0
        for i in range(dim):
            for j in range(i, dim):
                if i == j:
                    self._Q_vec[k] = self.Q_diag
                else:
                    self._Q_vec[k] = self.Q_offdiag
                k += 1

    def initial_state(self, q0: np.ndarray, rng: np.random.Generator = None) -> ThermostatState:
        if rng is None:
            rng = np.random.default_rng(42)
        p0 = rng.normal(0, np.sqrt(self.mass * self.kT), size=self.dim)
        xi0 = np.zeros(self.n_xi)
        return ThermostatState(q0.copy(), p0, xi0, 0)

    def _Xi_matrix(self, xi_vec):
        """Construct symmetric friction matrix from packed xi."""
        return unpack_symmetric(xi_vec, self.dim)

    def _driving_force(self, p):
        """Compute (p_i*p_j/m - kT*delta_{ij}), packed as vector."""
        outer = np.outer(p, p) / self.mass
        target = self.kT * np.eye(self.dim)
        deviation = outer - target
        return pack_symmetric(deviation, self.dim)

    def dqdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        return state.p / self.mass

    def dpdt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        Xi = self._Xi_matrix(state.xi)
        g_Xi = self.friction_fn(Xi)
        return -grad_U - g_Xi @ state.p

    def dxidt(self, state: ThermostatState, grad_U: np.ndarray) -> np.ndarray:
        driving = self._driving_force(state.p)
        return driving / self._Q_vec


# ---------------------------------------------------------------------------
# Custom integrator: BAOAB-style with matrix exponential for friction
# ---------------------------------------------------------------------------

class TensorNHIntegrator:
    """Velocity-Verlet style integrator for tensor NH thermostat.

    Splitting (palindromic):
        1. Half-step Xi:  xi += dt/2 * dxi/dt(p)
        2. Half-step p (friction): p = expm(-g(Xi)*dt/2) @ p
        3. Half-step p (kick):     p -= dt/2 * grad_U
        4. Full-step q:            q += dt * p/m
        5. Recompute grad_U
        6. Half-step p (kick):     p -= dt/2 * grad_U
        7. Half-step p (friction): p = expm(-g(Xi)*dt/2) @ p
        8. Half-step Xi:  xi += dt/2 * dxi/dt(p)

    The matrix exponential exp(-g(Xi)*dt/2) handles the tensor friction
    correctly, preserving time-reversibility.

    For 2x2 symmetric matrices, we use the analytical formula:
        exp(A) for symmetric A = V @ diag(exp(lambda)) @ V^T
    where V, lambda are the eigenvectors/values.
    """

    def __init__(self, dynamics, potential, dt: float,
                 kT: float = 1.0, mass: float = 1.0):
        self.dynamics = dynamics
        self.potential = potential
        self.dt = dt
        self.kT = kT
        self.mass = mass
        self._cached_grad_U = None

    def _matrix_exp_symmetric(self, A, scale):
        """Compute exp(scale * A) for symmetric A via eigendecomposition.

        More efficient and stable than scipy expm for small symmetric matrices.
        """
        if A.shape[0] == 1:
            return np.array([[np.exp(scale * A[0, 0])]])
        elif A.shape[0] == 2:
            # Analytical 2x2 symmetric eigendecomposition
            a, b, d = A[0, 0], A[0, 1], A[1, 1]
            trace = a + d
            det = a * d - b * b
            disc = max(0.0, trace * trace / 4.0 - det)
            sqrt_disc = np.sqrt(disc)
            lam1 = trace / 2.0 + sqrt_disc
            lam2 = trace / 2.0 - sqrt_disc

            e1 = np.exp(scale * lam1)
            e2 = np.exp(scale * lam2)
            # Clamp
            e1 = np.clip(e1, 1e-15, 1e15)
            e2 = np.clip(e2, 1e-15, 1e15)

            if abs(b) < 1e-30:
                # Diagonal
                return np.array([[np.exp(scale * a), 0.0],
                                 [0.0, np.exp(scale * d)]])

            # Eigenvectors for symmetric 2x2
            # v1 = [b, lam1 - a], v2 = [b, lam2 - a]
            v1 = np.array([b, lam1 - a])
            v2 = np.array([b, lam2 - a])
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-30 or n2 < 1e-30:
                return np.diag([np.exp(scale * a), np.exp(scale * d)])
            v1 /= n1
            v2 /= n2
            V = np.column_stack([v1, v2])
            return V @ np.diag([e1, e2]) @ V.T
        else:
            return expm(scale * A)

    def step(self, state: ThermostatState) -> ThermostatState:
        q, p, xi, n_evals = state
        dt = self.dt
        half_dt = 0.5 * dt
        dyn = self.dynamics

        # Get gradient (FSAL cache)
        if self._cached_grad_U is not None:
            grad_U = self._cached_grad_U
        else:
            grad_U = self.potential.gradient(q)
            n_evals += 1

        # 1. Half-step Xi
        xi_dot = dyn.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        # 2. Half-step p (friction via matrix exponential)
        Xi = dyn._Xi_matrix(xi)
        g_Xi = dyn.friction_fn(Xi)
        exp_neg = self._matrix_exp_symmetric(g_Xi, -half_dt)
        p = exp_neg @ p

        # 3. Half-step p (kick)
        p = p - half_dt * grad_U

        # 4. Full-step q
        q = q + dt * p / self.mass

        # NaN check
        if np.any(np.isnan(q)) or np.any(np.isnan(p)):
            self._cached_grad_U = None
            return ThermostatState(q, p, xi, n_evals)

        # 5. Recompute gradient
        grad_U = self.potential.gradient(q)
        n_evals += 1

        # 6. Half-step p (kick)
        p = p - half_dt * grad_U

        # 7. Half-step p (friction via matrix exponential)
        Xi = dyn._Xi_matrix(xi)
        g_Xi = dyn.friction_fn(Xi)
        exp_neg = self._matrix_exp_symmetric(g_Xi, -half_dt)
        p = exp_neg @ p

        # 8. Half-step Xi
        xi_dot = dyn.dxidt(ThermostatState(q, p, xi, n_evals), grad_U)
        xi = xi + half_dt * xi_dot

        # Cache
        self._cached_grad_U = grad_U

        return ThermostatState(q, p, xi, n_evals)


# ---------------------------------------------------------------------------
# Anisotropic Gaussian potential (primary test for tensor NH)
# ---------------------------------------------------------------------------

class AnisotropicGaussian2D:
    """2D anisotropic Gaussian: U(q) = 0.5 * (kappa_x * x^2 + kappa_y * y^2).

    At kT=1, the canonical distribution is:
        P(x,y) ~ exp(-U/kT) = exp(-0.5 * kappa_x * x^2 - 0.5 * kappa_y * y^2)

    So x ~ N(0, 1/kappa_x), y ~ N(0, 1/kappa_y).

    When kappa_x >> kappa_y (or vice versa), the distribution is highly
    anisotropic. A scalar NH thermostat monitors tr(p^2) = p_x^2 + p_y^2,
    which averages over both directions and cannot correct anisotropy.
    The tensor NH monitors each p_i*p_j separately.
    """

    name = "anisotropic_gaussian_2d"
    dim = 2

    def __init__(self, kappa_x: float = 1.0, kappa_y: float = 1.0):
        self.kappa_x = kappa_x
        self.kappa_y = kappa_y
        self.kappa = max(kappa_x, kappa_y) / min(kappa_x, kappa_y)
        self.name = f"aniso_gauss_kappa{self.kappa:.0f}"

    def energy(self, q: np.ndarray) -> float:
        return 0.5 * (self.kappa_x * q[0]**2 + self.kappa_y * q[1]**2)

    def gradient(self, q: np.ndarray) -> np.ndarray:
        return np.array([self.kappa_x * q[0], self.kappa_y * q[1]])

    def analytical_position_density(self, q: np.ndarray, kT: float) -> float:
        return np.exp(-self.energy(q) / kT)
