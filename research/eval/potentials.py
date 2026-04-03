"""Benchmark potential energy surfaces for thermostat sampler evaluation."""

import numpy as np


class Potential:
    """Base class for potential energy surfaces."""

    name: str
    dim: int

    def energy(self, q: np.ndarray) -> float:
        raise NotImplementedError

    def gradient(self, q: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def analytical_position_density(self, q: np.ndarray, kT: float) -> float:
        """Unnormalized Boltzmann density in position space: exp(-U(q)/kT)."""
        return np.exp(-self.energy(q) / kT)


class HarmonicOscillator1D(Potential):
    """1D harmonic oscillator: U(q) = 0.5 * omega^2 * q^2.

    Analytical canonical distribution:
        P(q) ~ N(0, kT/omega^2), P(p) ~ N(0, m*kT)
    Known: Nose-Hoover is non-ergodic here (KAM tori).
    """

    name = "harmonic_1d"
    dim = 1

    def __init__(self, omega: float = 1.0):
        self.omega = omega

    def energy(self, q: np.ndarray) -> float:
        return 0.5 * self.omega**2 * float(q[0] ** 2)

    def gradient(self, q: np.ndarray) -> np.ndarray:
        return np.array([self.omega**2 * q[0]])


class DoubleWell2D(Potential):
    """2D double-well: U(x, y) = (x^2 - 1)^2 + 0.5 * y^2.

    Two minima at (+-1, 0) separated by barrier of height 1.
    Tests barrier crossing ability of the thermostat.
    """

    name = "double_well_2d"
    dim = 2

    def __init__(self, barrier_height: float = 1.0, y_stiffness: float = 0.5):
        self.a = barrier_height
        self.b = y_stiffness

    def energy(self, q: np.ndarray) -> float:
        x, y = q[0], q[1]
        return self.a * (x**2 - 1) ** 2 + self.b * y**2

    def gradient(self, q: np.ndarray) -> np.ndarray:
        x, y = q[0], q[1]
        dUdx = 4.0 * self.a * x * (x**2 - 1)
        dUdy = 2.0 * self.b * y
        return np.array([dUdx, dUdy])


class GaussianMixture2D(Potential):
    """2D Gaussian mixture: U(q) = -kT * log(sum_k w_k * N(q; mu_k, Sigma_k)).

    Multi-modal distribution to test mode-hopping.
    Default: 5 modes arranged in a ring.
    """

    name = "gaussian_mixture_2d"
    dim = 2

    def __init__(self, n_modes: int = 5, radius: float = 3.0, sigma: float = 0.5):
        self.n_modes = n_modes
        angles = np.linspace(0, 2 * np.pi, n_modes, endpoint=False)
        self.centers = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        self.sigma = sigma
        self.weights = np.ones(n_modes) / n_modes

    def _component_densities(self, q: np.ndarray) -> np.ndarray:
        """Unnormalized Gaussian densities at q for each component."""
        diffs = self.centers - q[np.newaxis, :]  # (n_modes, 2)
        exponents = -0.5 * np.sum(diffs**2, axis=1) / self.sigma**2
        return self.weights * np.exp(exponents)

    def energy(self, q: np.ndarray) -> float:
        # U = -log(mixture density), so Boltzmann at kT=1 gives the mixture
        densities = self._component_densities(q)
        total = np.sum(densities)
        if total < 1e-300:
            return 700.0  # cap to avoid inf
        return -np.log(total)

    def gradient(self, q: np.ndarray) -> np.ndarray:
        densities = self._component_densities(q)
        total = np.sum(densities)
        if total < 1e-300:
            return np.zeros(2)
        # grad U = -grad log(sum) = -sum(w_k * N_k * (mu_k - q)/sigma^2) / sum(w_k * N_k)
        diffs = self.centers - q[np.newaxis, :]  # (n_modes, 2)
        weighted = densities[:, np.newaxis] * diffs / self.sigma**2
        return -np.sum(weighted, axis=0) / total


class Rosenbrock2D(Potential):
    """2D Rosenbrock banana: U(x, y) = (a - x)^2 + b*(y - x^2)^2.

    Default a=0, b=5: banana-shaped distribution centered near origin.
    Tests sampling of strongly curved, correlated distributions.
    """

    name = "rosenbrock_2d"
    dim = 2

    def __init__(self, a: float = 0.0, b: float = 5.0):
        self.a = a
        self.b = b

    def energy(self, q: np.ndarray) -> float:
        x, y = q[0], q[1]
        return (self.a - x) ** 2 + self.b * (y - x**2) ** 2

    def gradient(self, q: np.ndarray) -> np.ndarray:
        x, y = q[0], q[1]
        dUdx = -2.0 * (self.a - x) + self.b * 2.0 * (y - x**2) * (-2.0 * x)
        dUdy = self.b * 2.0 * (y - x**2)
        return np.array([dUdx, dUdy])


class LennardJonesCluster(Potential):
    """Lennard-Jones cluster in 2D (for tractability) or 3D.

    U = sum_{i<j} 4*eps * [(sigma/r_ij)^12 - (sigma/r_ij)^6]

    Stage 3 benchmark. N=7 (2D) or N=7,13 (3D).
    """

    name = "lennard_jones"

    def __init__(self, n_atoms: int = 7, spatial_dim: int = 2, eps: float = 1.0, sigma: float = 1.0):
        self.n_atoms = n_atoms
        self.spatial_dim = spatial_dim
        self.dim = n_atoms * spatial_dim
        self.eps = eps
        self.sigma = sigma

    def _pairwise(self, q: np.ndarray):
        """Reshape q and compute pairwise distances."""
        pos = q.reshape(self.n_atoms, self.spatial_dim)
        # pairwise difference vectors
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, d)
        r2 = np.sum(diff**2, axis=2)  # (N, N)
        return pos, diff, r2

    def energy(self, q: np.ndarray) -> float:
        _, _, r2 = self._pairwise(q)
        # upper triangle only
        i_upper, j_upper = np.triu_indices(self.n_atoms, k=1)
        r2_pairs = r2[i_upper, j_upper]
        r2_pairs = np.maximum(r2_pairs, 1e-12)  # avoid division by zero
        sr2 = (self.sigma**2) / r2_pairs
        sr6 = sr2**3
        sr12 = sr6**2
        return float(4.0 * self.eps * np.sum(sr12 - sr6))

    def gradient(self, q: np.ndarray) -> np.ndarray:
        pos, diff, r2 = self._pairwise(q)
        r2 = np.maximum(r2, 1e-12)
        sr2 = (self.sigma**2) / r2
        sr6 = sr2**3
        sr12 = sr6**2
        # Force magnitude factor: dU/dr_ij * (1/r_ij) for each pair
        # dU/dr2 = 4*eps * (-6*sr6/r2 + 12*sr12/r2) * (-sigma^2/r2^2) ... easier via chain rule
        # F_ij = -dU/dr_ij * rhat_ij = 24*eps/r2 * (2*sr12 - sr6) * diff_ij
        factor = 24.0 * self.eps * (2.0 * sr12 - sr6) / r2  # (N, N)
        np.fill_diagonal(factor, 0.0)
        # gradient on atom i = sum_j factor_ij * (r_i - r_j)
        grad_pos = np.sum(factor[:, :, np.newaxis] * diff, axis=1)  # (N, d)
        # Note: this gives dU/dq (gradient), not force. The sign:
        # diff[i,j] = pos[i] - pos[j], and dU/dq_i = sum_j dU/dr2_ij * 2*(q_i - q_j)
        # Let's be careful: U = 4eps * sum_{i<j} [sr12 - sr6]
        # dU/dq_i = sum_{j!=i} 4eps * [-12*sr12/r2 + 6*sr6/r2] * 2*(q_i - q_j)
        # = sum_{j!=i} -24eps * [2*sr12 - sr6] / r2 * (q_i - q_j)
        # So gradient = -factor * diff
        grad_pos = -np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        return grad_pos.flatten()


# Registry of all potentials by stage
STAGE_1 = [DoubleWell2D(), HarmonicOscillator1D()]
STAGE_2 = [GaussianMixture2D(), Rosenbrock2D()]
STAGE_3 = [LennardJonesCluster(n_atoms=7, spatial_dim=2)]
ALL_POTENTIALS = STAGE_1 + STAGE_2 + STAGE_3


def get_potentials_by_stage(stage: int) -> list[Potential]:
    if stage == 1:
        return STAGE_1
    elif stage == 2:
        return STAGE_2
    elif stage == 3:
        return STAGE_3
    else:
        raise ValueError(f"Unknown stage: {stage}")
