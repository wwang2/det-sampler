# Logarithmic Oscillator Thermostat: Derivation of Invariant Measure

## Extended Hamiltonian

We replace the standard quadratic thermostat kinetic energy $Q\xi^2/2$ in the Nose-Hoover framework with a logarithmic form:

$$H_{\text{ext}} = U(q) + \frac{p^2}{2m} + Q \log(1 + \xi^2)$$

The target invariant measure is:

$$\rho(q, p, \xi) \propto \exp\left(-\frac{H_{\text{ext}}}{k_B T}\right) = \exp\left(-\frac{U(q)}{k_B T}\right) \exp\left(-\frac{p^2}{2mk_BT}\right) (1 + \xi^2)^{-Q/k_BT}$$

Note that marginalizing over $\xi$ gives the canonical distribution in $(q, p)$, provided the $\xi$-integral converges. This requires $Q/k_BT > 1/2$, which is satisfied for any reasonable $Q$.

The thermostat marginal distribution $(1+\xi^2)^{-Q/k_BT}$ is a generalized Cauchy (Student's $t$) distribution, in contrast to the Gaussian distribution of standard NH.

## Equations of Motion

We seek equations of the form:
$$\dot{q} = p/m$$
$$\dot{p} = -\nabla U(q) - \alpha(\xi) \cdot p$$
$$\dot{\xi} = \frac{\beta(\xi)}{Q} \left(\frac{p^2}{m} - N k_B T\right)$$

where $N$ is the number of physical degrees of freedom.

## Liouville Equation for the Invariant Measure

For a flow $v = (\dot{q}, \dot{p}, \dot{\xi})$ to preserve the density $\rho$, we need:

$$\nabla \cdot v + v \cdot \nabla \log \rho = 0$$

### Computing the divergence

$$\frac{\partial \dot{q}}{\partial q} = 0$$

$$\frac{\partial \dot{p}}{\partial p} = -N\alpha(\xi) \quad \text{(sum over N DOF)}$$

$$\frac{\partial \dot{\xi}}{\partial \xi} = \frac{\beta'(\xi)}{Q}(K - Nk_BT)$$

where $K = \sum p_i^2/m$ is the kinetic energy.

### Computing $v \cdot \nabla \log \rho$

$$\log \rho = -\frac{U(q)}{k_BT} - \frac{p^2}{2mk_BT} - \frac{Q\log(1+\xi^2)}{k_BT} + \text{const}$$

The $(q, p)$-sector terms:

$$\dot{q} \cdot \frac{\partial \log\rho}{\partial q} + \dot{p} \cdot \frac{\partial \log\rho}{\partial p} = \frac{p}{m}\left(-\frac{\nabla U}{k_BT}\right) + \left(-\nabla U - \alpha p\right)\left(-\frac{p}{mk_BT}\right) = \frac{\alpha K}{k_BT}$$

The $\xi$-sector term:

$$\dot{\xi} \cdot \frac{\partial \log\rho}{\partial \xi} = \frac{\beta}{Q}(K - Nk_BT) \cdot \left(-\frac{2Q\xi}{(1+\xi^2)k_BT}\right) = -\frac{2\beta\xi(K-Nk_BT)}{(1+\xi^2)k_BT}$$

### Full Liouville condition

Combining all terms and factoring out $(K - Nk_BT)$:

$$(K - Nk_BT) \left[\frac{\alpha}{k_BT} + \frac{\beta'}{Q} - \frac{2\beta\xi}{(1+\xi^2)k_BT}\right] = 0$$

Since this must hold for **all** values of $K$ (all momenta), the bracket must vanish:

$$\frac{\alpha}{k_BT} + \frac{\beta'}{Q} = \frac{2\beta\xi}{(1+\xi^2)k_BT}$$

## Solution: $\beta(\xi) = 1$ (constant)

The simplest solution sets $\beta(\xi) = 1$ (constant), so $\beta' = 0$:

$$\alpha(\xi) = \frac{2\xi}{1+\xi^2} \equiv g(\xi)$$

This gives the **final equations of motion**:

$$\dot{q} = p/m$$
$$\dot{p} = -\nabla U(q) - g(\xi) p, \quad g(\xi) = \frac{2\xi}{1+\xi^2}$$
$$\dot{\xi} = \frac{1}{Q}\left(\sum_i \frac{p_i^2}{m} - Nk_BT\right)$$

## Properties of $g(\xi)$

The friction function $g(\xi) = 2\xi/(1+\xi^2)$ has remarkable properties:

1. **Bounded**: $|g(\xi)| \le 1$ for all $\xi$ (maximum at $\xi = \pm 1$)
2. **Linear for small $\xi$**: $g(\xi) \approx 2\xi$ as $\xi \to 0$ (twice the NH coupling)
3. **Decaying for large $\xi$**: $g(\xi) \approx 2/\xi \to 0$ as $\xi \to \infty$
4. **Odd function**: $g(-\xi) = -g(\xi)$

The boundedness means the thermostat cannot create arbitrarily strong friction, unlike NH where $\alpha(\xi) = \xi$ grows without bound. This should make the dynamics more "exploratory" -- when the thermostat variable becomes large, friction weakens, allowing the system to escape trapped regions.

## SymPy Verification

The Liouville condition was verified symbolically (see `derive.py`):

```
Liouville check: 0 = 0  (verified)
```

## Thermostat Interpretation (Tier 3)

The logarithmic thermostat potential $Q\log(1+\xi^2)$ can be interpreted as a "soft" heat bath with bounded coupling strength. In the standard NH thermostat, the quadratic potential $Q\xi^2/2$ creates a harmonic restoring force on $\xi$, leading to oscillatory (and potentially quasi-periodic) thermostat dynamics. The logarithmic potential creates an **anharmonic** restoring force that:

- Acts like a harmonic oscillator for small $\xi$ (near-equilibrium behavior)
- Becomes sublinear for large $\xi$ (the thermostat "loosens" when far from equilibrium)

This prevents the thermostat from getting trapped in regular oscillatory patterns (KAM tori), promoting more chaotic -- and thus more ergodic -- behavior.

## Numerical Integration

We use a custom velocity Verlet scheme adapted for the $g(\xi)$ friction:

1. Half-step thermostat: $\xi \leftarrow \xi + \frac{\Delta t}{2} \dot{\xi}$
2. Half-step momenta: $p \leftarrow p \cdot \exp(-g(\xi)\Delta t/2)$; $p \leftarrow p - \frac{\Delta t}{2}\nabla U$
3. Full-step positions: $q \leftarrow q + \Delta t \cdot p/m$
4. Recompute forces
5. Half-step momenta: $p \leftarrow p - \frac{\Delta t}{2}\nabla U$; $p \leftarrow p \cdot \exp(-g(\xi)\Delta t/2)$
6. Half-step thermostat: $\xi \leftarrow \xi + \frac{\Delta t}{2} \dot{\xi}$

The analytical $\exp(-g(\xi)\Delta t/2)$ rescaling ensures the friction is applied exactly in the split-operator sense, analogous to the Martyna et al. (1996) scheme for NHC.

## References

- [Nose, S. (1984). A unified formulation of the constant temperature molecular dynamics methods. J. Chem. Phys. 81, 511.](https://doi.org/10.1063/1.447334)
- [Hoover, W. G. (1985). Canonical dynamics: Equilibrium phase-space distributions. Phys. Rev. A, 31, 1695.](https://doi.org/10.1103/PhysRevA.31.1695)
- [Martyna, G. J., Klein, M. L., & Tuckerman, M. (1992). Nose-Hoover chains. J. Chem. Phys. 97, 2635.](https://doi.org/10.1063/1.463940)
- [Martyna, G. J., Tuckerman, M. E., Tobias, D. J., & Klein, M. L. (1996). Explicit reversible integrators for extended systems dynamics. Mol. Phys. 87, 1117.](https://doi.org/10.1080/00268979600100761)
- [KAM theorem](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold%E2%80%93Moser_theorem) -- explains non-ergodicity of standard NH on harmonic oscillator
