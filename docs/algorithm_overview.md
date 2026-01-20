# Algorithm Overview

## Introduction

Spiral-PRIME (Physics-based Reconstruction with Iterative Model-based Enhancement) implements advanced MRI reconstruction algorithms specifically designed for spiral k-space acquisitions. This document provides an overview of the mathematical foundations and algorithms.

## Signal Model

### Steady-State Spin-Echo

For a steady-state spin-echo sequence, the signal equation is:

```
S = M₀ · sin(α) · exp(-TE/T₂) · (1 - exp(-TR/T₁)) / (1 - cos(α)·exp(-TR/T₁))
```

Where:
- M₀: Equilibrium magnetization
- α: Flip angle
- TE: Echo time
- TR: Repetition time
- T₁, T₂: Tissue relaxation times

### Localized-Quadratic (LQ) Model

The LQ model captures spatially-varying field inhomogeneities:

```
B(r) = B₀ · B₁(r) · (1 + g₁·r + ½r^T·G·r)
```

Where:
- B₀: Static field strength
- B₁(r): Transmit field map
- g₁: Linear gradient terms
- G: Quadratic gradient moment matrix
- r = [x, y, z]^T: Spatial position

## Reconstruction Algorithms

### 1. Non-Cartesian Fourier Transform

#### Forward Model
The signal acquisition model for spiral MRI:

```
s(k) = ∫ ρ(r) · C(r) · exp(-i2π k·r) dr
```

Where:
- s(k): Measured k-space signal
- ρ(r): Image to be reconstructed
- C(r): Coil sensitivity
- k: k-space trajectory

#### Gridding
For non-Cartesian trajectories, we use gridding to interpolate data onto a Cartesian grid:

```
S_cart = W · F^(-1) · D · s_nonCart
```

Where:
- W: Gridding kernel (Kaiser-Bessel)
- F^(-1): Inverse FFT
- D: Density compensation
- s_nonCart: Non-Cartesian k-space data

### 2. SENSE Reconstruction

Parallel imaging using SENSE (Sensitivity Encoding):

#### Forward Operator
```
E(ρ) = F · C · ρ
```

Where:
- F: Fourier encoding operator
- C: Coil sensitivity operator
- ρ: Image

#### Reconstruction
Solve the least-squares problem:

```
ρ̂ = argmin_ρ ||E(ρ) - s||²₂
```

Normal equations:
```
(E^H·E)ρ = E^H·s
```

Where E^H is the adjoint operator.

### 3. Iterative Reconstruction

#### Conjugate Gradient (CG)
For quadratic optimization problems:

```
ρ^(k+1) = ρ^(k) + α_k · p^(k)
```

Where:
- α_k: Step size
- p^(k): Search direction

Algorithm:
1. Initialize: r = b - A·x₀, p = r
2. Iterate: 
   - α = (r^T·r) / (p^T·A·p)
   - x = x + α·p
   - r_new = r - α·A·p
   - β = (r_new^T·r_new) / (r^T·r)
   - p = r_new + β·p
3. Stop when ||r|| < tolerance

#### ADMM (Alternating Direction Method of Multipliers)

For problems with non-smooth regularization:

```
min_ρ ||E(ρ) - s||²₂ + λ·R(ρ)
```

ADMM formulation:
```
min_ρ,z ||E(ρ) - s||²₂ + λ·R(z)
subject to ρ = z
```

Augmented Lagrangian:
```
L(ρ,z,u) = ||E(ρ) - s||²₂ + λ·R(z) + (μ/2)||ρ - z + u||²₂
```

Algorithm:
1. ρ-update: Data consistency
2. z-update: Proximal operator of R
3. u-update: Dual variable

## Regularization

### Total Variation (TV)

Promotes piecewise-constant images:

```
TV(ρ) = ∫ |∇ρ(r)| dr ≈ Σ √(|∂ρ/∂x|² + |∂ρ/∂y|² + |∂ρ/∂z|²)
```

#### TV Proximal Operator
Solved using Chambolle-Pock algorithm:

```
prox_λTV(ρ) = argmin_x (1/2)||x - ρ||²₂ + λ·TV(x)
```

### L1 Regularization

Promotes sparsity:

```
||ρ||₁ = Σ |ρᵢ|
```

Proximal operator (soft thresholding):
```
prox_λL1(ρ) = sign(ρ) · max(|ρ| - λ, 0)
```

### L2 Regularization

Smoothness constraint:

```
||ρ||²₂ = Σ |ρᵢ|²
```

Proximal operator:
```
prox_λL2(ρ) = ρ / (1 + λ)
```

## Coil Sensitivity Estimation

### Sum-of-Squares (SOS)
```
ρ_SOS = √(Σ |Iᶜ|²)
C_SOS^c = Iᶜ / ρ_SOS
```

### Adaptive Combine
Uses smoothed images:
```
Ĩᶜ = G_σ * Iᶜ
C_adaptive^c = Ĩᶜ / √(Σ |Ĩᶜ|²)
```

Where G_σ is a Gaussian smoothing kernel.

### Walsh Method
Based on eigenvector decomposition of local covariance matrices.

## Spiral Trajectory Design

### Archimedean Spiral
```
k(t) = (k_max/T) · t · [cos(ωt), sin(ωt)]
```

Where:
- k_max: Maximum k-space radius
- T: Total readout time
- ω: Angular frequency

### Variable Density Spiral
```
k(θ) = k_max · (θ/θ_max)^α · [cos(θ), sin(θ)]
```

Where α controls density (α=1 for Archimedean).

### 3D Stack-of-Spirals
Combination of 2D spirals at different k_z positions:
```
k_3D = [k_z, k_spiral(t)]
```

## Density Compensation

### Voronoi Method
Weight proportional to Voronoi cell area:
```
w_i ∝ Area(Voronoi_cell_i)
```

### Jackson Method
Analytical weights based on trajectory:
```
w_i = |J(k_i)|
```
Where J is the Jacobian of the trajectory.

## Multi-Contrast Reconstruction

For multi-contrast acquisitions, each contrast can be reconstructed independently or jointly:

### Joint Reconstruction
```
min_ρ₁,...,ρ_N Σᵢ ||Eᵢ(ρᵢ) - sᵢ||²₂ + λ·R(ρ₁,...,ρ_N)
```

With coupled regularization promoting similarity between contrasts.

## Computational Complexity

- **Gridding**: O(N log N) where N is the number of k-space points
- **FFT**: O(N log N) where N is the image size
- **CG iteration**: O(N) per iteration
- **TV denoising**: O(N·I) where I is the number of inner iterations

## References

1. Pruessmann et al., "SENSE: Sensitivity Encoding for Fast MRI", MRM 1999
2. Lustig et al., "Sparse MRI: The Application of Compressed Sensing", MRM 2007
3. Boyd et al., "Distributed Optimization and Statistical Learning via ADMM", 2011
4. Chambolle & Pock, "A First-Order Primal-Dual Algorithm", JMIV 2011
5. Pipe, "Motion Correction With PROPELLER MRI", MRM 1999
