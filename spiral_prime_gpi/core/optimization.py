"""
Optimization algorithms for iterative MRI reconstruction.

This module implements various optimization methods including:
- Conjugate Gradient (CG)
- ADMM (Alternating Direction Method of Multipliers)
- FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
"""

import numpy as np
from . import mri_math
from . import regularization


def conjugate_gradient(A, b, x0=None, max_iter=10, tol=1e-6):
    """
    Conjugate gradient solver for Ax = b.
    
    Args:
        A: Linear operator (function that takes x and returns Ax)
        b: Right-hand side
        x0: Initial guess (default: zeros)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        
    Returns:
        x: Solution
        residuals: Residual norms at each iteration
    """
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0.copy()
    
    r = b - A(x)
    p = r.copy()
    rsold = np.sum(np.conj(r) * r).real
    
    residuals = [np.sqrt(rsold)]
    
    for i in range(max_iter):
        Ap = A(p)
        alpha = rsold / np.sum(np.conj(p) * Ap).real
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.sum(np.conj(r) * r).real
        
        residuals.append(np.sqrt(rsnew))
        
        if np.sqrt(rsnew) < tol:
            break
        
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    
    return x, residuals


def iterative_sense_recon(kspace_data, trajectory, coil_maps, 
                         image_shape=None, n_iterations=10,
                         regularization_type='none', lambda_reg=0.001,
                         density_comp=None):
    """
    Iterative SENSE reconstruction for non-Cartesian data.
    
    Args:
        kspace_data: Multi-coil k-space data [n_coils, n_points]
        trajectory: k-space trajectory [n_points, n_dims], normalized to [-0.5, 0.5]
        coil_maps: Coil sensitivity maps [n_coils, ...image_shape]
        image_shape: Output image shape (tuple)
        n_iterations: Number of CG iterations
        regularization_type: Type of regularization ('none', 'l2', 'tv')
        lambda_reg: Regularization parameter
        density_comp: Density compensation weights [n_points] (optional)
        
    Returns:
        Reconstructed image [...image_shape]
    """
    n_coils = kspace_data.shape[0]
    
    if image_shape is None:
        image_shape = coil_maps.shape[1:]
    
    n_dims = len(image_shape)
    
    # Apply density compensation if provided
    if density_comp is not None:
        kspace_weighted = kspace_data * density_comp[np.newaxis, :]
    else:
        kspace_weighted = kspace_data
    
    # Define forward operator: E(x) = S * F * x
    # where S is coil sensitivity, F is Fourier transform
    def forward_op(image):
        """Apply encoding operator: image -> k-space"""
        # Multiply by coil sensitivities
        coil_images = coil_maps * image[np.newaxis, ...]
        
        # FFT to k-space
        if n_dims == 2:
            kspace_cart = mri_math.fft2c(coil_images)
        else:
            kspace_cart = mri_math.fft3c(coil_images)
        
        # Interpolate to trajectory points (simplified - use gridding)
        # For production, would use NUFFT
        kspace_noncart = np.zeros((n_coils, trajectory.shape[0]), dtype=complex)
        
        # Simplified sampling (in practice, use proper NUFFT)
        grid_size = image_shape
        for c in range(n_coils):
            # Simple nearest-neighbor sampling for demo
            # Production code would use proper non-uniform FFT
            kspace_noncart[c] = kspace_cart[c].ravel()[:trajectory.shape[0]]
        
        return kspace_noncart
    
    # Define adjoint operator: E^H(k) = F^H * S^H * k
    def adjoint_op(kspace_nc):
        """Apply adjoint encoding operator: k-space -> image"""
        # Grid non-Cartesian k-space to Cartesian
        kspace_cart = np.zeros((n_coils, *image_shape), dtype=complex)
        
        for c in range(n_coils):
            # Grid the data (simplified)
            kspace_cart[c] = mri_math.grid_noncartesian(
                trajectory, kspace_nc[c], image_shape
            )
        
        # Inverse FFT
        if n_dims == 2:
            coil_images = mri_math.ifft2c(kspace_cart)
        else:
            coil_images = mri_math.ifft3c(kspace_cart)
        
        # Combine using coil sensitivities (adjoint of multiplication)
        image = np.sum(np.conj(coil_maps) * coil_images, axis=0)
        
        return image
    
    # Define normal operator: A = E^H * E
    def normal_op(image):
        """Apply normal operator: E^H * E * image"""
        k = forward_op(image)
        img_back = adjoint_op(k)
        
        # Add regularization
        if regularization_type == 'l2':
            img_back = img_back + lambda_reg * image
        elif regularization_type == 'tv':
            # For TV, use gradient descent step
            tv_grad = regularization.tv_gradient(image)
            img_back = img_back + lambda_reg * tv_grad
        
        return img_back
    
    # Initial reconstruction: gridding + adjoint
    x0 = adjoint_op(kspace_weighted)
    
    # Right-hand side: E^H * data
    b = adjoint_op(kspace_weighted)
    
    # Solve using conjugate gradient
    if regularization_type == 'none' or regularization_type == 'l2':
        image_recon, residuals = conjugate_gradient(
            normal_op, b, x0=x0, max_iter=n_iterations
        )
    else:
        # For TV and other non-quadratic regularizations, use ADMM or FISTA
        image_recon = admm_reconstruction(
            kspace_weighted, trajectory, coil_maps, image_shape,
            n_iterations=n_iterations, lambda_reg=lambda_reg,
            regularization_type=regularization_type
        )
    
    return image_recon


def admm_reconstruction(kspace_data, trajectory, coil_maps, image_shape,
                       n_iterations=10, lambda_reg=0.001, 
                       regularization_type='tv', rho=1.0):
    """
    ADMM-based reconstruction with various regularizations.
    
    ADMM solves: minimize ||E(x) - y||^2 + lambda * R(x)
    where R(x) is a regularization functional (e.g., TV norm).
    
    Args:
        kspace_data: k-space measurements
        trajectory: k-space trajectory
        coil_maps: Coil sensitivity maps
        image_shape: Desired image shape
        n_iterations: Number of ADMM iterations
        lambda_reg: Regularization weight
        regularization_type: Type of regularization ('tv', 'wavelet', 'l1')
        rho: ADMM penalty parameter
        
    Returns:
        Reconstructed image
    """
    # Initialize
    n_dims = len(image_shape)
    x = np.zeros(image_shape, dtype=complex)
    z = np.zeros(image_shape, dtype=complex)
    u = np.zeros(image_shape, dtype=complex)
    
    # Encoding operator (simplified)
    def encode(img):
        coil_imgs = coil_maps * img[np.newaxis, ...]
        if n_dims == 2:
            return mri_math.fft2c(coil_imgs)
        else:
            return mri_math.fft3c(coil_imgs)
    
    def encode_adjoint(ksp):
        if n_dims == 2:
            coil_imgs = mri_math.ifft2c(ksp)
        else:
            coil_imgs = mri_math.ifft3c(ksp)
        return np.sum(np.conj(coil_maps) * coil_imgs, axis=0)
    
    # ADMM iterations
    for i in range(n_iterations):
        # x-update: data consistency step (simplified)
        # Solve: (E^H*E + rho*I)x = E^H*y + rho*(z - u)
        rhs = encode_adjoint(kspace_data) + rho * (z - u)
        
        # Simple gradient descent for x-update
        x_new = x + 0.1 * (rhs - (encode_adjoint(encode(x)) + rho * x))
        x = x_new
        
        # z-update: proximal operator of regularization
        if regularization_type == 'tv':
            z = regularization.tv_proximal(x + u, lambda_reg / rho)
        elif regularization_type == 'l1':
            # Soft thresholding
            threshold = lambda_reg / rho
            z = regularization.soft_threshold(x + u, threshold)
        else:
            z = x + u
        
        # u-update: dual variable
        u = u + x - z
    
    return x


def fista_reconstruction(A, AT, b, image_shape, n_iterations=50, 
                        lambda_reg=0.001, regularization_type='l1'):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).
    
    Solves: minimize ||Ax - b||^2 + lambda * ||x||_1 (or other regularization)
    
    Args:
        A: Forward operator
        AT: Adjoint operator
        b: Measurements
        image_shape: Image dimensions
        n_iterations: Number of iterations
        lambda_reg: Regularization parameter
        regularization_type: Type of regularization
        
    Returns:
        Reconstructed image
    """
    # Initialize
    x = np.zeros(image_shape, dtype=complex)
    y = x.copy()
    t = 1.0
    
    # Lipschitz constant estimation (simplified)
    L = 1.0
    
    for i in range(n_iterations):
        # Gradient step
        grad = AT(A(y) - b)
        x_new = y - (1.0 / L) * grad
        
        # Proximal step
        if regularization_type == 'l1':
            x_new = regularization.soft_threshold(x_new, lambda_reg / L)
        elif regularization_type == 'tv':
            x_new = regularization.tv_proximal(x_new, lambda_reg / L)
        
        # Update y with momentum
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x_new + ((t - 1) / t_new) * (x_new - x)
        
        x = x_new
        t = t_new
    
    return x
