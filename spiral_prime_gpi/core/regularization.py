"""
Regularization functions for MRI reconstruction.

Implements various regularization penalties and their proximal operators:
- L1/L2 norms
- Total Variation (TV)
- Wavelet sparsity
"""

import numpy as np
from scipy import ndimage

# Optional PyWavelets import
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


def soft_threshold(x, threshold):
    """
    Soft thresholding operator (proximal operator of L1 norm).
    
    Args:
        x: Input array (complex or real)
        threshold: Threshold value
        
    Returns:
        Soft-thresholded array
    """
    if np.iscomplexobj(x):
        # Complex soft thresholding
        magnitude = np.abs(x)
        shrinkage = np.maximum(0, magnitude - threshold)
        # Avoid division by zero
        phase = np.where(magnitude > 0, x / magnitude, 0)
        return shrinkage * phase
    else:
        # Real soft thresholding
        return np.sign(x) * np.maximum(0, np.abs(x) - threshold)


def tv_norm(image, axis=None):
    """
    Compute total variation (TV) norm.
    
    TV = sum |gradient(image)|
    
    Args:
        image: Input image
        axis: Axes along which to compute gradients (default: all)
        
    Returns:
        TV norm (scalar)
    """
    if image.ndim == 2:
        # 2D TV
        grad_y, grad_x = np.gradient(image)
        tv = np.sum(np.sqrt(np.abs(grad_x)**2 + np.abs(grad_y)**2))
    elif image.ndim == 3:
        # 3D TV
        grad_z, grad_y, grad_x = np.gradient(image)
        tv = np.sum(np.sqrt(np.abs(grad_x)**2 + np.abs(grad_y)**2 + np.abs(grad_z)**2))
    else:
        # General case
        gradients = np.gradient(image)
        grad_mag = np.sqrt(sum(np.abs(g)**2 for g in gradients))
        tv = np.sum(grad_mag)
    
    return tv


def tv_gradient(image, epsilon=1e-8):
    """
    Compute gradient of TV norm (for gradient descent).
    
    This is the negative divergence of the normalized gradient.
    
    Args:
        image: Input image
        epsilon: Small constant for numerical stability
        
    Returns:
        TV gradient with same shape as input
    """
    if image.ndim == 2:
        # 2D case
        grad_y, grad_x = np.gradient(image)
        
        # Gradient magnitude
        grad_mag = np.sqrt(np.abs(grad_x)**2 + np.abs(grad_y)**2 + epsilon)
        
        # Normalized gradients
        nx = grad_x / grad_mag
        ny = grad_y / grad_mag
        
        # Divergence (negative of TV gradient)
        div_x = np.gradient(nx, axis=1)
        div_y = np.gradient(ny, axis=0)
        
        tv_grad = -(div_x + div_y)
        
    elif image.ndim == 3:
        # 3D case
        grad_z, grad_y, grad_x = np.gradient(image)
        
        # Gradient magnitude
        grad_mag = np.sqrt(np.abs(grad_x)**2 + np.abs(grad_y)**2 + 
                          np.abs(grad_z)**2 + epsilon)
        
        # Normalized gradients
        nx = grad_x / grad_mag
        ny = grad_y / grad_mag
        nz = grad_z / grad_mag
        
        # Divergence
        div_x = np.gradient(nx, axis=2)
        div_y = np.gradient(ny, axis=1)
        div_z = np.gradient(nz, axis=0)
        
        tv_grad = -(div_x + div_y + div_z)
    else:
        raise ValueError(f"Unsupported number of dimensions: {image.ndim}")
    
    return tv_grad


def tv_proximal(image, lambda_tv, n_iter=10, step_size=0.1):
    """
    Proximal operator of TV norm using gradient descent.
    
    Solves: argmin_x (1/2)||x - image||^2 + lambda_tv * TV(x)
    
    Args:
        image: Input image
        lambda_tv: TV regularization weight
        n_iter: Number of gradient descent iterations
        step_size: Step size for gradient descent
        
    Returns:
        Denoised image
    """
    x = image.copy()
    
    for i in range(n_iter):
        # Gradient of data term: x - image
        grad_data = x - image
        
        # Gradient of TV term
        grad_tv = tv_gradient(x)
        
        # Gradient descent step
        x = x - step_size * (grad_data + lambda_tv * grad_tv)
    
    return x


def tv_chambolle_pock(image, lambda_tv, n_iter=100, tau=0.1, sigma=0.1):
    """
    TV denoising using Chambolle-Pock algorithm (primal-dual).
    
    More efficient than gradient descent for TV minimization.
    
    Args:
        image: Input image
        lambda_tv: TV regularization weight
        n_iter: Number of iterations
        tau: Primal step size
        sigma: Dual step size
        
    Returns:
        Denoised image
    """
    # Initialize primal and dual variables
    x = image.copy()
    x_bar = x.copy()
    
    # Dual variables (one for each gradient component)
    if image.ndim == 2:
        p = np.zeros((*image.shape, 2))
    elif image.ndim == 3:
        p = np.zeros((*image.shape, 3))
    else:
        raise ValueError(f"Unsupported dimensions: {image.ndim}")
    
    for i in range(n_iter):
        # Update dual variable
        if image.ndim == 2:
            grad_y, grad_x = np.gradient(x_bar)
            p[:, :, 0] = p[:, :, 0] + sigma * grad_x
            p[:, :, 1] = p[:, :, 1] + sigma * grad_y
            
            # Project onto unit ball scaled by lambda_tv
            p_norm = np.sqrt(p[:, :, 0]**2 + p[:, :, 1]**2)
            p_norm = np.maximum(p_norm, lambda_tv)
            p[:, :, 0] = p[:, :, 0] * lambda_tv / p_norm
            p[:, :, 1] = p[:, :, 1] * lambda_tv / p_norm
        else:  # 3D
            grad_z, grad_y, grad_x = np.gradient(x_bar)
            p[:, :, :, 0] = p[:, :, :, 0] + sigma * grad_x
            p[:, :, :, 1] = p[:, :, :, 1] + sigma * grad_y
            p[:, :, :, 2] = p[:, :, :, 2] + sigma * grad_z
            
            p_norm = np.sqrt(p[:, :, :, 0]**2 + p[:, :, :, 1]**2 + p[:, :, :, 2]**2)
            p_norm = np.maximum(p_norm, lambda_tv)
            p[:, :, :, 0] = p[:, :, :, 0] * lambda_tv / p_norm
            p[:, :, :, 1] = p[:, :, :, 1] * lambda_tv / p_norm
            p[:, :, :, 2] = p[:, :, :, 2] * lambda_tv / p_norm
        
        # Update primal variable
        x_old = x.copy()
        
        if image.ndim == 2:
            div_p = np.gradient(p[:, :, 0], axis=1) + np.gradient(p[:, :, 1], axis=0)
        else:  # 3D
            div_p = (np.gradient(p[:, :, :, 0], axis=2) + 
                    np.gradient(p[:, :, :, 1], axis=1) + 
                    np.gradient(p[:, :, :, 2], axis=0))
        
        x = (x + tau * div_p + tau * image) / (1 + tau)
        
        # Extrapolation
        x_bar = 2 * x - x_old
    
    return x


def wavelet_threshold(image, threshold, wavelet='db4', level=3):
    """
    Wavelet soft thresholding.
    
    Args:
        image: Input image
        threshold: Threshold value
        wavelet: Wavelet type (default: Daubechies 4)
        level: Decomposition level
        
    Returns:
        Thresholded image
    """
    if not HAS_PYWT:
        # If PyWavelets not available, return simple soft threshold
        print("Warning: PyWavelets not available, using simple soft threshold")
        return soft_threshold(image, threshold)
    
    # Wavelet decomposition
    coeffs = pywt.wavedecn(image, wavelet=wavelet, level=level)
    
    # Threshold detail coefficients
    coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients
    for detail in coeffs[1:]:
        detail_thresh = {}
        for key, coeff in detail.items():
            detail_thresh[key] = soft_threshold(coeff, threshold)
        coeffs_thresh.append(detail_thresh)
    
    # Reconstruct
    image_thresh = pywt.waverecn(coeffs_thresh, wavelet=wavelet)
    
    return image_thresh


def l2_regularization(image, lambda_l2):
    """
    L2 regularization penalty.
    
    Args:
        image: Input image
        lambda_l2: L2 regularization weight
        
    Returns:
        Regularization penalty (scalar)
    """
    return lambda_l2 * np.sum(np.abs(image)**2)


def l1_regularization(image, lambda_l1):
    """
    L1 regularization penalty.
    
    Args:
        image: Input image
        lambda_l1: L1 regularization weight
        
    Returns:
        Regularization penalty (scalar)
    """
    return lambda_l1 * np.sum(np.abs(image))


def huber_loss(x, delta=1.0):
    """
    Huber loss function (smooth approximation of L1 norm).
    
    Args:
        x: Input
        delta: Threshold parameter
        
    Returns:
        Huber loss
    """
    abs_x = np.abs(x)
    quadratic = np.minimum(abs_x, delta)
    linear = abs_x - quadratic
    return np.sum(0.5 * quadratic**2 + delta * linear)


def apply_regularization(image, regularization_type, lambda_reg, **kwargs):
    """
    Apply regularization to image.
    
    Args:
        image: Input image
        regularization_type: Type ('l1', 'l2', 'tv', 'wavelet')
        lambda_reg: Regularization weight
        **kwargs: Additional arguments for specific regularization
        
    Returns:
        Regularized/denoised image
    """
    if regularization_type == 'l1':
        return soft_threshold(image, lambda_reg)
    elif regularization_type == 'l2':
        # L2 proximal operator: x / (1 + lambda)
        return image / (1 + lambda_reg)
    elif regularization_type == 'tv':
        return tv_proximal(image, lambda_reg, **kwargs)
    elif regularization_type == 'tv_cp':
        return tv_chambolle_pock(image, lambda_reg, **kwargs)
    elif regularization_type == 'wavelet':
        return wavelet_threshold(image, lambda_reg, **kwargs)
    else:
        raise ValueError(f"Unknown regularization type: {regularization_type}")
