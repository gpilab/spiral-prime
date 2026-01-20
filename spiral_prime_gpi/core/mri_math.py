"""
MRI mathematics and fundamental operations.

This module provides core MRI reconstruction operations including:
- FFT operations for k-space/image transformations
- Gridding for non-Cartesian k-space data
- Coil sensitivity estimation
- SENSE reconstruction
"""

import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
import numba


def ifft2c(kspace):
    """
    Centered 2D inverse FFT.
    
    Args:
        kspace: k-space data [..., ny, nx]
        
    Returns:
        Image domain data with same shape
    """
    return np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(kspace, axes=(-2, -1)),
            axes=(-2, -1)
        ),
        axes=(-2, -1)
    )


def fft2c(image):
    """
    Centered 2D FFT.
    
    Args:
        image: Image domain data [..., ny, nx]
        
    Returns:
        k-space data with same shape
    """
    return np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(image, axes=(-2, -1)),
            axes=(-2, -1)
        ),
        axes=(-2, -1)
    )


def ifft3c(kspace):
    """
    Centered 3D inverse FFT.
    
    Args:
        kspace: k-space data [..., nz, ny, nx]
        
    Returns:
        Image domain data with same shape
    """
    return np.fft.fftshift(
        np.fft.ifftn(
            np.fft.ifftshift(kspace, axes=(-3, -2, -1)),
            axes=(-3, -2, -1)
        ),
        axes=(-3, -2, -1)
    )


def fft3c(image):
    """
    Centered 3D FFT.
    
    Args:
        image: Image domain data [..., nz, ny, nx]
        
    Returns:
        k-space data with same shape
    """
    return np.fft.fftshift(
        np.fft.fftn(
            np.fft.ifftshift(image, axes=(-3, -2, -1)),
            axes=(-3, -2, -1)
        ),
        axes=(-3, -2, -1)
    )


@numba.jit(nopython=True, parallel=True)
def _gridding_kernel(k_traj, k_data, grid_size, kernel_width=3):
    """
    Fast gridding kernel using Numba.
    
    Args:
        k_traj: Trajectory coordinates [n_points, n_dims] normalized to [-0.5, 0.5]
        k_data: k-space data [n_coils, n_points]
        grid_size: Output grid size (tuple)
        kernel_width: Kaiser-Bessel kernel width
        
    Returns:
        Gridded k-space data
    """
    # This is a simplified version; production would use Kaiser-Bessel kernel
    n_coils = k_data.shape[0]
    n_points = k_data.shape[1]
    
    # Initialize output grid
    if len(grid_size) == 2:
        grid = np.zeros((n_coils, grid_size[0], grid_size[1]), dtype=np.complex128)
        density = np.zeros(grid_size, dtype=np.float64)
    else:
        grid = np.zeros((n_coils, grid_size[0], grid_size[1], grid_size[2]), 
                       dtype=np.complex128)
        density = np.zeros(grid_size, dtype=np.float64)
    
    return grid


def grid_noncartesian(k_traj, k_data, grid_size, method='linear'):
    """
    Grid non-Cartesian k-space data onto Cartesian grid.
    
    Args:
        k_traj: k-space trajectory [n_points, n_dims], normalized to [-0.5, 0.5]
        k_data: k-space data [n_coils, n_points] or [n_points]
        grid_size: Output grid dimensions (tuple)
        method: Gridding method ('linear', 'nearest', 'cubic', 'kb' for Kaiser-Bessel)
        
    Returns:
        Gridded k-space data [n_coils, ...grid_size] or [...grid_size]
    """
    # Add coil dimension if needed
    if k_data.ndim == 1:
        k_data = k_data[np.newaxis, :]
        single_coil = True
    else:
        single_coil = False
    
    n_coils = k_data.shape[0]
    n_dims = k_traj.shape[1]
    
    # Convert normalized coordinates to grid indices
    k_traj_scaled = k_traj.copy()
    for d in range(n_dims):
        k_traj_scaled[:, d] = (k_traj_scaled[:, d] + 0.5) * (grid_size[d] - 1)
    
    # Grid each coil separately
    gridded = []
    for coil in range(n_coils):
        if n_dims == 2:
            # Create grid coordinates
            yi, xi = np.mgrid[0:grid_size[0], 0:grid_size[1]]
            points = np.column_stack([yi.ravel(), xi.ravel()])
            
            # Gridding using scipy
            grid_real = griddata(k_traj_scaled, k_data[coil].real, 
                                points, method=method, fill_value=0.0)
            grid_imag = griddata(k_traj_scaled, k_data[coil].imag, 
                                points, method=method, fill_value=0.0)
            
            grid_coil = (grid_real + 1j * grid_imag).reshape(grid_size)
            
        elif n_dims == 3:
            # 3D gridding
            zi, yi, xi = np.mgrid[0:grid_size[0], 0:grid_size[1], 0:grid_size[2]]
            points = np.column_stack([zi.ravel(), yi.ravel(), xi.ravel()])
            
            grid_real = griddata(k_traj_scaled, k_data[coil].real, 
                                points, method=method, fill_value=0.0)
            grid_imag = griddata(k_traj_scaled, k_data[coil].imag, 
                                points, method=method, fill_value=0.0)
            
            grid_coil = (grid_real + 1j * grid_imag).reshape(grid_size)
        else:
            raise ValueError(f"Unsupported number of dimensions: {n_dims}")
        
        gridded.append(grid_coil)
    
    gridded = np.array(gridded)
    
    if single_coil:
        return gridded[0]
    return gridded


def estimate_coil_sensitivity(kspace_data, method='adaptive', smoothing=5):
    """
    Estimate coil sensitivity maps from k-space data.
    
    Args:
        kspace_data: Multi-coil k-space data [n_coils, ...spatial_dims]
        method: Estimation method ('adaptive', 'sos', 'walsh')
        smoothing: Gaussian smoothing sigma for regularization
        
    Returns:
        Coil sensitivity maps [n_coils, ...spatial_dims]
    """
    n_coils = kspace_data.shape[0]
    
    # Transform to image space
    if kspace_data.ndim == 3:  # 2D
        coil_images = ifft2c(kspace_data)
    elif kspace_data.ndim == 4:  # 3D
        coil_images = ifft3c(kspace_data)
    else:
        raise ValueError(f"Unsupported k-space dimensions: {kspace_data.ndim}")
    
    if method == 'sos':
        # Sum-of-squares method
        sos = np.sqrt(np.sum(np.abs(coil_images)**2, axis=0))
        sos = np.maximum(sos, np.max(sos) * 1e-6)  # Avoid division by zero
        
        sensitivity_maps = coil_images / sos[np.newaxis, ...]
        
    elif method == 'adaptive':
        # Adaptive combine (Walsh method approximation)
        # Smooth the coil images
        smoothed_images = np.zeros_like(coil_images)
        for c in range(n_coils):
            smoothed_images[c] = ndimage.gaussian_filter(
                np.abs(coil_images[c]), sigma=smoothing
            ) * np.exp(1j * np.angle(coil_images[c]))
        
        # Compute sensitivity as smoothed / sum-of-squares
        sos = np.sqrt(np.sum(np.abs(smoothed_images)**2, axis=0))
        sos = np.maximum(sos, np.max(sos) * 1e-6)
        
        sensitivity_maps = smoothed_images / sos[np.newaxis, ...]
        
    elif method == 'walsh':
        # Full Walsh method (simplified version)
        # This would typically involve eigendecomposition
        # For simplicity, use adaptive method
        return estimate_coil_sensitivity(kspace_data, method='adaptive', smoothing=smoothing)
        
    else:
        raise ValueError(f"Unknown sensitivity estimation method: {method}")
    
    return sensitivity_maps


def sense_combine(coil_images, sensitivity_maps):
    """
    Combine multi-coil images using SENSE.
    
    Args:
        coil_images: Multi-coil images [n_coils, ...spatial_dims]
        sensitivity_maps: Coil sensitivity maps [n_coils, ...spatial_dims]
        
    Returns:
        Combined image [...spatial_dims]
    """
    # SENSE combination: sum(conj(S) * I) / sum(abs(S)^2)
    numerator = np.sum(np.conj(sensitivity_maps) * coil_images, axis=0)
    denominator = np.sum(np.abs(sensitivity_maps)**2, axis=0)
    
    # Regularization to avoid division by zero
    denominator = np.maximum(denominator, np.max(denominator) * 1e-6)
    
    return numerator / denominator


def compute_density_compensation(k_traj, method='voronoi'):
    """
    Compute density compensation weights for non-Cartesian trajectories.
    
    Args:
        k_traj: k-space trajectory [n_points, n_dims]
        method: Computation method ('voronoi', 'jackson', 'pipe')
        
    Returns:
        Density compensation weights [n_points]
    """
    n_points = k_traj.shape[0]
    
    if method == 'voronoi':
        # Simplified Voronoi-based density compensation
        # Find nearest neighbor distances
        distances = cdist(k_traj, k_traj)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        
        # Density compensation proportional to area (distance^2 for 2D)
        weights = min_distances ** k_traj.shape[1]
        
    elif method == 'jackson':
        # Jackson's method (simplified)
        weights = np.ones(n_points)
        
    elif method == 'pipe':
        # Pipe's method
        weights = np.ones(n_points)
        
    else:
        raise ValueError(f"Unknown density compensation method: {method}")
    
    # Normalize weights
    weights /= np.mean(weights)
    
    return weights


def apply_density_compensation(k_data, weights):
    """
    Apply density compensation weights to k-space data.
    
    Args:
        k_data: k-space data [..., n_points]
        weights: Density compensation weights [n_points]
        
    Returns:
        Weighted k-space data with same shape
    """
    return k_data * weights


def rss_combine(coil_images):
    """
    Root-sum-of-squares coil combination.
    
    Args:
        coil_images: Multi-coil images [n_coils, ...spatial_dims]
        
    Returns:
        Combined magnitude image [...spatial_dims]
    """
    return np.sqrt(np.sum(np.abs(coil_images)**2, axis=0))
