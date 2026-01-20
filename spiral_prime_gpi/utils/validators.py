"""
Validation utilities for input parameters and data.
"""

import numpy as np
from pathlib import Path


def validate_kspace_data(kspace_data):
    """
    Validate k-space data format and properties.
    
    Args:
        kspace_data: k-space array to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If data is invalid
    """
    if not isinstance(kspace_data, np.ndarray):
        raise ValueError("k-space data must be a numpy array")
    
    if not np.iscomplexobj(kspace_data):
        raise ValueError("k-space data must be complex-valued")
    
    if kspace_data.ndim < 2:
        raise ValueError(f"k-space data must be at least 2D, got {kspace_data.ndim}D")
    
    if np.any(np.isnan(kspace_data)):
        raise ValueError("k-space data contains NaN values")
    
    if np.any(np.isinf(kspace_data)):
        raise ValueError("k-space data contains infinite values")
    
    return True


def validate_trajectory(trajectory, expected_dims=None):
    """
    Validate trajectory data.
    
    Args:
        trajectory: Trajectory array [n_points, n_dims]
        expected_dims: Expected number of dimensions (2 or 3)
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If trajectory is invalid
    """
    if not isinstance(trajectory, np.ndarray):
        raise ValueError("Trajectory must be a numpy array")
    
    if trajectory.ndim != 2:
        raise ValueError(f"Trajectory must be 2D [n_points, n_dims], got shape {trajectory.shape}")
    
    n_points, n_dims = trajectory.shape
    
    if expected_dims is not None and n_dims != expected_dims:
        raise ValueError(f"Expected {expected_dims}D trajectory, got {n_dims}D")
    
    if n_dims not in [2, 3]:
        raise ValueError(f"Trajectory must be 2D or 3D, got {n_dims}D")
    
    # Check if normalized to [-0.5, 0.5] (allow some margin)
    if np.any(trajectory < -0.6) or np.any(trajectory > 0.6):
        raise ValueError(
            "Trajectory should be normalized to [-0.5, 0.5]. "
            f"Got range [{trajectory.min():.3f}, {trajectory.max():.3f}]"
        )
    
    if np.any(np.isnan(trajectory)):
        raise ValueError("Trajectory contains NaN values")
    
    return True


def validate_coil_maps(coil_maps, expected_shape=None):
    """
    Validate coil sensitivity maps.
    
    Args:
        coil_maps: Coil maps array [n_coils, ...spatial_dims]
        expected_shape: Expected spatial dimensions
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If coil maps are invalid
    """
    if not isinstance(coil_maps, np.ndarray):
        raise ValueError("Coil maps must be a numpy array")
    
    if not np.iscomplexobj(coil_maps):
        raise ValueError("Coil maps must be complex-valued")
    
    if coil_maps.ndim < 3:
        raise ValueError(f"Coil maps must be at least 3D [n_coils, ...], got {coil_maps.ndim}D")
    
    if expected_shape is not None:
        if coil_maps.shape[1:] != tuple(expected_shape):
            raise ValueError(
                f"Coil maps spatial shape {coil_maps.shape[1:]} "
                f"does not match expected {expected_shape}"
            )
    
    if np.any(np.isnan(coil_maps)):
        raise ValueError("Coil maps contain NaN values")
    
    return True


def validate_image_shape(shape):
    """
    Validate image shape.
    
    Args:
        shape: Image dimensions tuple
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If shape is invalid
    """
    if not isinstance(shape, (tuple, list)):
        raise ValueError("Image shape must be a tuple or list")
    
    if len(shape) < 2 or len(shape) > 3:
        raise ValueError(f"Image shape must be 2D or 3D, got {len(shape)}D")
    
    if any(s <= 0 for s in shape):
        raise ValueError(f"Image dimensions must be positive, got {shape}")
    
    if any(not isinstance(s, int) for s in shape):
        raise ValueError(f"Image dimensions must be integers, got {shape}")
    
    return True


def validate_regularization_params(reg_type, lambda_reg):
    """
    Validate regularization parameters.
    
    Args:
        reg_type: Regularization type
        lambda_reg: Regularization weight
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If parameters are invalid
    """
    valid_types = ['none', 'l1', 'l2', 'tv', 'tv_cp', 'wavelet']
    
    if reg_type not in valid_types:
        raise ValueError(
            f"Invalid regularization type '{reg_type}'. "
            f"Valid options: {valid_types}"
        )
    
    if lambda_reg < 0:
        raise ValueError(f"Regularization weight must be non-negative, got {lambda_reg}")
    
    return True


def validate_file_path(filepath, must_exist=False, extension=None):
    """
    Validate file path.
    
    Args:
        filepath: Path to file
        must_exist: If True, file must exist
        extension: Expected file extension (e.g., '.h5')
        
    Returns:
        Path object if valid
        
    Raises:
        ValueError: If path is invalid
        FileNotFoundError: If file must exist but doesn't
    """
    path = Path(filepath)
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if extension is not None:
        if not filepath.endswith(extension):
            raise ValueError(
                f"Expected file with extension '{extension}', got '{path.suffix}'"
            )
    
    return path


def validate_sequence_params(params):
    """
    Validate MRI sequence parameters.
    
    Args:
        params: Dictionary of sequence parameters
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If parameters are invalid
    """
    required_keys = ['TR', 'TE', 'flip_angle']
    
    for key in required_keys:
        if key not in params:
            raise ValueError(f"Missing required sequence parameter: {key}")
    
    # Validate timing
    if params['TR'] <= 0:
        raise ValueError(f"TR must be positive, got {params['TR']}")
    
    if params['TE'] <= 0:
        raise ValueError(f"TE must be positive, got {params['TE']}")
    
    if params['TE'] >= params['TR']:
        raise ValueError(f"TE ({params['TE']}) must be less than TR ({params['TR']})")
    
    # Validate flip angle
    if not (0 < params['flip_angle'] <= 180):
        raise ValueError(
            f"Flip angle must be between 0 and 180 degrees, got {params['flip_angle']}"
        )
    
    return True


def validate_reconstruction_params(params):
    """
    Validate reconstruction parameters.
    
    Args:
        params: Dictionary of reconstruction parameters
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If parameters are invalid
    """
    # Validate number of iterations
    if 'n_iterations' in params:
        if params['n_iterations'] <= 0:
            raise ValueError(
                f"Number of iterations must be positive, got {params['n_iterations']}"
            )
    
    # Validate regularization
    if 'regularization' in params:
        reg = params['regularization']
        if isinstance(reg, dict):
            if 'type' in reg and 'lambda' in reg:
                validate_regularization_params(reg['type'], reg['lambda'])
    
    # Validate coil combine method
    if 'coil_combine' in params:
        valid_methods = ['sos', 'adaptive', 'walsh']
        if params['coil_combine'] not in valid_methods:
            raise ValueError(
                f"Invalid coil combine method '{params['coil_combine']}'. "
                f"Valid options: {valid_methods}"
            )
    
    return True


def check_data_consistency(kspace_data, trajectory, coil_maps=None):
    """
    Check consistency between k-space data, trajectory, and coil maps.
    
    Args:
        kspace_data: k-space data [n_coils, n_points]
        trajectory: Trajectory [n_points, n_dims]
        coil_maps: Optional coil maps [n_coils, ...image_shape]
        
    Returns:
        True if consistent
        
    Raises:
        ValueError: If data is inconsistent
    """
    # Check number of points matches
    if kspace_data.shape[-1] != trajectory.shape[0]:
        raise ValueError(
            f"Number of k-space points ({kspace_data.shape[-1]}) "
            f"does not match trajectory points ({trajectory.shape[0]})"
        )
    
    # Check number of coils matches
    if coil_maps is not None:
        if kspace_data.shape[0] != coil_maps.shape[0]:
            raise ValueError(
                f"Number of coils in k-space data ({kspace_data.shape[0]}) "
                f"does not match coil maps ({coil_maps.shape[0]})"
            )
    
    # Check trajectory dimensions
    n_dims = trajectory.shape[1]
    if coil_maps is not None:
        if coil_maps.ndim - 1 != n_dims:
            raise ValueError(
                f"Trajectory is {n_dims}D but coil maps are "
                f"{coil_maps.ndim - 1}D (excluding coil dimension)"
            )
    
    return True
