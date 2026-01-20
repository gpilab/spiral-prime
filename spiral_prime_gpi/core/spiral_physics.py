"""
Spiral MRI physics and signal modeling.

This module implements physics-based modeling for spiral acquisitions,
particularly for steady-state localized-quadratic (LQ) spin-echo sequences.
"""

import numpy as np


def design_spiral_trajectory(fov, resolution, n_interleaves=1, gmax=40, smax=150, 
                             dt=4e-6, gamma=4257.0):
    """
    Design a 2D spiral trajectory.
    
    Args:
        fov: Field of view in cm
        resolution: Spatial resolution in cm
        n_interleaves: Number of spiral interleaves
        gmax: Maximum gradient amplitude in mT/m
        smax: Maximum slew rate in T/m/s
        dt: Sampling interval in seconds
        gamma: Gyromagnetic ratio in Hz/T (default for 1H)
        
    Returns:
        k_traj: k-space trajectory [n_points, 2] in normalized units [-0.5, 0.5]
        gradient: Gradient waveforms [n_points, 2] in mT/m
        time: Time points in seconds
    """
    # Maximum k-space extent (in cycles/cm)
    kmax = 1.0 / (2.0 * resolution)
    
    # Design single spiral arm using variable density
    # This is a simplified Archimedean spiral
    n_points = int(np.pi * (fov / resolution)**2 / n_interleaves)
    
    # Spiral parameter
    theta = np.linspace(0, np.sqrt(n_points) * 2 * np.pi / n_interleaves, n_points)
    
    # Archimedean spiral: r = a * theta
    a = kmax / theta[-1]
    r = a * theta
    
    # k-space coordinates
    kx = r * np.cos(theta)
    ky = r * np.sin(theta)
    
    # Compute gradients (derivative of k-space trajectory)
    # k(t) = gamma/(2*pi) * integral(G(t)dt)
    gradient_x = np.gradient(kx, dt) / (gamma * 10)  # Convert to mT/m
    gradient_y = np.gradient(ky, dt) / (gamma * 10)
    
    # Check constraints
    g_mag = np.sqrt(gradient_x**2 + gradient_y**2)
    if np.max(g_mag) > gmax:
        # Scale down to meet gradient constraint
        scale = gmax / np.max(g_mag)
        gradient_x *= scale
        gradient_y *= scale
        kx *= scale
        ky *= scale
    
    # Normalize k-space to [-0.5, 0.5]
    kx_norm = kx / (2 * kmax)
    ky_norm = ky / (2 * kmax)
    
    k_traj = np.column_stack([kx_norm, ky_norm])
    gradient = np.column_stack([gradient_x, gradient_y])
    time = np.arange(n_points) * dt
    
    return k_traj, gradient, time


def create_3d_stack_of_spirals(fov_xy, fov_z, resolution_xy, resolution_z, 
                               n_interleaves=1, **kwargs):
    """
    Create a 3D stack-of-spirals trajectory.
    
    Args:
        fov_xy: In-plane field of view in cm
        fov_z: Through-plane field of view in cm
        resolution_xy: In-plane resolution in cm
        resolution_z: Through-plane resolution in cm
        n_interleaves: Number of spiral interleaves per slice
        **kwargs: Additional arguments passed to design_spiral_trajectory
        
    Returns:
        k_traj_3d: 3D k-space trajectory [n_points_total, 3]
        gradient_3d: Gradient waveforms [n_points_total, 3]
    """
    # Design 2D spiral
    k_traj_2d, gradient_2d, time = design_spiral_trajectory(
        fov_xy, resolution_xy, n_interleaves, **kwargs
    )
    
    # Number of slices
    n_slices = int(fov_z / resolution_z)
    
    # Create kz encoding
    kz_max = 1.0 / (2.0 * resolution_z)
    kz_positions = np.linspace(-0.5, 0.5, n_slices)
    
    # Stack spirals at different kz positions
    k_traj_list = []
    gradient_list = []
    
    for kz in kz_positions:
        # Repeat the 2D spiral for this slice
        for interleaf in range(n_interleaves):
            # Rotate spiral for interleaves
            angle = 2 * np.pi * interleaf / n_interleaves
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            kx_rot = k_traj_2d[:, 0] * cos_a - k_traj_2d[:, 1] * sin_a
            ky_rot = k_traj_2d[:, 0] * sin_a + k_traj_2d[:, 1] * cos_a
            kz_vals = np.full(len(k_traj_2d), kz)
            
            k_traj_3d_inter = np.column_stack([kz_vals, ky_rot, kx_rot])
            k_traj_list.append(k_traj_3d_inter)
            
            # Also rotate gradients
            gx_rot = gradient_2d[:, 0] * cos_a - gradient_2d[:, 1] * sin_a
            gy_rot = gradient_2d[:, 0] * sin_a + gradient_2d[:, 1] * cos_a
            gz_vals = np.zeros(len(gradient_2d))
            
            gradient_3d_inter = np.column_stack([gz_vals, gy_rot, gx_rot])
            gradient_list.append(gradient_3d_inter)
    
    k_traj_3d = np.vstack(k_traj_list)
    gradient_3d = np.vstack(gradient_list)
    
    return k_traj_3d, gradient_3d


def steady_state_signal(T1, T2, TR, TE, flip_angle, t_prep=None):
    """
    Calculate steady-state signal for a spin-echo sequence.
    
    Args:
        T1: Longitudinal relaxation time in ms
        T2: Transverse relaxation time in ms
        TR: Repetition time in ms
        TE: Echo time in ms
        flip_angle: Flip angle in degrees
        t_prep: Preparation time (optional) in ms
        
    Returns:
        Signal amplitude (arbitrary units, relative to M0)
    """
    # Convert flip angle to radians
    alpha = np.deg2rad(flip_angle)
    
    # Basic spin-echo steady-state signal
    # S = M0 * sin(alpha) * exp(-TE/T2) * (1 - exp(-TR/T1)) / (1 - cos(alpha)*exp(-TR/T1))
    
    E1 = np.exp(-TR / T1)
    E2 = np.exp(-TE / T2)
    
    signal = np.sin(alpha) * E2 * (1 - E1) / (1 - np.cos(alpha) * E1)
    
    if t_prep is not None:
        # Include preparation effects
        signal *= (1 - np.exp(-t_prep / T1))
    
    return signal


def localized_quadratic_model(position, B0, B1, gradient_moments, shim_coeffs=None):
    """
    Model localized quadratic (LQ) field variations.
    
    The LQ model captures spatially-varying B0 and B1 fields including
    linear and quadratic components for improved reconstruction.
    
    Args:
        position: Spatial position [x, y, z] in cm
        B0: Static field in Tesla
        B1: Transmit field relative amplitude
        gradient_moments: Gradient moment matrix [3, 3] for quadratic terms
        shim_coeffs: Optional shimming coefficients [n_coeffs]
        
    Returns:
        Field value at the given position
    """
    x, y, z = position
    
    # Linear gradient terms (first-order shim)
    linear_field = 0.0
    if shim_coeffs is not None and len(shim_coeffs) >= 3:
        linear_field = shim_coeffs[0] * x + shim_coeffs[1] * y + shim_coeffs[2] * z
    
    # Quadratic terms
    r = np.array([x, y, z])
    quadratic_field = 0.5 * r @ gradient_moments @ r
    
    # Total field
    total_field = B0 * B1 * (1 + linear_field + quadratic_field)
    
    return total_field


def multi_contrast_weighting(tissue_params, sequence_params):
    """
    Calculate multi-contrast weightings for different tissue types.
    
    Args:
        tissue_params: Dictionary with tissue T1, T2 values
                      e.g., {'WM': (800, 80), 'GM': (1200, 100), 'CSF': (4000, 2000)}
        sequence_params: Dictionary with TR, TE, flip_angle
        
    Returns:
        Dictionary of signal intensities for each tissue type
    """
    TR = sequence_params['TR']
    TE = sequence_params['TE']
    flip_angle = sequence_params.get('flip_angle', 90)
    
    signals = {}
    for tissue_name, (T1, T2) in tissue_params.items():
        signal = steady_state_signal(T1, T2, TR, TE, flip_angle)
        signals[tissue_name] = signal
    
    return signals


def estimate_off_resonance(field_map, TE):
    """
    Estimate signal phase from off-resonance (B0 field map).
    
    Args:
        field_map: Off-resonance map in Hz
        TE: Echo time in seconds
        
    Returns:
        Phase map in radians
    """
    phase = 2 * np.pi * field_map * TE
    return phase


def apply_off_resonance_correction(kspace_data, field_map, trajectory, TE):
    """
    Correct for off-resonance effects in non-Cartesian reconstruction.
    
    Args:
        kspace_data: k-space data
        field_map: Off-resonance map in Hz
        trajectory: k-space trajectory
        TE: Echo time in seconds
        
    Returns:
        Corrected k-space data
    """
    # This is a placeholder for off-resonance correction
    # Full implementation would use conjugate phase reconstruction or
    # multi-frequency interpolation
    
    # Simple phase correction
    phase_correction = np.exp(-1j * 2 * np.pi * field_map * TE)
    
    return kspace_data


def simulate_spiral_acquisition(image, trajectory, coil_maps, noise_std=0.0):
    """
    Simulate spiral k-space acquisition from an image.
    
    Args:
        image: Image to acquire [..., ny, nx]
        trajectory: Spiral trajectory [n_points, 2]
        coil_maps: Coil sensitivity maps [n_coils, ny, nx]
        noise_std: Standard deviation of complex Gaussian noise
        
    Returns:
        Simulated k-space data [n_coils, n_points]
    """
    from scipy.interpolate import RegularGridInterpolator
    
    n_coils = coil_maps.shape[0]
    ny, nx = image.shape[-2:]
    n_points = trajectory.shape[0]
    
    # Create grid coordinates
    y_coords = np.linspace(-0.5, 0.5, ny)
    x_coords = np.linspace(-0.5, 0.5, nx)
    
    # FFT of coil images
    kspace_full = np.zeros((n_coils, n_points), dtype=complex)
    
    for c in range(n_coils):
        # Multiply by coil sensitivity
        coil_image = image * coil_maps[c]
        
        # FFT to k-space
        from . import mri_math
        kspace_cart = mri_math.fft2c(coil_image)
        
        # Interpolate onto spiral trajectory
        real_interp = RegularGridInterpolator(
            (y_coords, x_coords), 
            kspace_cart.real,
            bounds_error=False,
            fill_value=0
        )
        imag_interp = RegularGridInterpolator(
            (y_coords, x_coords), 
            kspace_cart.imag,
            bounds_error=False,
            fill_value=0
        )
        
        kspace_full[c] = (real_interp(trajectory) + 
                         1j * imag_interp(trajectory))
    
    # Add noise
    if noise_std > 0:
        noise = (np.random.randn(*kspace_full.shape) + 
                1j * np.random.randn(*kspace_full.shape)) * noise_std / np.sqrt(2)
        kspace_full += noise
    
    return kspace_full
