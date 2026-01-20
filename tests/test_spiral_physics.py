"""
Unit tests for spiral physics and trajectory design.
"""

import pytest
import numpy as np
from spiral_prime_gpi.core import spiral_physics


class TestSpiralTrajectory:
    """Test spiral trajectory design."""
    
    def test_design_spiral_trajectory_2d(self):
        """Test 2D spiral trajectory design."""
        fov = 24.0  # cm
        resolution = 0.1  # cm
        n_interleaves = 4
        
        trajectory, gradient, time = spiral_physics.design_spiral_trajectory(
            fov, resolution, n_interleaves
        )
        
        # Check output shapes
        assert trajectory.shape[1] == 2  # 2D
        assert gradient.shape == trajectory.shape
        assert len(time) == trajectory.shape[0]
        
        # Check trajectory is normalized
        assert np.all(trajectory >= -0.6)
        assert np.all(trajectory <= 0.6)
        
        # Check gradients are reasonable
        assert np.all(np.abs(gradient) <= 50.0)  # mT/m
    
    def test_design_spiral_multiple_interleaves(self):
        """Test that more interleaves give different trajectories."""
        fov = 20.0
        resolution = 0.2
        
        traj1, _, _ = spiral_physics.design_spiral_trajectory(fov, resolution, n_interleaves=1)
        traj4, _, _ = spiral_physics.design_spiral_trajectory(fov, resolution, n_interleaves=4)
        
        # Different number of interleaves should give different results
        # (though for single interleaf case, they start the same)
        assert traj1.shape[0] > traj4.shape[0]  # More interleaves = shorter per interleaf


class TestStackOfSpirals:
    """Test 3D stack-of-spirals trajectory."""
    
    def test_create_3d_stack_of_spirals(self):
        """Test 3D stack-of-spirals creation."""
        fov_xy = 24.0
        fov_z = 16.0
        resolution_xy = 0.2
        resolution_z = 0.2
        n_interleaves = 2
        
        trajectory_3d, gradient_3d = spiral_physics.create_3d_stack_of_spirals(
            fov_xy, fov_z, resolution_xy, resolution_z, n_interleaves
        )
        
        # Check that trajectory is 3D
        assert trajectory_3d.shape[1] == 3
        
        # Check that we have multiple slices
        unique_kz = np.unique(np.round(trajectory_3d[:, 0], decimals=3))
        expected_slices = int(fov_z / resolution_z)
        assert len(unique_kz) >= expected_slices * 0.8  # Allow some tolerance
        
        # Check normalization
        assert np.all(trajectory_3d >= -0.6)
        assert np.all(trajectory_3d <= 0.6)


class TestSteadyStateSignal:
    """Test steady-state signal calculation."""
    
    def test_steady_state_signal_basic(self):
        """Test basic steady-state signal calculation."""
        T1 = 1000  # ms
        T2 = 100   # ms
        TR = 2000  # ms
        TE = 30    # ms
        flip_angle = 90  # degrees
        
        signal = spiral_physics.steady_state_signal(T1, T2, TR, TE, flip_angle)
        
        # Signal should be positive and less than 1
        assert signal > 0
        assert signal < 1
    
    def test_steady_state_signal_T2_decay(self):
        """Test that longer TE gives lower signal."""
        T1, T2 = 1000, 100
        TR, flip_angle = 2000, 90
        
        signal_short_te = spiral_physics.steady_state_signal(T1, T2, TR, TE=20, flip_angle=flip_angle)
        signal_long_te = spiral_physics.steady_state_signal(T1, T2, TR, TE=80, flip_angle=flip_angle)
        
        # Longer TE should give lower signal
        assert signal_long_te < signal_short_te
    
    def test_steady_state_signal_with_prep(self):
        """Test steady-state signal with preparation."""
        T1, T2 = 1000, 100
        TR, TE, flip_angle = 2000, 30, 90
        
        signal_no_prep = spiral_physics.steady_state_signal(T1, T2, TR, TE, flip_angle)
        signal_with_prep = spiral_physics.steady_state_signal(T1, T2, TR, TE, flip_angle, t_prep=1000)
        
        # Both should be valid
        assert signal_no_prep > 0
        assert signal_with_prep > 0


class TestLocalizedQuadraticModel:
    """Test LQ field modeling."""
    
    def test_localized_quadratic_model(self):
        """Test LQ field calculation."""
        position = [1.0, 2.0, 3.0]  # cm
        B0 = 3.0  # Tesla
        B1 = 1.0
        gradient_moments = np.eye(3) * 0.001
        
        field = spiral_physics.localized_quadratic_model(
            position, B0, B1, gradient_moments
        )
        
        # Field should be close to B0*B1
        assert field > 0
        assert np.abs(field - B0 * B1) < 0.1


class TestMultiContrastWeighting:
    """Test multi-contrast weighting."""
    
    def test_multi_contrast_weighting(self):
        """Test multi-contrast signal calculation."""
        tissue_params = {
            'WM': (800, 80),    # White matter
            'GM': (1200, 100),  # Gray matter
            'CSF': (4000, 2000) # CSF
        }
        
        sequence_params = {
            'TR': 2000,
            'TE': 30,
            'flip_angle': 90
        }
        
        signals = spiral_physics.multi_contrast_weighting(tissue_params, sequence_params)
        
        # Should have signal for each tissue
        assert 'WM' in signals
        assert 'GM' in signals
        assert 'CSF' in signals
        
        # All signals should be positive
        assert all(s > 0 for s in signals.values())
        
        # With typical TE=30ms, CSF (very long T2) might not have highest signal
        # Just check all are positive and reasonable
        assert 0 < signals['CSF'] < 1
        assert 0 < signals['WM'] < 1
        assert 0 < signals['GM'] < 1


class TestOffResonance:
    """Test off-resonance effects."""
    
    def test_estimate_off_resonance(self):
        """Test off-resonance phase estimation."""
        field_map = np.random.randn(32, 32) * 100  # Hz
        TE = 0.030  # seconds
        
        phase = spiral_physics.estimate_off_resonance(field_map, TE)
        
        # Check output shape
        assert phase.shape == field_map.shape
        
        # Check phase is in reasonable range (field_map is random, so allow larger range)
        # Phase = 2*pi*field_map*TE, with field_map up to ~300 Hz and TE=0.03s
        max_expected = 2 * np.pi * 500 * TE
        assert np.all(np.abs(phase) <= max_expected)


class TestSpiralSimulation:
    """Test spiral acquisition simulation."""
    
    def test_simulate_spiral_acquisition(self):
        """Test simulating spiral k-space acquisition."""
        image_size = (32, 32)
        n_coils = 4
        n_points = 100
        
        # Create simple image
        image = np.random.randn(*image_size) + 1j * np.random.randn(*image_size)
        
        # Create simple trajectory
        trajectory = np.random.rand(n_points, 2) - 0.5
        
        # Create uniform coil maps
        coil_maps = np.ones((n_coils, *image_size), dtype=complex) / np.sqrt(n_coils)
        
        # Simulate
        kspace = spiral_physics.simulate_spiral_acquisition(
            image, trajectory, coil_maps, noise_std=0.0
        )
        
        # Check output shape
        assert kspace.shape == (n_coils, n_points)
        
        # Check that output is complex
        assert np.iscomplexobj(kspace)
    
    def test_simulate_with_noise(self):
        """Test simulation with noise."""
        image_size = (16, 16)
        n_coils = 2
        n_points = 50
        
        image = np.ones(image_size, dtype=complex)
        trajectory = np.random.rand(n_points, 2) - 0.5
        coil_maps = np.ones((n_coils, *image_size), dtype=complex) / np.sqrt(n_coils)
        
        # With and without noise
        kspace_clean = spiral_physics.simulate_spiral_acquisition(
            image, trajectory, coil_maps, noise_std=0.0
        )
        kspace_noisy = spiral_physics.simulate_spiral_acquisition(
            image, trajectory, coil_maps, noise_std=0.1
        )
        
        # Noisy should be different from clean
        assert not np.allclose(kspace_clean, kspace_noisy)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
