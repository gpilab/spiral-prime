"""
Unit tests for MRI math operations.
"""

import pytest
import numpy as np
from spiral_prime_gpi.core import mri_math


class TestFFTOperations:
    """Test FFT operations."""
    
    def test_fft2c_ifft2c_inverse(self):
        """Test that FFT and inverse FFT are inverses."""
        # Create random image
        image = np.random.randn(64, 64) + 1j * np.random.randn(64, 64)
        
        # Forward and inverse transform
        kspace = mri_math.fft2c(image)
        image_recon = mri_math.ifft2c(kspace)
        
        # Check reconstruction
        np.testing.assert_allclose(image, image_recon, rtol=1e-10)
    
    def test_fft3c_ifft3c_inverse(self):
        """Test that 3D FFT and inverse FFT are inverses."""
        # Create random 3D image
        image = np.random.randn(16, 32, 32) + 1j * np.random.randn(16, 32, 32)
        
        # Forward and inverse transform
        kspace = mri_math.fft3c(image)
        image_recon = mri_math.ifft3c(kspace)
        
        # Check reconstruction
        np.testing.assert_allclose(image, image_recon, rtol=1e-10)
    
    def test_fft2c_centering(self):
        """Test that FFT is properly centered."""
        # Create delta function at center
        image = np.zeros((64, 64), dtype=complex)
        image[32, 32] = 1.0
        
        # Transform should be constant
        kspace = mri_math.fft2c(image)
        
        # Check that k-space is approximately uniform
        magnitude = np.abs(kspace)
        np.testing.assert_allclose(magnitude, magnitude[0, 0], rtol=0.01)


class TestGridding:
    """Test gridding operations."""
    
    def test_grid_noncartesian_2d(self):
        """Test 2D non-Cartesian gridding."""
        # Create simple radial trajectory
        n_points = 100
        angles = np.linspace(0, np.pi, n_points)
        r = np.linspace(0, 0.5, n_points)
        
        trajectory = np.column_stack([
            r * np.cos(angles),
            r * np.sin(angles)
        ])
        
        # Create simple data (constant)
        data = np.ones(n_points, dtype=complex)
        
        # Grid the data
        grid_size = (64, 64)
        gridded = mri_math.grid_noncartesian(trajectory, data, grid_size)
        
        # Check output shape
        assert gridded.shape == grid_size
        
        # Check that gridded data is complex
        assert np.iscomplexobj(gridded)
    
    def test_grid_noncartesian_multicoil(self):
        """Test multi-coil gridding."""
        n_coils = 4
        n_points = 50
        
        # Simple trajectory
        trajectory = np.random.rand(n_points, 2) - 0.5
        
        # Multi-coil data
        data = np.random.randn(n_coils, n_points) + 1j * np.random.randn(n_coils, n_points)
        
        # Grid
        grid_size = (32, 32)
        gridded = mri_math.grid_noncartesian(trajectory, data, grid_size)
        
        # Check output shape
        assert gridded.shape == (n_coils, *grid_size)


class TestCoilSensitivity:
    """Test coil sensitivity estimation."""
    
    def test_estimate_coil_sensitivity_sos(self):
        """Test SOS coil sensitivity estimation."""
        n_coils = 8
        image_size = (64, 64)
        
        # Create synthetic k-space data
        kspace = np.random.randn(n_coils, *image_size) + 1j * np.random.randn(n_coils, *image_size)
        
        # Estimate sensitivity maps
        coil_maps = mri_math.estimate_coil_sensitivity(kspace, method='sos')
        
        # Check output shape
        assert coil_maps.shape == (n_coils, *image_size)
        
        # Check that maps sum approximately to 1
        sum_of_squares = np.sum(np.abs(coil_maps)**2, axis=0)
        np.testing.assert_allclose(sum_of_squares, 1.0, rtol=0.1)
    
    def test_estimate_coil_sensitivity_adaptive(self):
        """Test adaptive coil sensitivity estimation."""
        n_coils = 4
        image_size = (32, 32)
        
        # Create synthetic k-space data
        kspace = np.random.randn(n_coils, *image_size) + 1j * np.random.randn(n_coils, *image_size)
        
        # Estimate sensitivity maps
        coil_maps = mri_math.estimate_coil_sensitivity(kspace, method='adaptive', smoothing=3)
        
        # Check output shape
        assert coil_maps.shape == (n_coils, *image_size)
        
        # Check that output is complex
        assert np.iscomplexobj(coil_maps)


class TestSENSECombine:
    """Test SENSE combination."""
    
    def test_sense_combine(self):
        """Test SENSE coil combination."""
        n_coils = 4
        image_size = (32, 32)
        
        # Create synthetic coil images
        coil_images = np.random.randn(n_coils, *image_size) + 1j * np.random.randn(n_coils, *image_size)
        
        # Create sensitivity maps (uniform for simplicity)
        coil_maps = np.ones((n_coils, *image_size), dtype=complex) / np.sqrt(n_coils)
        
        # Combine
        combined = mri_math.sense_combine(coil_images, coil_maps)
        
        # Check output shape
        assert combined.shape == image_size
        
        # Check that output is complex
        assert np.iscomplexobj(combined)


class TestDensityCompensation:
    """Test density compensation."""
    
    def test_compute_density_compensation_voronoi(self):
        """Test Voronoi density compensation."""
        n_points = 100
        
        # Create random trajectory
        trajectory = np.random.rand(n_points, 2) - 0.5
        
        # Compute weights
        weights = mri_math.compute_density_compensation(trajectory, method='voronoi')
        
        # Check output shape
        assert weights.shape == (n_points,)
        
        # Check that weights are positive
        assert np.all(weights > 0)
        
        # Check that mean is approximately 1 (normalized)
        np.testing.assert_allclose(np.mean(weights), 1.0, rtol=0.5)
    
    def test_apply_density_compensation(self):
        """Test applying density compensation."""
        n_points = 50
        
        # Create data and weights
        data = np.random.randn(n_points) + 1j * np.random.randn(n_points)
        weights = np.random.rand(n_points)
        
        # Apply weights
        weighted_data = mri_math.apply_density_compensation(data, weights)
        
        # Check shape preserved
        assert weighted_data.shape == data.shape
        
        # Check that multiplication worked
        np.testing.assert_allclose(weighted_data, data * weights)


class TestRSSCombine:
    """Test root-sum-of-squares combination."""
    
    def test_rss_combine(self):
        """Test RSS coil combination."""
        n_coils = 4
        image_size = (32, 32)
        
        # Create synthetic coil images
        coil_images = np.random.randn(n_coils, *image_size) + 1j * np.random.randn(n_coils, *image_size)
        
        # Combine
        combined = mri_math.rss_combine(coil_images)
        
        # Check output shape
        assert combined.shape == image_size
        
        # Check that output is real (magnitude)
        assert not np.iscomplexobj(combined)
        
        # Check that all values are non-negative
        assert np.all(combined >= 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
