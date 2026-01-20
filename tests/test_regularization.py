"""
Unit tests for regularization functions.
"""

import pytest
import numpy as np
from spiral_prime_gpi.core import regularization


class TestSoftThreshold:
    """Test soft thresholding."""
    
    def test_soft_threshold_real(self):
        """Test soft thresholding on real data."""
        x = np.array([-3.0, -1.0, 0.5, 2.0, 4.0])
        threshold = 1.5
        
        result = regularization.soft_threshold(x, threshold)
        expected = np.array([-1.5, 0.0, 0.0, 0.5, 2.5])
        
        np.testing.assert_allclose(result, expected)
    
    def test_soft_threshold_complex(self):
        """Test soft thresholding on complex data."""
        x = np.array([2+2j, 1+1j, 0.5+0.5j])
        threshold = 1.5
        
        result = regularization.soft_threshold(x, threshold)
        
        # Check that magnitude is reduced correctly
        for i, val in enumerate(x):
            mag = np.abs(val)
            if mag > threshold:
                expected_mag = mag - threshold
                assert np.abs(np.abs(result[i]) - expected_mag) < 1e-10
            else:
                assert np.abs(result[i]) < 1e-10
    
    def test_soft_threshold_zeros_below_threshold(self):
        """Test that values below threshold are zeroed."""
        x = np.array([0.5, 1.0, 1.5])
        threshold = 2.0
        
        result = regularization.soft_threshold(x, threshold)
        
        np.testing.assert_allclose(result, np.zeros_like(x))


class TestTVNorm:
    """Test total variation norm."""
    
    def test_tv_norm_2d(self):
        """Test TV norm on 2D image."""
        # Create image with edges
        image = np.zeros((32, 32))
        image[10:20, 10:20] = 1.0
        
        tv = regularization.tv_norm(image)
        
        # TV should be positive
        assert tv > 0
    
    def test_tv_norm_3d(self):
        """Test TV norm on 3D image."""
        image = np.zeros((16, 16, 16))
        image[5:10, 5:10, 5:10] = 1.0
        
        tv = regularization.tv_norm(image)
        
        assert tv > 0
    
    def test_tv_norm_smooth_image(self):
        """Test that smooth images have lower TV."""
        # Smooth image
        smooth = np.ones((32, 32))
        
        # Image with edges
        edges = np.zeros((32, 32))
        edges[10:20, :] = 1.0
        
        tv_smooth = regularization.tv_norm(smooth)
        tv_edges = regularization.tv_norm(edges)
        
        # Edges should have higher TV
        assert tv_edges > tv_smooth


class TestTVGradient:
    """Test TV gradient computation."""
    
    def test_tv_gradient_2d(self):
        """Test TV gradient on 2D image."""
        image = np.random.randn(32, 32)
        
        grad = regularization.tv_gradient(image)
        
        # Check output shape
        assert grad.shape == image.shape
    
    def test_tv_gradient_3d(self):
        """Test TV gradient on 3D image."""
        image = np.random.randn(16, 16, 16)
        
        grad = regularization.tv_gradient(image)
        
        assert grad.shape == image.shape


class TestTVProximal:
    """Test TV proximal operator."""
    
    def test_tv_proximal_denoising(self):
        """Test that TV proximal reduces noise."""
        # Create clean image
        clean = np.zeros((32, 32))
        clean[10:20, 10:20] = 1.0
        
        # Add noise
        noisy = clean + 0.1 * np.random.randn(32, 32)
        
        # Denoise
        denoised = regularization.tv_proximal(noisy, lambda_tv=0.05, n_iter=20)
        
        # Denoised should be closer to clean than noisy
        error_noisy = np.linalg.norm(noisy - clean)
        error_denoised = np.linalg.norm(denoised - clean)
        
        assert error_denoised < error_noisy
    
    def test_tv_proximal_preserves_shape(self):
        """Test that TV proximal preserves shape."""
        image = np.random.randn(24, 24)
        
        result = regularization.tv_proximal(image, lambda_tv=0.01)
        
        assert result.shape == image.shape


class TestL1L2Regularization:
    """Test L1 and L2 regularization."""
    
    def test_l1_regularization(self):
        """Test L1 norm computation."""
        image = np.array([1.0, -2.0, 3.0, -4.0])
        lambda_l1 = 0.5
        
        reg = regularization.l1_regularization(image, lambda_l1)
        
        expected = lambda_l1 * np.sum(np.abs(image))
        assert np.abs(reg - expected) < 1e-10
    
    def test_l2_regularization(self):
        """Test L2 norm computation."""
        image = np.array([1.0, 2.0, 3.0])
        lambda_l2 = 0.1
        
        reg = regularization.l2_regularization(image, lambda_l2)
        
        expected = lambda_l2 * np.sum(image**2)
        assert np.abs(reg - expected) < 1e-10


class TestHuberLoss:
    """Test Huber loss."""
    
    def test_huber_loss(self):
        """Test Huber loss computation."""
        x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        delta = 1.0
        
        loss = regularization.huber_loss(x, delta)
        
        # Loss should be positive
        assert loss > 0
    
    def test_huber_reduces_to_l2(self):
        """Test Huber reduces to L2 for small values."""
        x = np.array([0.1, 0.2, -0.15])
        delta = 1.0
        
        huber = regularization.huber_loss(x, delta)
        l2 = 0.5 * np.sum(x**2)
        
        # Should be close for small values
        np.testing.assert_allclose(huber, l2, rtol=0.1)


class TestApplyRegularization:
    """Test apply_regularization wrapper."""
    
    def test_apply_regularization_l1(self):
        """Test applying L1 regularization."""
        image = np.array([2.0, 1.0, 0.5, -1.5])
        
        result = regularization.apply_regularization(image, 'l1', lambda_reg=0.8)
        
        # Should be soft thresholded
        assert result.shape == image.shape
    
    def test_apply_regularization_l2(self):
        """Test applying L2 regularization."""
        image = np.array([2.0, 4.0, 6.0])
        lambda_reg = 0.5
        
        result = regularization.apply_regularization(image, 'l2', lambda_reg)
        
        # L2 proximal: x / (1 + lambda)
        expected = image / (1 + lambda_reg)
        np.testing.assert_allclose(result, expected)
    
    def test_apply_regularization_tv(self):
        """Test applying TV regularization."""
        image = np.random.randn(16, 16)
        
        result = regularization.apply_regularization(image, 'tv', lambda_reg=0.01, n_iter=5)
        
        assert result.shape == image.shape
    
    def test_apply_regularization_invalid_type(self):
        """Test that invalid regularization type raises error."""
        image = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError):
            regularization.apply_regularization(image, 'invalid_type', 0.1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
