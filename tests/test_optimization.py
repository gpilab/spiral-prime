"""
Unit tests for optimization algorithms.
"""

import pytest
import numpy as np
from spiral_prime_gpi.core import optimization, mri_math


class TestConjugateGradient:
    """Test conjugate gradient solver."""
    
    def test_cg_simple_system(self):
        """Test CG on a simple linear system."""
        # Create a simple positive definite system: Ax = b
        n = 10
        A_matrix = np.eye(n) + 0.1 * np.random.randn(n, n)
        A_matrix = A_matrix @ A_matrix.T  # Make positive definite
        
        b = np.random.randn(n)
        
        # Define linear operator
        def A(x):
            return A_matrix @ x
        
        # Solve
        x, residuals = optimization.conjugate_gradient(A, b, max_iter=20)
        
        # Check solution
        np.testing.assert_allclose(A(x), b, rtol=1e-3)
        
        # Check that residuals decrease
        assert residuals[-1] < residuals[0]
    
    def test_cg_convergence(self):
        """Test that CG converges for well-conditioned systems."""
        n = 20
        A_matrix = np.eye(n)
        b = np.random.randn(n)
        
        def A(x):
            return A_matrix @ x
        
        x, residuals = optimization.conjugate_gradient(A, b, max_iter=30, tol=1e-8)
        
        # Should converge to b (since A is identity)
        np.testing.assert_allclose(x, b, rtol=1e-6)


class TestIterativeSENSE:
    """Test iterative SENSE reconstruction."""
    
    def test_iterative_sense_recon_basic(self):
        """Test basic SENSE reconstruction."""
        # Create simple synthetic data
        n_coils = 4
        image_size = (32, 32)
        n_points = 200
        
        # True image
        true_image = np.random.randn(*image_size) + 1j * np.random.randn(*image_size)
        
        # Coil maps (uniform for simplicity)
        coil_maps = np.ones((n_coils, *image_size), dtype=complex) / np.sqrt(n_coils)
        
        # Simple trajectory (Cartesian subset)
        trajectory = np.random.rand(n_points, 2) - 0.5
        
        # Simulate k-space data (simplified)
        kspace_data = np.random.randn(n_coils, n_points) + 1j * np.random.randn(n_coils, n_points)
        
        # Reconstruct
        recon = optimization.iterative_sense_recon(
            kspace_data,
            trajectory,
            coil_maps,
            image_shape=image_size,
            n_iterations=5,
            regularization_type='none'
        )
        
        # Check output shape
        assert recon.shape == image_size
        
        # Check that output is complex
        assert np.iscomplexobj(recon)
    
    def test_iterative_sense_with_l2_reg(self):
        """Test SENSE with L2 regularization."""
        n_coils = 2
        image_size = (16, 16)
        n_points = 50
        
        coil_maps = np.ones((n_coils, *image_size), dtype=complex) / np.sqrt(n_coils)
        trajectory = np.random.rand(n_points, 2) - 0.5
        kspace_data = np.random.randn(n_coils, n_points) + 1j * np.random.randn(n_coils, n_points)
        
        # Reconstruct with L2 regularization
        recon = optimization.iterative_sense_recon(
            kspace_data,
            trajectory,
            coil_maps,
            image_shape=image_size,
            n_iterations=3,
            regularization_type='l2',
            lambda_reg=0.01
        )
        
        assert recon.shape == image_size


class TestADMM:
    """Test ADMM reconstruction."""
    
    def test_admm_reconstruction_basic(self):
        """Test basic ADMM reconstruction."""
        n_coils = 2
        image_size = (16, 16)
        n_points = 50
        
        # Create synthetic data
        coil_maps = np.ones((n_coils, *image_size), dtype=complex) / np.sqrt(n_coils)
        trajectory = np.random.rand(n_points, 2) - 0.5
        kspace_data = np.random.randn(n_coils, n_points) + 1j * np.random.randn(n_coils, n_points)
        
        # Reconstruct with ADMM
        recon = optimization.admm_reconstruction(
            kspace_data,
            trajectory,
            coil_maps,
            image_size,
            n_iterations=5,
            lambda_reg=0.001,
            regularization_type='tv'
        )
        
        # Check output
        assert recon.shape == image_size
        assert np.iscomplexobj(recon)


class TestFISTA:
    """Test FISTA reconstruction."""
    
    def test_fista_reconstruction(self):
        """Test FISTA algorithm."""
        image_size = (16, 16)
        
        # Simple forward and adjoint operators (identity for testing)
        def A(x):
            return x.ravel()
        
        def AT(y):
            return y.reshape(image_size)
        
        # Measurements
        b = np.random.randn(np.prod(image_size))
        
        # Reconstruct
        recon = optimization.fista_reconstruction(
            A, AT, b,
            image_size,
            n_iterations=10,
            lambda_reg=0.001,
            regularization_type='l1'
        )
        
        # Check output shape
        assert recon.shape == image_size


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
