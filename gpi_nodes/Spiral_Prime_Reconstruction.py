"""
Main Spiral-PRIME reconstruction GPI node.

This is the primary GPI node for performing 3D reconstruction of spiral k-space data.
"""

# GPI node imports (these would be provided by GPI framework)
try:
    import gpi
    from gpi import QtGui
    HAS_GPI = True
except ImportError:
    HAS_GPI = False
    # Fallback for when GPI is not available
    class gpi:
        @staticmethod
        def NodeAPI(x):
            return x

import numpy as np
import yaml
from pathlib import Path

# Spiral-PRIME imports
from ...core import mri_math, optimization, spiral_physics
from ...utils import logging_config, validators


class ExternalNode(gpi.NodeAPI):
    """
    Spiral-PRIME Reconstruction Node
    
    Performs iterative SENSE reconstruction for spiral k-space data with
    physics-based modeling and regularization.
    
    INPUT PORTS:
    - kspace: Multi-coil k-space data [n_coils, n_points]
    - trajectory: k-space trajectory [n_points, n_dims] (normalized to [-0.5, 0.5])
    - coil_maps: Coil sensitivity maps [n_coils, ...image_shape] (optional)
    - density_comp: Density compensation weights [n_points] (optional)
    
    OUTPUT PORTS:
    - image: Reconstructed image [...image_shape]
    - coil_maps_out: Estimated or input coil sensitivity maps
    
    WIDGETS:
    - Algorithm: Reconstruction algorithm selection
    - Iterations: Number of iterations
    - Regularization: Type of regularization
    - Lambda: Regularization weight
    - Image Size: Output image dimensions
    """
    
    def initUI(self):
        """Initialize the UI widgets."""
        # Input ports
        self.addInPort('kspace', 'NPYarray', dtype=[np.complex64, np.complex128],
                      obligation=gpi.REQUIRED)
        self.addInPort('trajectory', 'NPYarray', dtype=np.float32,
                      obligation=gpi.REQUIRED)
        self.addInPort('coil_maps', 'NPYarray', dtype=[np.complex64, np.complex128],
                      obligation=gpi.OPTIONAL)
        self.addInPort('density_comp', 'NPYarray', dtype=np.float32,
                      obligation=gpi.OPTIONAL)
        
        # Output ports
        self.addOutPort('image', 'NPYarray', dtype=np.complex64)
        self.addOutPort('coil_maps_out', 'NPYarray', dtype=np.complex64)
        
        # Widgets
        self.addWidget('PushButton', 'Reconstruct', toggle=False)
        
        self.addWidget('ComboBox', 'Algorithm', 
                      items=['Iterative SENSE', 'CG-SENSE', 'ADMM'],
                      val=0)
        
        self.addWidget('SpinBox', 'Iterations', val=10, min=1, max=100)
        
        self.addWidget('ComboBox', 'Regularization',
                      items=['None', 'L1', 'L2', 'TV', 'TV (Chambolle-Pock)'],
                      val=3)  # Default to TV
        
        self.addWidget('DoubleSpinBox', 'Lambda', val=0.001, min=0.0, max=1.0,
                      singlestep=0.0001, decimals=4)
        
        self.addWidget('ComboBox', 'Coil Combine',
                      items=['SOS', 'Adaptive', 'Walsh'],
                      val=1)  # Default to Adaptive
        
        self.addWidget('StringBox', 'Image Size', val='auto',
                      placeholder='auto or [nz,ny,nx]')
        
        self.addWidget('PushButton', 'Load Config', toggle=False)
        self.addWidget('PushButton', 'Save Config', toggle=False)
        
        # Status display
        self.addWidget('TextBox', 'Status', val='Ready')
    
    def validate(self):
        """Validate input data."""
        kspace = self.getData('kspace')
        trajectory = self.getData('trajectory')
        
        # Validate inputs
        try:
            validators.validate_kspace_data(kspace)
            validators.validate_trajectory(trajectory)
            
            # Check consistency
            validators.check_data_consistency(kspace, trajectory)
            
            self.setAttr('Status', val='Validation passed')
            return 0
            
        except ValueError as e:
            self.setAttr('Status', val=f'Validation error: {str(e)}')
            return 1
    
    def compute(self):
        """Perform the reconstruction."""
        # Set up logging
        logger = logging_config.get_logger('reconstruction_node')
        
        # Get input data
        kspace_data = self.getData('kspace')
        trajectory = self.getData('trajectory')
        coil_maps_in = self.getData('coil_maps')
        density_comp = self.getData('density_comp')
        
        # Get widget values
        algorithm = self.getVal('Algorithm')
        n_iterations = self.getVal('Iterations')
        reg_type = self.getVal('Regularization')
        lambda_reg = self.getVal('Lambda')
        coil_combine = self.getVal('Coil Combine')
        image_size_str = self.getVal('Image Size')
        
        # Map widget selections to internal names
        algorithm_map = {
            0: 'iterative_sense',
            1: 'cg_sense', 
            2: 'admm'
        }
        reg_map = {
            0: 'none',
            1: 'l1',
            2: 'l2',
            3: 'tv',
            4: 'tv_cp'
        }
        coil_map = {
            0: 'sos',
            1: 'adaptive',
            2: 'walsh'
        }
        
        algorithm_name = algorithm_map[algorithm]
        reg_name = reg_map[reg_type]
        coil_method = coil_map[coil_combine]
        
        # Parse image size safely
        if image_size_str.lower() == 'auto':
            image_shape = None
        else:
            try:
                import ast
                image_shape = ast.literal_eval(image_size_str)
                if not isinstance(image_shape, (list, tuple)):
                    image_shape = None
            except (ValueError, SyntaxError):
                logger.warning(f"Invalid image size format: {image_size_str}, using auto")
                image_shape = None
        
        logger.info(f"Starting {algorithm_name} reconstruction")
        logger.info(f"Iterations: {n_iterations}")
        logger.info(f"Regularization: {reg_name} (lambda={lambda_reg})")
        
        try:
            # Update status
            self.setAttr('Status', val='Estimating coil maps...')
            
            # Estimate coil sensitivity maps if not provided
            if coil_maps_in is None:
                logger.info("Estimating coil sensitivity maps")
                
                # Grid the k-space data for coil map estimation
                if image_shape is None:
                    # Auto-determine size from trajectory
                    image_shape = self._estimate_image_size(trajectory)
                
                kspace_cart = np.zeros((kspace_data.shape[0], *image_shape), 
                                      dtype=complex)
                for c in range(kspace_data.shape[0]):
                    kspace_cart[c] = mri_math.grid_noncartesian(
                        trajectory, kspace_data[c], image_shape
                    )
                
                coil_maps = mri_math.estimate_coil_sensitivity(
                    kspace_cart, method=coil_method
                )
            else:
                coil_maps = coil_maps_in
                if image_shape is None:
                    image_shape = coil_maps.shape[1:]
            
            # Update status
            self.setAttr('Status', val=f'Reconstructing ({algorithm_name})...')
            
            # Perform reconstruction
            if algorithm_name in ['iterative_sense', 'cg_sense']:
                image_recon = optimization.iterative_sense_recon(
                    kspace_data,
                    trajectory,
                    coil_maps,
                    image_shape=image_shape,
                    n_iterations=n_iterations,
                    regularization_type=reg_name,
                    lambda_reg=lambda_reg,
                    density_comp=density_comp
                )
            elif algorithm_name == 'admm':
                image_recon = optimization.admm_reconstruction(
                    kspace_data,
                    trajectory,
                    coil_maps,
                    image_shape=image_shape,
                    n_iterations=n_iterations,
                    lambda_reg=lambda_reg,
                    regularization_type=reg_name if reg_name != 'none' else 'tv'
                )
            else:
                raise ValueError(f"Unknown algorithm: {algorithm_name}")
            
            # Set outputs
            self.setData('image', image_recon)
            self.setData('coil_maps_out', coil_maps)
            
            logger.info("Reconstruction complete")
            self.setAttr('Status', val='Reconstruction complete!')
            
            return 0
            
        except Exception as e:
            logger.error(f"Reconstruction failed: {str(e)}")
            self.setAttr('Status', val=f'Error: {str(e)}')
            return 1
    
    def _estimate_image_size(self, trajectory):
        """Estimate appropriate image size from trajectory."""
        # Use trajectory extent to estimate size
        n_dims = trajectory.shape[1]
        
        # Estimate from k-space coverage
        # Assume normalized trajectory [-0.5, 0.5]
        # Use a typical oversampling factor of 2
        base_size = 128
        
        if n_dims == 2:
            return (base_size, base_size)
        else:
            return (base_size // 2, base_size, base_size)


# Allow running as standalone script
if __name__ == '__main__':
    if HAS_GPI:
        # Run as GPI node
        pass
    else:
        print("GPI framework not available. This node requires GPI to run.")
