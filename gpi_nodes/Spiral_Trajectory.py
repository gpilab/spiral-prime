"""
Spiral Trajectory design and loading GPI node.
"""

try:
    import gpi
    HAS_GPI = True
except ImportError:
    HAS_GPI = False
    class gpi:
        @staticmethod
        def NodeAPI(x):
            return x

import numpy as np
from ...core import spiral_physics
from ...utils import logging_config


class ExternalNode(gpi.NodeAPI):
    """
    Spiral Trajectory Node
    
    Design or load spiral k-space trajectories for reconstruction.
    
    INPUT PORTS:
    - trajectory_file: Path to trajectory file (optional, for loading)
    
    OUTPUT PORTS:
    - trajectory: k-space trajectory [n_points, n_dims]
    - gradient: Gradient waveforms [n_points, n_dims] (mT/m)
    - density_comp: Density compensation weights [n_points]
    
    WIDGETS:
    - Mode: Design or Load trajectory
    - Type: Trajectory type (2D spiral, 3D stack-of-spirals)
    - FOV: Field of view (cm)
    - Resolution: Spatial resolution (cm)
    - Interleaves: Number of spiral interleaves
    - Max Gradient: Maximum gradient amplitude (mT/m)
    - Max Slew: Maximum slew rate (T/m/s)
    """
    
    def initUI(self):
        """Initialize the UI."""
        # Input ports
        self.addInPort('trajectory_file', 'STRING', obligation=gpi.OPTIONAL)
        
        # Output ports
        self.addOutPort('trajectory', 'NPYarray', dtype=np.float32)
        self.addOutPort('gradient', 'NPYarray', dtype=np.float32)
        self.addOutPort('density_comp', 'NPYarray', dtype=np.float32)
        
        # Widgets
        self.addWidget('PushButton', 'Generate', toggle=False)
        
        self.addWidget('ComboBox', 'Mode',
                      items=['Design', 'Load from file'],
                      val=0)
        
        self.addWidget('ComboBox', 'Type',
                      items=['2D Spiral', '3D Stack-of-Spirals'],
                      val=0)
        
        self.addWidget('DoubleSpinBox', 'FOV (cm)', val=24.0, min=1.0, max=50.0,
                      singlestep=1.0, decimals=1)
        
        self.addWidget('DoubleSpinBox', 'FOV Z (cm)', val=16.0, min=1.0, max=50.0,
                      singlestep=1.0, decimals=1)
        
        self.addWidget('DoubleSpinBox', 'Resolution (mm)', val=1.0, min=0.1, max=10.0,
                      singlestep=0.1, decimals=2)
        
        self.addWidget('SpinBox', 'Interleaves', val=8, min=1, max=64)
        
        self.addWidget('DoubleSpinBox', 'Max Gradient (mT/m)', val=40.0, 
                      min=1.0, max=80.0, singlestep=1.0, decimals=1)
        
        self.addWidget('DoubleSpinBox', 'Max Slew (T/m/s)', val=150.0,
                      min=10.0, max=200.0, singlestep=10.0, decimals=1)
        
        self.addWidget('ComboBox', 'Density Comp',
                      items=['None', 'Voronoi', 'Jackson', 'Pipe'],
                      val=1)
        
        self.addWidget('TextBox', 'Status', val='Ready')
    
    def compute(self):
        """Design or load trajectory."""
        logger = logging_config.get_logger('trajectory_node')
        
        # Get widget values
        mode = self.getVal('Mode')
        traj_type = self.getVal('Type')
        fov = self.getVal('FOV (cm)')
        fov_z = self.getVal('FOV Z (cm)')
        resolution = self.getVal('Resolution (mm)') / 10.0  # Convert to cm
        n_interleaves = self.getVal('Interleaves')
        gmax = self.getVal('Max Gradient (mT/m)')
        smax = self.getVal('Max Slew (T/m/s)')
        density_comp_method = self.getVal('Density Comp')
        
        try:
            if mode == 0:  # Design
                self.setAttr('Status', val='Designing trajectory...')
                logger.info(f"Designing {'2D' if traj_type == 0 else '3D'} spiral trajectory")
                
                if traj_type == 0:  # 2D Spiral
                    trajectory, gradient, time = spiral_physics.design_spiral_trajectory(
                        fov=fov,
                        resolution=resolution,
                        n_interleaves=n_interleaves,
                        gmax=gmax,
                        smax=smax
                    )
                    
                else:  # 3D Stack-of-Spirals
                    trajectory, gradient = spiral_physics.create_3d_stack_of_spirals(
                        fov_xy=fov,
                        fov_z=fov_z,
                        resolution_xy=resolution,
                        resolution_z=resolution,
                        n_interleaves=n_interleaves,
                        gmax=gmax,
                        smax=smax
                    )
                
                logger.info(f"Trajectory designed: {trajectory.shape[0]} points, "
                          f"{trajectory.shape[1]}D")
                
            else:  # Load from file
                self.setAttr('Status', val='Loading trajectory...')
                trajectory_file = self.getData('trajectory_file')
                
                if trajectory_file is None or trajectory_file == '':
                    raise ValueError("No trajectory file specified")
                
                logger.info(f"Loading trajectory from {trajectory_file}")
                
                # Load from file (support .npy, .mat, etc.)
                if trajectory_file.endswith('.npy'):
                    data = np.load(trajectory_file)
                    if isinstance(data, dict):
                        trajectory = data['trajectory']
                        gradient = data.get('gradient', None)
                    else:
                        trajectory = data
                        gradient = None
                else:
                    raise ValueError(f"Unsupported file format: {trajectory_file}")
            
            # Compute density compensation if requested
            if density_comp_method > 0:
                method_map = {1: 'voronoi', 2: 'jackson', 3: 'pipe'}
                method = method_map[density_comp_method]
                
                logger.info(f"Computing density compensation using {method}")
                self.setAttr('Status', val='Computing density compensation...')
                
                density_comp = spiral_physics.compute_density_compensation(
                    trajectory, method=method
                )
            else:
                density_comp = np.ones(trajectory.shape[0], dtype=np.float32)
            
            # Set outputs
            self.setData('trajectory', trajectory.astype(np.float32))
            if gradient is not None:
                self.setData('gradient', gradient.astype(np.float32))
            self.setData('density_comp', density_comp)
            
            logger.info("Trajectory generation complete")
            self.setAttr('Status', 
                        val=f'Complete! {trajectory.shape[0]} points, '
                            f'{trajectory.shape[1]}D')
            
            return 0
            
        except Exception as e:
            logger.error(f"Trajectory generation failed: {str(e)}")
            self.setAttr('Status', val=f'Error: {str(e)}')
            return 1


if __name__ == '__main__':
    if not HAS_GPI:
        print("GPI framework not available. This node requires GPI to run.")
