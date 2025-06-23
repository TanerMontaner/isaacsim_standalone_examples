# Copyright (c) 2024, Peristaltic Conveyor Demo
# Demonstrating wave-based object transport using Isaac Sim

import numpy as np
import torch
import math
import carb
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.materials.deformable_material import DeformableMaterial
from isaacsim.core.prims import DeformablePrim, SingleDeformablePrim, GeometryPrim
from isaacsim.core.api.objects.ground_plane import GroundPlane
from pxr import Sdf, UsdLux, Gf, UsdGeom
from omni.physx.scripts import deformableUtils, physicsUtils
import isaacsim.core.utils.deformable_mesh_utils as deformableMeshUtils


class PeristalticConveyorDemo:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0, backend="torch", device="cuda")
        self.stage = omni.usd.get_context().get_stage()
        
        # Configuration
        self.actuator_grid_size = 20  # 20x20 grid = 400 actuators
        self.membrane_size = 1.0  # 1m x 1m membrane
        self.actuator_height = 0.05  # 5cm tall actuators
        self.actuator_radius = 0.02  # 2cm radius actuators
        self.membrane_height = 0.1  # Position membrane 10cm above ground
        
        # Wave parameters
        self.wave_amplitude = 0.03  # 3cm wave amplitude
        self.wave_frequency = 0.5   # Hz
        self.wave_speed = 0.2       # m/s
        
        # Actuator arrays
        self.actuators = []
        self.actuator_positions = []
        self.actuator_initial_heights = []
        
        self.setup_scene()
        self.create_membrane()
        self.create_actuator_grid()
        self.create_test_object()
        
    def setup_scene(self):
        """Phase 1: Basic Scene Setup"""
        print("Setting up basic scene...")
        
        # Add ground plane
        self.world.scene.add_default_ground_plane()
        
        # Add lighting
        stage = self.stage
        distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
        distant_light.CreateIntensityAttr(500)
        distant_light.CreateAngleAttr(45)
        
        # Add dome light for better illumination
        dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/DomeLight"))
        dome_light.CreateIntensityAttr(200)
        
        print("Scene setup complete")
        
    def create_membrane(self):
        """Phase 1: Create deformable membrane surface"""
        print("Creating deformable membrane...")
        
        # Create membrane mesh
        membrane_path = "/World/ConveyorMembrane"
        membrane_mesh = UsdGeom.Mesh.Define(self.stage, membrane_path)
        
        # Create a square mesh for the membrane (higher resolution for better deformation)
        resolution = 20  # 20x20 mesh resolution
        tri_points, tri_indices = deformableUtils.create_triangle_mesh_square(
            dimx=resolution, dimy=resolution, scale=self.membrane_size
        )
        
        membrane_mesh.GetPointsAttr().Set(tri_points)
        membrane_mesh.GetFaceVertexIndicesAttr().Set(tri_indices)
        membrane_mesh.GetFaceVertexCountsAttr().Set([3] * (len(tri_indices) // 3))
        
        # Position the membrane
        physicsUtils.setup_transform_as_scale_orient_translate(membrane_mesh)
        physicsUtils.set_or_add_translate_op(membrane_mesh, Gf.Vec3d(0.0, 0.0, self.membrane_height))
        
        # Create deformable material
        deformable_material_path = "/World/DeformableMaterial"
        self.deformable_material = DeformableMaterial(
            prim_path=deformable_material_path,
            dynamic_friction=0.7,
            static_friction=0.8,
            youngs_modulus=1e5,      # Moderate stiffness
            poissons_ratio=0.45,     # Rubber-like material
            damping_scale=0.05,      # Low damping for responsive movement
            elasticity_damping=0.02  # Minimal elasticity damping
        )
        
        # Create the deformable prim
        self.membrane = SingleDeformablePrim(
            name="conveyor_membrane",
            prim_path=membrane_path,
            deformable_material=self.deformable_material,
            vertex_velocity_damping=0.1,
            sleep_damping=0.5,
            sleep_threshold=0.01,
            settling_threshold=0.05,
            self_collision=False,  # Disable for performance
            solver_position_iteration_count=15,
            kinematic_enabled=False,
            simulation_hexahedral_resolution=3,
            collision_simplification=True
        )
        
        self.world.scene.add(self.membrane)
        print("Membrane created successfully")
        
    def create_actuator_grid(self):
        """Phase 2: Create 20x20 actuator grid"""
        print("Creating actuator grid...")
        
        # Calculate actuator positions
        spacing = self.membrane_size / (self.actuator_grid_size - 1)
        start_pos = -self.membrane_size / 2
        
        actuator_id = 0
        for i in range(self.actuator_grid_size):
            for j in range(self.actuator_grid_size):
                x = start_pos + i * spacing
                y = start_pos + j * spacing
                z = self.membrane_height - self.actuator_height / 2  # Position below membrane
                
                # Create actuator
                actuator_path = f"/World/Actuator_{actuator_id:03d}"
                actuator = DynamicCuboid(
                    prim_path=actuator_path,
                    name=f"actuator_{actuator_id}",
                    position=np.array([x, y, z]),
                    scale=np.array([self.actuator_radius * 2, self.actuator_radius * 2, self.actuator_height]),
                    size=1.0,
                    color=np.array([100, 150, 255]),  # Blue actuators
                    mass=0.1  # Light actuators for responsive movement
                )
                
                self.world.scene.add(actuator)
                self.actuators.append(actuator)
                self.actuator_positions.append([x, y, z])
                self.actuator_initial_heights.append(z)
                
                actuator_id += 1
        
        print(f"Created {len(self.actuators)} actuators in {self.actuator_grid_size}x{self.actuator_grid_size} grid")
        
    def create_test_object(self):
        """Phase 3: Add test object (20g payload)"""
        print("Creating test object...")
        
        # Create a small test object
        test_object_path = "/World/TestObject"
        self.test_object = DynamicCuboid(
            prim_path=test_object_path,
            name="test_payload",
            position=np.array([0.0, 0.0, self.membrane_height + 0.05]),  # Slightly above membrane
            scale=np.array([0.02, 0.02, 0.02]),  # 2cm cube
            size=1.0,
            color=np.array([255, 100, 100]),  # Red test object
            mass=0.02  # 20g mass
        )
        
        self.world.scene.add(self.test_object)
        print("Test object created")
        
    def generate_wave_pattern(self, time, direction=[1, 0]):
        """Phase 4: Generate wave pattern for actuator movement"""
        actuator_heights = []
        
        # Normalize direction vector
        dir_mag = math.sqrt(direction[0]**2 + direction[1]**2)
        if dir_mag > 0:
            direction = [direction[0]/dir_mag, direction[1]/dir_mag]
        else:
            direction = [1, 0]
        
        for i, (x, y, base_z) in enumerate(self.actuator_positions):
            # Calculate wave phase based on position and direction
            distance = x * direction[0] + y * direction[1]
            phase = (distance / self.wave_speed * self.wave_frequency * 2 * math.pi) - (time * self.wave_frequency * 2 * math.pi)
            
            # Calculate height offset
            height_offset = self.wave_amplitude * math.sin(phase)
            new_height = base_z + height_offset
            
            actuator_heights.append(new_height)
            
        return actuator_heights
        
    def update_actuators(self, target_heights):
        """Update actuator positions"""
        for i, (actuator, target_height) in enumerate(zip(self.actuators, target_heights)):
            current_pos = actuator.get_world_pose()[0]
            new_position = np.array([current_pos[0], current_pos[1], target_height])
            actuator.set_world_pose(position=new_position)
            
    def run_simulation(self):
        """Main simulation loop"""
        print("Starting simulation...")
        self.world.reset()
        
        frame_count = 0
        wave_direction = [1, 0]  # Initial wave direction (positive X)
        direction_change_interval = 500  # Change direction every 500 frames
        
        while simulation_app.is_running():
            if self.world.is_playing():
                current_time = self.world.current_time
                
                # Change wave direction periodically
                if frame_count % direction_change_interval == 0:
                    if frame_count % (direction_change_interval * 4) == 0:
                        wave_direction = [1, 0]   # +X direction
                    elif frame_count % (direction_change_interval * 3) == 0:
                        wave_direction = [0, -1]  # -Y direction
                    elif frame_count % (direction_change_interval * 2) == 0:
                        wave_direction = [-1, 0]  # -X direction
                    else:
                        wave_direction = [0, 1]   # +Y direction
                    
                    print(f"Wave direction changed to: {wave_direction}")
                
                # Generate wave pattern
                target_heights = self.generate_wave_pattern(current_time, wave_direction)
                
                # Update actuators
                self.update_actuators(target_heights)
                
                # Print status every 100 frames
                if frame_count % 100 == 0:
                    test_pos = self.test_object.get_world_pose()[0]
                    print(f"Frame {frame_count}: Test object at ({test_pos[0]:.3f}, {test_pos[1]:.3f}, {test_pos[2]:.3f})")
                
                frame_count += 1
            
            self.world.step(render=True)
        
        print("Simulation ended")
        simulation_app.close()


if __name__ == "__main__":
    demo = PeristalticConveyorDemo()
    demo.run_simulation()