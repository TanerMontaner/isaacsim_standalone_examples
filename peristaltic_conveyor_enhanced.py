# Copyright (c) 2024, Enhanced Peristaltic Conveyor Demo
# Advanced implementation with improved physics and RL preparation

import numpy as np
import torch
import math
import carb
from typing import List, Tuple, Dict, Optional
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.materials.deformable_material import DeformableMaterial
from isaacsim.core.api.materials.particle_material import ParticleMaterial
from isaacsim.core.prims import DeformablePrim, SingleDeformablePrim, ClothPrim, SingleClothPrim, SingleParticleSystem
from isaacsim.core.api.objects.ground_plane import GroundPlane
from pxr import Sdf, UsdLux, Gf, UsdGeom
from omni.physx.scripts import deformableUtils, physicsUtils
import isaacsim.core.utils.deformable_mesh_utils as deformableMeshUtils


class ActuatorController:
    """Phase 2 & 4: Advanced actuator control system with wave generation"""
    
    def __init__(self, actuators: List, positions: List[Tuple[float, float, float]], 
                 wave_params: Dict[str, float]):
        self.actuators = actuators
        self.positions = positions
        self.initial_heights = [pos[2] for pos in positions]
        
        # Wave parameters
        self.wave_amplitude = wave_params.get('amplitude', 0.03)
        self.wave_frequency = wave_params.get('frequency', 0.5)
        self.wave_speed = wave_params.get('speed', 0.2)
        self.wave_length = wave_params.get('length', 0.4)
        
        # Control state
        self.current_direction = [1, 0]
        self.current_time = 0.0
        
    def set_wave_direction(self, direction: List[float]):
        """Set wave propagation direction"""
        dir_mag = math.sqrt(direction[0]**2 + direction[1]**2)
        if dir_mag > 0:
            self.current_direction = [direction[0]/dir_mag, direction[1]/dir_mag]
        else:
            self.current_direction = [1, 0]
            
    def generate_wave_heights(self, time: float) -> List[float]:
        """Generate wave pattern heights for all actuators"""
        heights = []
        
        for i, (x, y, base_z) in enumerate(self.positions):
            # Calculate wave phase based on position and direction
            distance = x * self.current_direction[0] + y * self.current_direction[1]
            phase = (2 * math.pi * distance / self.wave_length) - (2 * math.pi * self.wave_frequency * time)
            
            # Multiple wave components for more complex patterns
            primary_wave = self.wave_amplitude * math.sin(phase)
            secondary_wave = 0.3 * self.wave_amplitude * math.sin(2 * phase + math.pi/4)
            
            height_offset = primary_wave + secondary_wave
            new_height = base_z + height_offset
            
            heights.append(new_height)
            
        return heights
    
    def update_actuators(self, target_heights: List[float], force_factor: float = 1.0):
        """Update actuator positions with smooth interpolation"""
        for i, (actuator, target_height) in enumerate(zip(self.actuators, target_heights)):
            current_pos = actuator.get_world_pose()[0]
            
            # Smooth interpolation for more realistic movement
            height_diff = target_height - current_pos[2]
            smooth_height = current_pos[2] + height_diff * 0.1 * force_factor
            
            new_position = np.array([current_pos[0], current_pos[1], smooth_height])
            actuator.set_world_pose(position=new_position)
            
            # Apply upward force to help with membrane interaction
            if height_diff > 0:
                force_vector = np.array([0, 0, height_diff * 100 * force_factor])
                actuator.apply_force(force_vector)


class PeristalticConveyorEnhanced:
    """Enhanced peristaltic conveyor with improved physics and RL preparation"""
    
    def __init__(self, config: Optional[Dict] = None):
        # Default configuration
        self.config = {
            'actuator_grid_size': 20,
            'membrane_size': 1.0,
            'actuator_height': 0.05,
            'actuator_radius': 0.02,
            'membrane_height': 0.15,
            'wave_amplitude': 0.03,
            'wave_frequency': 0.5,
            'wave_speed': 0.2,
            'wave_length': 0.4,
            'use_cloth': True,  # Use cloth instead of deformable for better interaction
            'physics_dt': 1.0/60.0,
            'render_dt': 1.0/60.0
        }
        
        if config:
            self.config.update(config)
            
        # Initialize Isaac Sim World
        self.world = World(
            stage_units_in_meters=1.0, 
            backend="torch", 
            device="cuda",
            physics_dt=self.config['physics_dt'],
            rendering_dt=self.config['render_dt']
        )
        self.stage = omni.usd.get_context().get_stage()
        
        # Components
        self.actuators = []
        self.actuator_positions = []
        self.actuator_controller = None
        self.membrane = None
        self.test_objects = []
        
        # RL Environment preparation
        self.observation_space = None
        self.action_space = None
        self.current_reward = 0.0
        
        # Performance tracking
        self.frame_count = 0
        self.performance_metrics = {
            'object_displacement': 0.0,
            'wave_efficiency': 0.0,
            'energy_consumption': 0.0
        }
        
        self.setup_scene()
        self.create_conveyor_surface()
        self.create_actuator_grid()
        self.create_test_objects()
        self.setup_rl_environment()
        
    def setup_scene(self):
        """Enhanced scene setup with better lighting and physics"""
        print("Setting up enhanced scene...")
        
        # Add ground plane
        self.world.scene.add_default_ground_plane()
        
        # Enhanced lighting setup
        stage = self.stage
        
        # Main directional light
        main_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/MainLight"))
        main_light.CreateIntensityAttr(800)
        main_light.CreateAngleAttr(30)
        main_light.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.9))
        
        # Fill light
        fill_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/FillLight"))
        fill_light.CreateIntensityAttr(300)
        fill_light.CreateAngleAttr(45)
        fill_light.CreateColorAttr(Gf.Vec3f(0.8, 0.9, 1.0))
        
        # Dome light for ambient illumination
        dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/DomeLight"))
        dome_light.CreateIntensityAttr(150)
        
        # Configure physics parameters
        physics_context = self.world.get_physics_context()
        physics_context.set_gravity(-9.81)
        
        print("Enhanced scene setup complete")
        
    def create_conveyor_surface(self):
        """Create cloth-based conveyor surface for better interaction"""
        print("Creating conveyor surface...")
        
        if self.config['use_cloth']:
            self._create_cloth_surface()
        else:
            self._create_deformable_surface()
            
    def _create_cloth_surface(self):
        """Create cloth-based conveyor surface"""
        membrane_path = "/World/ConveyorCloth"
        cloth_mesh = UsdGeom.Mesh.Define(self.stage, membrane_path)
        
        # Create high-resolution mesh for better interaction
        resolution = 25
        tri_points, tri_indices = deformableUtils.create_triangle_mesh_square(
            dimx=resolution, dimy=resolution, scale=self.config['membrane_size']
        )
        
        cloth_mesh.GetPointsAttr().Set(tri_points)
        cloth_mesh.GetFaceVertexIndicesAttr().Set(tri_indices)
        cloth_mesh.GetFaceVertexCountsAttr().Set([3] * (len(tri_indices) // 3))
        
        # Position the cloth
        physicsUtils.setup_transform_as_scale_orient_translate(cloth_mesh)
        physicsUtils.set_or_add_translate_op(cloth_mesh, Gf.Vec3d(0.0, 0.0, self.config['membrane_height']))
        
        # Create particle system for cloth
        particle_system_path = "/World/ParticleSystem"
        particle_material_path = "/World/ParticleMaterial"
        
        self.particle_material = ParticleMaterial(
            prim_path=particle_material_path,
            drag=0.05,
            lift=0.1,
            friction=0.8,
            particle_friction_scale=0.7,
            damping=0.1
        )
        
        radius = 0.5 * (self.config['membrane_size'] / resolution)
        rest_offset = radius * 0.8
        contact_offset = rest_offset * 1.2
        
        self.particle_system = SingleParticleSystem(
            prim_path=particle_system_path,
            simulation_owner=self.world.get_physics_context().prim_path,
            rest_offset=rest_offset,
            contact_offset=contact_offset,
            solid_rest_offset=rest_offset,
            fluid_rest_offset=rest_offset,
            particle_contact_offset=contact_offset,
        )
        
        # Create cloth prim
        self.membrane = SingleClothPrim(
            name="conveyor_cloth",
            prim_path=membrane_path,
            particle_system=self.particle_system,
            particle_material=self.particle_material,
        )
        
        self.world.scene.add(self.membrane)
        print("Cloth surface created successfully")
        
    def _create_deformable_surface(self):
        """Create deformable surface (fallback method)"""
        membrane_path = "/World/ConveyorMembrane"
        membrane_mesh = UsdGeom.Mesh.Define(self.stage, membrane_path)
        
        resolution = 20
        tri_points, tri_indices = deformableUtils.create_triangle_mesh_square(
            dimx=resolution, dimy=resolution, scale=self.config['membrane_size']
        )
        
        membrane_mesh.GetPointsAttr().Set(tri_points)
        membrane_mesh.GetFaceVertexIndicesAttr().Set(tri_indices)
        membrane_mesh.GetFaceVertexCountsAttr().Set([3] * (len(tri_indices) // 3))
        
        physicsUtils.setup_transform_as_scale_orient_translate(membrane_mesh)
        physicsUtils.set_or_add_translate_op(membrane_mesh, Gf.Vec3d(0.0, 0.0, self.config['membrane_height']))
        
        deformable_material_path = "/World/DeformableMaterial"
        self.deformable_material = DeformableMaterial(
            prim_path=deformable_material_path,
            dynamic_friction=0.8,
            static_friction=0.9,
            youngs_modulus=5e4,
            poissons_ratio=0.45,
            damping_scale=0.02,
            elasticity_damping=0.01
        )
        
        self.membrane = SingleDeformablePrim(
            name="conveyor_membrane",
            prim_path=membrane_path,
            deformable_material=self.deformable_material,
            vertex_velocity_damping=0.05,
            sleep_damping=0.3,
            sleep_threshold=0.005,
            settling_threshold=0.02,
            self_collision=False,
            solver_position_iteration_count=20,
            kinematic_enabled=False,
            simulation_hexahedral_resolution=4,
            collision_simplification=True
        )
        
        self.world.scene.add(self.membrane)
        print("Deformable surface created successfully")
        
    def create_actuator_grid(self):
        """Create optimized actuator grid with physics properties"""
        print("Creating optimized actuator grid...")
        
        grid_size = self.config['actuator_grid_size']
        spacing = self.config['membrane_size'] / (grid_size - 1)
        start_pos = -self.config['membrane_size'] / 2
        
        actuator_id = 0
        for i in range(grid_size):
            for j in range(grid_size):
                x = start_pos + i * spacing
                y = start_pos + j * spacing
                z = self.config['membrane_height'] - self.config['actuator_height'] * 0.8
                
                # Create actuator with optimized properties
                actuator_path = f"/World/Actuator_{actuator_id:03d}"
                actuator = DynamicCuboid(
                    prim_path=actuator_path,
                    name=f"actuator_{actuator_id}",
                    position=np.array([x, y, z]),
                    scale=np.array([
                        self.config['actuator_radius'] * 2, 
                        self.config['actuator_radius'] * 2, 
                        self.config['actuator_height']
                    ]),
                    size=1.0,
                    color=np.array([70, 130, 220]),  # Professional blue
                    mass=0.05,  # Lighter for better responsiveness
                    linear_damping=0.1,
                    angular_damping=0.5
                )
                
                self.world.scene.add(actuator)
                self.actuators.append(actuator)
                self.actuator_positions.append([x, y, z])
                
                actuator_id += 1
        
        # Initialize actuator controller
        wave_params = {
            'amplitude': self.config['wave_amplitude'],
            'frequency': self.config['wave_frequency'],
            'speed': self.config['wave_speed'],
            'length': self.config['wave_length']
        }
        
        self.actuator_controller = ActuatorController(
            self.actuators, self.actuator_positions, wave_params
        )
        
        print(f"Created {len(self.actuators)} actuators with controller")
        
    def create_test_objects(self):
        """Create multiple test objects for validation"""
        print("Creating test objects...")
        
        # Primary test object (20g)
        primary_test_path = "/World/TestObject_Primary"
        self.primary_test_object = DynamicCuboid(
            prim_path=primary_test_path,
            name="primary_payload",
            position=np.array([0.0, 0.0, self.config['membrane_height'] + 0.05]),
            scale=np.array([0.025, 0.025, 0.025]),
            size=1.0,
            color=np.array([255, 80, 80]),  # Red
            mass=0.02,  # 20g
            linear_damping=0.05,
            angular_damping=0.1
        )
        
        # Secondary test objects with different properties
        test_objects_config = [
            {"pos": [0.2, 0.2, 0.05], "mass": 0.01, "scale": 0.02, "color": [80, 255, 80]},  # Green, 10g
            {"pos": [-0.2, -0.2, 0.05], "mass": 0.05, "scale": 0.03, "color": [80, 80, 255]},  # Blue, 50g
        ]
        
        for i, config in enumerate(test_objects_config):
            test_path = f"/World/TestObject_{i}"
            test_obj = DynamicCuboid(
                prim_path=test_path,
                name=f"test_payload_{i}",
                position=np.array([
                    config["pos"][0], 
                    config["pos"][1], 
                    self.config['membrane_height'] + config["pos"][2]
                ]),
                scale=np.array([config["scale"]] * 3),
                size=1.0,
                color=np.array(config["color"]),
                mass=config["mass"],
                linear_damping=0.05,
                angular_damping=0.1
            )
            self.test_objects.append(test_obj)
            self.world.scene.add(test_obj)
        
        self.world.scene.add(self.primary_test_object)
        print(f"Created {len(self.test_objects) + 1} test objects")
        
    def setup_rl_environment(self):
        """Phase 5: Prepare for RL environment integration"""
        print("Setting up RL environment structure...")
        
        # Define observation space (example structure)
        self.observation_space = {
            'actuator_positions': len(self.actuators) * 3,  # x, y, z for each actuator
            'membrane_state': 100,  # Simplified membrane state
            'object_positions': len(self.test_objects) * 3,  # Position of each test object
            'object_velocities': len(self.test_objects) * 3,  # Velocity of each test object
            'wave_parameters': 4,  # amplitude, frequency, direction_x, direction_y
        }
        
        # Define action space
        self.action_space = {
            'wave_direction': 2,  # x, y components
            'wave_frequency': 1,  # frequency control
            'wave_amplitude': 1,  # amplitude control
            'selective_actuators': len(self.actuators),  # Individual actuator control
        }
        
        print("RL environment structure prepared")
        
    def get_observation(self):
        """Get current observation for RL"""
        obs = {}
        
        # Actuator positions
        actuator_positions = []
        for actuator in self.actuators:
            pos = actuator.get_world_pose()[0]
            actuator_positions.extend([pos[0], pos[1], pos[2]])
        obs['actuator_positions'] = np.array(actuator_positions)
        
        # Object positions and velocities
        object_positions = []
        object_velocities = []
        
        primary_pos = self.primary_test_object.get_world_pose()[0]
        primary_vel = self.primary_test_object.get_linear_velocity()
        object_positions.extend([primary_pos[0], primary_pos[1], primary_pos[2]])
        object_velocities.extend([primary_vel[0], primary_vel[1], primary_vel[2]])
        
        for test_obj in self.test_objects:
            pos = test_obj.get_world_pose()[0]
            vel = test_obj.get_linear_velocity()
            object_positions.extend([pos[0], pos[1], pos[2]])
            object_velocities.extend([vel[0], vel[1], vel[2]])
            
        obs['object_positions'] = np.array(object_positions)
        obs['object_velocities'] = np.array(object_velocities)
        
        # Wave parameters
        obs['wave_parameters'] = np.array([
            self.actuator_controller.wave_amplitude,
            self.actuator_controller.wave_frequency,
            self.actuator_controller.current_direction[0],
            self.actuator_controller.current_direction[1]
        ])
        
        return obs
        
    def calculate_reward(self):
        """Calculate reward for RL training"""
        # Simple reward based on object displacement towards target
        primary_pos = self.primary_test_object.get_world_pose()[0]
        
        # Reward for movement in desired direction
        target_direction = self.actuator_controller.current_direction
        displacement_reward = primary_pos[0] * target_direction[0] + primary_pos[1] * target_direction[1]
        
        # Penalty for object falling off
        height_penalty = max(0, self.config['membrane_height'] - primary_pos[2]) * 10
        
        # Energy efficiency (penalize excessive actuator movement)
        energy_penalty = sum(abs(h - self.actuator_controller.initial_heights[i]) 
                           for i, h in enumerate([pos[2] for pos in self.actuator_positions])) * 0.1
        
        reward = displacement_reward - height_penalty - energy_penalty
        return reward
        
    def run_simulation(self):
        """Enhanced simulation loop with performance monitoring"""
        print("Starting enhanced simulation...")
        self.world.reset()
        
        direction_patterns = [
            [1, 0],   # +X
            [0, 1],   # +Y
            [-1, 0],  # -X
            [0, -1],  # -Y
            [0.707, 0.707],   # Diagonal
            [-0.707, -0.707]  # Opposite diagonal
        ]
        
        pattern_index = 0
        pattern_duration = 400  # frames per pattern
        
        while simulation_app.is_running():
            if self.world.is_playing():
                current_time = self.world.current_time
                
                # Change wave direction periodically
                if self.frame_count % pattern_duration == 0:
                    new_direction = direction_patterns[pattern_index % len(direction_patterns)]
                    self.actuator_controller.set_wave_direction(new_direction)
                    print(f"Wave direction changed to: {new_direction}")
                    pattern_index += 1
                
                # Generate and apply wave pattern
                target_heights = self.actuator_controller.generate_wave_heights(current_time)
                self.actuator_controller.update_actuators(target_heights, force_factor=1.5)
                
                # Update performance metrics
                if self.frame_count % 60 == 0:  # Every second
                    self._update_performance_metrics()
                
                # Print status
                if self.frame_count % 120 == 0:  # Every 2 seconds
                    self._print_status()
                
                self.frame_count += 1
            
            self.world.step(render=True)
            
        print("Enhanced simulation ended")
        simulation_app.close()
        
    def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        primary_pos = self.primary_test_object.get_world_pose()[0]
        
        # Object displacement from origin
        displacement = math.sqrt(primary_pos[0]**2 + primary_pos[1]**2)
        self.performance_metrics['object_displacement'] = displacement
        
        # Wave efficiency (simplified)
        target_dir = self.actuator_controller.current_direction
        actual_dir = [primary_pos[0], primary_pos[1]]
        actual_mag = math.sqrt(actual_dir[0]**2 + actual_dir[1]**2)
        
        if actual_mag > 0:
            efficiency = (actual_dir[0] * target_dir[0] + actual_dir[1] * target_dir[1]) / actual_mag
            self.performance_metrics['wave_efficiency'] = max(0, efficiency)
        
    def _print_status(self):
        """Print current simulation status"""
        primary_pos = self.primary_test_object.get_world_pose()[0]
        primary_vel = self.primary_test_object.get_linear_velocity()
        
        print(f"Frame {self.frame_count}:")
        print(f"  Primary object: pos=({primary_pos[0]:.3f}, {primary_pos[1]:.3f}, {primary_pos[2]:.3f})")
        print(f"  Primary object: vel=({primary_vel[0]:.3f}, {primary_vel[1]:.3f}, {primary_vel[2]:.3f})")
        print(f"  Displacement: {self.performance_metrics['object_displacement']:.3f}m")
        print(f"  Wave efficiency: {self.performance_metrics['wave_efficiency']:.3f}")
        print(f"  Wave direction: {self.actuator_controller.current_direction}")


if __name__ == "__main__":
    # Configuration for the demo
    config = {
        'actuator_grid_size': 20,
        'membrane_size': 1.0,
        'wave_amplitude': 0.04,
        'wave_frequency': 0.8,
        'wave_speed': 0.25,
        'use_cloth': True,
        'physics_dt': 1.0/120.0,  # Higher physics rate for better simulation
        'render_dt': 1.0/60.0
    }
    
    demo = PeristalticConveyorEnhanced(config)
    demo.run_simulation()