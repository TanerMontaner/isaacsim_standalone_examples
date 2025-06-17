"""
Peristaltic Conveyor Standalone Simulation for Isaac Sim 4.5.0

This standalone script creates a 20x20 grid of actuators that generate coordinated waves
to transport objects across the surface. Each actuator can move up/down with 100mm stroke 
at 200mm/s speed.

Usage:
    ./python.bat peristaltic_conveyor_standalone.py

Author: Taner Adak Master's Thesis Project
"""

import argparse
import sys
import math
import asyncio
import numpy as np

# Isaac Sim imports (must be done AFTER SimulationApp is created)
from isaacsim import SimulationApp

def main():
    """Main function to run the peristaltic conveyor simulation"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Peristaltic Conveyor Simulation")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--test", action="store_true", help="Run in test mode with shorter duration")
    args, unknown = parser.parse_known_args()
    
    # Create simulation app
    config = {
        "headless": args.headless,
        "width": 1280,
        "height": 720,
    }
    
    print("=" * 60)
    print("PERISTALTIC CONVEYOR SIMULATION - Isaac Sim 4.5.0")
    print("=" * 60)
    print("Proof of Concept: 400 actuators generating coordinated waves")
    print("Goal: Transport a 20g object across 1m x 1m surface")
    print("Parameters: 20x20 grid, 100mm stroke, 200mm/s speed")
    print("=" * 60)
    
    # Start Isaac Sim
    simulation_app = SimulationApp(config)
    
    # Import omni modules AFTER SimulationApp is created
    import omni.usd
    import omni.timeline
    from omni.isaac.core import World
    from omni.isaac.core.objects import DynamicCuboid
    # Fixed import - use direct USD physics instead of deprecated materials
    from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, Sdf
    
    try:
        # Create the peristaltic conveyor
        conveyor = PeristalticConveyor(args.test)
        
        # Run the simulation synchronously (Isaac Sim 4.5.0 style)
        conveyor.run_simulation_sync()
        
    except Exception as e:
        print(f"Error in simulation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean shutdown
        simulation_app.close()


class PeristalticConveyor:
    """
    Modular Peristaltic Surface Conveyor Simulation for Isaac Sim 4.5.0
    
    This class creates a 20x20 grid of actuators that generate coordinated waves
    to transport objects across the surface.
    """
    
    def __init__(self, test_mode=False):
        # Import necessary modules
        import omni.usd
        from omni.isaac.core import World
        from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
        
        # Simulation parameters from specifications
        self.conveyor_size = 1.0  # 1m x 1m conveyor area
        self.grid_size = 20  # 20x20 matrix of actuators
        self.total_actuators = self.grid_size * self.grid_size  # 400 actuators
        self.actuator_stroke = 0.1  # 100mm stroke
        self.actuator_speed = 0.2  # 200mm/s
        self.stroke_time = self.actuator_stroke / self.actuator_speed  # 0.5s for full stroke
        
        # Derived parameters
        self.actuator_spacing = self.conveyor_size / self.grid_size  # 0.05m spacing
        self.actuator_base_height = 0.05  # Base height of actuators
        self.actuator_radius = 0.015  # Radius of actuator pillars
        
        # Wave parameters
        self.wave_frequency = 1.0  # Hz
        self.wave_amplitude = self.actuator_stroke / 2  # Half stroke for wave amplitude
        self.wave_speed = 0.2  # m/s wave propagation speed
        self.wave_length = self.wave_speed / self.wave_frequency
        
        # Object parameters
        self.object_mass = 0.02  # 20g polybag (simplified as cube for now)
        self.object_size = 0.08  # 8cm cube to represent polybag
        
        # Simulation state
        self.world = None
        self.actuators = []
        self.target_object = None
        self.time = 0.0
        self.is_running = False
        
        # Test mode parameters
        self.test_mode = test_mode
        self.max_simulation_time = 10.0 if test_mode else 30.0
        
    def run_simulation_sync(self):
        """Main simulation execution - synchronous version for Isaac Sim 4.5.0"""
        print("Setting up Peristaltic Conveyor Simulation...")
        
        # Setup simulation
        self.setup_simulation_sync()
        
        print("Starting simulation loop...")
        self.simulation_loop_sync()
        
        print("Simulation completed successfully!")
        
    def setup_simulation_sync(self):
        """Initialize the Isaac Sim world and create all simulation objects - synchronous"""
        from omni.isaac.core import World
        
        # Create world with proper timeline
        self.world = World(stage_units_in_meters=1.0)
        self.world.initialize_simulation_context()
        
        # Setup physics scene
        self._setup_physics_scene()
        
        # Create base platform
        self._create_base_platform()
        
        # Create actuator grid
        self._create_actuator_grid_sync()
        
        # Create target object
        self._create_target_object()
        
        # Setup lighting and camera
        self._setup_lighting()
        self._setup_camera()
        
        print(f"Simulation setup complete: {self.total_actuators} actuators created")
        
    def _setup_physics_scene(self):
        """Configure physics scene parameters"""
        import omni.usd
        from pxr import UsdPhysics, PhysxSchema, Gf
        
        stage = omni.usd.get_context().get_stage()
        
        # Create physics scene if it doesn't exist
        physics_scene_path = "/physicsScene"
        if not stage.GetPrimAtPath(physics_scene_path):
            physics_scene = UsdPhysics.Scene.Define(stage, physics_scene_path)
            physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            physics_scene.CreateGravityMagnitudeAttr().Set(9.81)
            
            # Add PhysX scene API for advanced features
            physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
            physx_scene_api.CreateEnableCCDAttr(True)
            physx_scene_api.CreateTimeStepsPerSecondAttr(60.0)
            physx_scene_api.CreateEnableGPUDynamicsAttr(False)  # Use CPU for stability
            
    def _create_base_platform(self):
        """Create the base platform for the conveyor"""
        import omni.usd
        from pxr import UsdGeom, UsdPhysics, Gf
        
        stage = omni.usd.get_context().get_stage()
        
        # Create base platform
        base_path = "/World/BasePlatform"
        base_prim = stage.DefinePrim(base_path, "Cube")
        
        cube_geom = UsdGeom.Cube(base_prim)
        cube_geom.CreateSizeAttr(1.0)
        cube_geom.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, -0.025))
        cube_geom.AddScaleOp().Set(Gf.Vec3f(self.conveyor_size * 1.2, self.conveyor_size * 1.2, 0.05))
        
        # Add physics properties
        UsdPhysics.CollisionAPI.Apply(base_prim)
        UsdPhysics.RigidBodyAPI.Apply(base_prim)
        rigid_body_api = UsdPhysics.RigidBodyAPI(base_prim)
        rigid_body_api.CreateRigidBodyEnabledAttr(True)
        rigid_body_api.CreateKinematicEnabledAttr(True)  # Static platform
        
        # Create physics material using USD physics directly
        self._create_physics_material(base_prim, "BaseMaterial", 0.8, 0.6, 0.1)
        
    def _create_physics_material(self, prim, material_name, static_friction, dynamic_friction, restitution):
        """Create physics material using USD physics (Isaac Sim 4.5.0 compatible)"""
        import omni.usd
        from pxr import UsdPhysics
        
        stage = omni.usd.get_context().get_stage()
        
        # Create physics material
        material_path = f"/World/Materials/{material_name}"
        material_prim = stage.DefinePrim(material_path, "Material")
        material = UsdPhysics.MaterialAPI.Apply(material_prim)
        material.CreateStaticFrictionAttr(static_friction)
        material.CreateDynamicFrictionAttr(dynamic_friction)
        material.CreateRestitutionAttr(restitution)
        
        # Bind material to prim
        UsdPhysics.MaterialBindingAPI.Apply(prim).Bind(material_prim)
        
        return material_prim
        
    def _create_actuator_grid_sync(self):
        """Create the 20x20 grid of actuators - synchronous version"""
        import omni.usd
        from pxr import UsdGeom, UsdPhysics, Gf
        
        print("Creating actuator grid...")
        
        stage = omni.usd.get_context().get_stage()
        
        # Create parent group for actuators
        actuators_group = stage.DefinePrim("/World/Actuators", "Scope")
        
        # Calculate starting position (center the grid)
        start_x = -self.conveyor_size / 2 + self.actuator_spacing / 2
        start_y = -self.conveyor_size / 2 + self.actuator_spacing / 2
        
        actuator_count = 0
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Calculate position
                x = start_x + i * self.actuator_spacing
                y = start_y + j * self.actuator_spacing
                z = self.actuator_base_height
                
                # Create actuator
                actuator = self._create_single_actuator(actuator_count, x, y, z)
                self.actuators.append(actuator)
                
                actuator_count += 1
                
                # Progress update
                if actuator_count % 50 == 0:
                    print(f"Created {actuator_count}/{self.total_actuators} actuators")
        
        print(f"Actuator grid creation complete: {len(self.actuators)} actuators")
        
    def _create_single_actuator(self, index, x, y, z):
        """Create a single actuator pillar"""
        import omni.usd
        from pxr import UsdGeom, UsdPhysics, Gf
        
        stage = omni.usd.get_context().get_stage()
        
        # Create actuator path
        actuator_path = f"/World/Actuators/Actuator_{index:03d}"
        
        # Create cylinder for actuator
        actuator_prim = stage.DefinePrim(actuator_path, "Cylinder")
        cylinder_geom = UsdGeom.Cylinder(actuator_prim)
        
        # Set geometry properties
        cylinder_geom.CreateRadiusAttr(self.actuator_radius)
        cylinder_geom.CreateHeightAttr(self.actuator_stroke * 2)  # Accommodate full stroke
        
        # Set initial position
        cylinder_geom.AddTranslateOp().Set(Gf.Vec3f(x, y, z))
        
        # Add physics properties
        UsdPhysics.CollisionAPI.Apply(actuator_prim)
        UsdPhysics.RigidBodyAPI.Apply(actuator_prim)
        rigid_body_api = UsdPhysics.RigidBodyAPI(actuator_prim)
        rigid_body_api.CreateRigidBodyEnabledAttr(True)
        rigid_body_api.CreateKinematicEnabledAttr(True)  # Kinematic control
        
        # Add material
        self._create_physics_material(actuator_prim, f"ActuatorMaterial_{index}", 0.9, 0.7, 0.2)
        
        # Create actuator data structure
        actuator_data = {
            'index': index,
            'prim_path': actuator_path,
            'prim': actuator_prim,
            'base_position': Gf.Vec3f(x, y, z),
            'current_height': z,
            'grid_x': index // self.grid_size,
            'grid_y': index % self.grid_size,
            'world_x': x,
            'world_y': y
        }
        
        return actuator_data
        
    def _create_target_object(self):
        """Create the target object (polybag represented as cube)"""
        from omni.isaac.core.objects import DynamicCuboid
        import numpy as np
        
        # Initial position above the conveyor
        initial_pos = np.array([0.2, 0.2, 0.2])
        
        # Create dynamic cube
        self.target_object = DynamicCuboid(
            prim_path="/World/TargetObject",
            name="polybag_cube",
            position=initial_pos,
            size=self.object_size,
            mass=self.object_mass,
            color=np.array([0.8, 0.2, 0.2])  # Red color
        )
        
        # Add to world
        self.world.scene.add(self.target_object)
        
        # Create physics material for object using USD physics
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        object_prim = stage.GetPrimAtPath("/World/TargetObject")
        self._create_physics_material(object_prim, "ObjectMaterial", 0.7, 0.5, 0.3)
        
        print(f"Target object created at position: {initial_pos}")
        
    def _setup_lighting(self):
        """Setup basic lighting for the scene"""
        import omni.usd
        from pxr import UsdLux, Gf
        
        stage = omni.usd.get_context().get_stage()
        
        # Create dome light
        dome_light_path = "/World/DomeLight"
        if not stage.GetPrimAtPath(dome_light_path):
            dome_light = UsdLux.DomeLight.Define(stage, dome_light_path)
            dome_light.CreateIntensityAttr(1000.0)
            
        # Create directional light
        dir_light_path = "/World/DirectionalLight"
        if not stage.GetPrimAtPath(dir_light_path):
            dir_light = UsdLux.DistantLight.Define(stage, dir_light_path)
            dir_light.CreateIntensityAttr(3000.0)
            dir_light.AddRotateXYZOp().Set(Gf.Vec3f(-45, 45, 0))
            
    def _setup_camera(self):
        """Setup camera for better viewing"""
        import omni.usd
        from pxr import UsdGeom, Gf
        
        stage = omni.usd.get_context().get_stage()
        
        # Create camera
        camera_path = "/World/Camera"
        if not stage.GetPrimAtPath(camera_path):
            camera = UsdGeom.Camera.Define(stage, camera_path)
            camera.AddTranslateOp().Set(Gf.Vec3f(1.5, 1.5, 1.0))
            camera.AddRotateXYZOp().Set(Gf.Vec3f(-20, 45, 0))
            
    def calculate_wave_height(self, x, y, time):
        """
        Calculate the wave height at position (x,y) at given time
        Creates a traveling wave moving in the +X direction
        """
        # Convert world coordinates to wave coordinates
        wave_x = x + self.wave_speed * time
        
        # Create sinusoidal wave
        wave_phase = 2.0 * math.pi * wave_x / self.wave_length
        height_offset = self.wave_amplitude * math.sin(wave_phase)
        
        # Add perpendicular wave component for more complex motion
        wave_y = y * 2.0  # Scale Y for wave effect
        cross_wave = 0.3 * self.wave_amplitude * math.sin(2.0 * math.pi * wave_y / self.wave_length + time * 3.0)
        
        return height_offset + cross_wave
        
    def update_actuators(self, dt):
        """Update all actuator positions based on wave function"""
        import omni.usd
        from pxr import UsdGeom, Gf
        
        self.time += dt
        
        stage = omni.usd.get_context().get_stage()
        
        for actuator in self.actuators:
            # Calculate target height based on wave function
            wave_height = self.calculate_wave_height(
                actuator['world_x'], 
                actuator['world_y'], 
                self.time
            )
            
            # Calculate new position
            base_z = actuator['base_position'][2]
            new_z = base_z + wave_height
            
            # Clamp to stroke limits
            min_z = base_z - self.wave_amplitude
            max_z = base_z + self.wave_amplitude
            new_z = max(min_z, min(max_z, new_z))
            
            # Update actuator position
            new_position = Gf.Vec3f(
                actuator['base_position'][0],
                actuator['base_position'][1],
                new_z
            )
            
            # Apply position to prim
            prim = stage.GetPrimAtPath(actuator['prim_path'])
            if prim.IsValid():
                xform = UsdGeom.Xformable(prim)
                translate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
                
                if translate_ops:
                    translate_ops[0].Set(new_position)
                else:
                    xform.AddTranslateOp().Set(new_position)
            
            # Update stored height
            actuator['current_height'] = new_z
            
    def simulation_loop_sync(self):
        """Main simulation loop - synchronous version"""
        print("Starting Peristaltic Conveyor Simulation...")
        print("Watch as the wave pattern transports the cube across the surface!")
        
        self.is_running = True
        
        # Reset world and start simulation
        self.world.reset()
        
        # Simulation parameters
        target_fps = 60
        dt = 1.0 / target_fps
        update_counter = 0
        
        try:
            while self.is_running and self.time < self.max_simulation_time:
                # Update actuators with wave motion
                self.update_actuators(dt)
                
                # Step physics simulation
                self.world.step(render=True)
                
                # Progress reporting
                update_counter += 1
                if update_counter % 60 == 0:  # Every second
                    if self.target_object:
                        obj_pos, _ = self.target_object.get_world_pose()
                        print(f"Time: {self.time:.1f}s, Object position: ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})")
                
                # Check if object moved significantly (success condition)
                if self.target_object:
                    obj_pos, _ = self.target_object.get_world_pose()
                    if obj_pos[0] > 0.4:  # Object moved 40cm in X direction
                        print("SUCCESS: Object successfully transported across the surface!")
                        print(f"Final object position: ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})")
                        break
                        
        except Exception as e:
            print(f"Simulation error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Simulation loop completed.")
            self.is_running = False


if __name__ == "__main__":
    main()