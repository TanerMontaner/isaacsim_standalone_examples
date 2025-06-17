from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import math
import numpy as np
import omni
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.prims import create_prim
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema


class PeristalticConveyor:
    """Simple peristaltic surface conveyor demo."""

    def __init__(self) -> None:
        # Conveyor parameters
        self.conveyor_size = 1.0
        self.grid_size = 20
        self.total_actuators = self.grid_size * self.grid_size
        self.actuator_stroke = 0.1
        self.actuator_speed = 0.2
        self.actuator_spacing = self.conveyor_size / self.grid_size
        self.actuator_base_height = 0.05
        self.actuator_radius = 0.015

        # Wave parameters
        self.wave_frequency = 1.0
        self.wave_amplitude = self.actuator_stroke / 2.0
        self.wave_speed = 0.2
        self.wave_length = self.wave_speed / self.wave_frequency

        # Object parameters
        self.object_mass = 0.02
        self.object_size = 0.08

        # Simulation state
        self.world: World | None = None
        self.actuators = []
        self.target_object: DynamicCuboid | None = None
        self.time = 0.0

    def setup_simulation(self) -> None:
        """Create the scene objects."""
        self.world = World(stage_units_in_meters=1.0)

        self._setup_physics_scene()
        self._create_base_platform()
        self._create_actuator_grid()
        self._create_target_object()
        self._setup_lighting()

        self.world.reset()

    def _setup_physics_scene(self) -> None:
        stage = omni.usd.get_context().get_stage()
        physics_scene_path = "/physicsScene"
        if not stage.GetPrimAtPath(physics_scene_path):
            physics_scene = UsdPhysics.Scene.Define(stage, physics_scene_path)
            physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            physics_scene.CreateGravityMagnitudeAttr().Set(9.81)
            physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
            physx_scene_api.CreateEnableCCDAttr(True)
            physx_scene_api.CreateTimeStepsPerSecondAttr(60.0)

    def _create_base_platform(self) -> None:
        stage = omni.usd.get_context().get_stage()
        base_path = "/World/BasePlatform"
        base_prim = create_prim(base_path, "Cube")
        cube_geom = UsdGeom.Cube(base_prim)
        cube_geom.CreateSizeAttr(1.0)
        cube_geom.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, -0.025))
        cube_geom.AddScaleOp().Set(Gf.Vec3f(self.conveyor_size * 1.2, self.conveyor_size * 1.2, 0.05))
        UsdPhysics.CollisionAPI.Apply(base_prim)
        UsdPhysics.RigidBodyAPI.Apply(base_prim)
        rigid_body_api = UsdPhysics.RigidBodyAPI(base_prim)
        rigid_body_api.CreateRigidBodyEnabledAttr(True)
        rigid_body_api.CreateKinematicEnabledAttr(True)
        try:
            material_path = "/World/Materials/BaseMaterial"
            material_prim = create_prim(material_path, "Material")
            physics_material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
            physics_material_api.CreateStaticFrictionAttr(0.8)
            physics_material_api.CreateDynamicFrictionAttr(0.6)
            physics_material_api.CreateRestitutionAttr(0.1)
            binding_api = UsdPhysics.MaterialBindingAPI.Apply(base_prim)
            binding_api.Bind(material_prim, UsdPhysics.Tokens.physics)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Could not create physics material: {exc}")

    def _create_actuator_grid(self) -> None:
        stage = omni.usd.get_context().get_stage()
        start_x = -self.conveyor_size / 2 + self.actuator_spacing / 2
        start_y = -self.conveyor_size / 2 + self.actuator_spacing / 2
        actuator_count = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = start_x + i * self.actuator_spacing
                y = start_y + j * self.actuator_spacing
                z = self.actuator_base_height
                actuator = self._create_single_actuator(actuator_count, x, y, z)
                self.actuators.append(actuator)
                actuator_count += 1

    def _create_single_actuator(self, index: int, x: float, y: float, z: float) -> dict:
        actuator_path = f"/World/Actuators/Actuator_{index:03d}"
        actuator_prim = create_prim(actuator_path, "Cylinder")
        cylinder_geom = UsdGeom.Cylinder(actuator_prim)
        cylinder_geom.CreateRadiusAttr(self.actuator_radius)
        cylinder_geom.CreateHeightAttr(self.actuator_stroke * 2)
        cylinder_geom.AddTranslateOp().Set(Gf.Vec3f(x, y, z))
        UsdPhysics.CollisionAPI.Apply(actuator_prim)
        UsdPhysics.RigidBodyAPI.Apply(actuator_prim)
        rigid_body_api = UsdPhysics.RigidBodyAPI(actuator_prim)
        rigid_body_api.CreateRigidBodyEnabledAttr(True)
        rigid_body_api.CreateKinematicEnabledAttr(True)
        return {
            "index": index,
            "prim_path": actuator_path,
            "base_position": Gf.Vec3f(x, y, z),
            "current_height": z,
            "world_x": x,
            "world_y": y,
        }

    def _create_target_object(self) -> None:
        initial_pos = np.array([0.2, 0.2, 0.2])
        self.target_object = DynamicCuboid(
            prim_path="/World/TargetObject",
            name="polybag_cube",
            position=initial_pos,
            size=self.object_size,
            mass=self.object_mass,
        )
        self.world.scene.add(self.target_object)
        try:
            stage = omni.usd.get_context().get_stage()
            object_prim = stage.GetPrimAtPath("/World/TargetObject")
            material_path = "/World/Materials/ObjectMaterial"
            if not stage.GetPrimAtPath(material_path):
                material_prim = create_prim(material_path, "Material")
                physics_material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
                physics_material_api.CreateStaticFrictionAttr(0.7)
                physics_material_api.CreateDynamicFrictionAttr(0.5)
                physics_material_api.CreateRestitutionAttr(0.3)
                binding_api = UsdPhysics.MaterialBindingAPI.Apply(object_prim)
                binding_api.Bind(material_prim, UsdPhysics.Tokens.physics)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Could not create object material: {exc}")

    def _setup_lighting(self) -> None:
        stage = omni.usd.get_context().get_stage()
        dome_light_path = "/World/DomeLight"
        if not stage.GetPrimAtPath(dome_light_path):
            dome_light = create_prim(dome_light_path, "DomeLight")
            dome_light.GetAttribute("intensity").Set(1000.0)
        dir_light_path = "/World/DirectionalLight"
        if not stage.GetPrimAtPath(dir_light_path):
            dir_light = create_prim(dir_light_path, "DistantLight")
            dir_light.GetAttribute("intensity").Set(3000.0)
            UsdGeom.Xformable(dir_light).AddRotateXYZOp().Set(Gf.Vec3f(-45, 45, 0))

    def calculate_wave_height(self, x: float, y: float, t: float) -> float:
        wave_x = x + self.wave_speed * t
        phase = 2.0 * math.pi * wave_x / self.wave_length
        height_offset = self.wave_amplitude * math.sin(phase)
        wave_y = y * 2.0
        cross_wave = 0.3 * self.wave_amplitude * math.sin(2.0 * math.pi * wave_y / self.wave_length + t * 3.0)
        return height_offset + cross_wave

    def update_actuators(self, dt: float) -> None:
        self.time += dt
        stage = omni.usd.get_context().get_stage()
        for actuator in self.actuators:
            wave_height = self.calculate_wave_height(actuator["world_x"], actuator["world_y"], self.time)
            base_z = actuator["base_position"][2]
            new_z = base_z + wave_height
            min_z = base_z - self.wave_amplitude
            max_z = base_z + self.wave_amplitude
            new_z = max(min_z, min(max_z, new_z))
            new_position = Gf.Vec3f(actuator["base_position"][0], actuator["base_position"][1], new_z)
            prim = stage.GetPrimAtPath(actuator["prim_path"])
            if prim.IsValid():
                xform = UsdGeom.Xformable(prim)
                translate_op = None
                for op in xform.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op
                        break
                if translate_op:
                    translate_op.Set(new_position)
                else:
                    xform.AddTranslateOp().Set(new_position)
            actuator["current_height"] = new_z

    def run_simulation(self) -> None:
        dt = 1.0 / 60.0
        max_time = 30.0
        update_counter = 0
        while simulation_app.is_running() and self.time < max_time:
            if self.world.is_playing():
                self.update_actuators(dt)
                self.world.step(render=True)
                update_counter += 1
                if update_counter % 60 == 0:
                    obj_pos = self.target_object.get_world_pose()[0]
                    print(
                        f"Time: {self.time:.1f}s, Object position: ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})"
                    )
                obj_pos = self.target_object.get_world_pose()[0]
                if obj_pos[0] > 0.4:
                    print("SUCCESS: Object successfully transported across the surface!")
                    print(
                        f"Final object position: ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})"
                    )
                    break
            simulation_app.update()


if __name__ == "__main__":
    print("=" * 60)
    print("PERISTALTIC CONVEYOR SIMULATION - Isaac Sim 4.5.0")
    print("=" * 60)
    conveyor = PeristalticConveyor()
    conveyor.setup_simulation()
    conveyor.run_simulation()
    simulation_app.close()
