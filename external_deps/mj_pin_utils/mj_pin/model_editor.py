import mujoco
import numpy as np
from typing import List, Dict, Optional
from mj_pin.abstract import Colors 

class ModelEditor():
    DEFAULT_NAME = "static"

    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.reset()
        self.colors = Colors()

    def _add_body_and_geom(
        self,
        geom_type,
        pos : np.ndarray,
        size : np.ndarray,
        euler : np.ndarray,
        rgba : List[float],
        name : str = "",
        allow_collision : bool = True,
        ) -> int:
        
        if name:
            if name in self.name2id:
                name = f"{name}_{self.id}"
        else:
            name = f"{ModelEditor.DEFAULT_NAME}_{self.id}"

        body = self.mj_spec.worldbody.add_body(name = name)
        geom = body.add_geom()

        geom.type = geom_type
        geom.pos = pos.copy()
        geom.size = size.copy()
        geom.quat = self.to_quat(euler)
        geom.rgba = rgba
        geom.name = name

        # Update maps
        self.id2name[self.id] = name
        self.name2id[name] = self.id
        self.id += 1
        if allow_collision:
            self.name_allowed_collisions.append(geom.name)
        # Return index
        return self.id - 1

    @staticmethod
    def to_quat(euler) -> np.array:
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, euler, "xyz")
        return quat
    
    def add_box(
        self,
        pos : np.ndarray,
        size : np.ndarray,
        euler : np.ndarray,
        rgba : Optional[List[float]] = None,
        color : Optional[str] = None,
        name : str = "",
        allow_collision : bool = True,
        ) -> int:
        if rgba is None and color is not None:
            rgba = self.colors.name(color)
        elif rgba is None:
            rgba = self.colors.WHITE
        return self._add_body_and_geom(
            geom_type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=size,
            euler=euler,
            rgba=rgba,
            name=name if name else "box",
            allow_collision=allow_collision,
        )

    def add_sphere(
        self,
        pos : np.ndarray,
        radius : float,
        rgba : Optional[List[float]] = None,
        color : Optional[str] = None,
        name : str = "",
        allow_collision : bool = True,
        ) -> int:
        if rgba is None and color is not None:
            rgba = self.colors.name(color)
        elif rgba is None:
            rgba = self.colors.WHITE
        
        size = np.array([radius, 0, 0])
        euler = np.zeros(3)
        return self._add_body_and_geom(
            geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=pos,
            size=size,
            euler=euler,
            rgba=rgba,
            name=name if name else "sphere",
            allow_collision=allow_collision
        )

    def add_cylinder(
        self,
        pos : np.ndarray,
        radius : float,
        height : float,
        euler : np.ndarray,
        rgba : Optional[List[float]] = None,
        color : Optional[str] = None,
        name : str = "",
        allow_collision : bool = True,
        ) -> int:
        if rgba is None and color is not None:
            rgba = self.colors.name(color)
        elif rgba is None:
            rgba = self.colors.WHITE
        
        size = np.array([radius / 2., height / 2., 0])
        return self._add_body_and_geom(
            geom_type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=pos,
            size=size,
            euler=euler,
            rgba=rgba,
            name=name if name else "cylinder",
            allow_collision=allow_collision,
        )
    
    def get_body(
        self, 
        name: Optional[str] = None, 
        id: Optional[int] = None
        ) -> None:

        if name is None and id is None:
            raise ValueError("get_body: provide a name or an id.")

        if id is not None:
            if id in self.id2name:
                name = self.id2name[id]

        if name is not None:
            if name in self.name2id:
                id = self.name2id[name]
                body = self.mj_spec.find_body(name)
                return body
        
        return None
    
    def remove(
        self, 
        name: Optional[str] = None, 
        id: Optional[int] = None
        ) -> None:
        body = self.get_body(name, id)
        if body:
            self.mj_spec.detach_body(body)

            if id is not None:
                name = self.id2name[id]
            if name is not None:
                id = self.name2id[name]
            
            del self.id2name[id]
            del self.name2id[name]
            if name in self.name_allowed_collisions:
                self.name_allowed_collisions.remove(name)

    def move(
        self, 
        new_pos: np.ndarray, 
        new_euler: Optional[np.ndarray] = None,
        name : Optional[str] = None, 
        id : Optional[int] = None, 
        ) -> None:
        body = self.get_body(name, id)
        if body:
            geom = body.first_geom()
            if geom:
                geom.pos = new_pos.copy()
                if new_euler is not None:
                    geom.quat = self.to_quat(new_euler)

    def set_color(self,
                  rgba : np.ndarray, 
                  name : Optional[str] = None, 
                  id : Optional[int] = None) -> None:
        body = self.get_body(name, id)
        if body:
            geom = body.first_geom()
            geom.rgba = rgba

    def reset(self):
        self.mj_spec = mujoco.MjSpec.from_file(self.xml_path)
        self.id : int = len(self.mj_spec.bodies)
        self.id2name : Dict[int, str] = {}
        self.name2id : Dict[str, int] = {}
        self.name_allowed_collisions : List[str] = []

    def get_model(self):
        return self.mj_spec.compile()

if __name__ == "__main__":
    from mj_pin.utils import get_robot_description
    from mj_pin.simulator import Simulator
    SIM_TIME = 3

    robot_description = get_robot_description("go2")
    sim = Simulator(robot_description.xml_scene_path)
    
    # Add custom geometries
    print("--- Adding custom geometries...")
    pos = np.array([1., 1., 1.])
    size = np.array([1., 1., 1.])
    euler = np.array([0., 0., .5])
    box_id = sim.edit.add_box(
        pos,
        size,
        euler,
        color="red"
   )
    pos = np.array([-2., 1., 1.])
    radius = 0.5
    sphere_id = sim.edit.add_sphere(
        pos,
        radius,
        color="blue",
        name="sphere",
        allow_collision=True,
   )
    pos = np.array([-1., -1., 1.])
    radius = 0.3
    height = 1.0
    euler = np.array([0., 0., 0.])
    cylinder_id = sim.edit.add_cylinder(
        pos,
        radius,
        height,
        euler,
        color="green",
        name="cylinder"
    )
    sim.run(sim_time=SIM_TIME)
    
    # Move geometries
    print("--- Move box...")
    new_pos = np.array([3., 1., 1.])
    new_euler = np.array([0., 0.5, .0])
    sim.edit.move(new_pos, new_euler, id = box_id)
    sim.run(sim_time=SIM_TIME)

    # Change color
    print("--- Change color sphere...")
    new_color = np.array([1., 0., 0., 1.])
    sim.edit.set_color(new_color, id = sphere_id)
    sim.run(sim_time=SIM_TIME)

    # Remove geometries
    print("--- Remove sphere...")
    sim.edit.remove(name="sphere")
    sim.run(sim_time=SIM_TIME)

    # Reset
    print("--- Reset scene...")
    sim.edit.reset()
    sim.run(sim_time=SIM_TIME)