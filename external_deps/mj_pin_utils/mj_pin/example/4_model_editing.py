import numpy as np

from mj_pin.simulator import Simulator
from mj_pin.utils import get_robot_description

if __name__ == "__main__":

    # Load robot information and paths
    robot_name = "go2"
    sim_time = 2.
    robot_description = get_robot_description(robot_name)
    sim = Simulator(robot_description.xml_scene_path)
    
    
    # Add custom geometries
    print("--- Adding custom geometries...")
    pos = np.array([1., 1., 1.]) /  4.
    size = np.array([1., 1., 1.]) / 10.
    euler = np.array([0., 0., .5])
    box_id = sim.edit.add_box(
        pos,
        size,
        euler,
        color="red"
   )
    pos = np.array([-2., 1., 1.]) / 4.
    radius = 0.07
    sphere_id = sim.edit.add_sphere(
        pos,
        radius,
        color="blue",
        name="sphere"
   )
    pos = np.array([-1., -1., 1.]) / 4.
    radius = 0.05
    height = 0.2
    euler = np.array([0., 0., 0.])
    cylinder_id = sim.edit.add_cylinder(
        pos,
        radius,
        height,
        euler,
        color="green",
        name="cylinder"
    )
    sim.run(sim_time=sim_time)
    
    # Move geometries
    print("--- Move box...")
    new_pos = np.array([-1., 1., 1.]) /  4.
    new_euler = np.array([0., 0.5, .0])
    sim.edit.move(new_pos, new_euler, id = box_id)
    sim.run(sim_time=sim_time)

    # Change color
    print("--- Change color sphere...")
    new_color = np.array([1., 0., 0., 1.])
    sim.edit.set_color(new_color, id = sphere_id)
    sim.run(sim_time=sim_time)

    # Remove geometries
    print("--- Remove sphere from name...")
    sim.edit.remove(name="sphere")
    sim.run(sim_time=sim_time)

    # Remove geometries
    print("--- Remove cylinder from id...")
    sim.edit.remove(id=cylinder_id)
    sim.run(sim_time=sim_time)
    
    # Reset
    print("--- Reset scene...")
    sim.edit.reset()
    sim.run(sim_time=sim_time)