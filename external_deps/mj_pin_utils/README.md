# mj_pin_utils
mj_pin_utils provides tools to ease the use of MuJoCo in robotics projects.

## Installation

mj_pin_utils has the following python dependencies.
- python >= 3.10
- mujoco
- pin/pinocchio
- opencv-python
- robot_descriptions

mj_pin_utils needs to be installed locally:

```
git clone https://github.com/Atarilab/mj_pin_utils.git
cd mj_pin_utils

# Using pip
pip install -e .

# Using your favorite conda environment
conda develop .
```

## Usage


### Robot description

One can use [`get_robot_description`](mj_pin/utils.py) to load the paths of your models (MJCF or URDF) based on the [robot_descriptions](https://github.com/robot-descriptions/robot_descriptions.py) package. It will return a [`RobotDescription`](mj_pin/utils.py) class with paths to the model files.

```python
from mj_pin.utils import get_robot_description

robot_description = get_robot_description("go2")
# URDF
urdf_path = robot_description.urdf_path
# MJCF/XML
xml_path = robot_description.xml_path
xml_scene_path = robot_description.xml_scene_path # With floor
```

### Simulator

mj_pin_utils provides a generic framework based on the [`Simulator`](mj_pin/simulator.py) class. To instantiate a simulator:
```python
from mj_pin.simulator import Simulator
from mj_pin.utils import get_robot_description

robot_description = get_robot_description("go2")
sim = Simulator(robot_description.xml_scene_path)
sim.run()
```

- The `Simulator` runs different threads in parallel: one for the **physics**, one for the **viewer**. One can specify the frequency of those threads as arguments to the class.
- Run the simulation with `sim.run()`. It will stop when the viewer is closed. To run the simulation for a given time (5 seconds) one can use `sim.run(sim_time=5)`.
- To run the simulation without viewer: `sim.run(sim_time=5, use_viewer=False)`. Note: if the viewer is not used, the physics thread runs at maximum speed.


### Controller

So far, nothing actuates the robot. Define your own controller by inheriting the [`Controller`](mj_pin/abstract.py) class. One just needs to override the `compute_torques_dof(self, mj_data)` method. This method is called each physics step in the simulator. This method should update `self.torques_dof` with the desired torque values on **all the DoF** (only actuated ones will be used in the simulator). One can use methods from the base class to do so:

- use `q, v = self.get_state(mj_data)` to get the position and velocity of your degrees of freedom (DoF).

See `example/0_pd_controller.py` as an example. In this framework, using a controller based on a pinocchio model is simplified, as shown in the example.
```sh
cd mj_pin/example
# PD controller based on MuJoCo model
python3 0_pd_controller.py --robot_name go2
# PD controller based on Pinocchio model
python3 0_pd_controller.py --robot_name go2 --pin
```

### Visual callback

It is sometimes usefull to visualize objects used by you controller for debugging purposes. Define your own [`VisualCallback`](mj_pin/abstract.py) class to do so. The `add_visuals(self, mj_data)` method needs to be inherited. This method is called every viewer step by default. One should add the geometries that will be rendered in the viewer in this method. One should use methods provided by the base class:

- `add_sphere(self, pos, radius, rgba)` to add a sphere.
- `add_box(self, pos, rot_euler, size, rgba)` to add a box.

See `example/1_visual_callback.py` as an example. Geometries are added at the robot's feet locations.
```
python3 1_visual_callback.py
```

### Data recorder

Some data of the robot needs to be recorded when working with data driven approaches. Define your own [`DataRecorder`](mj_pin/abstract.py) class to do so. The `record(self, mj_data)`, `reset(self)`, `save(self)` methods need to be inherited. The `record` method is called every simulation step by default (before the controller). The user is free to record and save data his own way.

See `example/2_data_recorder.py` as an example. States variables and torques are recorded in this example.
```
python3 2_data_recorder.py
```

### Record video

To record a video of the simulation: `sim.run(record_video=True)`.
Use `sim.vs` to change the video settings (see the [`VideoSettings`](mj_pin/simulator.py) class).

See `example/3_record_video.py` as an example. 
```sh
# Record with the viewer, track the user camera motion 
python3 3_record_video.py
# Record without viewer, track the base position
python3 3_record_video.py --track_base
```

### Model editing

To edit your MuJoCo model provided to the simulator, use methods provided by `sim.edit` (see [`ModelEditor`](mj_pin/model_editor.py)). One can add different static bodies (box, cylinder, sphere), move and delete them.

See `example/4_model_editing.py` as an example.
```sh
python3 example/4_model_editing.py
```

### Collision

In some cases, the simulation should be stopped as soon as the robot collides. Define a list of all the geometries (name or id in the MuJoCo model) that are allowed to collide and pass it to the simulator. The simulation will stop if there is a geometry that is not in this list but in a contact pair.

See `example/5_collision.py` as an example.
```sh
python3 example/5_collision.py
```

### Parallel execution

To make a data collection process faster, one could run several simulations in parallel. Define your own [`ParallelExecutorBase`](mj_pin/abstract.py) class to do so. The `create_job(self, id_job)`, `run_job(self, id_job, **kwargs)` methods need to be inherited. The parallel executor is based on a producer/consumer architecture:

- `create_job` returns a set of arguments (as a dictionary) that are needed to run a job (e.g. different goals). The jobs are added to a queue shared between processes.
- `run_job` executes the jobs in parallel (e.g. run the simulation) by taking the arguments from the queue (passed in `**kwargs`). It should return a boolean if the jobs succeeded.

See `example/6_parallel_execution.py` as an example. `create_job` computes random PD target. `run_job` runs the simulation with a PD controller and the produced PD target.
```sh
python3 example/6_parallel_execution.py
```

## To Do

- [ ] pip/conda installation
- [ ] use other simulators (MJX, genesis)

- **Simulator**
  - [ ] Random perturbations
  - [ ] Radom initialization
  - [ ] Add visuals in the video recording


