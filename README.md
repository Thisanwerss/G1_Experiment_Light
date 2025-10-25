
# G1 Humanoid Robot DDS Control Framework

This repository provides a complete framework for controlling the Unitree G1 humanoid robot using DDS (Data Distribution Service). It includes low-level DDS communication, safety layers, visualization tools, and examples for implementing custom controllers.

## Core Components

*   ** `sdk_controller/`**: The heart of the robot control system.
    *   `abstract_biped.py`: Defines the main `HGSDKController` base class, which handles all DDS communication, state subscription, and safety checks. This is the foundation for any new controller.
    *   `safety.py`: A crucial safety layer that prevents the robot from exceeding joint limits and torque limits, ensuring safe operation.
    *   `robots/G1.py`: Contains all robot-specific configurations, including joint mappings, PD gains, and default postures.

*   ** `dummy_dds_controller.py`**: A simple, ready-to-run example that sends a fixed standing command to the robot. It's the perfect starting point to verify your connection and setup.

*   ** `vicon_hg_publisher.py`**: Captures motion data from a Vicon system and broadcasts it over DDS. This provides real-time, high-precision localization for the robot.

*   ** `debug_toolbox/`**: A collection of useful utilities for development and debugging.
    *   `g1_state_visualizer.py`: A powerful GUI tool that displays the robot's real-time state, including joint positions, velocities, and IMU data.

*   ** `global_config.json` & `safety_config.json`**: Centralized configuration files for tuning robot behavior, network settings, and safety parameters without changing the code.

*   ** `g1_model/`**: Contains the MuJoCo XML models for the G1 robot, used for simulation and visualization.

##  Dependencies

Make sure you have the following dependencies installed:

*   **Python 3.8+**
*   **Core Libraries**:
    ```bash
    pip install numpy PySide6
    ```
*   **Robotics Libraries**:
    ```bash
    pip install mujoco pinocchio
    ```
*   **Vicon SDK**:
    ```bash
    pip install pyvicon_datastream
    ```
*   **Git Submodules**: This project uses submodules for external dependencies like the Unitree SDK. Initialize them with:
    ```bash
    git submodule update --init --recursive
    ```

---

##  Getting Started: A Guide for New Users

Follow these steps to get the robot up and running.

### 1. Initial Setup

First, clone the repository and install the necessary dependencies as listed above. Don't forget to initialize the git submodules, as this is crucial for the DDS communication libraries.

```bash
git clone <your-repo-url>
cd <repository-name>
git submodule update --init --recursive
pip install -r requirements.txt # Assuming you create one from the list above
```

### 2. Network Configuration

For the robot to communicate via DDS, your computer needs to be on the same network. This is typically done by connecting an Ethernet cable directly to the robot. Use the provided script to correctly configure your network interface.

```bash
# This script helps set up a static IP for the specified network interface
sudo ./setup_robot_net.sh
```
You will be prompted to select a network interface (e.g., `eth0`).

### 3. Launch Vicon Publisher (for Real Robot)

If you are working with the physical robot, you need a localization source like Vicon. Make sure the Vicon system is running and tracking the robot. Then, start the Vicon publisher to broadcast the robot's pose and velocity over DDS.

```bash
# Make sure to update the Vicon IP and robot object name in the command if needed
python vicon_hg_publisher.py --channel <your_network_interface>
```
Replace `<your_network_interface>` with the same interface you configured in the previous step (e.g., `eth0`).

### 4. Run the Controller

Now you can start sending commands to the robot. The `dummy_dds_controller.py` is a safe way to test the connection.

*   **For local testing (no robot needed)**:
    You can run in `lo` (loopback) mode to test the software stack without a physical connection.
    ```bash
    python dummy_dds_controller.py --channel lo
    ```

*   **For controlling the real robot**:
    Make sure the Vicon publisher is running in a separate terminal. Then, run the controller using your network interface.
    ```bash
    python dummy_dds_controller.py --channel <your_network_interface>
    ```
    The robot should now move to and hold a gentle standing posture.

### 5. Visualize the Robot's State

To see what the robot is doing in real-time, use the `g1_state_visualizer.py`.

```bash
python debug_toolbox/g1_state_visualizer.py
```
In the GUI, select the correct network interface and click "Connect". You will see real-time plots of all joint positions, velocities, and IMU data. You can also launch a 3D MuJoCo visualization from the GUI to see a live model of the robot.

---

##  Developer Guide: Creating a Custom Controller

This framework is designed to be extensible. Follow these steps to implement your own custom controller.

### 1. Understand the Core Components

Your custom controller will be built on top of the `HGSDKController` base class located in `sdk_controller/abstract_biped.py`. This class abstracts away the complexities of DDS communication and provides a clean interface for sending commands and receiving state data. It automatically handles:
-   Subscribing to low-level robot state (`LowState`).
-   Subscribing to Vicon pose and velocity data.
-   Publishing low-level commands (`LowCmd`).
-   Enforcing safety checks defined in `sdk_controller/safety.py`.

### 2. Create Your Controller Class

Create a new Python file and define a class that inherits from `HGSDKController`.

```python
# my_custom_controller.py
from sdk_controller.abstract_biped import HGSDKController
from sdk_controller.robots.G1 import G1_JOINT_CONFIG # Or other configs
import numpy as np

class MyController(HGSDKController):
    def __init__(self, **kwargs):
        # Always call the parent constructor
        super().__init__(**kwargs)
        
        # Your custom initializations here
        self.start_time = time.time()

    def update_motor_cmd(self, time: float):
        """
        This is the core logic of your controller.
        It is called periodically by the control loop.
        """
        # In this example, we generate a sine wave trajectory for the hip joints
        
        # Start with a default standing posture
        pd_targets = self.robot_config.STAND_UP_JOINT_POS.copy()
        
        # Calculate a sine wave motion
        elapsed_time = time - self.start_time
        amplitude = 0.2
        frequency = 0.5
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * elapsed_time)

        # Apply the motion to the left and right hip pitch joints
        # You need to know the mujoco_index for the joints you want to control
        left_hip_pitch_idx = 0  # from G1.py
        right_hip_pitch_idx = 6 # from G1.py
        
        pd_targets[left_hip_pitch_idx] += sine_wave
        pd_targets[right_hip_pitch_idx] -= sine_wave
        
        # The base class handles sending the command and safety checks
        self.update_motor_cmd_from_pd_targets(pd_targets)

```

### 3. Implement the `update_motor_cmd` Method

The only abstract method you must implement is `update_motor_cmd(self, time: float)`. This method is where you will define your control logic.

-   **Get State**: The most recent robot state (joint positions `_q`, velocities `_v`) is automatically updated and available as `self._q` and `self._v`.
-   **Calculate Targets**: Based on the current state and your algorithm (e.g., inverse kinematics, trajectory planning), calculate the desired target joint positions. This should be a NumPy array of size `NUM_ACTIVE_BODY_JOINTS` (27 for G1).
-   **Send Commands**: Call `self.update_motor_cmd_from_pd_targets(your_pd_targets_array)`. This method takes your desired joint positions and populates the DDS command message with the correct PD gains. The main control loop will then send this command to the robot. The safety layer is automatically checked before sending any command.

### 4. Create a Script to Run Your Controller

Finally, create a main script to initialize and run your new controller, similar to `dummy_dds_controller.py`.

```python
# run_my_controller.py
import time
from my_custom_controller import MyController

def main():
    # Initialize your controller
    controller = MyController(
        simulate=False,
        robot_config=None, # Uses G1 default
        xml_path="g1_model/g1_lab.xml",
        vicon_required=True,
        lo_mode=False, # Set to True for local testing
        kp_scale_factor=1.0,
        safety_profile="default"
    )

    # Main control loop
    try:
        while True:
            current_time = time.time()
            controller.update_motor_cmd(current_time)
            time.sleep(1.0 / controller.robot_config.CONTROL_FREQ) # 100Hz
            
    except KeyboardInterrupt:
        print("Stopping controller.")
    finally:
        # Ensure robot is safely stopped
        controller.damping_motor_cmd()

if __name__ == "__main__":
    main()
```

## TODO

-   **Experiment Validation**

-   **Troubleshooting**
