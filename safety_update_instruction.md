# Safety Layer Refactoring Guide

## 1. Overview

This document outlines the recent refactoring of the robot's safety layer. The primary goal of this update was to move away from hardcoded safety parameters and inflexible joint monitoring logic, replacing it with a configurable and more maintainable system.

The refactoring focused on two key areas:
1.  **Flexible Joint Monitoring**: Instead of relying on a fixed number of active joints (`num_active_joints`), the system now uses an explicit list of ignored joint names.
2.  **External Safety Configuration**: All numerical safety limits (e.g., angle limits, torque scales) have been moved from Python scripts into a dedicated JSON configuration file.

---

## 2. Key Changes

### 2.1 From `num_active_joints` to an Ignored Joints List

*   **Old System**: The safety layer only monitored the first `N` joints, defined by `NUM_ACTIVE_BODY_JOINTS` in `sdk_controller/robots/G1.py`.
*   **Limitation**: This approach was inflexible. It was impossible to ignore a specific joint (e.g., a faulty wrist) without also ignoring all subsequent joints in the model's hierarchy (e.g., the entire other arm).
*   **New System**: A new list named `safety_ignored_joints` has been added to `global_config.json`. Any joint name included in this list will be completely ignored by all safety checks.

### 2.2 From Hardcoded Parameters to `safety_config.json`

*   **Old System**: Safety parameters for "default" and "conservative" modes were hardcoded directly within `sdk_controller/safety.py` and `sdk_controller/abstract_biped.py`.
*   **Limitation**: Adjusting any safety threshold required direct modification of the Python source code, increasing the risk of errors.
*   **New System**: A new file, `safety_config.json`, has been created. This file contains different "profiles" (e.g., `default`, `conservative`) where all safety-related numerical values are defined. The desired profile can be selected at runtime.

---

## 3. File-by-File Breakdown

### `safety_config.json` (New File)
This is the new central hub for all safety parameters. It allows for easy tuning of safety limits without touching the code.

```json
{
  "_comment": "Safety layer configuration file. Defines different profiles for robot safety limits.",
  "default": {
    "_comment_block": {
      "group_description": "Default safety profile for normal operation.",
      "base_orientation_limit_deg": "Maximum allowed base tilt (roll/pitch) in degrees before triggering a safety stop.",
      "scale_joint_limit": "Multiplier for the physical joint limits from the MuJoCo model. E.g., 0.95 means using 95% of the full range.",
      "...": "..."
    },
    "base_orientation_limit_deg": 35.0,
    "scale_joint_limit": 0.95,
    "...": "..."
  },
  "conservative": {
    "_comment_block": {
      "group_description": "Conservative safety profile with tighter limits..."
    },
    "base_orientation_limit_deg": 25.0,
    "scale_joint_limit": 0.90,
    "...": "..."
  }
}
```

### `global_config.json` (Modified)
A new key was added to manage which joints are monitored.

```json
{
  "...": "...",
  "vicon_z_offset": -0.23,
  "safety_ignored_joints": [
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint"
  ]
}
```

### `sdk_controller/safety.py` (Modified)
The core `SafetyLayer` class was refactored. Its `__init__` constructor signature changed from:
`__init__(self, mj_model, conservative_safety: bool, num_active_joints: int)`
to:
`__init__(self, mj_model, safety_profile: str, inactive_joint_names: set)`

It now reads `safety_config.json` and filters joints based on the provided name set.

### `sdk_controller/abstract_biped.py` (Modified)
The intermediate `HGSafetyLayer` was updated to read from both new configuration sources and pass the correct parameters to the base `SafetyLayer`.

### `zmq_dds_bridge.py` (Modified)
The application entry point was updated. The command-line argument `--conservative_safety` (a boolean flag) was **removed** and replaced with `--safety_profile <name>` (a string option).

---

## 4. New Usage Guide

### How to Modify Monitored Joints
1.  Open `global_config.json`.
2.  Add or remove joint names (as defined in `sdk_controller/robots/G1.py`) to/from the `safety_ignored_joints` list.
3.  Restart the application.

### How to Adjust Safety Parameters
1.  Open `safety_config.json`.
2.  Modify the numerical values within the `default` or `conservative` profiles as needed. You can also create new profiles.
3.  Restart the application, selecting the appropriate profile if you created a new one.

### How to Run the Bridge with Different Safety Profiles
Use the `--safety_profile` command-line argument when launching the bridge:

```bash
# Run with default safety settings (this is the default if not specified)
python zmq_dds_bridge.py --channel <your_interface> --safety_profile default

# Run with conservative (tighter) safety settings
python zmq_dds_bridge.py --channel <your_interface> --safety_profile conservative
```
