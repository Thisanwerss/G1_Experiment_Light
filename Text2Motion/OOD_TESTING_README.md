# OOD Robustness Testing Framework

This document outlines the methodology and implementation of the Out-of-Distribution (OOD) robustness testing framework for the humanoid policy. The goal is to evaluate the policy's ability to generalize to starting positions and orientations not seen during training.

## 1. Motivation

A robust policy should be able to perform its task (e.g., standing, walking) even if the robot is initialized in a different location or with a different heading in the world. Testing this requires a framework to systematically introduce these initial state offsets.

We have implemented two primary methods for handling this offset, controlled by the `--alignment_mode` argument in `run_policy.py`.

## 2. Alignment Modes

### 2.1. `--alignment_mode reference` (The Old Method)

This was the initial approach to OOD testing.

-   **Logic**: The robot is initialized at an offset position and yaw in the simulation. The system then calculates this offset and shifts the *entire reference trajectory* to match the robot's new starting state.
-   **How it Works**: The `create_aligned_static_reference` method in `humanoid_standonly.py` is called. It computes a constant XY and Yaw offset and applies it to every frame of the reference motion.
-   **Problem**: While functional, this method can lead to performance degradation. By moving the reference trajectory away from the origin, the cost landscape for the CEM optimizer changes. The policy is forced to operate in a state-space that may be out-of-distribution from its training data, which can negatively impact stability and performance.

### 2.2. `--alignment_mode state_to_origin` (The New, Recommended Method)

This method was developed to overcome the limitations of the `reference` mode. It proved to be more robust and effective.

-   **Logic**: Instead of moving the reference trajectory, we create a "simulation illusion" for the policy. The robot starts at an offset in the world, but we transform its state feedback *back to the origin's coordinate frame* before feeding it into the policy (NN + CEM).
-   **How it Works**:
    1.  At initialization, the `calculate_and_store_initial_offset` method is called. It computes the same XY and Yaw offset as the old method but simply stores it.
    2.  In every control step, just before calling the policy, the `align_state_to_origin` method is used to transform the robot's current world-frame `qpos` and `qvel` into the reference's origin-frame.
    3.  The policy receives this "normalized" state and believes the robot is at the origin. It computes and returns a control action (target joint angles) in this origin frame.
    4.  **Crucially**, the output control signal is a set of target joint angles (PD targets). These are in the robot's own body frame and are independent of its world position or orientation. Therefore, this control signal can be applied **directly** to the physically offset robot without needing an inverse transformation.
-   **Benefits**:
    -   **Consistent Input Distribution**: The policy's input is always normalized, keeping it consistent with the training data (which is typically centered at the origin).
    -   **Superior Performance**: This method eliminates the performance degradation observed in the `reference` mode, resulting in much more stable standing behavior even with significant initial offsets.
    -   **Purer Robustness Test**: It provides a more accurate assessment of the policy's core generalization capabilities, isolated from the effects of a shifted cost landscape.

### 2.3. `--alignment_mode none`

-   **Logic**: This is a raw OOD test. The robot starts at an offset, and the reference trajectory remains at the origin. No alignment is performed.
-   **Use Case**: This is useful for observing the policy's absolute baseline performance when faced with a large, uncorrected state error. As expected, the robot will often try to "chase" the reference back to the origin.

## 3. How to Use

To test OOD robustness, run the policy script with the desired alignment mode.

**Recommended command for robust testing:**
```bash
python Text2Motion/scripts/run_policy.py --alignment_mode state_to_origin
```

You can modify the initial offsets for position and yaw directly within the `run_policy.py` script to test different scenarios. 