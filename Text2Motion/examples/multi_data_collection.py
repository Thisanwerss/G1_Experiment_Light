#!/usr/bin/env python3
"""
Collect multiple episode samples with perturbations and fall detection.
Automatically retries if robot falls down.
"""

import argparse
import numpy as np
import mujoco
import mujoco.viewer
import pickle
import os
import time
import jax
import jax.numpy as jnp
from mujoco import mjx
from typing import Sequence, Tuple
from datetime import datetime
from hydrax.algs import CEM
from hydrax.tasks.humanoid_standonly import HumanoidStand

def apply_joint_force_perturbation(mj_model: mujoco.MjModel,
                                   mj_data: mujoco.MjData,
                                   force_std: float = 10.0):
    """Apply random generalized forces to joints (not base)."""
    nq = mj_model.nq
    nbase = 7  # typically first 6 DoFs are base

    force_noise = np.random.normal(0, force_std, nq - nbase)
    mj_data.qfrc_applied[nbase:] = force_noise  # Skip base DoFs


def apply_base_force_perturbation(mj_model: mujoco.MjModel,
                                  mj_data: mujoco.MjData,
                                  body_name: str = "torso",
                                  force_magnitude_range: tuple = (600.0, 600.0),
                                  torque_std: float = 50.0):
    """Apply random spatial force (fx, fy, fz, tx, ty, tz) to the base body in a random direction."""
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"Body '{body_name}' not found in model.")

    # Choose randomly between x (0) or y (1) direction
    direction_index = np.random.choice([0, 1])
    sign = np.random.choice([-1, 1])

    # Sample force magnitude
    magnitude = np.random.uniform(*force_magnitude_range)

    # Build the force vector
    force = np.zeros(3)
    force[direction_index] = sign * magnitude

    # Sample torque
    torque = np.random.normal(0, torque_std, 3)

    # Apply force and (optionally) torque
    mj_data.xfrc_applied[body_id, :3] = force
    # mj_data.xfrc_applied[body_id, 3:] = torque

    print(f"Force applied in {'xy'[direction_index]}-direction: {force}")


def add_perturbation_to_state(mj_model: mujoco.MjModel,
                             mj_data: mujoco.MjData, 
                             joint_noise_std: float = 0.0,
                             base_pos_noise_std: float = 0.0,
                             base_orient_noise_std: float = 0.0) -> None:
    """Add random perturbations to robot state."""
    # Perturb joint positions (skip base)
    if joint_noise_std > 0 and mj_model.nq > 7:
        joint_noise = np.random.normal(0, joint_noise_std, mj_model.nq - 7)
        mj_data.qpos[7:] += joint_noise
    
    # Perturb base position
    if base_pos_noise_std > 0:
        pos_noise = np.random.normal(0, base_pos_noise_std, 3)
        mj_data.qpos[:3] += pos_noise
    
    # Perturb base orientation (small random rotation)
    if base_orient_noise_std > 0:
        # Generate small random euler angles
        euler_noise = np.random.normal(0, base_orient_noise_std, 3)
        
        # Convert to quaternion and compose with current orientation
        # This is a simplified approach - for small angles
        quat = mj_data.qpos[3:7]
        # Apply small rotation (simplified for small angles)
        quat[1] += euler_noise[0] * quat[0] / 2  # x component
        quat[2] += euler_noise[1] * quat[0] / 2  # y component
        quat[3] += euler_noise[2] * quat[0] / 2  # z component
        
        # Renormalize quaternion
        quat_norm = np.linalg.norm(quat)
        if quat_norm > 0:
            mj_data.qpos[3:7] = quat / quat_norm


def run_episode_with_logging(
    controller,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    frequency: float,
    initial_knots: jax.Array = None,
    reference: np.ndarray = None,
    reference_fps: float = 30.0,
    episode_data: dict = None,
    max_duration: float = None,
    fall_detection: bool = True,
    show_viewer: bool = True,
    args = None
) -> Tuple[bool, float]:
    """Run data collection and log state/action pair"""
    
    # Figure out timing
    replan_period = 1.0 / frequency
    sim_steps_per_replan = int(replan_period / mj_model.opt.timestep)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    
    # Initialize the controller
    mjx_data = mjx.put_data(mj_model, mj_data)
    mjx_data = mjx_data.replace(
        mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat
    )
    policy_params = controller.init_params(initial_knots=initial_knots)
    jit_optimize = jax.jit(controller.optimize)
    jit_interp_func = jax.jit(controller.interp_func)

    # Warm-up JIT
    policy_params, rollouts = jit_optimize(mjx_data, policy_params)
    tq = jnp.arange(0, sim_steps_per_replan) * mj_model.opt.timestep
    tk = policy_params.tk
    knots = policy_params.mean[None, ...]
    _ = jit_interp_func(tq, tk, knots)

    # Ghost reference setup
    if reference is not None:
        ref_data = mujoco.MjData(mj_model)
        ref_data.qpos[:] = reference[0, :]
        mujoco.mj_forward(mj_model, ref_data)
        vopt = mujoco.MjvOption()
        vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        pert = mujoco.MjvPerturb()
        catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC

    # Timing and state variables (define once before loop starts)
    last_perturb_time = 0
    perturb_interval = 2.0  # seconds between perturbations

    episode_start_time = time.time()
    
    if show_viewer:
        viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
        
        # Add ghost reference geometry
        if reference is not None:
            mujoco.mjv_addGeoms(mj_model, ref_data, vopt, pert, catmask, viewer.user_scn)
    else:
        viewer = None

    try:
        while True:
            # Apply perturbation only once and let the system recover
            # before applying back the disturbance again
            current_time = mj_data.time

            # Apply perturbation if enough time passed and robot is stable
            if (current_time - last_perturb_time) >= perturb_interval:
                if args.apply_perturbation:
                    print(f"\nApplying perturbation at t={mj_data.time:.2f}s")
                    # apply_joint_force_perturbation(mj_model, mj_data, force_std=5.0)
                    apply_base_force_perturbation(mj_model, 
                                                  mj_data, 
                                                  body_name="pelvis", 
                                                  force_magnitude_range = (200.0, 500.0),
                                                  torque_std=100.0)
                    last_perturb_time = mj_data.time

            # Check viewer status
            if viewer is not None and not viewer.is_running():
                break

            # Check if max duration reached
            if current_time >= max_duration:
                break

            start_time = time.time()

            # Update controller state
            mjx_data = mjx_data.replace(
                qpos=jnp.array(mj_data.qpos),
                qvel=jnp.array(mj_data.qvel),
                mocap_pos=jnp.array(mj_data.mocap_pos),
                mocap_quat=jnp.array(mj_data.mocap_quat),
                time=mj_data.time,
            )

            # Planning step
            plan_start = time.time()
            policy_params, rollouts = jit_optimize(mjx_data, policy_params)
            plan_time = time.time() - plan_start

            # Log knot data
            if episode_data is not None:
                knot_info = {
                    'knots': np.array(policy_params.mean),
                    'tk': np.array(policy_params.tk),
                    'timestamp': mj_data.time,
                    'planning_time': plan_time,
                    'qpos': mj_data.qpos.copy(),
                    'qvel': mj_data.qvel.copy()
                }
                episode_data['knots'].append(knot_info)
                episode_data['planning_times'].append(plan_time)

            # Update ghost reference
            if viewer is not None and reference is not None:
                t_ref = mj_data.time * reference_fps
                i_ref = min(int(t_ref), reference.shape[0] - 1)
                ref_data.qpos[:] = reference[i_ref]
                mujoco.mj_forward(mj_model, ref_data)
                mujoco.mjv_updateScene(
                    mj_model, ref_data, vopt, pert, viewer.cam, catmask, viewer.user_scn
                )

            # Interpolate controls
            sim_dt = mj_model.opt.timestep
            t_curr = mj_data.time
            tq = jnp.arange(0, sim_steps_per_replan) * sim_dt + t_curr
            tk = policy_params.tk
            knots = policy_params.mean[None, ...]
            us = np.asarray(jit_interp_func(tq, tk, knots))[0]

            # Simulate
            for i in range(sim_steps_per_replan):
                mj_data.ctrl[:] = np.array(us[i])
                mujoco.mj_step(mj_model, mj_data)

                # Log state data
                if episode_data is not None:
                    state_info = {
                        'qpos': mj_data.qpos.copy(),
                        'qvel': mj_data.qvel.copy(),
                        'ctrl': mj_data.ctrl.copy(),
                        'time': mj_data.time,
                        'actuator_force': mj_data.actuator_force.copy()
                    }
                    episode_data['trajectory'].append(state_info)
                    episode_data['controls'].append(mj_data.ctrl.copy())
                    episode_data['timestamps'].append(mj_data.time)

                if viewer is not None:
                    viewer.sync()

            # Timing control
            elapsed = time.time() - start_time
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

            # Print progress
            print(f"Time: {mj_data.time:.2f}s, Height: {mj_data.qpos[2]:.2f}m", end="\r")

            if len(episode_data['trajectory']) > 0:
                output_path = os.path.join(args.output_dir, args.filename + '.pkl')
                with open(output_path, 'wb') as f:
                    pickle.dump(episode_data, f)
            else:
                print(f"✗ Episode length is not enough.....")
    finally:
        if viewer is not None:
            viewer.close()
    
    duration = time.time() - episode_start_time

def main():
    parser = argparse.ArgumentParser(
        description="Collect multiple episode samples with perturbations."
    )
    # Basic parameters
    parser.add_argument(
        "--sequence",
        type=str,
        default="standing0",
        choices=["standing0", "standing1", "standing2", "standing3", "standing4", "and more...."],
        help="Reference sequence type.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./episode_data",
        help="Directory to save episode data.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="test",
        help="filename for the pickle.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10800,
        help="Max duration for the logging.",
    )
    
    # Perturbation parameters
    parser.add_argument(
        "--apply_perturbation",
        action="store_true",
        help="Apply perturbation to the sytem while tracking the reference",
    )
    parser.add_argument(
        "--joint_noise_std",
        type=float,
        default=0.05,
        help="Standard deviation for joint position noise (radians).",
    )
    parser.add_argument(
        "--base_pos_noise_std",
        type=float,
        default=0.02,
        help="Standard deviation for base position noise (meters).",
    )
    parser.add_argument(
        "--base_orient_noise_std",
        type=float,
        default=0.02,
        help="Standard deviation for base orientation noise (radians).",
    )
    parser.add_argument(
        "--perturbation_probability",
        type=float,
        default=0.8,
        help="Probability of applying perturbation to each episode.",
    )
    
    # Fall detection parameters
    parser.add_argument(
        "--min_height",
        type=float,
        default=0.4,
        help="Minimum base height before considering robot fallen (meters).",
    )
    parser.add_argument(
        "--max_tilt_deg",
        type=float,
        default=45.0,
        help="Maximum tilt angle before considering robot fallen (degrees).",
    )
    
    # Controller parameters
    parser.add_argument(
        "--cem_samples",
        type=int,
        default=500,
        help="Number of CEM samples.",
    )
    parser.add_argument(
        "--cem_elites",
        type=int,
        default=20,
        help="Number of CEM elite samples.",
    )
    parser.add_argument(
        "--plan_horizon",
        type=float,
        default=0.5,
        help="Planning horizon in seconds.",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=50.0,
        help="Control frequency in Hz.",
    )
    
    # Display options
    parser.add_argument(
        "--no_viewer",
        action="store_true",
        help="Run without visualization (faster).",
    )
    parser.add_argument(
        "--show_reference",
        action="store_true",
        help="Show reference trajectory.",
    )
    
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize task
    print(f"Creating HumanoidStand task with sequence: {args.sequence}")
    task = HumanoidStand(args.sequence)
    
    # Set up controller
    ctrl = CEM(
        task,
        num_samples=args.cem_samples,
        num_elites=args.cem_elites,
        sigma_start=0.4,
        sigma_min=0.1,
        explore_fraction=0.5,
        plan_horizon=args.plan_horizon,
        spline_type="zero",
        num_knots=4,
        iterations=2
    )

    # Configure model
    mj_model = task.mj_model
    mj_model.opt.timestep = 0.01
    mj_model.opt.iterations = 10
    mj_model.opt.ls_iterations = 50
    mj_model.opt.noslip_iterations = 2
    mj_model.opt.o_solimp = [0.0, 0.95, 0.01, 0.5, 2]
    mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE

    # Reference for visualization
    reference = task.reference if args.show_reference else None

    print(f"Perturbation settings:")
    print(f"  - Joint noise std: {args.joint_noise_std} rad")
    print(f"  - Base position noise std: {args.base_pos_noise_std} m")
    print(f"  - Base orientation noise std: {args.base_orient_noise_std} rad")
    print(f"  - Perturbation probability: {args.perturbation_probability}")
    print(f"Fall detection: height < {args.min_height}m, tilt > {args.max_tilt_deg}°")
    print()

    # Create fresh data object for each attempt
    mj_data = mujoco.MjData(mj_model)
    
    # Set initial pose from reference
    if task.reference.shape[1] == mj_model.nq:
        mj_data.qpos[:] = task.reference[0]
    else:
        mj_data.qpos[2] = 0.79
        if mj_model.nq > 6:
            mj_data.qpos[3] = 1.0
    
    # Initialize control knots
    initial_knots = None
    if ctrl.num_knots > 0 and mj_model.nu > 0:
        initial_knots = np.zeros((ctrl.num_knots, mj_model.nu))
    
    # Create episode data structure
    episode_data = {
        'metadata': {
            'sequence': args.sequence,
            'control_frequency': args.frequency,
            'simulation_timestep': mj_model.opt.timestep,
            'plan_horizon': args.plan_horizon,
            'num_knots': ctrl.num_knots,
            'ctrl_steps': ctrl.ctrl_steps,
            'nq': mj_model.nq,
            'nu': mj_model.nu,
            'reference_fps': task.reference_fps,
            'perturbations': {
                'joint_noise_std': args.joint_noise_std,
                'base_pos_noise_std': args.base_pos_noise_std,
                'base_orient_noise_std': args.base_orient_noise_std,
            },
        },
        'initial_state': {
            'qpos': mj_data.qpos.copy(),
            'qvel': mj_data.qvel.copy(),
            'time': mj_data.time,
        },
        'trajectory': [],
        'controls': [],
        'knots': [],
        'planning_times': [],
        'timestamps': [],
    }
    
    # Run episode
    run_episode_with_logging(
        ctrl,
        mj_model,
        mj_data,
        frequency=args.frequency,
        initial_knots=initial_knots,
        reference=reference,
        reference_fps=task.reference_fps,
        episode_data=episode_data,
        max_duration=args.duration,
        fall_detection=True,
        show_viewer=not args.no_viewer,
        args=args
    )

if __name__ == "__main__":
    main()