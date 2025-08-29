#!/usr/bin/env python3
"""
Replay recorded episode data from the humanoid simulation.
Supports both state replay (exact) and knot replay (using CEM controller logic).
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
from hydrax.algs import CEM
from hydrax.tasks.humanoid_standonly import HumanoidStand


def replay_states(episode_data, mj_model, mj_data, show_reference=False, reference=None):
    """Replay exact states from recorded trajectory."""
    
    trajectory = episode_data['trajectory']
    metadata = episode_data['metadata']
    
    # Set initial state
    initial_state = episode_data['initial_state']
    mj_data.qpos[:] = initial_state['qpos']
    mj_data.qvel[:] = initial_state['qvel']
    mj_data.time = initial_state['time']
    
    # Ghost reference setup
    if show_reference and reference is not None:
        ref_data = mujoco.MjData(mj_model)
        ref_data.qpos[:] = reference[0, :]
        mujoco.mj_forward(mj_model, ref_data)
        
        vopt = mujoco.MjvOption()
        vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        pert = mujoco.MjvPerturb()
        catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC
    
    print(f"Replaying {len(trajectory)} timesteps...")
    
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # Add geometry for ghost reference if needed
        if show_reference and reference is not None:
            mujoco.mjv_addGeoms(mj_model, ref_data, vopt, pert, catmask, viewer.user_scn)
        
        start_time = time.time()
        
        for i, state in enumerate(trajectory):
            # Set state directly
            mj_data.qpos[:] = state['qpos']
            mj_data.qvel[:] = state['qvel']
            mj_data.time = state['time']
            
            # Set control if available
            if 'ctrl' in state:
                mj_data.ctrl[:] = state['ctrl']
            
            # Forward dynamics to update visualization
            mujoco.mj_forward(mj_model, mj_data)
            
            # Update ghost reference
            if show_reference and reference is not None:
                t_ref = mj_data.time * metadata['reference_fps']
                i_ref = int(t_ref)
                i_ref = min(i_ref, reference.shape[0] - 1)
                ref_data.qpos[:] = reference[i_ref]
                mujoco.mj_forward(mj_model, ref_data)
                mujoco.mjv_updateScene(
                    mj_model, ref_data, vopt, pert, viewer.cam, catmask, viewer.user_scn
                )
            
            viewer.sync()
            
            # Try to maintain original timing
            if i < len(trajectory) - 1:
                dt = trajectory[i + 1]['time'] - state['time']
                elapsed = time.time() - start_time
                target_time = state['time'] - initial_state['time']
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)
            
            if not viewer.is_running():
                break
            
            # Print progress
            if i % 100 == 0:
                print(f"Timestep {i}/{len(trajectory)}, Time: {state['time']:.2f}s", end="\r")
    
    print(f"\nReplay completed. Total time: {trajectory[-1]['time']:.2f}s")


def replay_knots(episode_data, controller, mj_model, mj_data, show_reference=False, reference=None):
    """Replay using recorded knots with CEM controller interpolation logic."""
    
    # Figure out timing
    replan_period = 1.0 / 50
    sim_steps_per_replan = int(replan_period / mj_model.opt.timestep)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    
    # Initialize the controller
    mjx_data = mjx.put_data(mj_model, mj_data)
    mjx_data = mjx_data.replace(
        mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat
    )
    policy_params = controller.init_params(initial_knots=None)
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

    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
    
    # Add ghost reference geometry
    if reference is not None:
        mujoco.mjv_addGeoms(mj_model, ref_data, vopt, pert, catmask, viewer.user_scn)

    while True:
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


def main():
    parser = argparse.ArgumentParser(
        description="Replay recorded episode data from humanoid simulation."
    )
    parser.add_argument(
        "episode_path",
        type=str,
        help="Path to the episode pickle file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="states",
        choices=["states", "knots"],
        help="Replay mode: 'states' for exact state replay, 'knots' for control generation from knots.",
    )
    parser.add_argument(
        "--show_reference",
        action="store_true",
        help="Show the reference trajectory as a ghost.",
    )
    parser.add_argument(
        "--show_traces",
        action="store_true",
        help="Show trajectory traces (only for knot replay).",
    )
    
    args = parser.parse_args()
    
    # Load episode data
    if not os.path.exists(args.episode_path):
        raise FileNotFoundError(f"Episode file not found: {args.episode_path}")
    
    print(f"Loading episode data from: {args.episode_path}")
    with open(args.episode_path, 'rb') as f:
        episode_data = pickle.load(f)
    
    metadata = episode_data['metadata']
    print(f"\nEpisode metadata:")
    print(f"  Sequence: {metadata['sequence']}")
    print(f"  Control frequency: {metadata['control_frequency']} Hz")
    print(f"  Simulation timestep: {metadata['simulation_timestep']}s")
    print(f"  Model DOFs: {metadata['nq']}")
    print(f"  Actuators: {metadata['nu']}")
    print(f"  Trajectory length: {len(episode_data['trajectory'])} timesteps")
    print(f"  Planning steps: {len(episode_data['knots'])}")
    
    # Create task and model
    task = HumanoidStand(reference_sequence=metadata['sequence'])
    
    # Set up model parameters (same as in collection)
    mj_model = task.mj_model
    mj_model.opt.timestep = metadata['simulation_timestep']
    mj_model.opt.iterations = 10
    mj_model.opt.ls_iterations = 50
    mj_model.opt.noslip_iterations = 2
    mj_model.opt.o_solimp = [0.0, 0.95, 0.01, 0.5, 2]
    mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE
    
    # Create data object
    mj_data = mujoco.MjData(mj_model)
    
    # Get reference if needed
    reference = task.reference if args.show_reference else None
    
    if args.mode == "states":
        # Exact state replay
        replay_states(episode_data, mj_model, mj_data, args.show_reference, reference)
    else:
        # Knot-based replay - need to create controller
        print("\nSetting up CEM controller for knot replay...")
        ctrl = CEM(
            task,
            num_samples=2000,  # These don't matter for replay
            num_elites=20,
            sigma_start=0.3,
            sigma_min=0.05,
            explore_fraction=0.3,
            plan_horizon=metadata['plan_horizon'],
            spline_type="zero",
            num_knots=metadata['num_knots'],
            iterations=1
        )
        
        replay_knots(episode_data, ctrl, mj_model, mj_data, args.show_reference, reference)


if __name__ == "__main__":
    main()