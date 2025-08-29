#!/usr/bin/env python3
"""
Collect episode data from the humanoid simulation for replay.
Records states, controls, knots, and timing information.
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
from typing import Sequence
from datetime import datetime
from hydrax.algs import CEM
from hydrax.tasks.humanoid_standonly import HumanoidStand


def run_interactive_with_logging(
    controller,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    frequency: float,
    initial_knots: jax.Array = None,
    fixed_camera_id: int = None,
    show_traces: bool = True,
    max_traces: int = 5,
    trace_width: float = 5.0,
    trace_color: Sequence = [1.0, 1.0, 1.0, 0.1],
    reference: np.ndarray = None,
    reference_fps: float = 30.0,
    episode_data: dict = None,
    max_duration: float = None,
) -> None:
    """Modified version of run_interactive that logs episode data."""
    
    # Report the planning horizon in seconds for debugging
    print(
        f"Planning with {controller.ctrl_steps} steps "
        f"over a {controller.plan_horizon} second horizon "
        f"with {controller.num_knots} knots."
    )

    # Figure out how many sim steps to run before replanning
    replan_period = 1.0 / frequency
    sim_steps_per_replan = int(replan_period / mj_model.opt.timestep)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    actual_frequency = 1.0 / step_dt
    print(
        f"Planning at {actual_frequency} Hz, "
        f"simulating at {1.0 / mj_model.opt.timestep} Hz"
    )

    # Initialize the controller
    mjx_data = mjx.put_data(mj_model, mj_data)
    mjx_data = mjx_data.replace(
        mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat
    )
    policy_params = controller.init_params(initial_knots=initial_knots)
    jit_optimize = jax.jit(controller.optimize)
    jit_interp_func = jax.jit(controller.interp_func)

    # Warm-up the controller
    print("Jitting the controller...")
    st = time.time()
    policy_params, rollouts = jit_optimize(mjx_data, policy_params)
    policy_params, rollouts = jit_optimize(mjx_data, policy_params)

    tq = jnp.arange(0, sim_steps_per_replan) * mj_model.opt.timestep
    tk = policy_params.tk
    knots = policy_params.mean[None, ...]
    _ = jit_interp_func(tq, tk, knots)
    _ = jit_interp_func(tq, tk, knots)
    print(f"Time to jit: {time.time() - st:.3f} seconds")
    num_traces = min(rollouts.controls.shape[1], max_traces)

    # Ghost reference setup
    if reference is not None:
        ref_data = mujoco.MjData(mj_model)
        assert reference.shape[1] == mj_model.nq
        ref_data.qpos[:] = reference[0, :]
        mujoco.mj_forward(mj_model, ref_data)

        vopt = mujoco.MjvOption()
        vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        pert = mujoco.MjvPerturb()
        catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC

    # Track start time for duration limit
    episode_start_time = time.time()

    # Start the simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        if fixed_camera_id is not None:
            viewer.cam.fixedcamid = fixed_camera_id
            viewer.cam.type = 2

        # Set up rollout traces
        if show_traces:
            num_trace_sites = len(controller.task.trace_site_ids)
            for i in range(
                num_trace_sites * num_traces * controller.ctrl_steps
            ):
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_LINE,
                    size=np.zeros(3),
                    pos=np.zeros(3),
                    mat=np.eye(3).flatten(),
                    rgba=np.array(trace_color),
                )
                viewer.user_scn.ngeom += 1

        # Add geometry for the ghost reference
        if reference is not None:
            mujoco.mjv_addGeoms(
                mj_model, ref_data, vopt, pert, catmask, viewer.user_scn
            )

        while viewer.is_running():
            # Check duration limit
            if max_duration is not None:
                if time.time() - episode_start_time > max_duration:
                    print(f"\nReached maximum duration of {max_duration} seconds.")
                    break

            start_time = time.time()

            # Set the start state for the controller
            mjx_data = mjx_data.replace(
                qpos=jnp.array(mj_data.qpos),
                qvel=jnp.array(mj_data.qvel),
                mocap_pos=jnp.array(mj_data.mocap_pos),
                mocap_quat=jnp.array(mj_data.mocap_quat),
                time=mj_data.time,
            )

            # Do a replanning step
            plan_start = time.time()
            policy_params, rollouts = jit_optimize(mjx_data, policy_params)
            plan_time = time.time() - plan_start

            # Log knot data
            if episode_data is not None:
                knot_info = {
                    'knots': np.array(policy_params.mean),  # Shape: (num_knots, nu)
                    'tk': np.array(policy_params.tk),       # Knot times
                    'timestamp': mj_data.time,
                    'planning_time': plan_time,
                }
                episode_data['knots'].append(knot_info)
                episode_data['planning_times'].append(plan_time)

            # Visualize the rollouts
            if show_traces:
                ii = 0
                for k in range(num_trace_sites):
                    for i in range(num_traces):
                        for j in range(controller.ctrl_steps):
                            mujoco.mjv_connector(
                                viewer.user_scn.geoms[ii],
                                mujoco.mjtGeom.mjGEOM_LINE,
                                trace_width,
                                rollouts.trace_sites[i, j, k],
                                rollouts.trace_sites[i, j + 1, k],
                            )
                            ii += 1

            # Update the ghost reference
            if reference is not None:
                t_ref = mj_data.time * reference_fps
                i_ref = int(t_ref)
                i_ref = min(i_ref, reference.shape[0] - 1)
                ref_data.qpos[:] = reference[i_ref]
                mujoco.mj_forward(mj_model, ref_data)
                mujoco.mjv_updateScene(
                    mj_model,
                    ref_data,
                    vopt,
                    pert,
                    viewer.cam,
                    catmask,
                    viewer.user_scn,
                )

            # Query the control spline at the sim frequency
            sim_dt = mj_model.opt.timestep
            t_curr = mj_data.time

            tq = jnp.arange(0, sim_steps_per_replan) * sim_dt + t_curr
            tk = policy_params.tk
            knots = policy_params.mean[None, ...]
            us = np.asarray(jit_interp_func(tq, tk, knots))[0]  # (ss, nu)

            # Simulate the system between spline replanning steps
            for i in range(sim_steps_per_replan):
                mj_data.ctrl[:] = np.array(us[i])
                mujoco.mj_step(mj_model, mj_data)

                # Log state and control data
                if episode_data is not None:
                    state_info = {
                        'qpos': mj_data.qpos.copy(),
                        'qvel': mj_data.qvel.copy(),
                        'ctrl': mj_data.ctrl.copy(),
                        'time': mj_data.time,
                        'actuator_force': mj_data.actuator_force.copy(),
                    }
                    episode_data['trajectory'].append(state_info)
                    episode_data['controls'].append(mj_data.ctrl.copy())
                    episode_data['timestamps'].append(mj_data.time)

                viewer.sync()

            # Try to run in roughly realtime
            elapsed = time.time() - start_time
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

            # Print some timing information
            rtr = step_dt / (time.time() - start_time)
            print(
                f"Realtime rate: {rtr:.2f}, plan time: {plan_time:.4f}s, time: {mj_data.time:.2f}s",
                end="\r",
            )

    # Preserve the last printout
    print("")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Collect episode data from humanoid simulation."
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="simple_stand",
        choices=["simple_stand", "balance"],
        help="Reference sequence type.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./episode_data",
        help="Directory to save episode data.",
    )
    parser.add_argument(
        "--episode_name",
        type=str,
        default=None,
        help="Name for the episode file (auto-generated if not provided).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration of episode to record in seconds.",
    )
    parser.add_argument(
        "--show_reference",
        action="store_true",
        help="Show the reference trajectory.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of CEM iterations.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2000,
        help="Number of CEM samples.",
    )
    parser.add_argument(
        "--num_elites",
        type=int,
        default=20,
        help="Number of CEM elite samples.",
    )
    parser.add_argument(
        "--plan_horizon",
        type=float,
        default=0.8,
        help="Planning horizon in seconds.",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=30.0,
        help="Control frequency in Hz.",
    )
    parser.add_argument(
        "--show_traces",
        action="store_true",
        help="Show trajectory traces in the simulation.",
    )
    
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Define the task
    print(f"Creating HumanoidStand task with sequence: {args.sequence}")
    task = HumanoidStand(reference_sequence=args.sequence)
    
    print(f"Model has {task.mj_model.nq} DOFs and {task.mj_model.nu} actuators")

    # Set up the controller
    ctrl = CEM(
        task,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        sigma_start=0.3,
        sigma_min=0.05,
        explore_fraction=0.3,
        plan_horizon=args.plan_horizon,
        spline_type="zero",
        num_knots=4,
        iterations=args.iterations
    )

    # Define the model used for simulation
    mj_model = task.mj_model
    mj_model.opt.timestep = 0.01
    mj_model.opt.iterations = 10
    mj_model.opt.ls_iterations = 50
    mj_model.opt.noslip_iterations = 2
    mj_model.opt.o_solimp = [0.0, 0.95, 0.01, 0.5, 2]
    mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE

    # Set the initial state
    mj_data = mujoco.MjData(mj_model)
    
    # Set initial pose
    if task.reference.shape[1] == mj_model.nq:
        mj_data.qpos[:] = task.reference[0]
        print("Set initial pose from reference")
    else:
        print(f"Warning: Reference pose size mismatch")
        mj_data.qpos[2] = 0.79
        if mj_model.nq > 6:
            mj_data.qpos[3] = 1.0
    
    # Initialize control knots
    initial_knots = None
    if ctrl.num_knots > 0 and mj_model.nu > 0:
        initial_knots = np.zeros((ctrl.num_knots, mj_model.nu))
    
    # Set up reference visualization
    reference = task.reference if args.show_reference else None

    # Generate episode name if not provided
    if args.episode_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.episode_name = f"episode_{args.sequence}_{timestamp}"

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
            'initial_knots_shape': initial_knots.shape if initial_knots is not None else None,
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

    print(f"Starting data collection for {args.duration} seconds")
    print(f"Data will be saved to: {os.path.join(args.output_dir, args.episode_name + '.pkl')}")
    
    # Run the interactive simulation with logging
    run_interactive_with_logging(
        ctrl,
        mj_model,
        mj_data,
        frequency=args.frequency,
        show_traces=args.show_traces,
        reference=reference,
        reference_fps=task.reference_fps,
        initial_knots=initial_knots,
        episode_data=episode_data,
        max_duration=args.duration,
    )

    # Save the episode data
    output_path = os.path.join(args.output_dir, args.episode_name + '.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(episode_data, f)
    
    print(f"\nEpisode data saved to: {output_path}")
    print(f"Total timesteps recorded: {len(episode_data['trajectory'])}")
    print(f"Total planning steps: {len(episode_data['knots'])}")
    if episode_data['trajectory']:
        print(f"Episode duration: {episode_data['trajectory'][-1]['time']:.2f} seconds")


if __name__ == "__main__":
    main()