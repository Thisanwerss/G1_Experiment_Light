#!/usr/bin/env python3
"""
Run an interactive simulation of the humanoid standing/balancing task.

This example demonstrates simplified full-body qpos tracking without 
complex object manipulation or detailed end-effector constraints.
"""

import argparse
import numpy as np
import mujoco
from hydrax.algs import CEM
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.humanoid_standonly import HumanoidStand


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run an interactive simulation of standing/balancing with the G1."
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="simple_stand",
        choices=["simple_stand", "balance"],
        help="Reference sequence type: 'simple_stand' for static pose, 'balance' for dynamic balancing.",
    )
    parser.add_argument(
        "--show_reference",
        action="store_true",
        help="Show the reference trajectory as a 'ghost' in the simulation.",
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
        "--record_video",
        action="store_true",
        help="Record video of the simulation.",
    )
    parser.add_argument(
        "--show_traces",
        action="store_true",
        help="Show trajectory traces in the simulation.",
    )
    
    args = parser.parse_args()

    # Define the task (cost and dynamics)
    print(f"Creating HumanoidStand task with sequence: {args.sequence}")
    task = HumanoidStand(reference_sequence=args.sequence)
    
    print(f"Model has {task.mj_model.nq} DOFs and {task.mj_model.nu} actuators")
    print(f"Reference trajectory has {len(task.reference)} frames at {task.reference_fps} fps")
    print(f"Reference pose shape: {task.reference.shape}")
    print(f"Total reference duration: {len(task.reference) / task.reference_fps:.2f} seconds")

    # Set up the controller
    print(f"Setting up CEM controller with {args.num_samples} samples, {args.num_elites} elites")
    ctrl = CEM(
        task,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        sigma_start=0.3,        # Slightly lower for standing task
        sigma_min=0.05,         # Lower minimum for more precise control
        explore_fraction=0.3,   # Less exploration needed for standing
        plan_horizon=args.plan_horizon,
        spline_type="zero",
        num_knots=4,           # Fewer knots needed for standing
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
    
    # Check compatibility and set initial pose
    if task.reference.shape[1] == mj_model.nq:
        mj_data.qpos[:] = task.reference[0]  # Start from first reference pose
        print("Set initial pose from reference")
    else:
        print(f"Warning: Reference pose size ({task.reference.shape[1]}) != model nq ({mj_model.nq})")
        print("Using default initial pose")
        # Set a basic standing pose manually
        mj_data.qpos[2] = 0.79  # Set base height
        if mj_model.nq > 6:  # If we have quaternion
            mj_data.qpos[3] = 1.0  # w component of quaternion
    
    # Initialize control knots - for standing task, create zero control inputs
    initial_knots = None
    if ctrl.num_knots > 0 and mj_model.nu > 0:
        # Create zero control inputs with correct dimensions
        initial_knots = np.zeros((ctrl.num_knots, mj_model.nu))
        print(f"Using zero initial knots with shape: {initial_knots.shape}")
        print(f"Model expects nu={mj_model.nu} control inputs per knot")
    
    # Set up reference visualization
    reference = task.reference if args.show_reference else None

    print(f"Starting simulation at {args.frequency} Hz control frequency")
    print("Controls:")
    print("  - Space: Pause/unpause")
    print("  - Mouse: Rotate view") 
    print("  - Scroll: Zoom")
    print("  - Ctrl+Mouse: Pan")
    
    # Run the interactive simulation
    run_interactive(
        ctrl,
        mj_model,
        mj_data,
        frequency=args.frequency,
        show_traces=args.show_traces,
        reference=reference,
        reference_fps=task.reference_fps,
        initial_knots=initial_knots,
        record_video=args.record_video,
    )


if __name__ == "__main__":
    main()