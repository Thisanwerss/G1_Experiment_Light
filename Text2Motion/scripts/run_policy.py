#!/usr/bin/env python3
"""
Run NN policy that predicts knots.
Run:
    python run_policy.py --model_path overfit_model.pth --show_reference
"""
import argparse
import os
import time
import pickle

import numpy as np
import torch
import torch.nn as nn
import mujoco
import mujoco.viewer
import jax
import jax.numpy as jnp
from mujoco import mjx
import pytorch_lightning as pl
from scipy.spatial.transform import Rotation as R


from hydrax.algs import CEM
from hydrax.tasks.humanoid_standonly import HumanoidStand
from replay_episode import replay_knots

# Define the MLP regressor model
class MLPRegressor(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim=1, learning_rate=1e-3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.BatchNorm1d(hidden_dim3),
            nn.ReLU(),
            
            nn.Linear(hidden_dim3, output_dim)
        )

    def forward(self, x):
        return self.model(x)


def load_model(model_path, device):
    """Load the model from disk."""
    checkpoint = torch.load(model_path, map_location=device)
    
    net = MLPRegressor(95, 512, 512, 512, 164)
    net.load_state_dict(checkpoint['state_dict'])
    net.to(device).eval()

    print("Model loaded successfully")
    
    return net
    

def predict_knots(net, qpos, qvel, device):
    """
    Run NN knots prediction
    """
    with torch.no_grad():
        net.eval()  # Ensure network is in evaluation mode

        # Network inference
        state = np.concatenate([qpos, qvel], axis=0).astype(np.float32)
        state = torch.from_numpy(state).to(device).float().unsqueeze(0)

        knots = net(state)
        knots = knots.squeeze(0).cpu().numpy().reshape(4, 41)

        return knots
            

def main():
    parser = argparse.ArgumentParser(
        description="Replay episode using NN-predicted knots instead of recorded ones."
    )
    parser.add_argument(
        "--model_path", type=str, required=False,default="nn_ckpt/model-epoch=72-val_loss=0.003503.ckpt",
        help="Path to the overfit model checkpoint (overfit_model.pth)."
    )
    parser.add_argument(
        "--show_reference", action="store_true",
        help="Show the reference trajectory as a transparent ghost."
    )
    parser.add_argument(
        "--hybrid", action="store_true",
        help="Use the network to warm-start the sampling approach"
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=50.0,
        help="Planning frequency.",
    )

    # CEM parameters
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Number of rollouts to perform.",
    )
    parser.add_argument(
        "--num_elites",
        type=int,
        default=20,
        help="Number of elites.",
    )
    parser.add_argument(
        "--plan_horizon",
        type=float,
        default=0.5,
        help="Horizon length.",
    )
    parser.add_argument(
        "--num_knots",
        type=int,
        default=4,
        help="Number of spline knots.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
        help="Sampling iterations to perform.",
    )
    parser.add_argument(
        "--alignment_mode",
        type=str,
        default="reference",
        choices=["reference", "state_to_origin", "none"],
        help="How to handle initial state offset for OOD testing. "
             "'reference': shifts the reference trajectory to the robot. "
             "'state_to_origin': shifts the robot state to the reference origin for NN input. "
             "'none': do nothing, run raw OOD."
    )

    args = parser.parse_args()

    # Setup device and load network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    net = load_model(args.model_path, device)

    # Setup Mujoco model and data
    task = HumanoidStand()
    mj_model = task.mj_model
    mj_model.opt.timestep = 0.01
    mj_model.opt.iterations = 10
    mj_model.opt.ls_iterations = 50
    mj_model.opt.noslip_iterations = 2
    mj_model.opt.o_solimp = [0.0, 0.95, 0.01, 0.5, 2]
    mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE
    mj_data = mujoco.MjData(mj_model)

    #OOD Test
    offset_xy = np.array([6.0, -6.0])
    print(f"🔬 OOD测试：施加初始位置偏移 {offset_xy}")
    mj_data.qpos[:2] += offset_xy

    # 增加基座的yaw偏转，使其“斜着站”
    yaw_offset_deg = 30.0
    yaw_offset_rad = np.deg2rad(yaw_offset_deg)
    print(f"🔬 OOD测试：施加初始Yaw偏转 {yaw_offset_deg}°")
    
    # 获取当前姿态四元数
    initial_quat_wxyz = mj_data.qpos[3:7]
    # Scipy使用 [x, y, z, w] 格式
    initial_rotation = R.from_quat([initial_quat_wxyz[1], initial_quat_wxyz[2], initial_quat_wxyz[3], initial_quat_wxyz[0]])
    
    # 创建一个代表yaw偏转的旋转
    yaw_rotation = R.from_euler('z', yaw_offset_rad)
    
    # 将yaw偏转应用到当前姿态
    new_rotation = yaw_rotation * initial_rotation
    
    # 转换回MuJoCo使用的 [w, x, y, z] 格式
    new_quat_xyzw = new_rotation.as_quat()
    mj_data.qpos[3:7] = np.array([new_quat_xyzw[3], new_quat_xyzw[0], new_quat_xyzw[1], new_quat_xyzw[2]])

    # Handle alignment based on chosen mode
    if args.alignment_mode == "reference":
        print("🚀 Alignment Mode: REFERENCE. Shifting reference trajectory to robot's initial state.")
        task.create_aligned_static_reference(mj_data.qpos)
    elif args.alignment_mode == "state_to_origin":
        print("🚀 Alignment Mode: STATE_TO_ORIGIN. Storing offset to align state for NN input.")
        task.calculate_and_store_initial_offset(mj_data.qpos)
    elif args.alignment_mode == "none":
        print("🚀 Alignment Mode: NONE. No alignment will be performed.")
    else:
        raise ValueError(f"Invalid alignment mode: {args.alignment_mode}")

    reference = task.reference if args.show_reference else None

    # Instantiate CEM controller (only for interpolation)
    ctrl = CEM(
        task,
        num_samples=args.num_samples, 
        num_elites=args.num_elites,
        sigma_start=0.3, 
        sigma_min=0.05,
        explore_fraction=0.3,
        plan_horizon=args.plan_horizon,
        spline_type="zero",
        num_knots=args.num_knots,
        iterations=args.iterations
    )
    print("Starting simulation")

    # Figure out timing
    replan_period = 1.0 / args.frequency
    sim_steps_per_replan = int(replan_period / mj_model.opt.timestep)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    
    # Initialize the controller
    mjx_data = mjx.put_data(mj_model, mj_data)
    mjx_data = mjx_data.replace(
        mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat
    )

    # 在 state_to_origin 模式下，初次暖启动要用对齐后的输入
    if args.alignment_mode == "state_to_origin":
        qpos0, qvel0 = task.align_state_to_origin(mj_data.qpos, mj_data.qvel)
    else:
        qpos0, qvel0 = mj_data.qpos, mj_data.qvel
    initial_knots = predict_knots(net, qpos0, qvel0, device) if args.hybrid else None



    initial_knots = predict_knots(net, mj_data.qpos, mj_data.qvel, device) if args.hybrid else None
    policy_params = ctrl.init_params(initial_knots=initial_knots)
    jit_optimize = jax.jit(ctrl.optimize)
    jit_interp_func = jax.jit(ctrl.interp_func)

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

    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

    start_time = time.time()
    
    # Add ghost reference geometry
    if reference is not None:
        mujoco.mjv_addGeoms(mj_model, ref_data, vopt, pert, catmask, viewer.user_scn)

    while True:
        # Check viewer status
        if viewer is not None and not viewer.is_running():
            break
        
        # Update controller state
        mjx_data = mjx_data.replace(
            qpos=jnp.array(mj_data.qpos),
            qvel=jnp.array(mj_data.qvel),
            mocap_pos=jnp.array(mj_data.mocap_pos),
            mocap_quat=jnp.array(mj_data.mocap_quat),
            time=mj_data.time,
        )

        # --- State Alignment for NN ---
        qpos_for_nn = mj_data.qpos
        qvel_for_nn = mj_data.qvel
        if args.alignment_mode == "state_to_origin":
            qpos_for_nn, qvel_for_nn = task.align_state_to_origin(mj_data.qpos, mj_data.qvel)
            # 同样使用对齐后的状态更新CEM控制器的输入，确保优化器和NN在同一坐标系下工作
            mjx_data = mjx_data.replace(
                qpos=jnp.array(qpos_for_nn),
                qvel=jnp.array(qvel_for_nn)
            )

        # Planning step
        plan_time = None
        if args.hybrid:
            plan_start = time.time()
            new_knots = predict_knots(net, qpos_for_nn, qvel_for_nn, device)
            #policy_params.replace(mean=new_knots)
            policy_params=policy_params.replace(mean=jnp.asarray(new_knots))
            policy_params, rollouts = jit_optimize(mjx_data, policy_params)
            plan_time = time.time() - plan_start
        else:
            plan_start = time.time()
            new_knots = predict_knots(net, qpos_for_nn, qvel_for_nn, device)
            policy_params.replace(mean=new_knots)
            plan_time = time.time() - plan_start

        print(f"Plan time took {plan_time}")

        # Update ghost reference
        if viewer is not None and reference is not None:
            t_ref = mj_data.time * 30
            i_ref = min(int(t_ref), reference.shape[0] - 1)
            ref_data.qpos[:] = reference[i_ref]
            mujoco.mj_forward(mj_model, ref_data)
            mujoco.mjv_updateScene(
                mj_model, ref_data, vopt, pert, viewer.cam, catmask, viewer.user_scn
            )

        # Interpolate controls
        sim_dt = mj_model.opt.timestep
        t_curr = mj_data.time

        # Querying parameters
        us = None
        if args.hybrid:
            tq = jnp.arange(0, sim_steps_per_replan) * sim_dt + t_curr
            tk = policy_params.tk
            knots = policy_params.mean[None, ...]
            us = np.asarray(jit_interp_func(tq, tk, knots))[0]
        else:
            tq = jnp.arange(0, sim_steps_per_replan) * sim_dt
            tk = jnp.linspace(0, 0.5, 4)
            knots = policy_params.mean[None, ...]
            us = np.asarray(jit_interp_func(tq, tk, knots))[0]

        # Simulate
        for i in range(sim_steps_per_replan):
            print(us[i])
            # The control signal is a set of target joint angles from the controller.
            # These are in the robot's own body frame and are independent of the world
            # coordinate system. They do not need to be transformed back as they are
            # relative to the robot's own linkage, not its world position.
            control_signal = np.array(us[i])

            mj_data.ctrl[:] = control_signal
            mujoco.mj_step(mj_model, mj_data)

            if viewer is not None:
                viewer.sync()

        # Timing control
        elapsed = time.time() - start_time
        if elapsed < step_dt:
            time.sleep(step_dt - elapsed)

if __name__ == "__main__":
    main()