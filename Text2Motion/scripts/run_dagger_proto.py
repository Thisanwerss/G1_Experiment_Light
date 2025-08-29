#!/usr/bin/env python3
"""
DAgger-style interactive simulation that can run in two modes:
1. NN-predicted knots mode: Use neural network to predict control knots
2. CEM-sampled knots mode: Use CEM algorithm to sample control knots

The initial state is randomly picked from episode data samples.
"""

import argparse
import os
import time
import pickle
import glob
import numpy as np
import torch
import mujoco
import mujoco.viewer
import jax
import jax.numpy as jnp
from mujoco import mjx
import random
from collections import deque
from datetime import datetime
import pytorch_lightning as pl

from hydrax.algs import CEM
from hydrax.tasks.humanoid_standonly import HumanoidStand
from learning.train_overfit import ConfigurableOverfitNet
import torch.nn as nn
import math

# Try to import TransformerController
try:
    from learning.train import TransformerController
except ImportError:
    TransformerController = None
    print("⚠️ TransformerController not available in learning.train")

# For backward compatibility, define ConfigurableNet
class ConfigurableNet(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden_dims, dropout_rate=0.05):
        super().__init__()
        
        layers = []
        prev_dim = inp_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, out_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Residual connection if dimensions match
        self.use_residual = (inp_dim == out_dim)
        if not self.use_residual and inp_dim < out_dim:
            self.input_projection = nn.Linear(inp_dim, out_dim)
            self.use_projection = True
        else:
            self.use_projection = False
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x):
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        out = self.net(x)
        
        # Residual connection
        if self.use_residual:
            out = out + x
        elif self.use_projection:
            out = out + self.input_projection(x)
        
        return out.squeeze(0) if batch_size == 1 else out

# Compatibility for older network versions
class OverfitNet(torch.nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(inp_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, out_dim),
        )

    def forward(self, x):
        return self.net(x)
    

class MLPRegressor(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim=1, learning_rate=1e-3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.BatchNorm1d(hidden_dim3),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim3, output_dim)
        )

    def forward(self, x):
        return self.model(x)


def load_model(model_path, device):
    """Load the neural network model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check for model architecture type
    has_normalization = 'X_mean' in checkpoint
    is_full_dataset_model = 'train_metrics' in checkpoint or 'val_metrics' in checkpoint
    is_transformer_model = 'model_config' in checkpoint and checkpoint.get('model_config', {}).get('architecture') == 'transformer'
    
    # If checkpoint file doesn't have model_config, try loading from model_config.json in same directory
    external_model_config = None
    if not is_transformer_model and is_full_dataset_model:
        config_path = os.path.join(os.path.dirname(model_path), 'model_config.json')
        if os.path.exists(config_path):
            import json
            try:
                with open(config_path, 'r') as f:
                    external_model_config = json.load(f)
                if external_model_config.get('architecture') == 'transformer':
                    is_transformer_model = True
                    print(f"📁 Loading Transformer config from external config file: {config_path}")
            except Exception as e:
                print(f"⚠️ Unable to load external config file: {e}")
    
    # For state dict detection, if transformer-related keys are found, also consider it a transformer model
    if not is_transformer_model and 'model_state_dict' in checkpoint:
        state_dict_keys = list(checkpoint['model_state_dict'].keys())
        has_transformer_keys = any('transformer_encoder' in key or 'input_embedding' in key or 'output_decoder' in key 
                                 for key in state_dict_keys)
        if has_transformer_keys:
            is_transformer_model = True
            print("🔍 Detected Transformer model based on state dict keys")
    
    if is_transformer_model:
        print("🔄 Loading Transformer model (trained with transformer-based train.py)")
        
        # Prioritize config from checkpoint, otherwise use external config
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
        elif external_model_config is not None:
            model_config = external_model_config
        else:
            raise ValueError("Unable to find Transformer model configuration")
        
        # Create Transformer model
        if TransformerController is None:
            raise ImportError("TransformerController not available. Please check learning.train module.")
        
        net = TransformerController(
            state_dim=model_config['state_dim'],
            action_dim=model_config['action_dim'],
            model_dim=model_config['model_dim'],
            num_heads=model_config['num_heads'],
            num_layers=model_config['num_layers'],
            feedforward_dim=model_config['feedforward_dim'],
            sequence_length=model_config['sequence_length'],
            dropout=model_config['dropout']
        )
        
        print(f"  - Using TransformerController")
        print(f"  - State dimension: {model_config['state_dim']}")
        print(f"  - Action dimension: {model_config['action_dim']}")
        print(f"  - Sequence length: {model_config['sequence_length']}")
        
        # Extract normalization parameters from Transformer model
        if 'stats_dict' in checkpoint:
            normalization_params = {
                'X_mean': np.concatenate([
                    checkpoint['stats_dict']['qpos_mean'],
                    checkpoint['stats_dict']['qvel_mean']
                ]),
                'X_std': np.concatenate([
                    checkpoint['stats_dict']['qpos_std'],
                    checkpoint['stats_dict']['qvel_std']
                ]),
                'Y_mean': checkpoint['stats_dict']['action_mean'],
                'Y_std': checkpoint['stats_dict']['action_std']
            }
        else:
            # Try loading from external normalization stats file
            stats_path = os.path.join(os.path.dirname(model_path), 'normalization_stats.json')
            if os.path.exists(stats_path):
                try:
                    import json
                    with open(stats_path, 'r') as f:
                        stats_dict = json.load(f)
                    # Convert to numpy arrays
                    for key, value in stats_dict.items():
                        if isinstance(value, list):
                            stats_dict[key] = np.array(value)
                    
                    normalization_params = {
                        'X_mean': np.concatenate([
                            stats_dict['qpos_mean'],
                            stats_dict['qvel_mean']
                        ]),
                        'X_std': np.concatenate([
                            stats_dict['qpos_std'],
                            stats_dict['qvel_std']
                        ]),
                        'Y_mean': stats_dict['action_mean'],
                        'Y_std': stats_dict['action_std']
                    }
                    print(f"📁 Loading normalization parameters from external stats file: {stats_path}")
                except Exception as e:
                    print(f"  ⚠️ Unable to load external normalization stats file: {e}")
                    normalization_params = None
            else:
                print("  ⚠️ No normalization parameters found in Transformer model")
                normalization_params = None
        
    elif is_full_dataset_model:
        print("🔄 Loading full dataset model (trained with train.py)")
        if 'hidden_dims' in checkpoint:
            hidden_dims = checkpoint['hidden_dims']
            net = ConfigurableNet(
                checkpoint['input_dim'], 
                checkpoint['output_dim'], 
                hidden_dims, 
                dropout_rate=checkpoint.get('config_used', {}).get('dropout', 0.05)
            )
            print(f"  - Using ConfigurableNet with architecture: {hidden_dims}")
        else:
            print("  - No hidden_dims found, using default network structure")
            net = OverfitNet(checkpoint.get('input_dim', checkpoint.get('inp_dim', 89)), 
                           checkpoint.get('output_dim', checkpoint.get('out_dim', 41)))
            print(f"  - Using OverfitNet as fallback")
            
        # Extract normalization parameters from train.py model
        if 'stats_dict' in checkpoint:
            if 'knots_mean' in checkpoint['stats_dict']:
                y_mean_key = 'knots_mean'
                y_std_key = 'knots_std'
            else:
                y_mean_key = 'action_mean'
                y_std_key = 'action_std'
            
            normalization_params = {
                'X_mean': np.concatenate([
                    checkpoint['stats_dict']['qpos_mean'],
                    checkpoint['stats_dict']['qvel_mean']
                ]),
                'X_std': np.concatenate([
                    checkpoint['stats_dict']['qpos_std'],
                    checkpoint['stats_dict']['qvel_std']
                ]),
                'Y_mean': checkpoint['stats_dict'][y_mean_key],
                'Y_std': checkpoint['stats_dict'][y_std_key]
            }
        else:
            print("  ⚠️ No normalization parameters found in full dataset model")
            normalization_params = None
    elif has_normalization:
        print("🔄 Loading overfit model with normalization")
        if 'hidden_dims' in checkpoint:
            hidden_dims = checkpoint['hidden_dims']
            net = ConfigurableOverfitNet(
                checkpoint['inp_dim'], 
                checkpoint['out_dim'], 
                hidden_dims, 
                dropout_rate=checkpoint.get('config_used', {}).get('dropout', 0.05)
            )
            print(f"  - Using ConfigurableOverfitNet with architecture: {hidden_dims}")
        else:
            net = OverfitNet(checkpoint['inp_dim'], checkpoint['out_dim'])
            
        normalization_params = {
            'X_mean': checkpoint['X_mean'],
            'X_std': checkpoint['X_std'],
            'Y_mean': checkpoint['Y_mean'],
            'Y_std': checkpoint['Y_std']
        }
    else:
        print("🔄 Loading old simple model (no data normalization)")
        net = OverfitNet(checkpoint['inp_dim'], checkpoint['out_dim'])
        normalization_params = None
    
    net.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint['state_dict'])
    net.to(device).eval()
    
    return net, normalization_params


def predict_knots(net, current_state, device, normalization_params=None, num_knots=None, nu=None, state_history=None):
    """
    Predict knots using neural network
    """
    with torch.no_grad():
        net.eval()
        
        # Check if it's a Transformer model
        is_transformer = TransformerController is not None and isinstance(net, TransformerController)
        
        if is_transformer:
            if state_history is None:
                raise ValueError("Transformer model requires state history buffer")
            
            # Ensure state history length is sufficient
            if len(state_history) < net.sequence_length:
                for _ in range(net.sequence_length - len(state_history)):
                    if len(state_history) > 0:
                        state_history.appendleft(state_history[0])
                    else:
                        zero_state = {
                            'qpos': np.zeros_like(current_state['qpos']),
                            'qvel': np.zeros_like(current_state['qvel'])
                        }
                        state_history.appendleft(zero_state)
            
            # Build state sequence
            state_sequence = []
            for state in list(state_history)[-net.sequence_length:]:
                qpos = state['qpos']
                qvel = state['qvel']
                
                if normalization_params is not None:
                    qpos_norm = (qpos - normalization_params['X_mean'][:qpos.shape[0]]) / normalization_params['X_std'][:qpos.shape[0]]
                    qvel_norm = (qvel - normalization_params['X_mean'][qpos.shape[0]:qpos.shape[0]+qvel.shape[0]]) / normalization_params['X_std'][qpos.shape[0]:qpos.shape[0]+qvel.shape[0]]
                    state_vec = np.concatenate([qpos_norm, qvel_norm]).astype(np.float32)
                else:
                    state_vec = np.concatenate([qpos, qvel]).astype(np.float32)
                
                state_sequence.append(state_vec)
            
            # Convert to tensor [1, sequence_length, state_dim]
            state_tensor = torch.from_numpy(np.stack(state_sequence, axis=0)).unsqueeze(0).to(device)
            
            # Network inference
            y_t = net(state_tensor)[0]  # Remove batch dimension
            
            # Denormalize output
            if normalization_params is not None:
                y_np = y_t.cpu().numpy()
                y_np = y_np * normalization_params['Y_std'] + normalization_params['Y_mean']
                knots = y_np.reshape(num_knots, nu)
            else:
                knots = y_t.cpu().numpy().reshape(num_knots, nu)
            
            return knots
        
        else:
            # MLP model prediction logic
            # Get expected input dimension from model
            if hasattr(net, 'input_dim'):
                expected_input_dim = net.input_dim
            elif hasattr(net, 'inp_dim'):
                expected_input_dim = net.inp_dim
            else:
                for module in net.modules():
                    if isinstance(module, torch.nn.Linear):
                        expected_input_dim = module.in_features
                        break
                else:
                    expected_input_dim = None
            
            x = np.concatenate([current_state['qpos'], current_state['qvel']], axis=0).astype(np.float32)
            
            # Check if input dimensions match
            if expected_input_dim and x.shape[0] != expected_input_dim:
                if x.shape[0] < expected_input_dim:
                    padding = np.zeros(expected_input_dim - x.shape[0], dtype=np.float32)
                    x = np.concatenate([x, padding])
                else:
                    x = x[:expected_input_dim]
            
            # Apply data normalization
            if normalization_params is not None:
                if normalization_params['X_mean'].shape[0] != x.shape[0]:
                    if normalization_params['X_mean'].shape[0] < x.shape[0]:
                        padding_mean = np.zeros(x.shape[0] - normalization_params['X_mean'].shape[0])
                        padding_std = np.ones(x.shape[0] - normalization_params['X_std'].shape[0])
                        normalization_params['X_mean'] = np.concatenate([normalization_params['X_mean'], padding_mean])
                        normalization_params['X_std'] = np.concatenate([normalization_params['X_std'], padding_std])
                    else:
                        normalization_params['X_mean'] = normalization_params['X_mean'][:x.shape[0]]
                        normalization_params['X_std'] = normalization_params['X_std'][:x.shape[0]]
                
                x = (x - normalization_params['X_mean']) / normalization_params['X_std']
            
            x_t = torch.from_numpy(x).to(device).float()
            
            # Network inference
            y_t = net(x_t)
            
            # De-normalize output
            if normalization_params is not None:
                y_np = y_t.cpu().numpy()
                y_np = y_np * normalization_params['Y_std'] + normalization_params['Y_mean']
                knots = y_np.reshape(num_knots, nu)
            else:
                knots = y_t.cpu().numpy().reshape(num_knots, nu)
            
            return knots


def load_random_initial_state(episode_data_dir):
    """
    Load a random initial state from episode data samples
    """
    # Find all episode data files
    episode_files = glob.glob(os.path.join(episode_data_dir, "*.pkl"))
    
    if not episode_files:
        raise FileNotFoundError(f"No episode data files found in {episode_data_dir}")
    
    # Randomly select an episode file
    selected_file = random.choice(episode_files)
    print(f"📂 Selecting initial state from file: {selected_file}")
    
    with open(selected_file, 'rb') as f:
        episode_data = pickle.load(f)
    
    # Randomly select a state from the trajectory as initial state
    trajectory = episode_data['trajectory']
    if not trajectory:
        # If trajectory is empty, use initial state from metadata
        return episode_data['initial_state']
    
    # Randomly select a state from the trajectory
    random_state = random.choice(trajectory)
    initial_state = {
        'qpos': random_state['qpos'],
        'qvel': random_state['qvel'],
        'time': 0.0  # Reset time
    }
    
    print(f"🎲 Randomly selected state at time {random_state['time']:.2f}s as initial state")
    
    return initial_state


def run_dagger_simulation(
    mode: str,
    task: HumanoidStand,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    controller: CEM = None,
    net=None,
    device=None,
    normalization_params=None,
    show_reference=False,
    reference=None,
    frequency=30.0
):
    """
    Run DAgger-style simulation in either NN or CEM mode
    """
    print(f"🚀 Starting DAgger simulation, mode: {mode}")
    
    # Calculate timing parameters
    replan_period = 1.0 / frequency
    sim_timestep = mj_model.opt.timestep
    sim_steps_per_replan = int(replan_period / sim_timestep)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    
    # Check if it's a Transformer model, create state history buffer if needed
    is_transformer = TransformerController is not None and isinstance(net, TransformerController)
    state_history = None
    if mode == "nn" and is_transformer:
        state_history = deque(maxlen=net.sequence_length)
        print(f"ℹ️ Creating state history buffer for Transformer model (max length: {net.sequence_length})")
    
    # Initialize controller for CEM mode
    if mode == "cem":
        mjx_data = mjx.put_data(mj_model, mj_data)
        mjx_data = mjx_data.replace(mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat)
        
        # Create initial knots (zero matrix, start from rest)
        initial_knots = np.zeros((controller.num_knots, mj_model.nu))
        policy_params = controller.init_params(initial_knots=initial_knots)
        
        jit_optimize = jax.jit(controller.optimize)
        jit_interp_func = jax.jit(controller.interp_func)
        
        # Multiple warmup runs to ensure CEM converges to stable distribution
        print("Warming up CEM controller...")
        policy_params, rollouts = jit_optimize(mjx_data, policy_params)
        policy_params, rollouts = jit_optimize(mjx_data, policy_params)
        print(f"Warmup complete, using {controller.num_knots} knots, {controller.iterations} iterations")
    
    # Reference trajectory setup
    if show_reference and reference is not None:
        ref_data = mujoco.MjData(mj_model)
        ref_data.qpos[:] = reference[0, :]
        mujoco.mj_forward(mj_model, ref_data)
        
        vopt = mujoco.MjvOption()
        vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        pert = mujoco.MjvPerturb()
        catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC
    
    print(f"Starting simulation (control frequency: {frequency} Hz)")
    print("Control instructions:")
    print("  - Space: pause/resume")
    print("  - Mouse: rotate view")
    print("  - Mouse wheel: zoom")
    print("  - Ctrl+Mouse: pan")
    
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # Add reference trajectory geometry (if needed)
        if show_reference and reference is not None:
            mujoco.mjv_addGeoms(mj_model, ref_data, vopt, pert, catmask, viewer.user_scn)
        
        step_count = 0
        
        while viewer.is_running():
            start_time = time.time()
            
            # Create current state dictionary
            current_state = {
                'qpos': mj_data.qpos.copy(),
                'qvel': mj_data.qvel.copy(),
                'time': mj_data.time
            }
            
            if mode == "nn":
                # Neural network mode
                # Update state history for Transformer model
                if is_transformer:
                    state_history.append({
                        'qpos': current_state['qpos'].copy(),
                        'qvel': current_state['qvel'].copy()
                    })
                
                # Use neural network to predict knots
                predicted_knots = predict_knots(
                    net, current_state, device, normalization_params,
                    controller.num_knots, mj_model.nu, state_history
                )
                
                # Create time nodes
                tk = jnp.linspace(0.0, replan_period, controller.num_knots)
                knots_jax = jnp.array(predicted_knots)[None, ...]
                tk_jax = jnp.array(tk)
                
                # Interpolate to get control sequence (use global timestamp for continuity)
                t_curr = mj_data.time
                tq = jnp.arange(0, sim_steps_per_replan) * sim_timestep + t_curr
                us = np.asarray(controller.interp_func(tq, tk_jax, knots_jax))[0]
                
            else:
                # CEM mode
                # Set the start state for the controller
                mjx_data = mjx_data.replace(
                    qpos=jnp.array(mj_data.qpos),
                    qvel=jnp.array(mj_data.qvel),
                    time=mj_data.time,
                )
                
                # CEM optimization
                policy_params, rollouts = jit_optimize(mjx_data, policy_params)
                
                # Interpolate to get control sequence (use global timestamp for continuity)
                t_curr = mj_data.time
                tq = jnp.arange(0, sim_steps_per_replan) * sim_timestep + t_curr
                tk = policy_params.tk
                knots = policy_params.mean[None, ...]
                us = np.asarray(jit_interp_func(tq, tk, knots))[0]
            
            # Execute control
            for i in range(sim_steps_per_replan):
                mj_data.ctrl[:] = us[i]
                mujoco.mj_step(mj_model, mj_data)
                
                # Update reference trajectory
                if show_reference and reference is not None:
                    t_ref = mj_data.time * task.reference_fps
                    i_ref = int(t_ref)
                    i_ref = min(i_ref, reference.shape[0] - 1)
                    ref_data.qpos[:] = reference[i_ref]
                    mujoco.mj_forward(mj_model, ref_data)
                    mujoco.mjv_updateScene(
                        mj_model, ref_data, vopt, pert, viewer.cam, catmask, viewer.user_scn
                    )
                
                viewer.sync()
            
            # Maintain real-time speed
            elapsed = time.time() - start_time
            if elapsed < replan_period:
                time.sleep(replan_period - elapsed)
            
            step_count += 1
            if step_count % 10 == 0:
                if mode == "cem":
                    # Simple stability check
                    base_height = mj_data.qpos[2]
                    base_stable = "stable" if base_height > 0.5 else "⚠️unstable"
                    print(f"Time: {mj_data.time:.2f}s, Mode: CEM, Height: {base_height:.3f}m ({base_stable})", end="\r")
                else:
                    print(f"Time: {mj_data.time:.2f}s, Mode: {mode.upper()}", end="\r")


def main():
    parser = argparse.ArgumentParser(
        description="DAgger-style simulation with NN or CEM control modes"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["nn", "cem"],
        default="nn",
        help="Control mode: 'nn' for neural network predicted knots, 'cem' for CEM sampled knots"
    )
    
    # Data and model paths
    parser.add_argument(
        "--episode_data_dir",
        type=str,
        default="./episode_data",
        help="Directory containing episode data files for random initial state selection"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="dagger_rone/best_dagger_model.pth",
        help="Path to neural network model (only used in nn mode)"
    )
    
    # Task configuration
    parser.add_argument(
        "--sequence",
        type=str,
        default="simple_stand",
        choices=["simple_stand", "balance"],
        help="Reference sequence type"
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=30.0,
        help="Control frequency in Hz"
    )
    parser.add_argument(
        "--show_reference",
        action="store_true",
        help="Show reference trajectory"
    )
    
    # CEM parameters (only used in cem mode)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2000,
        help="Number of CEM samples (CEM mode only)"
    )
    parser.add_argument(
        "--num_elites",
        type=int,
        default=20,
        help="Number of CEM elite samples (CEM mode only)"
    )
    parser.add_argument(
        "--plan_horizon",
        type=float,
        default=0.8,
        help="Planning horizon in seconds (CEM mode only)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of CEM optimization iterations (CEM mode only)"
    )
    
    args = parser.parse_args()
    
    print(f"🎯 DAgger simulation startup")
    print(f"📋 Control mode: {args.mode.upper()}")
    print(f"📂 Episode data directory: {args.episode_data_dir}")
    if args.mode == "cem":
        print(f"🔧 CEM parameters: samples={args.num_samples}, elites={args.num_elites}, iterations={args.iterations}")
        print(f"⏱️ Planning parameters: horizon={args.plan_horizon}s, frequency={args.frequency}Hz")
    
    # Create task
    task = HumanoidStand(reference_sequence=args.sequence)
    mj_model = task.mj_model
    mj_model.opt.timestep = 0.01
    mj_model.opt.iterations = 10
    mj_model.opt.ls_iterations = 50
    mj_model.opt.noslip_iterations = 2
    mj_model.opt.o_solimp = [0.0, 0.95, 0.01, 0.5, 2]
    mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE
    
    mj_data = mujoco.MjData(mj_model)
    
    # Load random initial state
    try:
        initial_state = load_random_initial_state(args.episode_data_dir)
        mj_data.qpos[:] = initial_state['qpos']
        mj_data.qvel[:] = initial_state['qvel']
        mj_data.time = initial_state['time']
        print(f"✅ Successfully set random initial state")
    except Exception as e:
        print(f"⚠️ Unable to load random initial state: {e}")
        print("Using default initial state")
        if task.reference.shape[1] == mj_model.nq:
            mj_data.qpos[:] = task.reference[0]
        else:
            mj_data.qpos[2] = 0.79
            if mj_model.nq > 6:
                mj_data.qpos[3] = 1.0
    
    # Initialize controller (using parameters consistent with humanoid_standonly.py)
    controller = CEM(
        task,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        sigma_start=0.3,        # Consistent with original version
        sigma_min=0.05,         # Consistent with original version
        explore_fraction=0.3,   # Consistent with original version
        plan_horizon=args.plan_horizon,
        spline_type="zero",     # Consistent with original version
        num_knots=4,           # Consistent with original version
        iterations=args.iterations  # Use user-specified iterations
    )
    
    # Load neural network model (if nn mode)
    net = None
    normalization_params = None
    device = None
    
    if args.mode == "nn":
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found: {args.model_path}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {device}")
        net, normalization_params = load_model(args.model_path, device)
    
    # Get reference trajectory
    reference = task.reference if args.show_reference else None
    
    # Run simulation
    run_dagger_simulation(
        mode=args.mode,
        task=task,
        mj_model=mj_model,
        mj_data=mj_data,
        controller=controller,
        net=net,
        device=device,
        normalization_params=normalization_params,
        show_reference=args.show_reference,
        reference=reference,
        frequency=args.frequency
    )


if __name__ == "__main__":
    main()