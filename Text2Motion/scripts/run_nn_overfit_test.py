#!/usr/bin/env python3
"""
Replay recorded episode using an overfit neural network to predict knots instead of using recorded ones.
Run:
    python replay_episode_nn.py --episode_path episode_data_overfit/episode_overfit.pkl \
        --model_path overfit_model.pth --show_reference
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

from hydrax.algs import CEM
from hydrax.tasks.humanoid_standonly import HumanoidStand
from replay_episode import replay_knots

# Define the advanced network architecture (must match training)
class AdvancedOverfitNet(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        
        # Significantly expand the network scale - deeper and wider
        hidden_dims = [2048, 4096, 8192, 8192, 4096, 2048, 1024]
        
        layers = []
        prev_dim = inp_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # LayerNorm helps stabilize training
                nn.GELU(),  # GELU activation function usually performs better than ReLU
                nn.Dropout(0.05 if i < len(hidden_dims)//2 else 0.02)  # Slightly higher dropout for the first half
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, out_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Multi-scale residual connection
        self.use_residual = (inp_dim == out_dim)
        if not self.use_residual:
            # Multi-layer projection to better match dimensions
            self.input_projection = nn.Sequential(
                nn.Linear(inp_dim, min(inp_dim * 2, out_dim)),
                nn.GELU(),
                nn.Linear(min(inp_dim * 2, out_dim), out_dim)
            )
            self.use_projection = True
        else:
            self.use_projection = False
        
        # Attention mechanism - helps the network focus on important features
        # Ensure embed_dim is divisible by num_heads
        attention_dim = min(inp_dim, 512)
        num_heads = 8
        # Adjust attention_dim to be divisible by num_heads
        attention_dim = (attention_dim // num_heads) * num_heads
        if attention_dim == 0:
            attention_dim = num_heads
            
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.attention_proj = nn.Linear(inp_dim, attention_dim)
        self.attention_out_proj = nn.Linear(attention_dim, out_dim)
        
        # He initialization (better for GELU)
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
        
        # Main network path
        main_out = self.net(x)
        
        # Attention path
        attn_input = self.attention_proj(x).unsqueeze(1)  # [B, 1, D]
        attn_out, _ = self.attention(attn_input, attn_input, attn_input)
        attn_out = self.attention_out_proj(attn_out.squeeze(1))
        
        # Combine outputs
        out = 0.8 * main_out + 0.2 * attn_out
        
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


def load_model(model_path, device):
    """Load the overfit model from disk."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check for normalization parameters (new model version)
    has_normalization = 'X_mean' in checkpoint
    
    if has_normalization:
        print("🔄 Loading large-scale improved model (includes data normalization and attention mechanism)")
        net = AdvancedOverfitNet(checkpoint['inp_dim'], checkpoint['out_dim'])
        normalization_params = {
            'X_mean': checkpoint['X_mean'],
            'X_std': checkpoint['X_std'],
            'Y_mean': checkpoint['Y_mean'],
            'Y_std': checkpoint['Y_std']
        }
        print(f"📊 Model Information:")
        print(f"  - Input Dimension: {checkpoint['inp_dim']}")
        print(f"  - Output Dimension: {checkpoint['out_dim']}")
        print(f"  - Final Loss: {checkpoint.get('final_loss', 'N/A'):.15e}")
        print(f"  - Best Optimizer: {checkpoint.get('best_optimizer', 'N/A')}")
        
        # Calculate network parameter count
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"  - Total Parameters: {total_params:,}")
        print(f"  - Trainable Parameters: {trainable_params:,}")
    else:
        print("🔄 Loading old simple model (no data normalization)")
        net = OverfitNet(checkpoint['inp_dim'], checkpoint['out_dim'])
        normalization_params = None
        total_params = sum(p.numel() for p in net.parameters())
        print(f"  - Total Parameters: {total_params:,}")
    
    net.load_state_dict(checkpoint['state_dict'])
    net.to(device).eval()
    
    return net, normalization_params


def predict_knots(net, episode_data, device, normalization_params=None):
    """
    Predict knots sequence for each knot timestamp using the neural network.
    Returns a new list of knot_info dicts matching original structure.
    """
    traj = episode_data['trajectory']
    original_knots = episode_data['knots']
    metadata = episode_data['metadata']
    num_knots = metadata['num_knots']
    nu = metadata['nu']

    predicted = []
    print(f"🔮 Starting prediction for {len(original_knots)} knot timestamps...")
    
    with torch.no_grad():
        net.eval()  # Ensure network is in evaluation mode
        
        for i, kinfo in enumerate(original_knots):
            t_k = kinfo['timestamp']
            # find closest state
            st = min(traj, key=lambda s: abs(s['time'] - t_k))
            x = np.concatenate([st['qpos'], st['qvel']], axis=0).astype(np.float32)
            
            # Apply data normalization (if any)
            if normalization_params is not None:
                x = (x - normalization_params['X_mean']) / normalization_params['X_std']
            
            x_t = torch.from_numpy(x).to(device).float()
            
            # Network inference
            y_t = net(x_t)
            
            # De-normalize output (if any)
            if normalization_params is not None:
                y_np = y_t.cpu().numpy()
                y_np = y_np * normalization_params['Y_std'] + normalization_params['Y_mean']
                knots = y_np.reshape(num_knots, nu)
            else:
                knots = y_t.cpu().numpy().reshape(num_knots, nu)
            
            predicted.append({
                'knots': knots,
                'tk': kinfo['tk'],
                'timestamp': t_k
            })
            
            # Show progress
            if (i + 1) % 10 == 0 or i == len(original_knots) - 1:
                print(f"  Predicted: {i+1}/{len(original_knots)} ({100*(i+1)/len(original_knots):.1f}%)")
    
    print("✅ Knot prediction complete!")
    return predicted


def main():
    parser = argparse.ArgumentParser(
        description="Replay episode using NN-predicted knots instead of recorded ones."
    )
    parser.add_argument(
        "--episode_path", type=str, required=False, default="episode_data_overfit/episode_overfit.pkl",
        help="Path to the episode pickle file."
    )
    parser.add_argument(
        "--model_path", type=str, required=False, default="overfit_model.pth",
        help="Path to the overfit model checkpoint (overfit_model.pth)."
    )
    parser.add_argument(
        "--show_reference", action="store_true",
        help="Show the reference trajectory as a transparent ghost."
    )

    parser.add_argument(
        "--use_nn_results", action="store_true",
        help="Use the results from the overfit network instead of the recorded knots."
    )
    args = parser.parse_args()

    if not os.path.exists(args.episode_path):
        raise FileNotFoundError(f"Episode file not found: {args.episode_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
        
    if args.use_nn_results:
        print("Predicting knots with the overfit network...")
    else:
        print("Using recorded knots instead of predicted knots.")
    # Load episode data
    print(f"Loading episode data from: {args.episode_path}")
    with open(args.episode_path, 'rb') as f:
        episode_data = pickle.load(f)

    metadata = episode_data['metadata']
    print(f"Episode metadata: sequence={metadata['sequence']}, \
          control_freq={metadata['control_frequency']}Hz, \
          sim_timestep={metadata['simulation_timestep']}s, \
          nq={metadata['nq']}, nu={metadata['nu']}, \
          traj_len={len(episode_data['trajectory'])}, \
          orig_knots={len(episode_data['knots'])}")

    # Setup device and load network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    net, normalization_params = load_model(args.model_path, device)

    # Predict knots
    if args.use_nn_results:
        episode_data['knots'] = predict_knots(net, episode_data, device, normalization_params)
    else:
        pass

    # Setup Mujoco model and data
    task = HumanoidStand(reference_sequence=metadata['sequence'])
    mj_model = task.mj_model
    mj_model.opt.timestep = metadata['simulation_timestep']
    mj_model.opt.iterations = 10
    mj_model.opt.ls_iterations = 50
    mj_model.opt.noslip_iterations = 2
    mj_model.opt.o_solimp = [0.0, 0.95, 0.01, 0.5, 2]
    mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE
    mj_data = mujoco.MjData(mj_model)
    reference = task.reference if args.show_reference else None

    # Instantiate CEM controller (only for interpolation)
    ctrl = CEM(
        task,
        num_samples=2000, num_elites=20,
        sigma_start=0.3, sigma_min=0.05,
        explore_fraction=0.3,
        plan_horizon=metadata['plan_horizon'],
        spline_type="zero",
        num_knots=metadata['num_knots'],
        iterations=1
    )
    print("Starting knot-based replay with predicted knots...")
    replay_knots(episode_data, ctrl, mj_model, mj_data, args.show_reference, reference)

if __name__ == "__main__":
    main()
