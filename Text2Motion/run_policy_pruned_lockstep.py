#!/usr/bin/env python3
"""
Isolated control computation service - hybrid mode based on run_policy.py
Receives robot state via ZeroMQ and returns PD control targets

Usage:
    python run_policy_pruned.py --model_path nn_ckpt/model-epoch=72-val_loss=0.003503.ckpt

Workflow:
1. Initialize NN+CEM pipeline and warm up JIT
2. Start ZeroMQ server and wait for state requests
3. Receive state -> NN prediction + CEM optimization + interpolation -> return control sequence
4. Internally maintain a strict 50Hz loop to keep JIT hot
"""

import argparse
import time
import pickle
from typing import Tuple, Optional, Dict, Any
import threading
import queue

import numpy as np
import torch
import torch.nn as nn
import mujoco
import jax
import jax.numpy as jnp
from mujoco import mjx
import pytorch_lightning as pl
import zmq

from hydrax.algs import CEM
from hydrax.tasks.humanoid_standonly import HumanoidStand


class MLPRegressor(pl.LightningModule):
    """Neural network model definition (same as original)"""
    
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
    """Load model from disk"""
    checkpoint = torch.load(model_path, map_location=device)
    
    net = MLPRegressor(95, 512, 512, 512, 164)
    net.load_state_dict(checkpoint['state_dict'])
    net.to(device).eval()

    print("Model loaded successfully")
    return net


def predict_knots(net, qpos, qvel, device):
    """Predict knots using neural network"""
    with torch.no_grad():
        net.eval()
        
        # Concatenate state
        state = np.concatenate([qpos, qvel], axis=0).astype(np.float32)
        state = torch.from_numpy(state).to(device).float().unsqueeze(0)

        # Network inference
        knots = net(state)
        knots = knots.squeeze(0).cpu().numpy().reshape(4, 41)

        return knots


class ControllerService:
    """Isolated control computation service"""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        frequency: float = 50.0,
        zmq_port: int = 5555,
        # CEM parameters
        num_samples: int = 500,
        num_elites: int = 20,
        sigma_start: float = 0.3,
        sigma_min: float = 0.05,
        plan_horizon: float = 0.5,
        num_knots: int = 4,
        iterations: int = 1
    ):
        """Initialize control service"""
        self.device = torch.device(device)
        self.frequency = frequency
        self.replan_period = 1.0 / frequency
        self.zmq_port = zmq_port
        
        print(f"🚀 Initializing control computation service")
        print(f"   Device: {self.device}")
        print(f"   Control frequency: {frequency} Hz")
        print(f"   ZeroMQ port: {zmq_port}")
        
        # 1. Load PyTorch network
        print("📦 Loading neural network...")
        self.net = load_model(model_path, self.device)
        
        # 2. Set up task and model
        print("🤖 Setting up robot task...")
        self.task = HumanoidStand()
        self.mj_model = self.task.mj_model
        
        # Configure MuJoCo parameters (same as run_policy.py)
        self.mj_model.opt.timestep = 0.01
        self.mj_model.opt.iterations = 10
        self.mj_model.opt.ls_iterations = 50
        self.mj_model.opt.noslip_iterations = 2
        self.mj_model.opt.o_solimp = [0.0, 0.95, 0.01, 0.5, 2]
        self.mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE
        
        # 3. Set up CEM controller
        print("🧠 Setting up CEM controller...")
        self.ctrl = CEM(
            self.task,
            num_samples=num_samples, 
            num_elites=num_elites,
            sigma_start=sigma_start, 
            sigma_min=sigma_min,
            explore_fraction=0.3,
            plan_horizon=plan_horizon,
            spline_type="zero",
            num_knots=num_knots,
            iterations=iterations
        )
        
        # 4. Precompute simulation parameters
        self.sim_steps_per_replan = max(int(self.replan_period / self.mj_model.opt.timestep), 1)
        self.step_dt = self.sim_steps_per_replan * self.mj_model.opt.timestep
        
        print(f"   Simulation steps per plan: {self.sim_steps_per_replan}")
        print(f"   Planning period: {self.step_dt:.4f}s")
        
        # 5. Initialize dummy state and JAX data
        print("🎭 Initializing dummy state...")
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mjx_data = mjx.put_data(self.mj_model, self.mj_data)
        self.mjx_data = self.mjx_data.replace(
            mocap_pos=self.mj_data.mocap_pos, 
            mocap_quat=self.mj_data.mocap_quat
        )
        
        # 6. Initialize policy parameters
        initial_knots = predict_knots(self.net, self.mj_data.qpos, self.mj_data.qvel, self.device)
        self.policy_params = self.ctrl.init_params(initial_knots=initial_knots)
        
        # 7. Precompile JAX functions
        print("⚡ Precompiling JAX functions...")
        self.jit_optimize = jax.jit(self.ctrl.optimize)
        self.jit_interp_func = jax.jit(self.ctrl.interp_func)
        
        # JIT warmup
        print("🔥 Warming up JIT...")
        self.policy_params, rollouts = self.jit_optimize(self.mjx_data, self.policy_params)
        tq = jnp.arange(0, self.sim_steps_per_replan) * self.mj_model.opt.timestep
        tk = self.policy_params.tk
        knots = self.policy_params.mean[None, ...]
        _ = self.jit_interp_func(tq, tk, knots)
        print("✅ JIT warmup complete")
        
        # 8. Set up ZeroMQ
        print("🌐 Setting up ZeroMQ service...")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{zmq_port}")
        print(f"✅ ZeroMQ server started on port {zmq_port}")
        
        # 9. State management
        self.current_request = None
        self.running = False
        self.stats = {
            'total_requests': 0,
            'total_compute_time': 0.0,
            'max_compute_time': 0.0,
            'min_compute_time': float('inf')
        }
        
    def compute_controls(
        self, 
        qpos: np.ndarray, 
        qvel: np.ndarray,
        mocap_pos: Optional[np.ndarray] = None,
        mocap_quat: Optional[np.ndarray] = None,
        current_time: float = 0.0
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Compute control outputs (core hybrid logic)"""
        start_time = time.time()
        timing_info = {}
        
        # 1. Neural network prediction
        nn_start = time.time()
        predicted_knots = predict_knots(self.net, qpos, qvel, self.device)
        timing_info['nn_time'] = time.time() - nn_start
        
        # 2. Prepare JAX data
        prep_start = time.time()
        mjx_data = self.mjx_data.replace(
            qpos=jnp.array(qpos),
            qvel=jnp.array(qvel),
            time=current_time
        )
        
        if mocap_pos is not None:
            mjx_data = mjx_data.replace(mocap_pos=jnp.array(mocap_pos))
        if mocap_quat is not None:
            mjx_data = mjx_data.replace(mocap_quat=jnp.array(mocap_quat))
        
        # Update policy parameters
        policy_params = self.policy_params.replace(mean=predicted_knots)
        timing_info['prep_time'] = time.time() - prep_start
        
        # 3. CEM optimization
        cem_start = time.time()
        policy_params, rollouts = self.jit_optimize(mjx_data, policy_params)
        timing_info['cem_time'] = time.time() - cem_start
        
        # 4. Interpolate to generate control sequence
        interp_start = time.time()
        tq = jnp.arange(0, self.sim_steps_per_replan) * self.mj_model.opt.timestep + current_time
        tk = policy_params.tk
        knots = policy_params.mean[None, ...]
        controls_jax = self.jit_interp_func(tq, tk, knots)[0]
        
        controls = np.asarray(controls_jax)
        timing_info['interp_time'] = time.time() - interp_start
        timing_info['total_time'] = time.time() - start_time
        
        # Update internal policy parameters to maintain continuity
        self.policy_params = policy_params
        
        return controls, timing_info
    
    def run_dummy_loop(self):
        """Background low-frequency maintenance loop (to avoid resource contention with main request handler)"""
        print("🎭 Starting background maintenance loop...")
        
        cycle_count = 0
        start_time = time.time()
        
        while self.running:
            cycle_start = time.time()
            
            # Low-frequency maintenance (1Hz): only lightweight JIT keep-alive
            dummy_mjx_data = self.mjx_data.replace(time=cycle_count * 1.0)
            tq = jnp.arange(0, self.sim_steps_per_replan) * self.mj_model.opt.timestep
            tk = self.policy_params.tk
            knots = self.policy_params.mean[None, ...]
            _ = self.jit_interp_func(tq, tk, knots)
            
            cycle_count += 1
            
            # Print status every 60 cycles (~1 minute)
            if cycle_count % 60 == 0:
                elapsed = time.time() - start_time
                avg_freq = cycle_count / elapsed
                print(f"🔄 Background maintenance loop #{cycle_count}: avg freq={avg_freq:.1f}Hz (target: 1Hz)")
            
            # Control loop frequency at 1Hz (avoid contention with main thread)
            cycle_elapsed = time.time() - cycle_start
            target_cycle_time = 1.0  # 1 second per cycle
            if cycle_elapsed < target_cycle_time:
                time.sleep(target_cycle_time - cycle_elapsed)
    
    def handle_zmq_requests(self):
        """Handle ZeroMQ requests (blocking mode, focus on requests)"""
        print("🌐 Starting to handle ZeroMQ requests...")
        
        while self.running:
            try:
                # Blocking wait for request
                message = self.socket.recv_pyobj()
                
                # Parse request
                qpos = message['qpos']
                qvel = message['qvel']
                mocap_pos = message.get('mocap_pos', None)
                mocap_quat = message.get('mocap_quat', None)
                current_time = message.get('time', 0.0)
                
                # Compute control outputs
                controls, timing_info = self.compute_controls(
                    qpos, qvel, mocap_pos, mocap_quat, current_time
                )
                
                # Update statistics
                self.stats['total_requests'] += 1
                self.stats['total_compute_time'] += timing_info['total_time']
                self.stats['max_compute_time'] = max(self.stats['max_compute_time'], timing_info['total_time'])
                self.stats['min_compute_time'] = min(self.stats['min_compute_time'], timing_info['total_time'])
                
                # Send response
                response = {
                    'controls': controls,
                    'timing': timing_info,
                    'stats': self.stats.copy()
                }
                self.socket.send_pyobj(response)
                
                # Print request info
                if self.stats['total_requests'] % 10 == 1:
                    avg_time = self.stats['total_compute_time'] / self.stats['total_requests'] * 1000
                    print(f"📨 Request #{self.stats['total_requests']}: "
                          f"Compute time={timing_info['total_time']*1000:.1f}ms, "
                          f"Avg={avg_time:.1f}ms")
                        
            except zmq.ContextTerminated:
                # Context terminated, exit loop
                break
            except Exception as e:
                print(f"❌ ZeroMQ handling error: {e}")
                if not self.running:
                    break
                
    def start(self):
        """Start service"""
        print("🚀 Starting control computation service...")
        self.running = True
        
        # Warmup phase: run several dummy cycles to ensure JIT is fully warmed up
        print("🔥 Warming up JIT...")
        for i in range(20):  # Warm up 20 cycles
            dummy_qpos = self.mj_data.qpos.copy()
            dummy_qvel = self.mj_data.qvel.copy()
            dummy_qpos += np.random.normal(0, 0.001, dummy_qpos.shape)
            dummy_qvel += np.random.normal(0, 0.001, dummy_qvel.shape)
            _, timing_info = self.compute_controls(dummy_qpos, dummy_qvel, current_time=i * 0.02)
            if (i + 1) % 5 == 0:
                print(f"   Warmup progress: {i+1}/20, compute time: {timing_info['total_time']*1000:.1f}ms")
        
        print("✅ JIT warmup complete!")
        
        # Start background dummy loop thread (low-frequency maintenance)
        # dummy_thread = threading.Thread(target=self.run_dummy_loop, daemon=True)
        # dummy_thread.start()
        
        print("✅ Control computation pipeline started and running!")
        print("💡 System status:")
        print("   - JIT functions fully warmed up")
        print("   - Background maintenance loop running")
        print(f"   - ZeroMQ server listening on port {self.zmq_port}")
        print("   - Ready to receive robot state and return control commands")
        print("\n🎯 You can now start isolated_simulation.py!")
        
        try:
            # Main thread focuses on handling ZeroMQ requests
            self.handle_zmq_requests()
        except KeyboardInterrupt:
            print("\n🛑 Received interrupt signal, shutting down service...")
            self.stop()
    
    def stop(self):
        """Stop service"""
        self.running = False
        self.socket.close()
        self.context.term()
        
        # Print final statistics
        if self.stats['total_requests'] > 0:
            avg_time = self.stats['total_compute_time'] / self.stats['total_requests'] * 1000
            print(f"\n📊 Final statistics:")
            print(f"   Total requests: {self.stats['total_requests']}")
            print(f"   Average compute time: {avg_time:.2f}ms")
            print(f"   Max compute time: {self.stats['max_compute_time']*1000:.2f}ms")
            print(f"   Min compute time: {self.stats['min_compute_time']*1000:.2f}ms")
        
        print("✅ Control computation service stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Isolated control computation service - provides hybrid control via ZeroMQ"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=False, 
        default="nn_ckpt/model-epoch=72-val_loss=0.003503.ckpt",
        help="PyTorch model path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="PyTorch device (cuda/cpu)"
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=50.0,
        help="Control frequency (Hz)"
    )
    parser.add_argument(
        "--zmq_port",
        type=int,
        default=5555,
        help="ZeroMQ port"
    )
    
    # CEM parameters (default values tuned to avoid GPU memory issues)
    parser.add_argument("--num_samples", type=int, default=90, help="CEM sample count")
    parser.add_argument("--num_elites", type=int, default=20, help="CEM elite count")
    parser.add_argument("--plan_horizon", type=float, default=0.5, help="Planning horizon")
    parser.add_argument("--num_knots", type=int, default=4, help="Spline knot count")
    parser.add_argument("--iterations", type=int, default=1, help="CEM iteration count")

    args = parser.parse_args()

    # Create and start control service
    service = ControllerService(
        model_path=args.model_path,
        device=args.device,
        frequency=args.frequency,
        zmq_port=args.zmq_port,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        plan_horizon=args.plan_horizon,
        num_knots=args.num_knots,
        iterations=args.iterations
    )
    
    service.start()


if __name__ == "__main__":
    main() 