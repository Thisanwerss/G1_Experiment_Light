#!/usr/bin/env python3
"""
Isolated control computation node - Hybrid mode based on run_policy.py
Bidirectional asynchronous ZeroMQ to receive robot state and return PD control targets

Usage:
    python run_policy_pruned.py --model_path nn_ckpt/model-epoch=72-val_loss=0.003503.ckpt

Workflow:
1. Initialize NN+CEM pipeline and warm up JIT
2. Start bidirectional asynchronous ZeroMQ service (PULL/PUSH mode)
3. Use cycle_id barrier to receive state -> NN prediction + CEM optimization + interpolation -> return control sequence
4. Collect detailed communication and computation timing statistics
"""

import argparse
import time
import pickle
from typing import Tuple, Optional, Dict, Any, List
import struct

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


class OutlierFilteredStats:
    """Statistics tracker with outlier filtering"""
    
    def __init__(self, window_size=100, outlier_threshold=3.0):
        self.window_size = window_size
        self.outlier_threshold = outlier_threshold
        self.samples = []
        
    def add_sample(self, value: float):
        """Add a sample"""
        self.samples.append(value)
        if len(self.samples) > self.window_size:
            self.samples.pop(0)
    
    def get_filtered_stats(self) -> Tuple[float, float, int, int]:
        """Get stats after outlier filtering (mean, std, valid_count, outlier_count)"""
        if len(self.samples) < 3:
            return 0.0, 0.0, len(self.samples), 0
            
        # Compute initial mean and std
        arr = np.array(self.samples)
        mean = np.mean(arr)
        std = np.std(arr)
        
        # Filter outliers
        lower_bound = mean - self.outlier_threshold * std
        upper_bound = mean + self.outlier_threshold * std
        
        valid_mask = (arr >= lower_bound) & (arr <= upper_bound)
        valid_samples = arr[valid_mask]
        
        if len(valid_samples) == 0:
            return mean, std, 0, len(arr)
            
        # Recompute filtered stats
        filtered_mean = np.mean(valid_samples)
        filtered_std = np.std(valid_samples) if len(valid_samples) > 1 else 0.0
        
        return filtered_mean, filtered_std, len(valid_samples), len(arr) - len(valid_samples)


class AsyncControllerService:
    """Asynchronous control computation service"""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        frequency: float = 50.0,
        zmq_state_port: int = 5555,
        zmq_ctrl_port: int = 5556,
        # CEM parameters
        num_samples: int = 500,
        num_elites: int = 20,
        sigma_start: float = 0.3,
        sigma_min: float = 0.05,
        plan_horizon: float = 0.5,
        num_knots: int = 4,
        iterations: int = 1
    ):
        """Initialize asynchronous control service"""
        self.device = torch.device(device)
        self.frequency = frequency
        self.replan_period = 1.0 / frequency
        self.zmq_state_port = zmq_state_port
        self.zmq_ctrl_port = zmq_ctrl_port
        
        print(f"🚀 Initializing asynchronous control computation service")
        print(f"   Device: {self.device}")
        print(f"   Control frequency: {frequency} Hz")
        print(f"   State port: {zmq_state_port} (Sim PUSH → Control PULL)")
        print(f"   Control port: {zmq_ctrl_port} (Control PUSH → Sim PULL)")
        
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
        
        # 8. Set up bidirectional asynchronous ZeroMQ
        print("🌐 Setting up bidirectional asynchronous ZeroMQ service...")
        self.context = zmq.Context()
        
        # High performance config
        self.context.setsockopt(zmq.MAX_SOCKETS, 1024)
        self.context.setsockopt(zmq.IO_THREADS, 2)
        
        # State receive socket (PULL)
        self.socket_state = self.context.socket(zmq.PULL)
        self.socket_state.setsockopt(zmq.RCVHWM, 10)
        self.socket_state.setsockopt(zmq.RCVBUF, 1048576)
        self.socket_state.setsockopt(zmq.LINGER, 0)
        self.socket_state.bind(f"tcp://*:{zmq_state_port}")
        
        # Control send socket (PUSH)
        self.socket_ctrl = self.context.socket(zmq.PUSH)
        self.socket_ctrl.setsockopt(zmq.SNDHWM, 10)
        self.socket_ctrl.setsockopt(zmq.SNDBUF, 1048576)
        self.socket_ctrl.setsockopt(zmq.LINGER, 0)
        self.socket_ctrl.bind(f"tcp://*:{zmq_ctrl_port}")
        
        # Poller setup
        self.poller = zmq.Poller()
        self.poller.register(self.socket_state, zmq.POLLIN)
        
        print(f"✅ Asynchronous ZeroMQ server started")
        
        # 9. Preallocate buffer to avoid repeated allocation
        self.controls_buffer = np.zeros((self.sim_steps_per_replan, 41), dtype=np.float32)
        
        # 10. State management
        self.current_request = None
        self.running = False
        self.cycle_id = 0
        
        # 11. Enhanced timing statistics
        self.timing_history = []
        
        # Detailed timing variables
        self.t_state_recv_start = 0.0
        self.t_state_recv_end = 0.0
        self.t_ctrl_compute_start = 0.0
        self.t_ctrl_compute_end = 0.0
        self.t_ctrl_send_start = 0.0
        self.t_ctrl_send_end = 0.0
        
        # Statistics trackers
        self.state_recv_stats = OutlierFilteredStats()
        self.ctrl_compute_stats = OutlierFilteredStats()
        self.ctrl_send_stats = OutlierFilteredStats()
        self.cycle_latency_stats = OutlierFilteredStats()
        
    def compute_controls(
        self, 
        qpos: np.ndarray, 
        qvel: np.ndarray,
        mocap_pos: Optional[np.ndarray] = None,
        mocap_quat: Optional[np.ndarray] = None,
        current_time: float = 0.0
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Compute control outputs (core hybrid logic)"""
        self.t_ctrl_compute_start = time.time()
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
        
        # Reuse preallocated buffer to avoid memory allocation
        np.copyto(self.controls_buffer, np.asarray(controls_jax))
        timing_info['interp_time'] = time.time() - interp_start
        
        self.t_ctrl_compute_end = time.time()
        timing_info['total_time'] = self.t_ctrl_compute_end - self.t_ctrl_compute_start
        
        # Record compute time
        compute_time = self.t_ctrl_compute_end - self.t_ctrl_compute_start
        self.ctrl_compute_stats.add_sample(compute_time)
        
        # Update internal policy parameters for continuity
        self.policy_params = policy_params
        
        return self.controls_buffer, timing_info
    
    def recv_state_async(self, timeout_ms: int = 100) -> Optional[Tuple[int, Dict[str, Any]]]:
        """Asynchronously receive state"""
        try:
            # Check if state is available
            socks = dict(self.poller.poll(timeout_ms))
            
            if self.socket_state in socks:
                self.t_state_recv_start = time.time()
                
                # Receive multipart message [cycle_id, state_bytes]
                parts = self.socket_state.recv_multipart(zmq.NOBLOCK)
                
                if len(parts) != 2:
                    print(f"⚠️ Received invalid state message format, parts={len(parts)}")
                    return None
                
                # Parse cycle_id
                recv_cycle_id = struct.unpack('I', parts[0])[0]
                
                # Deserialize state
                state = pickle.loads(parts[1])
                
                # Convert lists back to numpy arrays
                for key, value in state.items():
                    if isinstance(value, list) and key in ['qpos', 'qvel', 'mocap_pos', 'mocap_quat']:
                        state[key] = np.array(value, dtype=np.float64)
                
                self.t_state_recv_end = time.time()
                
                # Record receive time
                recv_time = self.t_state_recv_end - self.t_state_recv_start
                self.state_recv_stats.add_sample(recv_time)
                
                return recv_cycle_id, state
            
            return None
            
        except zmq.Again:
            return None
        except Exception as e:
            print(f"❌ State receive error: {e}")
            return None
    
    def send_controls_async(self, cycle_id: int, controls: np.ndarray, timing_info: Dict[str, float]) -> bool:
        """Asynchronously send control commands"""
        try:
            self.t_ctrl_send_start = time.time()
            
            # Prepare response - convert numpy arrays to lists for compatibility
            response = {
                'controls': controls.tolist() if isinstance(controls, np.ndarray) else controls,
                'timing': timing_info
            }
            
            # Serialize response
            response_bytes = pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Construct message with cycle_id
            cycle_id_bytes = struct.pack('I', cycle_id)  # uint32
            
            # Send multipart message [cycle_id, response_bytes]
            self.socket_ctrl.send_multipart([cycle_id_bytes, response_bytes], zmq.NOBLOCK)
            
            self.t_ctrl_send_end = time.time()
            
            # Record send time
            send_time = self.t_ctrl_send_end - self.t_ctrl_send_start
            self.ctrl_send_stats.add_sample(send_time)
            
            return True
            
        except zmq.Again:
            print(f"⚠️ Control send queue full, cycle_id={cycle_id}")
            return False
        except Exception as e:
            print(f"❌ Control send error: {e}")
            return False
    
    def calculate_cycle_metrics(self, cycle_id: int):
        """Calculate cycle metrics"""
        # Cycle latency = control send end - state receive start
        cycle_latency = self.t_ctrl_send_end - self.t_state_recv_start
        self.cycle_latency_stats.add_sample(cycle_latency)
        
        # Save detailed timing record
        timing_record = {
            'cycle_id': cycle_id,
            'state_recv_time': self.t_state_recv_end - self.t_state_recv_start,
            'ctrl_compute_time': self.t_ctrl_compute_end - self.t_ctrl_compute_start,
            'ctrl_send_time': self.t_ctrl_send_end - self.t_ctrl_send_start,
            'cycle_latency': cycle_latency
        }
        self.timing_history.append(timing_record)
    
    def print_periodic_stats(self):
        """Print periodic statistics"""
        print(f"\n🔍 === Asynchronous Control Statistics Report (Cycle #{self.cycle_id}) ===")
        
        # State receive stats
        recv_mean, recv_std, recv_valid, recv_outliers = self.state_recv_stats.get_filtered_stats()
        print(f"📥 State receive: {recv_mean*1000:.2f}±{recv_std*1000:.2f}ms (valid:{recv_valid}, outliers:{recv_outliers})")
        
        # Control compute stats
        compute_mean, compute_std, compute_valid, compute_outliers = self.ctrl_compute_stats.get_filtered_stats()
        print(f"🧠 Control compute: {compute_mean*1000:.2f}±{compute_std*1000:.2f}ms (valid:{compute_valid}, outliers:{compute_outliers})")
        
        # Control send stats
        send_mean, send_std, send_valid, send_outliers = self.ctrl_send_stats.get_filtered_stats()
        print(f"📤 Control send: {send_mean*1000:.2f}±{send_std*1000:.2f}ms (valid:{send_valid}, outliers:{send_outliers})")
        
        # Cycle latency stats
        cycle_mean, cycle_std, cycle_valid, cycle_outliers = self.cycle_latency_stats.get_filtered_stats()
        print(f"🔄 Cycle latency: {cycle_mean*1000:.2f}±{cycle_std*1000:.2f}ms (valid:{cycle_valid}, outliers:{cycle_outliers})")
        
        # Estimated frequency
        if cycle_mean > 0:
            estimated_freq = 1.0 / cycle_mean
            print(f"📈 Estimated frequency: {estimated_freq:.1f} Hz (target: {self.frequency:.1f} Hz)")
        
        print("=" * 60)
        
    def handle_async_requests(self):
        """Main loop to handle asynchronous requests"""
        print("🌐 Start handling asynchronous ZeroMQ requests...")
        print("💡 Working mode:")
        print("   - Bidirectional asynchronous ZeroMQ communication (PULL/PUSH)")
        print("   - Cycle ID barrier synchronization mechanism")
        print("   - Enhanced timing statistics and outlier filtering")
        
        # Wait for simulation side to connect for the first time
        print("🔄 Waiting for simulation side to connect...")
        
        while self.running:
            try:
                # ========== Barrier A: Receive state ==========
                state_result = self.recv_state_async(timeout_ms=1000)  # 1 second timeout
                
                if state_result is None:
                    # Timeout or no message, continue waiting
                    continue
                
                recv_cycle_id, state = state_result
                
                # Fix: synchronize cycle_id logic
                # On first receive, directly sync to received cycle_id
                if self.cycle_id == 0 or recv_cycle_id > self.cycle_id:
                    self.cycle_id = recv_cycle_id
                    print(f"🔄 Sync to Cycle #{self.cycle_id}")
                elif recv_cycle_id < self.cycle_id - 5:
                    # If received cycle_id is too old, possibly network delay, skip
                    print(f"⚠️ Skip outdated state: current cycle={self.cycle_id}, received={recv_cycle_id}")
                    continue
                
                print(f"📥 Cycle #{self.cycle_id}: State received")
                
                # ========== Compute control outputs ==========
                qpos = state['qpos']
                qvel = state['qvel']
                mocap_pos = state.get('mocap_pos', None)
                mocap_quat = state.get('mocap_quat', None)
                current_time = state.get('time', 0.0)
                
                controls, timing_info = self.compute_controls(
                    qpos, qvel, mocap_pos, mocap_quat, current_time
                )
                
                # ========== Barrier B: Send control commands ==========
                if not self.send_controls_async(self.cycle_id, controls, timing_info):
                    print(f"❌ Cycle #{self.cycle_id}: Control send failed")
                    continue
                
                print(f"📤 Cycle #{self.cycle_id}: Control command sent")
                
                # ========== Calculate metrics ==========
                self.calculate_cycle_metrics(self.cycle_id)
                
                # ========== Periodically print statistics ==========
                if (self.cycle_id + 1) % 20 == 0:
                    self.print_periodic_stats()
                    
            except Exception as e:
                print(f"❌ Asynchronous request handling error: {e}")
                if not self.running:
                    break
                
    def start(self):
        """Start service"""
        print("🚀 Starting asynchronous control computation service...")
        self.running = True
        
        # Warmup phase: run several dummy loops to ensure JIT is fully warmed up
        print("🔥 Warming up JIT...")
        for i in range(30):  # Increase warmup rounds
            dummy_qpos = self.mj_data.qpos.copy()
            dummy_qvel = self.mj_data.qvel.copy()
            dummy_qpos += np.random.normal(0, 0.001, dummy_qpos.shape)
            dummy_qvel += np.random.normal(0, 0.001, dummy_qvel.shape)
            _, timing_info = self.compute_controls(dummy_qpos, dummy_qvel, current_time=i * 0.02)
            if (i + 1) % 10 == 0:
                print(f"   Warmup progress: {i+1}/30, compute time: {timing_info['total_time']*1000:.1f}ms")
        
        print("✅ JIT warmup complete!")
        print("✅ Asynchronous control computation pipeline started and running!")
        print("💡 System status:")
        print("   - JIT functions fully warmed up")
        print("   - Asynchronous ZeroMQ config optimized")
        print(f"   - State port: {self.zmq_state_port} (PULL)")
        print(f"   - Control port: {self.zmq_ctrl_port} (PUSH)")
        print("   - Ready to receive robot state and return control commands")
        print("\n🎯 Now you can start isolated_simulation.py!")
        
        try:
            # Main thread focuses on handling async requests
            self.handle_async_requests()
        except KeyboardInterrupt:
            print("\n🛑 Received interrupt signal, shutting down service...")
            self.stop()
    
    def stop(self):
        """Stop service"""
        self.running = False
        self.socket_state.close()
        self.socket_ctrl.close()
        self.context.term()
        
        # Print final statistics
        print(f"\n🏁 === Final Asynchronous Control Report ===")
        print(f"📊 Overall statistics:")
        print(f"   Total cycles: {self.cycle_id}")
        print(f"   Timing records: {len(self.timing_history)}")
        
        if len(self.timing_history) > 0:
            # Final filtered statistics
            recv_mean, recv_std, _, _ = self.state_recv_stats.get_filtered_stats()
            compute_mean, compute_std, _, _ = self.ctrl_compute_stats.get_filtered_stats()
            send_mean, send_std, _, _ = self.ctrl_send_stats.get_filtered_stats()
            cycle_mean, cycle_std, _, _ = self.cycle_latency_stats.get_filtered_stats()
            
            print(f"\n📈 Final performance metrics (after outlier filtering):")
            print(f"   State receive time: {recv_mean*1000:.2f}±{recv_std*1000:.2f}ms")
            print(f"   Control compute time: {compute_mean*1000:.2f}±{compute_std*1000:.2f}ms")
            print(f"   Control send time: {send_mean*1000:.2f}±{send_std*1000:.2f}ms")
            print(f"   Cycle latency: {cycle_mean*1000:.2f}±{cycle_std*1000:.2f}ms")
            
            if cycle_mean > 0:
                estimated_freq = 1.0 / cycle_mean
                print(f"   Estimated system frequency: {estimated_freq:.2f} Hz")
                print(f"   Target frequency achievement: {estimated_freq/self.frequency*100:.1f}%")
        
        print("✅ Asynchronous control computation service stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Asynchronous control computation node - Bidirectional asynchronous ZeroMQ hybrid control service"
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
        "--zmq_state_port",
        type=int,
        default=5555,
        help="State channel port (Sim PUSH → Control PULL)"
    )
    parser.add_argument(
        "--zmq_ctrl_port",
        type=int,
        default=5556,
        help="Control channel port (Control PUSH → Sim PULL)"
    )
    
    # CEM parameters (default values tuned to avoid GPU memory issues and improve speed)
    parser.add_argument("--num_samples", type=int, default=300, help="CEM sample count (lower for speed)")
    parser.add_argument("--num_elites", type=int, default=15, help="CEM elite count")
    parser.add_argument("--plan_horizon", type=float, default=0.4, help="Planning horizon (shorter for speed)")
    parser.add_argument("--num_knots", type=int, default=4, help="Spline knot count")
    parser.add_argument("--iterations", type=int, default=1, help="CEM iteration count")

    args = parser.parse_args()

    # Create and start asynchronous control service
    service = AsyncControllerService(
        model_path=args.model_path,
        device=args.device,
        frequency=args.frequency,
        zmq_state_port=args.zmq_state_port,
        zmq_ctrl_port=args.zmq_ctrl_port,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        plan_horizon=args.plan_horizon,
        num_knots=args.num_knots,
        iterations=args.iterations
    )
    
    service.start()


if __name__ == "__main__":
    main() 