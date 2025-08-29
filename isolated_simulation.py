#!/usr/bin/env python3
"""
Isolated MuJoCo Simulation Endpoint - Bidirectional Asynchronous Communication with Control Node via ZeroMQ

Usage:
    python isolated_simulation.py --zmq_port 5555

Workflow:
1. Initialize MuJoCo simulation environment (same as run_policy.py)
2. Establish bidirectional asynchronous ZeroMQ channels (PUSH/PULL mode)
3. Use cycle_id barrier mechanism for synchronization, while enabling asynchronous communication
4. Run simulation and collect detailed communication and computation timing statistics
"""

import argparse
import time
from typing import Optional, Dict, Any, List, Tuple
import struct

import numpy as np
import mujoco
import mujoco.viewer
import zmq
import pickle


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
            
        # Recompute stats after filtering
        filtered_mean = np.mean(valid_samples)
        filtered_std = np.std(valid_samples) if len(valid_samples) > 1 else 0.0
        
        return filtered_mean, filtered_std, len(valid_samples), len(arr) - len(valid_samples)


class AsyncSimulation:
    """Asynchronous MuJoCo Simulation"""
    
    def __init__(
        self,
        zmq_state_port: int = 5555,
        zmq_ctrl_port: int = 5556,
        control_frequency: float = 50.0,
        no_viewer: bool = False,
        zmq_timeout: int = 100  # ms
    ):
        """Initialize asynchronous simulation"""
        print(f"🚀 Initializing asynchronous MuJoCo simulation")
        print(f"   State port: {zmq_state_port} (Sim PUSH → Control PULL)")
        print(f"   Control port: {zmq_ctrl_port} (Control PUSH → Sim PULL)")
        print(f"   Control frequency: {control_frequency} Hz")
        print(f"   ZeroMQ timeout: {zmq_timeout} ms")
        
        self.zmq_state_port = zmq_state_port
        self.zmq_ctrl_port = zmq_ctrl_port
        self.control_frequency = control_frequency
        self.no_viewer = no_viewer
        self.zmq_timeout = zmq_timeout
        
        # 1. Setup MuJoCo environment
        print("🤖 Setting up robot environment...")
        from hydrax import ROOT
        self.mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/g1/scene.xml")
        
        # Configure MuJoCo parameters
        self.mj_model.opt.timestep = 0.01
        self.mj_model.opt.iterations = 10
        self.mj_model.opt.ls_iterations = 50
        self.mj_model.opt.noslip_iterations = 2
        self.mj_model.opt.o_solimp = [0.0, 0.95, 0.01, 0.5, 2]
        self.mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE
        
        # Compute simulation parameters
        replan_period = 1.0 / control_frequency
        sim_steps_per_replan = int(replan_period / self.mj_model.opt.timestep)
        self.sim_steps_per_replan = max(sim_steps_per_replan, 1)
        self.actual_step_dt = self.sim_steps_per_replan * self.mj_model.opt.timestep
        
        print(f"   MuJoCo timestep: {self.mj_model.opt.timestep:.4f}s")
        print(f"   MuJoCo steps per control cycle: {self.sim_steps_per_replan}")
        print(f"   Actual control period: {self.actual_step_dt:.4f}s")
        
        self.mj_data = mujoco.MjData(self.mj_model)
        
        # 2. Setup viewer
        self.viewer = None
        if not no_viewer:
            self._setup_viewer()
        else:
            print("🚫 Viewer disabled")
            
        # 3. Setup bidirectional asynchronous ZeroMQ
        print("🌐 Setting up bidirectional asynchronous ZeroMQ...")
        self.context = zmq.Context()
        
        # High performance config
        self.context.setsockopt(zmq.MAX_SOCKETS, 1024)
        self.context.setsockopt(zmq.IO_THREADS, 2)
        
        # State send socket (PUSH)
        self.socket_state = self.context.socket(zmq.PUSH)
        self.socket_state.setsockopt(zmq.SNDHWM, 10)
        self.socket_state.setsockopt(zmq.SNDBUF, 1048576)
        self.socket_state.setsockopt(zmq.LINGER, 0)
        self.socket_state.connect(f"tcp://localhost:{zmq_state_port}")
        
        # Control receive socket (PULL)
        self.socket_ctrl = self.context.socket(zmq.PULL)
        self.socket_ctrl.setsockopt(zmq.RCVHWM, 10)
        self.socket_ctrl.setsockopt(zmq.RCVBUF, 1048576)
        self.socket_ctrl.setsockopt(zmq.LINGER, 0)
        self.socket_ctrl.connect(f"tcp://localhost:{zmq_ctrl_port}")
        
        # Poller setup
        self.poller = zmq.Poller()
        self.poller.register(self.socket_ctrl, zmq.POLLIN)
        
        print(f"✅ Asynchronous ZeroMQ connection established")
        
        # 4. State management
        self.running = False
        self.cycle_id = 0
        self.current_controls = None
        
        # 5. Enhanced timing statistics
        self.stats = OutlierFilteredStats(window_size=100)
        self.timing_history = []
        
        # Detailed timing variables
        self.t_exchange_start = 0.0
        self.t_state_send_end = 0.0
        self.t_ctrl_recv_start = 0.0
        self.t_ctrl_recv_end = 0.0
        self.t_sim_start = 0.0
        self.t_sim_end = 0.0
        
        # Statistics trackers
        self.state_send_stats = OutlierFilteredStats()
        self.ctrl_recv_stats = OutlierFilteredStats()
        self.sim_compute_stats = OutlierFilteredStats()
        self.cycle_latency_stats = OutlierFilteredStats()
        
    def _setup_viewer(self):
        """Setup viewer"""
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        print("👁️ Viewer setup complete")
    
    def get_robot_state(self) -> Dict[str, Any]:
        """Get current robot state"""
        return {
            'qpos': self.mj_data.qpos.copy(),
            'qvel': self.mj_data.qvel.copy(),
            'mocap_pos': self.mj_data.mocap_pos.copy(),
            'mocap_quat': self.mj_data.mocap_quat.copy(),
            'time': self.mj_data.time
        }
    
    def send_state_async(self, state: Dict[str, Any]) -> bool:
        """Asynchronously send state"""
        try:
            self.t_exchange_start = time.time()
            
            # Serialize state
            state_bytes = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Construct message with cycle_id
            cycle_id_bytes = struct.pack('I', self.cycle_id)  # uint32
            
            # Send multipart message [cycle_id, state_bytes]
            self.socket_state.send_multipart([cycle_id_bytes, state_bytes], zmq.NOBLOCK)
            
            self.t_state_send_end = time.time()
            
            # Record send time
            send_time = self.t_state_send_end - self.t_exchange_start
            self.state_send_stats.add_sample(send_time)
            
            return True
            
        except zmq.Again:
            print(f"⚠️ State send queue full, cycle_id={self.cycle_id}")
            return False
        except Exception as e:
            print(f"❌ State send error: {e}")
            return False
    
    def recv_controls_async(self) -> Optional[np.ndarray]:
        """Asynchronously receive control commands"""
        try:
            # Check if control command is available
            socks = dict(self.poller.poll(self.zmq_timeout))
            
            if self.socket_ctrl in socks:
                self.t_ctrl_recv_start = time.time()
                
                # Receive multipart message [cycle_id, controls_bytes]
                parts = self.socket_ctrl.recv_multipart(zmq.NOBLOCK)
                
                if len(parts) != 2:
                    print(f"⚠️ Received invalid control message format, parts={len(parts)}")
                    return None
                
                # Parse cycle_id
                recv_cycle_id = struct.unpack('I', parts[0])[0]
                
                # Fix: more relaxed cycle_id matching logic
                # Accept current or slightly older cycle_id (to account for async delay)
                if recv_cycle_id <= self.cycle_id and recv_cycle_id >= self.cycle_id - 2:
                    # Deserialize control command
                    response = pickle.loads(parts[1])
                    controls = response['controls']
                    
                    self.t_ctrl_recv_end = time.time()
                    
                    # Record receive time
                    recv_time = self.t_ctrl_recv_end - self.t_ctrl_recv_start
                    self.ctrl_recv_stats.add_sample(recv_time)
                    
                    print(f"✅ Cycle #{self.cycle_id}: Successfully received control command (from cycle#{recv_cycle_id})")
                    return controls
                else:
                    print(f"⚠️ Cycle ID deviation too large: expected~{self.cycle_id}, received={recv_cycle_id}")
                    return None
            
            return None
            
        except zmq.Again:
            return None
        except Exception as e:
            print(f"❌ Control receive error: {e}")
            return None
    
    def execute_simulation_steps(self, controls: np.ndarray):
        """Execute simulation steps"""
        self.t_sim_start = time.time()
        
        for i in range(self.sim_steps_per_replan):
            # Apply control command
            if i < len(controls):
                self.mj_data.ctrl[:] = controls[i]
            
            # Step simulation
            mujoco.mj_step(self.mj_model, self.mj_data)
            
            # Update viewer
            if self.viewer is not None:
                self.viewer.sync()
        
        self.t_sim_end = time.time()
        
        # Record simulation compute time
        sim_time = self.t_sim_end - self.t_sim_start
        self.sim_compute_stats.add_sample(sim_time)
    
    def calculate_cycle_metrics(self):
        """Calculate cycle metrics"""
        # Cycle latency = max(control receive end, sim end) - exchange start
        cycle_latency = max(self.t_ctrl_recv_end, self.t_sim_end) - self.t_exchange_start
        self.cycle_latency_stats.add_sample(cycle_latency)
        
        # Save detailed timing record
        timing_record = {
            'cycle_id': self.cycle_id,
            'exchange_start': self.t_exchange_start,
            'state_send_time': self.t_state_send_end - self.t_exchange_start,
            'ctrl_recv_time': self.t_ctrl_recv_end - self.t_ctrl_recv_start if self.t_ctrl_recv_end > 0 else 0,
            'sim_compute_time': self.t_sim_end - self.t_sim_start,
            'cycle_latency': cycle_latency
        }
        self.timing_history.append(timing_record)
    
    def print_periodic_stats(self):
        """Print periodic statistics"""
        print(f"\n🔍 === Asynchronous Simulation Statistics Report (Cycle #{self.cycle_id}) ===")
        
        # State send stats
        send_mean, send_std, send_valid, send_outliers = self.state_send_stats.get_filtered_stats()
        print(f"📤 State send: {send_mean*1000:.2f}±{send_std*1000:.2f}ms (valid:{send_valid}, outliers:{send_outliers})")
        
        # Control receive stats
        recv_mean, recv_std, recv_valid, recv_outliers = self.ctrl_recv_stats.get_filtered_stats()
        print(f"📥 Control receive: {recv_mean*1000:.2f}±{recv_std*1000:.2f}ms (valid:{recv_valid}, outliers:{recv_outliers})")
        
        # Simulation compute stats
        sim_mean, sim_std, sim_valid, sim_outliers = self.sim_compute_stats.get_filtered_stats()
        print(f"⚙️  Simulation compute: {sim_mean*1000:.2f}±{sim_std*1000:.2f}ms (valid:{sim_valid}, outliers:{sim_outliers})")
        
        # Cycle latency stats
        cycle_mean, cycle_std, cycle_valid, cycle_outliers = self.cycle_latency_stats.get_filtered_stats()
        print(f"🔄 Cycle latency: {cycle_mean*1000:.2f}±{cycle_std*1000:.2f}ms (valid:{cycle_valid}, outliers:{cycle_outliers})")
        
        # Estimated frequency
        if cycle_mean > 0:
            estimated_freq = 1.0 / cycle_mean
            print(f"📈 Estimated frequency: {estimated_freq:.1f} Hz (target: {self.control_frequency:.1f} Hz)")
        
        print("=" * 60)
    
    def run_simulation(self, duration: float = 10.0):
        """Run asynchronous simulation main loop"""
        print(f"🎬 Starting asynchronous simulation (duration: {duration}s)")
        print("💡 Working mode:")
        print("   - Bidirectional asynchronous ZeroMQ communication (PUSH/PULL)")
        print("   - Cycle ID barrier synchronization mechanism")
        print("   - Enhanced timing statistics and outlier filtering")
        
        self.running = True
        start_time = time.time()
        
        # Initialize default control command (PD control for standing pose)
        print("🦾 Setting default standing control...")
        # Use standing reference pose as default control target
        standing_qpos = np.array([
            0, 0, 0.75,  # root position (x, y, z)
            1, 0, 0, 0,  # root quaternion (w, x, y, z)
            0, 0, 0,     # waist joints
            0, 0, 0, 0, 0, 0,     # left arm
            0, 0, 0, 0, 0, 0,     # right arm
            0, 0, -0.3, 0.6, -0.3, 0,  # left leg (hip, knee, ankle)
            0, 0, -0.3, 0.6, -0.3, 0,  # right leg
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # fingers
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0   # fingers
        ])[:41]  # Ensure length matches
        
        # Create default PD control command (repeat standing pose)
        default_controls = np.tile(standing_qpos, (self.sim_steps_per_replan, 1)).astype(np.float32)
        self.current_controls = default_controls
        
        # Wait for control node to initialize
        print("🔄 Waiting for control node to be ready...")
        time.sleep(1.0)  # Give control node enough time to start
        
        try:
            while time.time() - start_time < duration:
                cycle_start_time = time.time()
                
                # Check viewer status
                if self.viewer is not None and not self.viewer.is_running():
                    break
                
                # ========== Barrier A: Try to receive new control command first ==========
                new_controls = self.recv_controls_async()
                if new_controls is not None:
                    self.current_controls = new_controls
                    print(f"🎮 Cycle #{self.cycle_id}: Updated control command")
                
                # ========== Barrier B: Send current state ==========
                state = self.get_robot_state()
                if not self.send_state_async(state):
                    print(f"❌ Cycle #{self.cycle_id}: State send failed")
                    continue
                
                # ========== Execute simulation steps ==========
                self.execute_simulation_steps(self.current_controls)
                
                # ========== Calculate metrics ==========
                # Fix: ensure t_ctrl_recv_end has a valid value
                if self.t_ctrl_recv_end == 0:
                    self.t_ctrl_recv_end = self.t_sim_end  # Use sim end time as fallback
                
                self.calculate_cycle_metrics()
                
                # ========== Print statistics periodically ==========
                if (self.cycle_id + 1) % 20 == 0:
                    self.print_periodic_stats()
                
                # ========== Advance to next cycle ==========
                self.cycle_id += 1
                
                # ========== Frequency control ==========
                cycle_elapsed = time.time() - cycle_start_time
                target_cycle_time = 1.0 / self.control_frequency
                
                sleep_time = target_cycle_time - cycle_elapsed
                if sleep_time > 0.001:
                    time.sleep(sleep_time)
                elif sleep_time < -0.005:
                    if self.cycle_id % 50 == 0:
                        print(f"⚠️ Cycle #{self.cycle_id}: Delay {-sleep_time*1000:.1f}ms")
                        
        except KeyboardInterrupt:
            print("\n🛑 Received interrupt signal, stopping simulation...")
        
        finally:
            self.stop()
    
    def stop(self):
        """Stop simulation"""
        self.running = False
        
        # Close network connections
        self.socket_state.close()
        self.socket_ctrl.close()
        self.context.term()
        
        # Print final statistics
        print(f"\n🏁 === Asynchronous Simulation Final Report ===")
        print(f"📊 Overall statistics:")
        print(f"   Total cycles: {self.cycle_id}")
        print(f"   Timing records: {len(self.timing_history)}")
        
        if len(self.timing_history) > 0:
            # Final filtered statistics
            send_mean, send_std, _, _ = self.state_send_stats.get_filtered_stats()
            recv_mean, recv_std, _, _ = self.ctrl_recv_stats.get_filtered_stats()
            sim_mean, sim_std, _, _ = self.sim_compute_stats.get_filtered_stats()
            cycle_mean, cycle_std, _, _ = self.cycle_latency_stats.get_filtered_stats()
            
            print(f"\n📈 Final performance metrics (after outlier filtering):")
            print(f"   State send time: {send_mean*1000:.2f}±{send_std*1000:.2f}ms")
            print(f"   Control receive time: {recv_mean*1000:.2f}±{recv_std*1000:.2f}ms")
            print(f"   Simulation compute time: {sim_mean*1000:.2f}±{sim_std*1000:.2f}ms")
            print(f"   Cycle latency: {cycle_mean*1000:.2f}±{cycle_std*1000:.2f}ms")
            
            if cycle_mean > 0:
                estimated_freq = 1.0 / cycle_mean
                print(f"   Estimated system frequency: {estimated_freq:.2f} Hz")
                print(f"   Target frequency achievement: {estimated_freq/self.control_frequency*100:.1f}%")
        
        print("✅ Asynchronous simulation stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Asynchronous MuJoCo Simulation Endpoint - Bidirectional Asynchronous ZeroMQ Communication"
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
    parser.add_argument(
        "--control_frequency",
        type=float,
        default=50.0,
        help="Control frequency (Hz)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="Simulation duration (seconds)"
    )
    parser.add_argument(
        "--no_viewer",
        action="store_true",
        help="Disable viewer"
    )
    parser.add_argument(
        "--zmq_timeout",
        type=int,
        default=100,
        help="ZeroMQ timeout (milliseconds)"
    )

    args = parser.parse_args()

    print(f"🚀 Starting asynchronous MuJoCo simulation")
    print("Please ensure the control server is running")
    print("(Run first: python run_policy_pruned.py)")
    
    # Create and run asynchronous simulation
    simulation = AsyncSimulation(
        zmq_state_port=args.zmq_state_port,
        zmq_ctrl_port=args.zmq_ctrl_port,
        control_frequency=args.control_frequency,
        no_viewer=args.no_viewer,
        zmq_timeout=args.zmq_timeout
    )
    
    simulation.run_simulation(duration=args.duration)


if __name__ == "__main__":
    main() 