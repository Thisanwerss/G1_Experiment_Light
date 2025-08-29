#!/usr/bin/env python3
"""
Isolated MuJoCo simulation client - communicates with control server via ZeroMQ

Usage:
    python isolated_simulation.py --zmq_port 5555

Workflow:
1. Initialize MuJoCo simulation environment (same as run_policy.py)
2. Connect to ZeroMQ control server
3. Send robot state and receive control commands at 50Hz
4. Run simulation at 100Hz, strictly following timing
5. Render visualization (optional)

This script is fully standalone, requires only MuJoCo and ZeroMQ (no JAX dependency)
"""

import argparse
import time
from typing import Optional, Dict, Any

import numpy as np
import mujoco
import mujoco.viewer
import zmq


class FrequencyMonitor:
    """Simulation frequency monitor (simplified)"""
    
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.sim_step_times = []
        self.control_request_times = []
        self.control_receive_times = []
        
        self.total_sim_steps = 0
        self.total_control_requests = 0
        self.start_time = None
        
    def start_monitoring(self):
        self.start_time = time.time()
        print("🔍 === Simulation Frequency Monitor Started ===")
        
    def record_sim_step(self):
        self.sim_step_times.append(time.time())
        self.total_sim_steps += 1
        
    def record_control_request(self):
        self.control_request_times.append(time.time())
        self.total_control_requests += 1
        
    def record_control_receive(self):
        self.control_receive_times.append(time.time())
    
    def calculate_frequency(self, timestamps):
        if len(timestamps) < 2:
            return 0.0, 0.0
            
        recent_times = timestamps[-self.window_size:] if len(timestamps) >= self.window_size else timestamps
        
        if len(recent_times) < 2:
            return 0.0, 0.0
            
        intervals = np.diff(recent_times)
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        freq = 1.0 / avg_interval if avg_interval > 0 else 0.0
        freq_std = std_interval / (avg_interval**2) if avg_interval > 0 else 0.0
        
        return freq, freq_std
        
    def get_overall_frequencies(self):
        if self.start_time is None:
            return {}
            
        elapsed = time.time() - self.start_time
        
        return {
            'sim_steps': self.total_sim_steps / elapsed if elapsed > 0 else 0.0,
            'control_requests': self.total_control_requests / elapsed if elapsed > 0 else 0.0,
        }
        
    def print_report(self, target_sim_freq=100.0, target_control_freq=50.0):
        sim_freq, sim_std = self.calculate_frequency(self.sim_step_times)
        control_freq, control_std = self.calculate_frequency(self.control_request_times)
        
        overall_freqs = self.get_overall_frequencies()
        
        print(f"\n🔍 === Simulation Frequency Report ===")
        print(f"📊 Instantaneous Frequency:")
        print(f"   ⚙️  Simulation Step Frequency: {sim_freq:.2f} ± {sim_std:.2f} Hz")
        print(f"   📡 Control Request Frequency: {control_freq:.2f} ± {control_std:.2f} Hz")
        
        print(f"\n📈 Overall Average Frequency:")
        print(f"   ⚙️  Simulation Step Frequency: {overall_freqs['sim_steps']:.2f} Hz")
        print(f"   📡 Control Request Frequency: {overall_freqs['control_requests']:.2f} Hz")
        
        print(f"\n🎯 Target Comparison:")
        print(f"   Simulation Frequency Achievement: {sim_freq/target_sim_freq*100:.1f}% (Target: {target_sim_freq:.1f} Hz)")
        print(f"   Control Frequency Achievement: {control_freq/target_control_freq*100:.1f}% (Target: {target_control_freq:.1f} Hz)")
        print("=" * 50)


class IsolatedSimulation:
    """Isolated MuJoCo Simulation"""
    
    def __init__(
        self,
        zmq_server_address: str = "tcp://localhost:5555",
        control_frequency: float = 50.0,
        simulation_frequency: float = 100.0,
        show_reference: bool = False,
        no_viewer: bool = False,
        zmq_timeout: int = 1000  # ms
    ):
        """Initialize isolated simulation"""
        print(f"🚀 Initializing Isolated MuJoCo Simulation")
        print(f"   Control Server: {zmq_server_address}")
        print(f"   Control Frequency: {control_frequency} Hz")
        print(f"   Simulation Frequency: {simulation_frequency} Hz")
        print(f"   ZeroMQ Timeout: {zmq_timeout} ms")
        
        self.zmq_server_address = zmq_server_address
        self.control_frequency = control_frequency
        self.simulation_frequency = simulation_frequency
        self.show_reference = show_reference
        self.no_viewer = no_viewer
        self.zmq_timeout = zmq_timeout
        
        # Timing parameters
        self.control_frequency = control_frequency
        self.simulation_frequency = simulation_frequency
        
        # 1. Setup lightweight MuJoCo environment (avoid re-initializing HumanoidStand)
        print("🤖 Setting up robot environment...")
        print("   (Directly loading MuJoCo model, no need to regenerate reference sequence)")
        
        # Directly load MuJoCo model, do not initialize full HumanoidStand task
        from hydrax import ROOT
        self.mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/g1/scene.xml")
        
        # Configure MuJoCo parameters (identical to run_policy.py)
        self.mj_model.opt.timestep = 0.01  # Keep same timestep as run_policy.py
        self.mj_model.opt.iterations = 10
        self.mj_model.opt.ls_iterations = 50
        self.mj_model.opt.noslip_iterations = 2
        self.mj_model.opt.o_solimp = [0.0, 0.95, 0.01, 0.5, 2]
        self.mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE
        
        # Compute actual simulation parameters (same as run_policy.py)
        replan_period = 1.0 / control_frequency
        sim_steps_per_replan = int(replan_period / self.mj_model.opt.timestep)
        self.sim_steps_per_replan = max(sim_steps_per_replan, 1)
        self.actual_step_dt = self.sim_steps_per_replan * self.mj_model.opt.timestep
        
        print(f"   MuJoCo Timestep: {self.mj_model.opt.timestep:.4f}s")
        print(f"   MuJoCo Steps per Control Cycle: {self.sim_steps_per_replan}")
        print(f"   Actual Control Cycle: {self.actual_step_dt:.4f}s")
        
        self.mj_data = mujoco.MjData(self.mj_model)
        
        # 2. Setup reference trajectory and ghost
        # Reference trajectory is not needed on simulation side, as control is computed on server
        self.reference = None
        self.viewer = None
        self.ref_data = None
        self.vopt = None
        self.pert = None
        self.catmask = None
        
        if not no_viewer:
            self._setup_viewer()
        else:
            print("🚫 Rendering disabled")
            
        # 3. Setup ZeroMQ client
        print("🌐 Setting up ZeroMQ client...")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, zmq_timeout)
        self.socket.setsockopt(zmq.SNDTIMEO, zmq_timeout)
        self.socket.connect(zmq_server_address)
        print(f"✅ Connected to control server: {zmq_server_address}")
        
        # 4. State management
        self.running = False
        self.current_controls = None
        self.control_step = 0
        self.last_control_time = 0.0
        
        # 5. Statistics
        self.stats = {
            'total_control_requests': 0,
            'total_control_timeouts': 0,
            'total_communication_time': 0.0,
            'max_communication_time': 0.0,
            'total_sim_steps': 0
        }
        
        # 6. Frequency monitor
        self.freq_monitor = FrequencyMonitor(window_size=20)
        
    def _setup_viewer(self):
        """Setup renderer (simplified, no ghost reference)"""
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        print("👁️ Renderer setup complete (no ghost reference)")
    
    def get_robot_state(self) -> Dict[str, Any]:
        """Get current robot state"""
        return {
            'qpos': self.mj_data.qpos.copy(),
            'qvel': self.mj_data.qvel.copy(),
            'mocap_pos': self.mj_data.mocap_pos.copy(),
            'mocap_quat': self.mj_data.mocap_quat.copy(),
            'time': self.mj_data.time
        }
    
    def request_controls(self, state: Dict[str, Any]) -> Optional[np.ndarray]:
        """Request control commands via ZeroMQ"""
        self.freq_monitor.record_control_request()
        
        try:
            comm_start = time.time()
            
            # Send state
            self.socket.send_pyobj(state)
            
            # Receive control command
            response = self.socket.recv_pyobj()
            
            comm_time = time.time() - comm_start
            
            self.freq_monitor.record_control_receive()
            
            # Update statistics
            self.stats['total_control_requests'] += 1
            self.stats['total_communication_time'] += comm_time
            self.stats['max_communication_time'] = max(self.stats['max_communication_time'], comm_time)
            
            # Extract control command
            controls = response['controls']
            timing_info = response.get('timing', {})
            server_stats = response.get('stats', {})
            
            # Print communication info
            if self.stats['total_control_requests'] % 10 == 1:
                avg_comm_time = self.stats['total_communication_time'] / self.stats['total_control_requests'] * 1000
                server_compute_time = timing_info.get('total_time', 0) * 1000
                print(f"📡 Control Request #{self.stats['total_control_requests']}: "
                      f"Comm Time={comm_time*1000:.1f}ms, "
                      f"Server Compute={server_compute_time:.1f}ms, "
                      f"Avg Comm={avg_comm_time:.1f}ms")
            
            return controls
            
        except zmq.Again:
            # Timeout
            self.stats['total_control_timeouts'] += 1
            print(f"⚠️ Control request timeout (#{self.stats['total_control_timeouts']})")
            return None
            
        except Exception as e:
            print(f"❌ Control request error: {e}")
            return None
    
    def update_ghost_reference(self):
        """Update ghost reference trajectory (no-op in simulation client)"""
        # No ghost reference on simulation client, reference data is on control server
        pass
    
    def run_simulation(self, duration: float = 10.0):
        """Run main loop of isolated simulation (strictly follows run_policy.py logic)"""
        print(f"🎬 Starting isolated simulation (duration: {duration}s)")
        print("💡 Working mode:")
        print("   - MuJoCo simulation: local")
        print("   - Control computation: remote ZeroMQ server")
        print("   - Decoupled architecture: fully separated")
        print(f"   - Control frequency: {self.control_frequency}Hz ({self.sim_steps_per_replan} steps/control)")
        print(f"   - Strict frequency alignment: request control first, then simulate")
        
        self.freq_monitor.start_monitoring()
        self.running = True
        
        start_time = time.time()
        cycle_count = 0
        
        try:
            while time.time() - start_time < duration:
                cycle_start_time = time.time()
                
                # Check viewer status
                if self.viewer is not None and not self.viewer.is_running():
                    break
                
                # ========== 1. Control planning phase ==========
                # Request new control command (every loop, strictly at 50Hz)
                print(f"📡 Cycle #{cycle_count+1}: Requesting control command...")
                state = self.get_robot_state()
                controls = self.request_controls(state)
                
                if controls is None:
                    print("❌ Control request failed, terminating simulation")
                    break
                
                # ========== 2. Simulation execution phase ==========
                # Execute fixed number of simulation steps (identical to run_policy.py)
                for i in range(self.sim_steps_per_replan):
                    # Apply control command
                    if i < len(controls):
                        self.mj_data.ctrl[:] = controls[i]
                    
                    # Step simulation
                    self.freq_monitor.record_sim_step()
                    mujoco.mj_step(self.mj_model, self.mj_data)
                    self.stats['total_sim_steps'] += 1
                    
                    # Update rendering
                    if self.viewer is not None:
                        self.update_ghost_reference()
                        self.viewer.sync()
                
                cycle_count += 1
                
                # ========== 3. Frequency control phase ==========
                # Frequency monitoring
                if cycle_count % 20 == 0:
                    self.freq_monitor.print_report()
                    elapsed_total = time.time() - start_time
                    actual_freq = cycle_count / elapsed_total
                    print(f"⚡ Cycle #{cycle_count}: Actual frequency={actual_freq:.1f}Hz")
                
                # Strict timing control (same as run_policy.py)
                cycle_elapsed = time.time() - cycle_start_time
                target_cycle_time = 1.0 / self.control_frequency  # Strictly follow control frequency
                if cycle_elapsed < target_cycle_time:
                    time.sleep(target_cycle_time - cycle_elapsed)
                    
        except KeyboardInterrupt:
            print("\n🛑 Received interrupt signal, stopping simulation...")
        
        finally:
            self.stop()
    
    def stop(self):
        """Stop simulation"""
        self.running = False
        
        # Close network connection
        self.socket.close()
        self.context.term()
        
        # Print final statistics
        elapsed_total = time.time() - self.freq_monitor.start_time if self.freq_monitor.start_time else 1.0
        
        print(f"\n🏁 === Final Simulation Report ===")
        print(f"📊 Overall Statistics:")
        print(f"   Run Time: {elapsed_total:.2f}s")
        print(f"   Total Simulation Steps: {self.stats['total_sim_steps']}")
        print(f"   Total Control Requests: {self.stats['total_control_requests']}")
        print(f"   Control Timeouts: {self.stats['total_control_timeouts']}")
        
        if self.stats['total_control_requests'] > 0:
            avg_comm_time = self.stats['total_communication_time'] / self.stats['total_control_requests'] * 1000
            timeout_rate = self.stats['total_control_timeouts'] / self.stats['total_control_requests'] * 100
            print(f"   Average Communication Time: {avg_comm_time:.2f}ms")
            print(f"   Max Communication Time: {self.stats['max_communication_time']*1000:.2f}ms")
            print(f"   Timeout Rate: {timeout_rate:.1f}%")
        
        avg_sim_freq = self.stats['total_sim_steps'] / elapsed_total
        avg_control_freq = self.stats['total_control_requests'] / elapsed_total
        print(f"   Average Simulation Frequency: {avg_sim_freq:.1f} Hz")
        print(f"   Average Control Frequency: {avg_control_freq:.1f} Hz")
        
        self.freq_monitor.print_report()
        print("✅ Isolated simulation stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Isolated MuJoCo simulation client - communicates with control server via ZeroMQ"
    )
    parser.add_argument(
        "--zmq_server",
        type=str,
        default="tcp://localhost:5555",
        help="ZeroMQ control server address"
    )
    parser.add_argument(
        "--control_frequency",
        type=float,
        default=50.0,
        help="Control request frequency (Hz)"
    )
    parser.add_argument(
        "--simulation_frequency",
        type=float,
        default=100.0,
        help="MuJoCo simulation frequency (Hz)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Simulation duration (seconds)"
    )
    # Note: Simulation client no longer supports showing reference trajectory, as reference data is on control server
    # parser.add_argument(
    #     "--show_reference",
    #     action="store_true", 
    #     help="Show reference trajectory ghost"
    # )
    parser.add_argument(
        "--no_viewer",
        action="store_true",
        help="Disable renderer"
    )
    parser.add_argument(
        "--zmq_timeout",
        type=int,
        default=1000,
        help="ZeroMQ timeout (ms)"
    )

    args = parser.parse_args()

    print(f"🚀 Starting Isolated MuJoCo Simulation")
    print(f"Please ensure the control server is running at {args.zmq_server}")
    print("(Start first: python run_policy_pruned.py)")
    
    # Create and run isolated simulation
    simulation = IsolatedSimulation(
        zmq_server_address=args.zmq_server,
        control_frequency=args.control_frequency,
        simulation_frequency=args.simulation_frequency,
        show_reference=False,  # Simulation client no longer supports ghost reference
        no_viewer=args.no_viewer,
        zmq_timeout=args.zmq_timeout
    )
    
    simulation.run_simulation(duration=args.duration)


if __name__ == "__main__":
    main() 