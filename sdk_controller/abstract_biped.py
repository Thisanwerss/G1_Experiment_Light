import time
import numpy as np
import mujoco
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import json
import os
import threading

# --- Color Printing Utility ---
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(tag, message):
    tag_map = {
        "ERROR": bcolors.FAIL,
        "WARNING": bcolors.WARNING,
        "SUCCEED": bcolors.OKGREEN,
        "INFO": bcolors.OKBLUE,
    }
    color = tag_map.get(tag, bcolors.ENDC)
    print(f"{color}[{tag}]{bcolors.ENDC} {message}")


# HG series DDS message imports
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_, unitree_hg_msg_dds__HandState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_, HandCmd_, HandState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_, TwistStamped_

from mj_pin.utils import get_robot_description, mj_joint_name2act_id, mj_joint_name2dof
from sdk_controller.robots.G1 import *
from sdk_controller.topics import *

# Import basic safety layer
try:
    from sdk_controller.safety import SafetyLayer, _load_safety_config
except:
    from safety import SafetyLayer, _load_safety_config

# HG series specific DDS topics
HG_TOPIC_LOWCMD = "rt/lowcmd"
HG_TOPIC_LOWSTATE = "rt/lowstate"
HG_TOPIC_HANDCMD = "rt/handcmd"  
HG_TOPIC_HANDSTATE = "rt/handstate"
# Note: G1 doesn't have SportModeState and WirelessController, these features are implemented through other means


def load_global_config():
    """Load global configuration from JSON file"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "global_config.json")
    with open(config_path, 'r') as f:
        return json.load(f)


class HGSafetyLayer(SafetyLayer):
    """G1 specific safety layer - inherits from base SafetyLayer, adds torque limiting based on PD gains and position errors"""
    
    def __init__(self, mj_model, robot_config=None, safety_profile: str = "default"):
        """
        Initialize G1 safety layer
        
        Args:
            mj_model: MuJoCo model
            robot_config: G1 robot configuration (G1.py module), if None uses G1 default configuration
            safety_profile: The safety profile to use.
        """
        # Load global configuration for ignored joints and robot specific configs
        self.global_config = load_global_config()
        ignored_joints = set(self.global_config.get("safety_ignored_joints", []))
        
        # Initialize base SafetyLayer, passing the profile and ignored joints
        super().__init__(mj_model, safety_profile=safety_profile, inactive_joint_names=ignored_joints)
        
        self.robot_config = robot_config
        
        # Load G1-specific safety parameters from the dedicated config file
        safety_config = _load_safety_config()
        profile_config = safety_config[safety_profile]
        
        if safety_profile == "conservative":
            print("======================================================================")
            print_colored("WARNING", "G1 Safety Layer running in [Conservative Mode] - thresholds tightened!")
            print("======================================================================")
        
        self.max_position_error = profile_config["max_position_error_rad"]
        self.torque_static_scale = profile_config["torque_static_scale"]
        self.torque_mode_scale = profile_config["torque_mode_scale"]
        self.joint_limit_static_scale = profile_config["joint_limit_static_scale"]
        self.joint_limit_mode_scale = profile_config["joint_limit_mode_scale"]

        # Re-setup G1 specific torque limits, overriding base class settings
        self._setup_g1_torque_limits()
        # Setup G1 specific joint limits with scaling
        self._setup_g1_joint_limits()
        
    def _setup_g1_torque_limits(self):
        """Setup torque limits for each joint based on global configuration"""
        # Clear base class torque limits, use G1 specific settings
        self.torque_limits.clear()
        
        joint_config = self.global_config["g1_joint_config"]
        total_scale_factor = self.torque_static_scale * self.torque_mode_scale
        
        for joint_name, config in joint_config.items():
            mj_idx = config["mujoco_index"]
            dds_idx = config["dds_index"]
            max_torque = config["max_torque"]
            
            if mj_idx < NUM_ACTIVE_BODY_JOINTS:
                self.torque_limits[dds_idx] = max_torque * total_scale_factor
        
        print_colored("SUCCEED", f"G1 Safety Layer: Set torque limits for {len(self.torque_limits)} joints (total scale: {total_scale_factor:.3f})")

    def get_joint_limits_from_model(self) -> Dict[int, Tuple[float, float]]:
        """Helper to read raw joint limits from the MuJoCo model."""
        limits = {}
        base_dofs = 6  # Floating base
        for id in range(self.mj_model.njnt):
            if self.mj_model.jnt_limited[id]:
                dof = int(self.mj_model.joint(id).dofadr)
                if dof >= base_dofs:
                    # Filter out inactive joints using the name list
                    joint_name = self.mj_model.joint(id).name
                    if joint_name not in self.inactive_joint_names:
                        relative_joint_id = dof - base_dofs
                        min_val = self.mj_model.jnt_range[id][0]
                        max_val = self.mj_model.jnt_range[id][1]
                        limits[relative_joint_id] = (min_val, max_val)
        return limits

    def _setup_g1_joint_limits(self):
        """
        Reads the original joint limits from the MuJoCo model and applies scaling factors.
        This allows for dynamically tightening the operational range of joints for added safety.
        """
        total_scale_factor = self.joint_limit_static_scale * self.joint_limit_mode_scale

        original_limits = self.get_joint_limits_from_model()
        self.joint_limits.clear()

        for joint_id, (q_min, q_max) in original_limits.items():
            # For symmetric ranges, scale them towards zero
            if np.isclose(q_min, -q_max):
                scaled_min = q_min * total_scale_factor
                scaled_max = q_max * total_scale_factor
            else:
                # For asymmetric ranges, shrink the range towards its center
                center = (q_min + q_max) / 2
                span = (q_max - q_min) / 2
                scaled_span = span * total_scale_factor
                scaled_min = center - scaled_span
                scaled_max = center + scaled_span

            self.joint_limits[joint_id] = (scaled_min, scaled_max)

        print_colored("SUCCEED", f"G1 Safety Layer: Set scaled joint limits for {len(self.joint_limits)} joints (total scale: {total_scale_factor:.2f})")
    
    def check_safety(self, q_current: np.ndarray, q_target: np.ndarray, 
                    kp_gains: np.ndarray, base_quaternion: np.ndarray) -> bool:
        """
        Check safety: inherits base class method and adds G1 specific checks
        Calculate potential torques based on PD gains and position errors
        
        Args:
            q_current: Current joint positions
            q_target: Target joint positions
            kp_gains: PD gains
            base_quaternion: Base quaternion
            
        Returns:
            bool: Whether it is safe
        """
        # 1. First execute base class safety check
        # Note: base class check_safety needs torques parameter, we calculate potential torques here
        potential_torques = self._calculate_potential_torques(q_current, q_target, kp_gains)
        base_safe = super().check_safety(q_current, potential_torques, base_quaternion)
        
        if not base_safe:
            return False
            
        # 2. Check G1 specific position errors
        if not self._check_position_errors(q_current, q_target):
            return False
            
        # 3. Check G1 specific potential torques (more detailed check)
        if not self._check_potential_torques(q_current, q_target, kp_gains):
            return False
            
        return True
    
    def _calculate_potential_torques(self, q_current: np.ndarray, q_target: np.ndarray, 
                                   kp_gains: np.ndarray) -> list:
        """Calculate potential torques for base class safety check"""
        torques = [0.0] * len(self.torque_limits)
        
        for mj_idx, dds_idx in BODY_MUJOCO_TO_DDS.items():
            if (mj_idx < len(q_current) and mj_idx < len(q_target) and 
                mj_idx < len(kp_gains) and dds_idx in self.torque_limits):
                
                potential_torque = kp_gains[mj_idx] * (q_target[mj_idx] - q_current[mj_idx])
                if dds_idx < len(torques):
                    torques[dds_idx] = potential_torque
        
        return torques
    
    def _check_position_errors(self, q_current: np.ndarray, q_target: np.ndarray) -> bool:
        """Check if joint position errors are too large"""
        for mj_idx, dds_idx in BODY_MUJOCO_TO_DDS.items():
            if mj_idx < len(q_current) and mj_idx < len(q_target):
                error = abs(q_target[mj_idx] - q_current[mj_idx])
                if error > self.max_position_error:
                    joint_name = MUJOCO_JOINT_NAMES[mj_idx] if mj_idx < len(MUJOCO_JOINT_NAMES) else f"MJ ID {mj_idx}"
                    print_colored("ERROR", f"[Safety Layer] Joint target position jump too large! Joint: {joint_name} | Error: {error:.3f} > Limit: {self.max_position_error:.3f} rad")
                    return False
        return True
    
    def _check_potential_torques(self, q_current: np.ndarray, q_target: np.ndarray, 
                               kp_gains: np.ndarray) -> bool:
        """Check if potential torques generated by PD control exceed limits"""
        for mj_idx, dds_idx in BODY_MUJOCO_TO_DDS.items():
            if (mj_idx < len(q_current) and mj_idx < len(q_target) and 
                mj_idx < len(kp_gains) and dds_idx in self.torque_limits):
                
                potential_torque = abs(kp_gains[mj_idx] * (q_target[mj_idx] - q_current[mj_idx]))
                max_torque = self.torque_limits[dds_idx]
                
                if potential_torque > max_torque:
                    joint_name = MUJOCO_JOINT_NAMES[mj_idx] if mj_idx < len(MUJOCO_JOINT_NAMES) else f"MJ ID {mj_idx}"
                    print_colored("ERROR", f"[Safety Layer] Potential torque exceeds limit! Joint: {joint_name} | Estimated: {potential_torque:.1f} > Limit: {max_torque:.1f} Nm")
                    return False
        return True
    
    # Inherit other methods from base class:
    # - check_joint_limits()
    # - check_torque_limits()  
    # - check_base_orientation()
    # These methods can be used directly, can be overridden if G1 specific behavior is needed


class HGSDKControllerBase(ABC):
    """HG series DDS communication base class"""
    
    def __init__(self, wait_for_subscribers: bool = True):
        super().__init__()
        self.wait_for_subscribers = wait_for_subscribers
        
        self.last_low_state = None
        self.last_hand_state = None
        self.last_vicon_pose = None
        self.last_vicon_twist = None
        
        self.crc = CRC()
        
        # Initialize body control command
        self.cmd = unitree_hg_msg_dds__LowCmd_()
        self.cmd.mode_pr = 0
        self.cmd.mode_machine = 0
        for i in range(35):  # HG LowCmd has 35 motor slots
            self.cmd.motor_cmd[i].mode = 1  # Enable mode
            self.cmd.motor_cmd[i].q = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = 0.0
        
        # Initialize hand control command
        self.hand_cmd = unitree_hg_msg_dds__HandCmd_()
        # HandCmd's motor_cmd is sequence type, need to add elements dynamically
        
        # Create DDS publishers
        self.pub = ChannelPublisher(HG_TOPIC_LOWCMD, LowCmd_)
        self.pub.Init()
        
        self.hand_pub = ChannelPublisher(HG_TOPIC_HANDCMD, HandCmd_)
        self.hand_pub.Init()
        
        # Create DDS subscribers
        low_state_sub = ChannelSubscriber(HG_TOPIC_LOWSTATE, LowState_)
        hand_state_sub = ChannelSubscriber(HG_TOPIC_HANDSTATE, HandState_)
        low_state_sub.Init(self.low_state_handler, 10)
        hand_state_sub.Init(self.hand_state_handler, 10)
        
        self.wait_subscriber(required=self.wait_for_subscribers)
        print_colored("INFO", "HG Controller Ready!")
        
    def wait_subscriber(self, required: bool = True) -> bool:
        """Wait for DDS subscribers to connect"""
        if not required:
            print_colored("WARNING", "Lo mode: Skip DDS subscriber waiting")
            return True
            
        timeout = 15.0
        t = 0.0
        sleep = 0.1
        while t < timeout:
            if self.last_low_state is not None:
                return True
            t += sleep    
            time.sleep(sleep)
            
        raise TimeoutError("Did not receive G1 robot state message")
            
    def low_state_handler(self, msg: LowState_):
        """Low-level state message handler"""
        self.last_low_state = msg
        
    def hand_state_handler(self, msg: HandState_):
        """Hand state message handler"""
        self.last_hand_state = msg
        
    def vicon_pose_handler(self, msg: PoseStamped_):
        """Vicon pose message handler"""
        self.last_vicon_pose = msg

    def vicon_twist_handler(self, msg: TwistStamped_):
        """Vicon velocity message handler"""
        self.last_vicon_twist = msg
        
    # TODO: Gamepad input handling - future direct control from PC terminal
    # def wireless_handler(self, msg):
    #     """Gamepad input handler - PLACEHOLDER"""
    #     print("⚠️ TODO: Implement PC terminal gamepad input interface")
    #     pass


class HGSDKController(HGSDKControllerBase):
    """HG series SDK controller - G1 specific implementation"""
    
    def __init__(self,
                 simulate: bool,
                 robot_config,
                 xml_path: str = "",
                 vicon_required: bool = True,
                 lo_mode: bool = False,
                 kp_scale_factor: float = 1.0,
                 safety_profile: str = "default"):
        
        self.simulate = simulate
        self.robot_config = robot_config
        self.vicon_required = vicon_required
        self.lo_mode = lo_mode
        self.kp_scale_factor = kp_scale_factor
        print_colored("SUCCEED", f"G1 Kp scale factor: {self.kp_scale_factor}")
        
        # Load global configuration
        self.global_config = load_global_config()
        
        # --- NEW: Setup for disabled joints ---
        self.disabled_joint_names = set(self.global_config.get("disabled_joints", []))
        self.disabled_mj_indices = set()
        if self.disabled_joint_names:
            print_colored("INFO", f"Disabling control for joints: {list(self.disabled_joint_names)}")
            # G1_JOINT_CONFIG is imported from G1.py
            for joint_name, config in G1_JOINT_CONFIG.items():
                if joint_name in self.disabled_joint_names:
                    self.disabled_mj_indices.add(config["mujoco_index"])
        # --- END NEW ---
        
        # If Vicon is required but not provided, raise exception
        if self.vicon_required and not simulate:
            print_colored("INFO", "Vicon is required. Setting up subscribers...")
            
        # Initialize robot model
        if not xml_path:
            if robot_config is not None:
                desc = get_robot_description(robot_config.ROBOT_NAME)
                xml_path = desc.xml_scene_path
            else:
                # Use G1 default configuration
                desc = get_robot_description(ROBOT_NAME)
                xml_path = desc.xml_scene_path
        
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.nu = mj_model.nu
        self.nq = mj_model.nq
        self.nv = mj_model.nv
        self.off = self.nq - self.nv
        
        # State vectors
        self._q = np.zeros(self.nq)
        self._v = np.zeros(self.nv)
        
        # Build joint mapping - use mapping defined in G1.py
        self.joint_dof2act_id = {}
        self.joint_act_id2dof = {}
        
        # Body joint mapping (MuJoCo DoF -> DDS index)
        for mj_idx, dds_idx in BODY_MUJOCO_TO_DDS.items():
            if mj_idx < NUM_ACTIVE_BODY_JOINTS:  # 27 active body joints
                dof_idx = 6 + mj_idx  # First 6 DoF in MuJoCo are base
                self.joint_dof2act_id[dof_idx] = dds_idx
                self.joint_act_id2dof[dds_idx] = dof_idx
        
        print_colored("SUCCEED", f"G1 Joint Mapping: {len(self.joint_dof2act_id)} body joints")
        
        # Safety layer
        # Use configuration from current module, if no robot_config is passed
        config_to_use = robot_config if robot_config is not None else None
        self.safety = HGSafetyLayer(mj_model, config_to_use, safety_profile=safety_profile)
        
        # Control state management
        self.controller_running = False
        self.damping_running = False
        
        # Initialize DDS communication
        super().__init__(wait_for_subscribers=not self.lo_mode)

        # Set up additional subscribers for Vicon data
        if self.vicon_required and not self.simulate:
            vicon_pose_sub = ChannelSubscriber(TOPIC_VICON_POSE, PoseStamped_)
            vicon_twist_sub = ChannelSubscriber(TOPIC_VICON_TWIST, TwistStamped_)
            vicon_pose_sub.Init(self.vicon_pose_handler, 10)
            vicon_twist_sub.Init(self.vicon_twist_handler, 10)
            print_colored("SUCCEED", "Vicon subscribers for Pose and Twist are set up.")
        
        # In lo mode, don't wait for real DDS messages, but need to publish dummy state
        if self.lo_mode:
            print_colored("INFO", "Lo mode: DDS communication initialized, but not waiting for real state messages")
            self._setup_lo_mode_dummy_publisher()

    def update_q_v_from_lowstate(self):
        """Extract joint states from HG LowState message"""
        if self.last_low_state is None:
            return
            
        # TODO: IMU data processing - currently left empty, rely on Vicon
        # if not self.simulate:
        #     # Base orientation (from IMU)
        #     self._q[3:7] = self.last_low_state.imu_state.quaternion
        #     self._v[3:6] = self.last_low_state.imu_state.gyroscope
        
        # Body joint states (using G1 mapping)
        for mj_idx, dds_idx in BODY_MUJOCO_TO_DDS.items():
            if mj_idx < NUM_ACTIVE_BODY_JOINTS and dds_idx < len(self.last_low_state.motor_state):
                dof_idx = 6 + mj_idx  # MuJoCo DoF index
                self._q[dof_idx] = self.last_low_state.motor_state[dds_idx].q
                self._v[dof_idx] = self.last_low_state.motor_state[dds_idx].dq
    
    def update_hand_q_v_from_handstate(self):
        """Extract hand joint states from HandState message"""
        if self.last_hand_state is None:
            return
            
        # Hand joint state update (if needed)
        # TODO: Implement hand state parsing according to actual needs
        pass
    
    def send_motor_command(self, time: float, pd_targets: Optional[np.ndarray] = None):
        """
        Send motor control commands
        
        Args:
            time: Current time
            pd_targets: PD target positions [27,] for body joints, if None use internal logic
        """
        
        if pd_targets is not None:
            # External PD target mode (from ZMQ bridge)
            self.update_motor_cmd_from_pd_targets(pd_targets)
        else:
            # Internal control logic mode
            if self.controller_running:
                self.update_motor_cmd(time)
            else:
                self.damping_motor_cmd()
        
        # Safety check
        # Always perform safety check when external PD targets are present
        if pd_targets is not None:
            current_q = self._get_current_body_positions()
            target_q = pd_targets[:NUM_ACTIVE_BODY_JOINTS]
            kp_gains = self._get_current_kp_gains()
            
            safe = self.safety.check_safety(current_q, target_q, kp_gains, self._q[3:7])
            if not safe:
                print_colored("WARNING", "G1 safety check failed, switching to damping mode")
                self.damping_motor_cmd()
                self.controller_running = False
                self.damping_running = True
        
        # CRITICAL: Echo the mode_machine from the last received state
        if self.last_low_state:
            self.cmd.mode_machine = self.last_low_state.mode_machine

        # Send body control command
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)
        
        # --- MODIFIED: Disable hand command sending as per user request ---
        # The original code sent a soft damping command to the hands.
        # This is now disabled to ensure no hand commands are sent.
        # if not self.lo_mode:  # Only send hand commands in real robot mode
        #     self.send_hand_damping_command()
        # else:
        #     # Lo mode: send dummy hand command for testing
        #     self._send_dummy_hand_command()
        # --- END MODIFICATION ---
    
    def update_motor_cmd_from_pd_targets(self, pd_targets: np.ndarray):
        """Update motor commands based on external PD targets"""
        if len(pd_targets) < NUM_ACTIVE_BODY_JOINTS:
            print_colored("WARNING", f"PD target length insufficient: {len(pd_targets)} < {NUM_ACTIVE_BODY_JOINTS}")
            return
            
        # Body joint PD control
        for mj_idx, dds_idx in BODY_MUJOCO_TO_DDS.items():
            if mj_idx < NUM_ACTIVE_BODY_JOINTS:
                # Get corresponding PD gains
                kp, kd = self._get_joint_gains(mj_idx)
                
                self.cmd.motor_cmd[dds_idx].mode = 1
                self.cmd.motor_cmd[dds_idx].q = pd_targets[mj_idx]
                self.cmd.motor_cmd[dds_idx].kp = kp
                self.cmd.motor_cmd[dds_idx].dq = 0.0  # Target velocity is 0
                self.cmd.motor_cmd[dds_idx].kd = kd
                self.cmd.motor_cmd[dds_idx].tau = 0.0  # Don't use feedforward torque
        
        # --- NEW: Override commands for disabled joints ---
        if self.disabled_mj_indices:
            for mj_idx in self.disabled_mj_indices:
                if mj_idx in BODY_MUJOCO_TO_DDS:
                    dds_idx = BODY_MUJOCO_TO_DDS[mj_idx]
                    self.cmd.motor_cmd[dds_idx].q = 0.0
                    self.cmd.motor_cmd[dds_idx].kp = 0.0
                    self.cmd.motor_cmd[dds_idx].dq = 0.0
                    self.cmd.motor_cmd[dds_idx].kd = 0.0
                    self.cmd.motor_cmd[dds_idx].tau = 0.0
        # --- END NEW ---
    
    def _get_joint_gains(self, mj_idx: int) -> tuple:
        """Get PD gains for specified joint from global configuration"""
        joint_config = self.global_config["g1_joint_config"]
        
        # Find the joint name by mujoco index
        for joint_name, config in joint_config.items():
            if config["mujoco_index"] == mj_idx:
                return (config["kp"] * self.kp_scale_factor, config["kd"])
        
        # Fallback to default values if joint not found
        return (0.0, 3.0)
    
    def _get_current_body_positions(self) -> np.ndarray:
        """Get current body joint positions"""
        positions = np.zeros(NUM_ACTIVE_BODY_JOINTS)
        for mj_idx in range(NUM_ACTIVE_BODY_JOINTS):
            dof_idx = 6 + mj_idx
            if dof_idx < len(self._q):
                positions[mj_idx] = self._q[dof_idx]
        return positions
    
    def _get_current_kp_gains(self) -> np.ndarray:
        """Get current PD gains"""
        gains = np.zeros(NUM_ACTIVE_BODY_JOINTS)
        for mj_idx in range(NUM_ACTIVE_BODY_JOINTS):
            gains[mj_idx], _ = self._get_joint_gains(mj_idx)
        return gains
    
    def send_hand_damping_command(self):
        """Send hand soft damping control command"""
        if self.last_hand_state is None:
            return
        
        # Build hand control command - apply soft damping to current position
        hand_motor_cmds = []
        
        # Create damping command for each finger joint
        for i, motor_state in enumerate(self.last_hand_state.motor_state):
            motor_cmd = unitree_hg_msg_dds__MotorCmd_()
            motor_cmd.mode = 1
            motor_cmd.q = motor_state.q  # Target position = current position
            motor_cmd.kp = 0.0  # P gain is 0 (pure damping)
            motor_cmd.dq = 0.0  # Target velocity is 0
            motor_cmd.kd = HAND_KP  # Use damping gain
            motor_cmd.tau = 0.0
            hand_motor_cmds.append(motor_cmd)
        
        # Update hand command and send
        self.hand_cmd.motor_cmd = hand_motor_cmds
        self.hand_pub.Write(self.hand_cmd)
    
    def damping_motor_cmd(self):
        """Body joint damping mode"""
        for dds_idx in BODY_MUJOCO_TO_DDS.values():
            if dds_idx < 35:  # Ensure valid index
                self.cmd.motor_cmd[dds_idx].mode = 1
                self.cmd.motor_cmd[dds_idx].q = 0.0
                self.cmd.motor_cmd[dds_idx].kp = 0.0
                self.cmd.motor_cmd[dds_idx].dq = 0.0
                self.cmd.motor_cmd[dds_idx].kd = 2.0  # Light damping
                self.cmd.motor_cmd[dds_idx].tau = 0.0
    
    @abstractmethod
    def update_motor_cmd(self, time: float):
        """Motor control update method that subclasses need to implement"""
        pass
    
    def _setup_lo_mode_dummy_publisher(self):
        """Setup dummy state publisher for lo mode"""
        
        # Create dummy LowState message
        self.dummy_low_state = unitree_hg_msg_dds__LowState_()
        
        # Initialize dummy motor states
        for i in range(35):
            self.dummy_low_state.motor_state[i].q = 0.0
            self.dummy_low_state.motor_state[i].dq = 0.0 
            self.dummy_low_state.motor_state[i].tau_est = 0.0
        
        # Set dummy IMU state
        self.dummy_low_state.imu_state.quaternion[0] = 1.0  # w
        self.dummy_low_state.imu_state.quaternion[1] = 0.0  # x
        self.dummy_low_state.imu_state.quaternion[2] = 0.0  # y
        self.dummy_low_state.imu_state.quaternion[3] = 0.0  # z
        
        # Create DDS state publisher
        self.dummy_state_pub = ChannelPublisher(HG_TOPIC_LOWSTATE, LowState_)
        self.dummy_state_pub.Init()
        
        # Start dummy state publishing thread (100Hz)
        self.dummy_state_running = True
        self.dummy_state_thread = threading.Thread(target=self._dummy_state_publisher, daemon=True)
        self.dummy_state_thread.start()
        
        print_colored("INFO", "Lo mode: Start dummy state publisher (100Hz)")
    
    def _dummy_state_publisher(self):
        """Dummy state publishing thread"""
        
        dt = 1.0 / 100.0  # 100Hz
        while self.dummy_state_running:
            try:
                self.dummy_state_pub.Write(self.dummy_low_state)
                time.sleep(dt)
            except:
                break
    
    def _send_dummy_hand_command(self):
        """Send dummy hand control command for testing"""
        # Create simple dummy hand command
        hand_motor_cmds = []
        for i in range(14):  # 14 finger joints
            motor_cmd = unitree_hg_msg_dds__MotorCmd_()
            motor_cmd.mode = 1
            motor_cmd.q = 0.0
            motor_cmd.kp = 0.0
            motor_cmd.dq = 0.0
            motor_cmd.kd = 2.0
            motor_cmd.tau = 0.0
            hand_motor_cmds.append(motor_cmd)
        
        self.hand_cmd.motor_cmd = hand_motor_cmds
        self.hand_pub.Write(self.hand_cmd)
     
    def reset_controller(self):
        """Reset controller state"""
        print_colored("INFO", "Reset G1 Controller")
        self.controller_running = False
        self.damping_running = True
        
        # Stop dummy publisher in lo mode
        if hasattr(self, 'dummy_state_running'):
            self.dummy_state_running = False


if __name__ == "__main__":
    print_colored("INFO", "HG SDK Controller Abstract Base Class")
    print("Usage:")
    print("  1. Inherit from HGSDKController")
    print("  2. Implement update_motor_cmd() method")
    print("  3. Call send_motor_command() to send control commands") 