import numpy as np
import mujoco
from mj_pin.utils import mj_joint_name2act_id, mj_joint_name2dof

class SafetyLayer:
    def __init__(self, mj_model, conservative_safety: bool = False):
        """
        Safety controller to enforce kinematic and torque limits.

        Args:
            mj_model : MuJoCo model with joint and ctrl limits.
            conservative_safety: Whether to use more conservative safety limits.
        """
        joint_name2act_id = mj_joint_name2act_id(mj_model)
        joint_name2dof = mj_joint_name2dof(mj_model)
        self.joint_act_id2dof = {v:joint_name2dof[k] for k, v in joint_name2act_id.items()}
        self.joint_dof2act_id = {v : k for k, v in self.joint_act_id2dof.items()}
        self.joint_limits = {}
        self.torque_limits = {}
        
        if conservative_safety:
            self.base_orientation_limit = 25 * np.pi / 180. # 25 degrees
            self.scale_joint_limit = 0.90 # 90% of physical limit
        else:
            self.base_orientation_limit = 35 * np.pi / 180.
            self.scale_joint_limit = 0.95

        self.scale_torque_limit = 0.9

        for id in range(mj_model.njnt):
            dof = int(mj_model.joint(id).dofadr)
            if not dof in self.joint_dof2act_id:
                continue
            act_id = self.joint_dof2act_id[dof]
            q_dof = int(mj_model.jnt_qposadr[id])
            if mj_model.jnt_limited[id]:
                min = mj_model.jnt_range[id][0]
                max = mj_model.jnt_range[id][1]
                self.joint_limits[q_dof] = (min * self.scale_joint_limit, max * self.scale_joint_limit)
            if mj_model.actuator_ctrllimited[act_id]:
                self.torque_limits[act_id] = mj_model.actuator_ctrlrange[act_id][1] * self.scale_torque_limit
                
    def check_joint_limits(self, joint_positions):
        """Check if joint positions exceed limits."""
        if not self.joint_limits:
            return True
        
        for joint_id, (q_min, q_max) in self.joint_limits.items():
            if not (q_min < joint_positions[joint_id] < q_max):
                dof_name = mj_model.joint(mj_model.dof_jntid[joint_id]).name if joint_id < mj_model.nv else f"Dof {joint_id}"
                print(f"❌❌❌ [安全层] 关节物理极限超限! 关节: {dof_name} | 位置: {joint_positions[joint_id]:.2f} (范围: {q_min:.2f} ~ {q_max:.2f}) ❌❌❌")
                return False
        return True

    def check_torque_limits(self, torques):
        """Check if torques exceed limits."""
        if not self.torque_limits:
            return True
        
        for act_id, max_torque in self.torque_limits.items():
            if abs(torques[act_id]) > max_torque:
                act_name = mj_model.actuator(act_id).name
                print(f"❌❌❌ [安全层] 预估力矩超限! 执行器: {act_name} | 力矩: {abs(torques[act_id]):.1f} > 上限: {max_torque:.1f} Nm ❌❌❌")
                return False
        return True
    
    def check_base_orientation(self, base_quaternion):
        """Check if base orientation exceeds the safety threshold."""
        if self.base_orientation_limit is None:
            return True
        
        _, x, y, z = base_quaternion  # Assuming quaternion (w, x, y, z)
        roll = np.arctan2(2 * (y * z + x), 1 - 2 * (x ** 2 + y ** 2))
        pitch = np.arcsin(2 * (x * z - y))
        
        if abs(roll) > self.base_orientation_limit or abs(pitch) > self.base_orientation_limit:
            print(f"❌❌❌ [安全层] 机器人基座倾角过大! Roll: {roll*180/np.pi:.1f}°, Pitch: {pitch*180/np.pi:.1f}° (上限: {self.base_orientation_limit*180/np.pi:.1f}°) ❌❌❌")
            return False
        return True

    def check_safety(self, joint_positions, torques, base_quaternion) -> bool:
        """Enforce safety by zeroing torques if any limit is exceeded."""
        if not (self.check_joint_limits(joint_positions) and 
                self.check_torque_limits(torques) and
                self.check_base_orientation(base_quaternion)):
            print("--- 触发安全保护! 切换至阻尼模式 ---")
            return False
        return True
