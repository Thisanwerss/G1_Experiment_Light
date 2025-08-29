import joblib
from typing import Dict

import jax
import jax.numpy as jnp
from interpax import interp1d
import mujoco
import numpy as np
from huggingface_hub import hf_hub_download
from mujoco import mjx
from mujoco.mjx._src.math import quat_sub

from hydrax import ROOT
from hydrax.task_base import Task

from hydrax.sequences import *
from hydrax.helpers import interpolate_poses, rpy_to_quaternion_wxyz, interpolate_foot_position
#tasks/humanoid_mocap.py
class HumanoidMocap(Task):
    """The Unitree G1 humanoid tracks a reference from motion capture.

    Retargeted motion capture data comes from the LocoMuJoCo dataset:
    https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main.
    """

    def __init__(
        self,
        reference_filename: str = "Lafan1/mocap/UnitreeG1/walk1_subject1.npz",
    ) -> None:
        """Load the MuJoCo model and set task parameters.

        The list of available reference files can be found at
        https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main.
        """
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/g1/scene.xml"
        )
        super().__init__(
            mj_model,
            trace_sites=["imu_in_torso", "left_foot", "right_foot"],
        )

        # Get sensor IDs
        self.left_foot_pos_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "left_foot_position"
        )
        self.left_foot_quat_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "left_foot_orientation"
        )
        self.right_foot_pos_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "right_foot_position"
        )
        self.right_foot_quat_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "right_foot_orientation"
        )

        self.left_palm_pos_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "left_hand_position"
        )
        self.left_palm_quat_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "left_hand_orientation"
        )
        self.right_palm_pos_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "right_hand_position"
        )
        self.right_palm_quat_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "right_hand_orientation"
        )

        self.left_thumb_pos_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "left_thumb_position"
        )
        self.left_middle_pos_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "left_middle_position"
        )
        self.left_index_pos_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "left_index_position"
        )

        self.right_thumb_pos_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "right_thumb_position"
        )
        self.right_middle_pos_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "right_middle_position"
        )
        self.right_index_pos_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "right_index_position"
        )

        # # Create interpolated reference sequence
        # reference = []
        # sequence = standing0
        # keys = list(sequence.keys())
        
        # for i in range(len(keys) - 1):
        #     current_key = keys[i]
        #     next_key = keys[i + 1]
            
        #     # Get robot current and next reference poses
        #     current_pose = jnp.array(sequence[current_key]["robot"])
        #     next_pose = jnp.array(sequence[next_key]["robot"])
            
        #     # Get execution time from the sequence
        #     execution_time = sequence[next_key]["execution_time"]
            
        #     # Interpolate between poses
        #     # interpolated_poses = self._interpolate_reference(current_pose, next_pose, execution_time)
        #     interpolated_poses = interpolate_poses(current_pose, next_pose, int(execution_time * 30))

        #     # Get robot current and next reference poses
        #     current_opose = jnp.array(sequence[current_key]["object"])
        #     next_opose = jnp.array(sequence[next_key]["object"])
            
        #     # Interpolate between poses
        #     # interpolated_oposes = self._interpolate_reference(current_opose, next_opose, execution_time)
        #     interpolated_oposes = interpolate_poses(current_opose, next_opose, int(execution_time * 30))

        #     # Concatenate interpolated robot and object states
        #     concatenated = np.concatenate([interpolated_poses, interpolated_oposes], axis=1)

        #     # Concatenate robot and object poses at each time step
        #     for state in concatenated:
        #         reference.append(state)

        # # Concatenate all interpolated poses into a single array
        # self.reference_fps = 30.0
        # self.reference = jnp.array(reference)
        # print(self.reference.shape)

        # === Load sequence from joblib dump ===
        sequence = joblib.load("/home/ilyass/workspace/motion_retargeting/motion.pkl")  # update with actual path
        reference = sequence["retargeted_motion"]
        self.reference_fps = 30.0
        self.reference = jnp.array(reference)
        print(self.reference.shape)

        # Precompute reference feet and palms positions and orientations
        mj_data = mujoco.MjData(mj_model)
        n_frames = len(self.reference)
        ref_left_foot_pos = np.zeros((n_frames, 3))
        ref_left_foot_quat = np.zeros((n_frames, 4))
        ref_right_foot_pos = np.zeros((n_frames, 3))
        ref_right_foot_quat = np.zeros((n_frames, 4))

        ref_left_palm_pos = np.zeros((n_frames, 3))
        ref_left_palm_quat = np.zeros((n_frames, 4))
        ref_right_palm_pos = np.zeros((n_frames, 3))
        ref_right_palm_quat = np.zeros((n_frames, 4))

        for i in range(n_frames):
            mj_data.qpos[:] = self.reference[i]
            mujoco.mj_forward(mj_model, mj_data)
            ref_left_foot_pos[i] = mj_data.site_xpos[mj_model.site("left_foot").id]
            ref_right_foot_pos[i] = mj_data.site_xpos[mj_model.site("right_foot").id]
            ref_left_foot_pos[i, 2] = 0.035
            ref_right_foot_pos[i, 2] = 0.035
            mujoco.mju_mat2Quat(
                ref_left_foot_quat[i],
                mj_data.site_xmat[mj_model.site("left_foot").id].flatten(),
            )
            mujoco.mju_mat2Quat(
                ref_right_foot_quat[i],
                mj_data.site_xmat[mj_model.site("right_foot").id].flatten(),
            )

            ref_left_palm_pos[i] = mj_data.site_xpos[mj_model.site("left_hand").id]
            ref_right_palm_pos[i] = mj_data.site_xpos[mj_model.site("right_hand").id]
            mujoco.mju_mat2Quat(
                ref_left_palm_quat[i],
                mj_data.site_xmat[mj_model.site("left_hand").id].flatten(),
            )
            mujoco.mju_mat2Quat(
                ref_right_palm_quat[i],
                mj_data.site_xmat[mj_model.site("right_hand").id].flatten(),
            )

        # Convert reference data to jax arrays
        self.ref_left_foot_pos = jnp.array(ref_left_foot_pos)
        self.ref_left_foot_quat = jnp.array(ref_left_foot_quat)
        self.ref_right_foot_pos = jnp.array(ref_right_foot_pos)
        self.ref_right_foot_quat = jnp.array(ref_right_foot_quat)

        self.ref_left_palm_pos = jnp.array(ref_left_palm_pos)
        self.ref_left_palm_quat = jnp.array(ref_left_palm_quat)
        self.ref_right_palm_pos = jnp.array(ref_right_palm_pos)
        self.ref_right_palm_quat = jnp.array(ref_right_palm_quat)

        self.right_hand_object_contact_pos = jnp.array([0.0, -0.1, 0.0])
        self.right_hand_object_contact_quat = jnp.array([1, 0, 0, 0])
        
        self.left_hand_object_contact_pos = jnp.array([0.0, 0.1, 0.0])
        self.left_hand_object_contact_quat = jnp.array([1, 0, 0, 0])

        # Cost weights
        cost_weights = np.ones(mj_model.nq)
        cost_weights[:7] = 5.0  # Base pose
        cost_weights[-7:] = 10.0  # Object pose
        self.cost_weights = jnp.array(cost_weights)

    def _get_reference_configuration(self, t: jax.Array) -> jax.Array:
        """Get the reference position (q) at time t."""
        i = jnp.int32(t * self.reference_fps)
        i = jnp.clip(i, 0, self.reference.shape[0] - 1)
        return self.reference[i, :]

    def _get_reference_foot_data(self, t: jax.Array) -> tuple[jax.Array, ...]:
        """Get the reference foot positions and orientations at time t."""
        i = jnp.int32(t * self.reference_fps)
        i = jnp.clip(i, 0, self.reference.shape[0] - 1)
        return (
            self.ref_left_foot_pos[i],
            self.ref_left_foot_quat[i],
            self.ref_right_foot_pos[i],
            self.ref_right_foot_quat[i],
        )
    
    def _get_reference_hand_data(self, t: jax.Array) -> tuple[jax.Array, ...]:
        """Get the reference foot positions and orientations at time t."""
        i = jnp.int32(t * self.reference_fps)
        i = jnp.clip(i, 0, self.reference.shape[0] - 1)
        return (
            self.ref_left_palm_pos[i],
            self.ref_left_palm_quat[i],
            self.ref_right_palm_pos[i],
            self.ref_right_palm_quat[i],
        )

    def _get_foot_position_errors(
        self, state: mjx.Data
    ) -> tuple[jax.Array, jax.Array]:
        """Get position errors for both feet."""
        ref_left_foot_pos, _, ref_right_foot_pos, _ = self._get_reference_foot_data(
            state.time
        )

        left_pos_adr = self.model.sensor_adr[self.left_foot_pos_sensor]
        right_pos_adr = self.model.sensor_adr[self.right_foot_pos_sensor]

        left_err = (
            state.sensordata[left_pos_adr : left_pos_adr + 3] - ref_left_foot_pos
        )
        right_err = (
            state.sensordata[right_pos_adr : right_pos_adr + 3] - ref_right_foot_pos
        )

        return left_err, right_err

    def _get_foot_orientation_errors(
        self, state: mjx.Data
    ) -> tuple[jax.Array, jax.Array]:
        """Get orientation errors for both feet."""
        _, ref_left_foot_quat, _, ref_right_foot_quat = self._get_reference_foot_data(
            state.time
        )

        left_quat_adr = self.model.sensor_adr[self.left_foot_quat_sensor]
        right_quat_adr = self.model.sensor_adr[self.right_foot_quat_sensor]

        left_quat = state.sensordata[left_quat_adr : left_quat_adr + 4]
        right_quat = state.sensordata[right_quat_adr : right_quat_adr + 4]

        left_err = quat_sub(left_quat, ref_left_foot_quat)
        right_err = quat_sub(right_quat, ref_right_foot_quat)

        return left_err, right_err
    
    def _get_hand_position_errors(
        self, state: mjx.Data
    ) -> tuple[jax.Array, jax.Array]:
        """Get position errors for both feet."""
        ref_left_foot_pos, _, ref_right_foot_pos, _ = self._get_reference_hand_data(
            state.time
        )

        left_pos_adr = self.model.sensor_adr[self.left_palm_pos_sensor]
        right_pos_adr = self.model.sensor_adr[self.right_palm_pos_sensor]

        left_err = (
            state.sensordata[left_pos_adr : left_pos_adr + 3] - ref_left_foot_pos
        )
        right_err = (
            state.sensordata[right_pos_adr : right_pos_adr + 3] - ref_right_foot_pos
        )

        return left_err, right_err

    def _get_hand_orientation_errors(
        self, state: mjx.Data
    ) -> tuple[jax.Array, jax.Array]:
        """Get orientation errors for both feet."""
        _, ref_left_foot_quat, _, ref_right_foot_quat = self._get_reference_hand_data(
            state.time
        )

        left_quat_adr = self.model.sensor_adr[self.left_palm_quat_sensor]
        right_quat_adr = self.model.sensor_adr[self.right_palm_quat_sensor]

        left_quat = state.sensordata[left_quat_adr : left_quat_adr + 4]
        right_quat = state.sensordata[right_quat_adr : right_quat_adr + 4]

        left_err = quat_sub(left_quat, ref_left_foot_quat)
        right_err = quat_sub(right_quat, ref_right_foot_quat)

        return left_err, right_err
    
    def _object_to_world_frame(self, pos_object_frame: jax.Array, object_pos: jax.Array, object_quat: jax.Array) -> jax.Array:
        """Convert position from object frame to world frame."""
        # Convert quaternion to rotation matrix
        w, x, y, z = object_quat
        R = jnp.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        
        # Transform position from object to world frame
        pos_world_frame = object_pos + R @ pos_object_frame
        return pos_world_frame
    
    def _get_palm_position_errors(self, q_ref: jax.Array, state: mjx.Data) -> float:
        object_pos = q_ref[-7:-4]
        object_quat = q_ref[-4:]

        target_right_hand_object_contact_pose_world_frame = self._object_to_world_frame(
            self.right_hand_object_contact_pos,
            object_pos,
            object_quat
        )

        target_left_hand_object_contact_pose_world_frame = self._object_to_world_frame(
            self.left_hand_object_contact_pos,
            object_pos,
            object_quat
        )

        right_palm_pos_adr = self.model.sensor_adr[self.right_palm_pos_sensor]
        left_palm_pos_adr = self.model.sensor_adr[self.left_palm_pos_sensor]

        right_palm_pos_err = (
            state.sensordata[right_palm_pos_adr : right_palm_pos_adr + 3] - target_right_hand_object_contact_pose_world_frame
        )
        left_palm_pos_err = (
            state.sensordata[left_palm_pos_adr : left_palm_pos_adr + 3] - target_left_hand_object_contact_pose_world_frame
        )

        return right_palm_pos_err, left_palm_pos_err
    
    def _get_palm_orientation_errors(self, q: jax.Array, state: mjx.Data) -> float:
        object_quat = q[-4:]
        
        right_palm_quat_adr = self.model.sensor_adr[self.right_palm_quat_sensor]
        left_palm_quat_adr = self.model.sensor_adr[self.left_palm_quat_sensor]

        right_palm_quat_err = quat_sub(
            state.sensordata[left_palm_quat_adr : left_palm_quat_adr + 4], 
            object_quat)
        left_palm_quat_err = quat_sub(
            state.sensordata[right_palm_quat_adr : right_palm_quat_adr + 4], 
            object_quat)

        return right_palm_quat_err, left_palm_quat_err
    
    def _interpolate_reference(self, ref1: jax.Array, ref2: jax.Array, execution_time: float) -> jax.Array:
        """Interpolate between two consecutive reference poses using a cubic spline at 30Hz.
        
        Args:
            ref1: First reference pose
            ref2: Second reference pose
            execution_time: Time in seconds between the two reference poses
            
        Returns:
            Array of interpolated poses at 30Hz
        """
        
        # Calculate number of interpolation points based on execution time and 30Hz
        num_points = int(execution_time * 30)
        
        # Create time points for interpolation
        t = jnp.array([0.0, execution_time])  # Actual time points
        t_interp = jnp.linspace(0.0, execution_time, num_points)  # Interpolation points at 30Hz
        
        # Stack the reference poses
        refs = jnp.stack([ref1, ref2])
        
        # Interpolate each dimension separately using cubic spline
        interp_poses = []
        for i in range(ref1.shape[0]):
            interp_poses.append(interp1d(t_interp, t, refs[:, i], method="cubic2", extrap=True))
            
        return jnp.stack(interp_poses, axis=1)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        # Configuration error weighs the base pose more heavily
        q_ref = self._get_reference_configuration(state.time)
        q = state.qpos

        # Base error
        base_position_cost = jnp.sum(self.cost_weights[:3] * jnp.square(q[:3] - q_ref[:3]))
        base_orientation_cost = jnp.sum(self.cost_weights[3] * jnp.square(quat_sub(q[3:7], q_ref[3:7])))
        base_cost = base_position_cost + base_orientation_cost

        # Joint error
        joint_cost = jnp.sum(self.cost_weights[7:-7] * jnp.square(q[7:-7] - q_ref[7:-7]))

        # Object error
        object_position_cost = jnp.sum(self.cost_weights[-7:-4] * jnp.square(q[-7:-4] - q_ref[-7:-4]))
        object_orientation_cost = jnp.sum(self.cost_weights[-4] * jnp.square(quat_sub(q[-4:], q_ref[-4:])))
        object_cost = object_position_cost + object_orientation_cost

        # Foot tracking costs
        left_pos_err, right_pos_err = self._get_foot_position_errors(state)
        left_ori_err, right_ori_err = self._get_foot_orientation_errors(state)

        foot_position_cost = jnp.sum(jnp.square(left_pos_err)) + jnp.sum(
            jnp.square(right_pos_err))
        foot_orientation_cost = jnp.sum(jnp.square(left_ori_err)) + jnp.sum(
            jnp.square(right_ori_err)
        )
        foot_cost = 5.0 * foot_position_cost + 0.1 * foot_orientation_cost

        # Palm tracking costs
        left_pos_err, right_pos_err = self._get_palm_position_errors(q, state)
        left_ori_err, right_ori_err = self._get_palm_orientation_errors(q, state)

        palm_position_cost = jnp.sum(jnp.square(left_pos_err)) + jnp.sum(
            jnp.square(right_pos_err))
        palm_orientation_err = jnp.sum(jnp.square(left_ori_err)) + jnp.sum(
            jnp.square(right_ori_err)
        )
        palm_cost = 5.0 * palm_position_cost + 0.1 * palm_orientation_err

        # Action rate cost
        action_rate_cost = jnp.sum(jnp.square(control - q[7:-7]))

        return (
            1.0 * joint_cost
            + 1.0 * base_cost
            + 0.0 * object_cost
            + 1.0 * foot_cost
            + 0.0 * palm_cost
            + 1.0 * action_rate_cost
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        q_ref = self._get_reference_configuration(state.time)
        q = state.qpos

        # Base error
        base_position_cost = jnp.sum(self.cost_weights[:3] * jnp.square(q[:3] - q_ref[:3]))
        base_orientation_cost = jnp.sum(self.cost_weights[3] * jnp.square(quat_sub(q[3:7], q_ref[3:7])))
        base_cost = base_position_cost + base_orientation_cost

        # Joint error
        joint_cost = jnp.sum(self.cost_weights[7:-7] * jnp.square(q[7:-7] - q_ref[7:-7]))

        # Object error
        object_position_cost = jnp.sum(self.cost_weights[-7:-4] * jnp.square(q[-7:-4] - q_ref[-7:-4]))
        object_orientation_cost = jnp.sum(self.cost_weights[-4] * jnp.square(quat_sub(q[-4:], q_ref[-4:])))
        object_cost = object_position_cost + object_orientation_cost

        # Add foot tracking costs to terminal cost
        left_pos_err, right_pos_err = self._get_foot_position_errors(state)
        left_ori_err, right_ori_err = self._get_foot_orientation_errors(state)

        foot_position_cost = jnp.sum(jnp.square(left_pos_err)) + jnp.sum(
            jnp.square(right_pos_err))
        foot_orientation_cost = jnp.sum(jnp.square(left_ori_err)) + jnp.sum(
            jnp.square(right_ori_err)
        )
        foot_cost = 5.0 * foot_position_cost + 0.1 * foot_orientation_cost

        # Palm tracking costs
        left_pos_err, right_pos_err = self._get_palm_position_errors(q_ref, state)
        left_ori_err, right_ori_err = self._get_palm_orientation_errors(q_ref, state)

        palm_position_cost = jnp.sum(jnp.square(left_pos_err)) + jnp.sum(
            jnp.square(right_pos_err))
        palm_orientation_err = jnp.sum(jnp.square(left_ori_err)) + jnp.sum(
            jnp.square(right_ori_err)
        )
        palm_cost = 5.0 * palm_position_cost + 0.1 * palm_orientation_err

        return (
            0.0 * joint_cost
            + 1.0 * base_cost
            + 0.0 * object_cost
            + 0.0 * palm_cost
            + 0.0 * foot_cost
        )

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize the friction parameters."""
        n_geoms = self.model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.5, maxval=2.0)
        new_frictions = self.model.geom_friction.at[:, 0].set(
            self.model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Randomly perturb the measured base position and velocities."""
        rng, q_rng, v_rng = jax.random.split(rng, 3)
        q_err = 0.01 * jax.random.normal(q_rng, (7,))
        v_err = 0.01 * jax.random.normal(v_rng, (6,))

        qpos = data.qpos.at[0:7].set(data.qpos[0:7] + q_err)
        qvel = data.qvel.at[0:6].set(data.qvel[0:6] + v_err)

        return {"qpos": qpos, "qvel": qvel}
