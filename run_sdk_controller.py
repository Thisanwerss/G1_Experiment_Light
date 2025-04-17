import time
import sys
import numpy as np

from mpc_controller.mpc import LocomotionMPC
from mpc_controller.config.quadruped.utils import get_quadruped_config
from sdk_controller.abstract import SDKController
from mj_pin.utils import get_robot_description

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.utils.crc import CRC

class MPC_SDK(SDKController):
    def __init__(self,
                 simulate : bool,
                 mpc : LocomotionMPC,
                 robot_config,
                 xml_path = "",
                 v_max=0.5,
                 w_max=0.5):
        self.mpc = mpc
        self.mpc.scale_joint = np.repeat([1.4, 1.2, 1], 4)

        self.v_max = v_max
        self.w_max = w_max
        self.v_des = np.zeros(3)
        self.w_des = 0.
        super().__init__(simulate, robot_config, xml_path)

    def wireless_handler(self, msg : WirelessController_):
        super().wireless_handler(msg)
        self.mpc.set_command(
            v_des = np.round([msg.ly * self.v_max, -msg.lx * self.v_max, 0.], 2),
            w_yaw = -round(msg.rx * self.w_max, 1)
            )
            
    def update_motor_cmd(self, time):
        torques_ff = self.mpc._compute_torques_ff(time, self._q, self._v)
        if self.mpc.first_solve:
            # Stand up
            phase = 1.
            for i in range(self.nu):
                self.cmd.motor_cmd[i].q = phase * self.robot_config.STAND_UP_JOINT_POS[i] + (
                    1 - phase) * self.robot_config.STAND_DOWN_JOINT_POS[i]
                self.cmd.motor_cmd[i].kp = phase * 50.0 + (1 - phase) * 20.0
                self.cmd.motor_cmd[i].dq = 0.0
                self.cmd.motor_cmd[i].kd = 3.5
                self.cmd.motor_cmd[i].tau = 0.0
        else:
            self.mpc.tau_full.append(torques_ff)
            step = self.mpc.plan_step + 2
            scale = 1. if self.simulate else self.robot_config.scale_gains
            for i, tau in enumerate(torques_ff, start=6):
                i_act = self.joint_dof2act_id[i]
                self.cmd.motor_cmd[i_act].q = self.mpc.q_plan[step, i]
                self.cmd.motor_cmd[i_act].kp = self.mpc.scale_joint[i-6] * self.robot_config.Kp * scale
                self.cmd.motor_cmd[i_act].dq = self.mpc.v_plan[step, i]
                self.cmd.motor_cmd[i_act].kd = self.mpc.scale_joint[i-6] * self.robot_config.Kd * scale
                max_tau = self.safety.torque_limits[i_act]
                self.cmd.motor_cmd[i_act].tau = np.clip(tau, -max_tau, max_tau)

    def reset_controller(self):
        print("reset controller")
        time.sleep(0.1)
        self.mpc.reset()
        
input("Press enter to start")
runing_time = 0.0

VICON_IP = "192.168.123.100:801"
       
if __name__ == '__main__':
    from sdk_controller.robots import Go2
    from sdk_controller.joystick import JoystickPublisher
    from sdk_controller.vicon_publisher import ViconHighStatePublisher
    
    
    
    if len(sys.argv) <2:
        ChannelFactoryInitialize(1, "lo")
        simulate = True
        
    else:
        ChannelFactoryInitialize(0, sys.argv[1])
        
        joystick = JoystickPublisher(device_id=0, js_type="xbox")
        vicon = ViconHighStatePublisher(
            vicon_ip=VICON_IP,
            object_name=Go2.OBJECT_NAME,
            publish_freq=Go2.CONTROL_FREQ,
        )
        simulate = False
    
    dt = 1 / Go2.CONTROL_FREQ
    robot_desc = get_robot_description(Go2.ROBOT_NAME)
    feet_frame_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    gait_name = "trot"
    config_gait, config_opt, config_cost = get_quadruped_config(gait_name, Go2.ROBOT_NAME)
    config_gait.nominal_period = 0.5
    config_opt.recompile = False

    mpc = LocomotionMPC(
        path_urdf=robot_desc.urdf_path,
        feet_frame_names = feet_frame_names,
        config_opt=config_opt,
        config_gait=config_gait,
        config_cost=config_cost,
        joint_ref = robot_desc.q0,
        sim_dt=dt,
        print_info=False,
        solve_async=True,
        )
    
    sdk_controller = MPC_SDK(simulate, mpc, Go2)

    try:
        while True:
            step_start = time.perf_counter()
            
            sdk_controller.send_motor_command(runing_time)

            runing_time += dt
            time_until_next_step = dt - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            
    except KeyboardInterrupt:
        print(mpc.print_timings())
        mpc.plot_traj("q")
        mpc.plot_traj("v")
        mpc.plot_traj("f")
        mpc.plot_traj("tau")
        mpc.show_plots()