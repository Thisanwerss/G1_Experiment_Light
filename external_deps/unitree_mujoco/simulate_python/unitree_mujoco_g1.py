import time
import mujoco
import mujoco.viewer
import numpy as np
from collections import deque
from threading import Thread
from config_g1 import G1Config
from unitree_sdk2py_bridge_g1 import *


class UnitreeSdk2BridgeG1:
    def __init__(self, mj_model, mj_data) -> None:
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.config = G1Config()
        
        # 配置参数
        self.dt = 0.001
        self.control_dt = 0.001
        self.mj_step_counter = 0
        self.cmd_cur = np.zeros(G1Config.NUM_MOTOR)
        self.tau_est = np.zeros(G1Config.NUM_MOTOR)
        self.time_start = time.time()
        self.state_time = 0
        self.last_update_time = 0
        self.log_interval = 100
        
        # 扩展控制命令数组以包含手部
        self.cmd_cur_extended = np.zeros(self.mj_model.nu)  # 65 个控制输入
        
        # 手部控制初始化
        self.hand_control_enabled = False
        self.left_hand_joints = []
        self.right_hand_joints = []
        
        # 收集手部关节索引
        for i in range(self.mj_model.njnt):
            joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name and 'left_' in joint_name and ('finger' in joint_name or 'thumb' in joint_name):
                qpos_addr = self.mj_model.jnt_qposadr[i]
                if qpos_addr < self.mj_model.nq:
                    self.left_hand_joints.append(qpos_addr)
            elif joint_name and 'right_' in joint_name and ('finger' in joint_name or 'thumb' in joint_name):
                qpos_addr = self.mj_model.jnt_qposadr[i]
                if qpos_addr < self.mj_model.nq:
                    self.right_hand_joints.append(qpos_addr)
        
        # 控制索引映射
        self.ctrl_indices_body = list(range(G1Config.NUM_MOTOR))
        self.ctrl_indices_left_hand = list(range(G1Config.NUM_MOTOR, G1Config.NUM_MOTOR + len(self.left_hand_joints)))
        self.ctrl_indices_right_hand = list(range(G1Config.NUM_MOTOR + len(self.left_hand_joints), 
                                                   G1Config.NUM_MOTOR + len(self.left_hand_joints) + len(self.right_hand_joints)))
        
        # 初始化传感器标志
        self.have_imu_ = False
        self.have_frame_sensor_ = False
        self.dim_motor_sensor = len(G1Config.MOTOR_NAMES)
        
        # 检查传感器
        for i in range(self.dim_motor_sensor, self.mj_model.nsensor):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i
            )
            if name == "imu_quat" or name == "imu_in_torso_quat":
                self.have_imu_ = True
            if name == "frame_pos":
                self.have_frame_sensor_ = True

        # Unitree SDK2 消息
        self.low_state = LowState_default()
        self.low_state_puber = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self.low_state_puber.Init()
        self.lowStateThread = RecurrentThread(
            interval=self.dt, target=self.PublishLowState, name="sim_lowstate"
        )
        self.lowStateThread.Start()
        
        # 手部状态 - 仅在需要时发布
        if len(self.left_hand_joints) > 0 or len(self.right_hand_joints) > 0:
            self.hand_control_enabled = True
            self.hand_state = HandState_default()
            self.hand_state_puber = ChannelPublisher(TOPIC_HANDSTATE, HandState_)
            self.hand_state_puber.Init()
            self.handStateThread = RecurrentThread(
                interval=self.dt, target=self.PublishHandState, name="sim_handstate"
            )
            self.handStateThread.Start()

        self.high_state = unitree_go_msg_dds__SportModeState_()
        self.high_state_puber = ChannelPublisher(TOPIC_HIGHSTATE, SportModeState_)
        self.high_state_puber.Init()
        self.HighStateThread = RecurrentThread(
            interval=self.dt, target=self.PublishHighState, name="sim_highstate"
        )
        self.HighStateThread.Start()

        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.wireless_controller_puber = ChannelPublisher(
            TOPIC_WIRELESS_CONTROLLER, WirelessController_
        )
        self.wireless_controller_puber.Init()
        self.WirelessControllerThread = RecurrentThread(
            interval=0.01,
            target=self.PublishWirelessController,
            name="sim_wireless_controller",
        )
        self.WirelessControllerThread.Start()

        # 订阅控制命令
        self.low_cmd_suber = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
        self.low_cmd_suber.Init(self.LowCmdHandler, 10)
        
        print(f"🤖 G1 仿真初始化完成")
        print(f"   身体关节数: {G1Config.NUM_MOTOR}")
        print(f"   左手关节数: {len(self.left_hand_joints)}")
        print(f"   右手关节数: {len(self.right_hand_joints)}")
        print(f"   总控制维度: {self.mj_model.nu}")
        print(f"   IMU 传感器: {'✓' if self.have_imu_ else '✗'}")
        print(f"   Frame 传感器: {'✓' if self.have_frame_sensor_ else '✗'}")

    def PublishLowState(self):
        # 发布低层状态
        for i in range(G1Config.NUM_MOTOR):
            motor_state = self.low_state.motor_state[i]
            qpos_idx = G1Config.MOTOR_QPOS_IDX[i]
            qvel_idx = G1Config.MOTOR_QVEL_IDX[i]
            motor_state.q = self.mj_data.qpos[qpos_idx]
            motor_state.dq = self.mj_data.qvel[qvel_idx]
            motor_state.tau_est = self.tau_est[i]

        # IMU 数据
        if self.have_imu_:
            imu_quat_id = mujoco.mj_name2id(
                self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, "imu_quat"
            )
            self.low_state.imu_state.quaternion[0] = self.mj_data.sensordata[
                self.mj_model.sensor_adr[imu_quat_id] + 0
            ]
            self.low_state.imu_state.quaternion[1] = self.mj_data.sensordata[
                self.mj_model.sensor_adr[imu_quat_id] + 1
            ]
            self.low_state.imu_state.quaternion[2] = self.mj_data.sensordata[
                self.mj_model.sensor_adr[imu_quat_id] + 2
            ]
            self.low_state.imu_state.quaternion[3] = self.mj_data.sensordata[
                self.mj_model.sensor_adr[imu_quat_id] + 3
            ]

        self.low_state_puber.Write(self.low_state)

    def PublishHandState(self):
        # 仅在启用手部控制时发布
        if not self.hand_control_enabled:
            return
            
        # 发布手部状态（如果有手部关节）
        # 左手
        for i, qpos_idx in enumerate(self.left_hand_joints):
            if i < 7:  # DEX_HAND_LEFT_MAX_IDX = 7
                self.hand_state.left_hand_position[i] = self.mj_data.qpos[qpos_idx]
                # 假设速度索引与位置索引相同
                if qpos_idx < len(self.mj_data.qvel):
                    self.hand_state.left_hand_velocity[i] = self.mj_data.qvel[qpos_idx]
        
        # 右手
        for i, qpos_idx in enumerate(self.right_hand_joints):
            if i < 7:  # DEX_HAND_RIGHT_MAX_IDX = 7
                self.hand_state.right_hand_position[i] = self.mj_data.qpos[qpos_idx]
                # 假设速度索引与位置索引相同
                if qpos_idx < len(self.mj_data.qvel):
                    self.hand_state.right_hand_velocity[i] = self.mj_data.qvel[qpos_idx]
        
        self.hand_state_puber.Write(self.hand_state)

    def PublishHighState(self):
        # Frame 位置
        if self.have_frame_sensor_:
            frame_pos_id = mujoco.mj_name2id(
                self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, "frame_pos"
            )
            sensor_adr = self.mj_model.sensor_adr[frame_pos_id]
            
            self.high_state.position[0] = self.mj_data.sensordata[sensor_adr + 0]
            self.high_state.position[1] = self.mj_data.sensordata[sensor_adr + 1]
            self.high_state.position[2] = self.mj_data.sensordata[sensor_adr + 2]

        # IMU 数据
        if self.have_imu_:
            imu_quat_id = mujoco.mj_name2id(
                self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, "imu_in_torso_quat"
            )
            sensor_adr = self.mj_model.sensor_adr[imu_quat_id]
            
            self.high_state.imu_state.quaternion[0] = self.mj_data.sensordata[sensor_adr + 0]
            self.high_state.imu_state.quaternion[1] = self.mj_data.sensordata[sensor_adr + 1]
            self.high_state.imu_state.quaternion[2] = self.mj_data.sensordata[sensor_adr + 2]
            self.high_state.imu_state.quaternion[3] = self.mj_data.sensordata[sensor_adr + 3]

        self.high_state_puber.Write(self.high_state)

    def PublishWirelessController(self):
        self.wireless_controller_puber.Write(self.wireless_controller)

    def LowCmdHandler(self, msg: LowCmd_):
        # 处理低层控制命令
        current_time = time.time() - self.time_start
        
        # 只更新身体电机命令
        for i in range(G1Config.NUM_MOTOR):
            cmd = msg.motor_cmd[i]
            # PD控制
            qpos_idx = G1Config.MOTOR_QPOS_IDX[i]
            qvel_idx = G1Config.MOTOR_QVEL_IDX[i]
            
            q_actual = self.mj_data.qpos[qpos_idx]
            dq_actual = self.mj_data.qvel[qvel_idx]
            
            # PD控制律
            tau = cmd.kp * (cmd.q - q_actual) + cmd.kd * (cmd.dq - dq_actual) + cmd.tau
            
            # 存储计算的扭矩
            self.cmd_cur[i] = tau
            self.tau_est[i] = tau
        
        # 定期日志
        if self.mj_step_counter % self.log_interval == 0:
            print(f"⚡ t={current_time:.3f}s | 收到控制命令 | "
                  f"电机[0] q_des={msg.motor_cmd[0].q:.3f} tau={self.cmd_cur[0]:.3f}")
        
        self.last_update_time = current_time

    def Step(self):
        # 执行一步仿真
        # 设置身体电机控制
        for i in range(G1Config.NUM_MOTOR):
            self.cmd_cur_extended[self.ctrl_indices_body[i]] = self.cmd_cur[i]
        
        # 手部控制保持为0（或设置默认值）
        # 左手
        for i in self.ctrl_indices_left_hand:
            self.cmd_cur_extended[i] = 0.0
        
        # 右手  
        for i in self.ctrl_indices_right_hand:
            self.cmd_cur_extended[i] = 0.0
        
        # 应用控制到MuJoCo
        self.mj_data.ctrl[:] = self.cmd_cur_extended
        
        # 步进仿真
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.mj_step_counter += 1
        self.state_time = self.mj_data.time


def main():
    print("🚀 启动 G1 机器人 MuJoCo 仿真")
    print("=" * 50)
    
    # 加载正确的场景文件（包含地面）
    model_path = "../unitree_robots/g1/scene.xml"
    
    # 检查文件是否存在，如果不存在，尝试其他路径
    if not os.path.exists(model_path):
        # 尝试使用 g1_lab.xml
        model_path = "../unitree_robots/g1/g1_lab.xml"
        if not os.path.exists(model_path):
            print(f"❌ 找不到模型文件: {model_path}")
            print("请确保 unitree_robots/g1/ 目录下有 scene.xml 或 g1_lab.xml")
            return
    
    print(f"📂 加载模型: {model_path}")
    
    # 加载 MuJoCo 模型
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)
    
    # 创建 SDK 桥接器
    bridge = UnitreeSdk2BridgeG1(mj_model, mj_data)
    
    # 设置初始位置（站立姿态）
    mj_data.qpos[0] = 0.0   # x
    mj_data.qpos[1] = 0.0   # y
    mj_data.qpos[2] = 0.75  # z - 抬高一点避免穿透地面
    mj_data.qpos[3] = 1.0   # qw
    mj_data.qpos[4] = 0.0   # qx
    mj_data.qpos[5] = 0.0   # qy
    mj_data.qpos[6] = 0.0   # qz
    
    # 设置腿部初始角度（轻微弯曲）
    for i in range(len(G1Config.STANDING_POSE)):
        qpos_idx = G1Config.MOTOR_QPOS_IDX[i]
        mj_data.qpos[qpos_idx] = G1Config.STANDING_POSE[i]
    
    # 前向运动学更新
    mujoco.mj_forward(mj_model, mj_data)
    
    # 创建查看器
    print("🖼️  启动可视化界面...")
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
    
    print("✅ 仿真已启动！等待来自 DDS 的控制命令...")
    print("💡 提示: 请确保已启动策略节点和 SDK 桥接控制器")
    print("-" * 50)
    
    # 仿真主循环
    try:
        while viewer.is_running():
            # 执行仿真步骤
            bridge.Step()
            
            # 更新查看器
            viewer.sync()
            
            # 保持仿真频率
            time.sleep(bridge.dt)
            
    except KeyboardInterrupt:
        print("\n🛑 收到中断信号，停止仿真...")
    
    # 清理
    viewer.close()
    print("✅ 仿真已停止")


if __name__ == "__main__":
    main()
