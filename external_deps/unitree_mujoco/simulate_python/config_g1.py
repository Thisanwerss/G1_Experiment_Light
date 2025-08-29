import os

class G1Config:
    # G1 机器人配置
    ROBOT = "g1"  # 机器人名称
    ROBOT_SCENE = "../unitree_robots/" + ROBOT + "/g1_lab.xml"  # 使用 g1_lab.xml

    # 检查场景文件是否存在，如果不存在则使用备用路径
    if not os.path.exists(ROBOT_SCENE):
        # 尝试使用 g1 目录下的 g1_lab.xml
        ROBOT_SCENE = "../g1/g1_lab.xml"
        if not os.path.exists(ROBOT_SCENE):
            print(f"警告: 找不到 g1_lab.xml，使用默认场景文件")
            ROBOT_SCENE = "../unitree_robots/" + ROBOT + "/scene.xml"

    DOMAIN_ID = 1  # Domain id
    INTERFACE = "lo"  # 网络接口

    USE_JOYSTICK = 1  # 使用手柄控制
    JOYSTICK_TYPE = "xbox"  # 支持 "xbox" 和 "switch" 手柄布局
    JOYSTICK_DEVICE = 0  # 手柄编号

    PRINT_SCENE_INFORMATION = True  # 打印机器人链接、关节和传感器信息
    ENABLE_ELASTIC_BAND = False  # 虚拟弹簧带（用于提升 H1）

    # 仿真参数 - 100Hz 控制频率
    SIMULATE_DT = 0.01  # 仿真时间步长 (100Hz)
    VIEWER_DT = 0.02    # 可视化刷新率 (50 fps)

    # 电机名称
    MOTOR_NAMES = [
        "left_hip_roll_joint", "left_hip_yaw_joint", "left_hip_pitch_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_roll_joint", "right_hip_yaw_joint", "right_hip_pitch_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]
    NUM_MOTOR = len(MOTOR_NAMES)
