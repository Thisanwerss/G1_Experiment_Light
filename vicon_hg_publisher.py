import pyvicon_datastream as pv
import numpy as np
import time
import pinocchio as pin
import json

from sdk_controller.topics import TOPIC_VICON_POSE, TOPIC_VICON_TWIST

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_, TwistStamped_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import Header_

from unitree_sdk2py.utils.thread import RecurrentThread

# --- 全局配置加载 ---
try:
    with open("global_config.json", "r") as f:
        GLOBAL_CONFIG = json.load(f)
    VICON_Z_OFFSET = GLOBAL_CONFIG.get("vicon_z_offset", 0.0)
    VICON_VELOCITY_FILTER_ALPHA = GLOBAL_CONFIG.get("vicon_velocity_filter_alpha", 0.4)
    print(f"✅ 从 global_config.json 加载配置, VICON_Z_OFFSET={VICON_Z_OFFSET}, VICON_VELOCITY_FILTER_ALPHA={VICON_VELOCITY_FILTER_ALPHA}")
except FileNotFoundError:
    print("⚠️ global_config.json 未找到, 使用默认值。")
    VICON_Z_OFFSET = 0.0
    VICON_VELOCITY_FILTER_ALPHA = 0.4
except json.JSONDecodeError:
    print("❌ global_config.json 解析失败, 使用默认值。")
    VICON_Z_OFFSET = 0.0
    VICON_VELOCITY_FILTER_ALPHA = 0.4


class ViconPosePublisherHG:
    def __init__(self,
                 vicon_ip : str,
                 object_name : str,
                 publish_freq : int = 100,
                 ):
        self.vicon_ip = vicon_ip
        self.object_name = object_name
        self.publish_freq = publish_freq
        self.publish_dt = 1.0 / self.publish_freq
        
        self.is_connected = False
        self.vicon_client = pv.PyViconDatastream()
        self.connect()

        if self.is_connected == True:
            self.vicon_client.enable_segment_data()
            self.vicon_client.set_stream_mode(pv.StreamMode.ServerPush)
            self.vicon_client.set_axis_mapping(pv.Direction.Forward, pv.Direction.Left, pv.Direction.Up)
            self.vicon_frame_rate = self.vicon_client.get_frame_rate()
            if self.vicon_frame_rate > 0:
                self.vicon_dt = (1.0 / self.vicon_frame_rate)
            else:
                self.vicon_dt = self.publish_dt * 1.5
        else:
            raise RuntimeError(f"Failed to connect to Vicon at {self.vicon_ip}")
        
        # Create message objects
        self.pose_msg = PoseStamped_()
        self.twist_msg = TwistStamped_()

        # Create publishers
        self.pose_puber = ChannelPublisher(TOPIC_VICON_POSE, PoseStamped_)
        self.twist_puber = ChannelPublisher(TOPIC_VICON_TWIST, TwistStamped_)
        self.pose_puber.Init()
        self.twist_puber.Init()

        # Create threads
        self.PublishThread = RecurrentThread(
            interval=self.publish_dt, target=self.PublishData, name="vicon_publisher"
        )
        self.ViconClientThread = RecurrentThread(
            interval=self.vicon_dt, target=self.UpdateViconClient, name="vicon_client"
        )

        self.frame_number = -1
        self.t, self.prev_t, self.prev_prev_t = 0., 0., 0.
        self.p, self.prev_p, self.prev_prev_p = np.zeros(3), np.zeros(3), np.zeros(3)
        self.q, self.prev_q, self.prev_prev_q = np.array([1., 0., 0., 0.]), np.array([1., 0., 0., 0.]), np.array([1., 0., 0., 0.]) # w, x, y, z
        self.v = np.zeros(3)
        self.prev_v = np.zeros(3)
        self.w = np.zeros(3)
        
        self.start()
        
    def start(self):
        self.ViconClientThread.Start()
        self.PublishThread.Start()
        
        timout = 5
        t = 0
        sleep = 0.1
        
        while t < timout:
            if self.frame_number > 0:
                print("Vicon Pose Publisher (HG) started")
                return
            time.sleep(sleep)
            t += sleep
            
        raise TimeoutError(f"Vicon Pose Publisher (HG) failed to start. No messages received.")       
        
    def connect(self, ip=None):
        if ip is not None: # set
            print(f"Changing IP of Vicon Host to: {ip}")
            self.vicon_ip = ip
        
        ret = self.vicon_client.connect(self.vicon_ip)
        if ret != pv.Result.Success:
            print(f"Connection to {self.vicon_ip} failed")
            self.is_connected = False
        else:
            print(f"Connection to {self.vicon_ip} successful")
            self.is_connected = True
        return self.is_connected
    
    def _update_angular_velocity(self):
        # local angular velocity
        dt = self.t - self.prev_t
        if dt < 1e-6: return

        q_current = pin.Quaternion(self.q[1], self.q[2], self.q[3], self.q[0])
        q_prev = pin.Quaternion(self.prev_q[1], self.prev_q[2], self.prev_q[3], self.prev_q[0])
        
        # Calculate angular velocity using pinocchio
        self.w = pin.log3(q_prev.inverse() * q_current) / dt
        self.w[np.abs(self.w) < 0.04] = 0.0
        
    def _update_linear_velocity(self):
        # global linear velocity
        # 使用二阶后向差分以提高稳定性
        avg_dt = (self.t - self.prev_prev_t) / 2.
        if avg_dt < 1e-6: return
        
        self.prev_v = self.v
        self.v = (3 * self.p - 4 * self.prev_p + self.prev_prev_p) / (2 * avg_dt)

        # 应用一个轻量的指数平滑滤波器来减少抖动
        self.v = VICON_VELOCITY_FILTER_ALPHA * self.v + (1 - VICON_VELOCITY_FILTER_ALPHA) * self.prev_v
        
    def _update_p_q_t(self, new_p : np.ndarray, new_q : np.ndarray, new_t : float):
        # time
        self.prev_prev_t = self.prev_t
        self.prev_t = self.t
        self.t = new_t
        # position
        self.prev_prev_p = self.prev_p
        self.prev_p = self.p
        self.p = new_p / 1000.
        self.p[2] += VICON_Z_OFFSET
        # quaternion
        self.prev_prev_q = self.prev_q
        self.prev_q = self.q
        self.q = new_q
        
    def UpdateViconClient(self):
        if self.is_connected == True:
            
            frame = self.vicon_client.get_frame()
            if frame == pv.Result.Success:
                frame_number = self.vicon_client.get_frame_number()
                # if new frame
                if frame_number > self.frame_number:
                    self.frame_number = frame_number
                        
                    subject_count = self.vicon_client.get_subject_count()

                    for subj_idx in range(subject_count):
                        subject_name = self.vicon_client.get_subject_name(subj_idx)
                        if subject_name != self.object_name: #Skip objects we are not interessted in
                            continue
                        
                        segment_count = self.vicon_client.get_segment_count(self.object_name)
                        for seg_idx in range(segment_count):

                            segment_name = self.vicon_client.get_segment_name(self.object_name, seg_idx)
                            segment_global_translation = self.vicon_client.get_segment_global_translation(self.object_name, segment_name)
                            segment_global_quaternion     = self.vicon_client.get_segment_global_quaternion(self.object_name, segment_name)
                            latency = self.vicon_client.get_latency_total()

                            if segment_global_translation is not None and segment_global_quaternion is not None:
                                t = time.time() - latency
                                self._update_p_q_t(segment_global_translation, segment_global_quaternion, t)

                                if self.prev_t > 0.:
                                    self._update_angular_velocity()
                                if self.prev_prev_t > 0.:
                                    self._update_linear_velocity()
                                    
                                return True
        return False
                            
    def PublishData(self):
        # Create header
        header = Header_()
        header.stamp.sec = int(self.t)
        header.stamp.nanosec = int((self.t % 1) * 1e9)
        header.frame_id = self.object_name
        
        # Populate Pose message
        self.pose_msg.header = header
        self.pose_msg.pose.position.x = self.p[0]
        self.pose_msg.pose.position.y = self.p[1]
        self.pose_msg.pose.position.z = self.p[2]
        
        self.pose_msg.pose.orientation.w = self.q[0]
        self.pose_msg.pose.orientation.x = self.q[1]
        self.pose_msg.pose.orientation.y = self.q[2]
        self.pose_msg.pose.orientation.z = self.q[3]
        
        # Populate Twist message
        self.twist_msg.header = header
        self.twist_msg.twist.linear.x = self.v[0]
        self.twist_msg.twist.linear.y = self.v[1]
        self.twist_msg.twist.linear.z = self.v[2]

        self.twist_msg.twist.angular.x = self.w[0]
        self.twist_msg.twist.angular.y = self.w[1]
        self.twist_msg.twist.angular.z = self.w[2]

        # Publish
        self.pose_puber.Write(self.pose_msg)
        self.twist_puber.Write(self.twist_msg)

        
if __name__ == "__main__":
    import argparse
    from sdk_controller.robots import G1

    parser = argparse.ArgumentParser(description="Vicon Pose Publisher for HG-series robots")
    parser.add_argument("--vicon_ip", type=str, default="192.168.123.100:801", help="Vicon server IP and port")
    parser.add_argument("--object_name", type=str, default=G1.OBJECT_NAME, help="Object name in Vicon")
    parser.add_argument("--freq", type=int, default=100, help="Publishing frequency (Hz)")
    parser.add_argument("--channel", type=str, default="lo", help="DDS channel for communication")
    args = parser.parse_args()

    if args.channel == "lo":
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, args.channel)

    print(f"Publishing Vicon pose for '{args.object_name}' from {args.vicon_ip} at {args.freq}Hz on channel '{args.channel}'")

    vicon_publisher = ViconPosePublisherHG(
        vicon_ip=args.vicon_ip,
        object_name=args.object_name,
        publish_freq=args.freq
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping Vicon Publisher.") 