import pyvicon_datastream as pv
import numpy as np
import time

from sdk_controller.topics import TOPIC_HIGHSTATE

from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_

from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.utils.thread import RecurrentThread

class ViconHighStatePublisher:
    def __init__(self,
                 vicon_ip : str,
                 object_name : str,
                 publish_freq : int = 100,
                 ):
        self.vicon_ip = vicon_ip
        self.object_name = object_name
        self.publish_freq = publish_freq
        self.hight_state_dt = 1.0 / self.publish_freq
        
        self.is_connected = False
        self.vicon_client = pv.PyViconDatastream()
        self.connect()

        if self.is_connected == True:
            self.vicon_client.enable_segment_data()
            self.vicon_client.set_stream_mode(pv.StreamMode.ServerPush)
            self.vicon_client.set_axis_mapping(pv.Direction.Forward, pv.Direction.Left, pv.Direction.Up)
            self.vicon_frame_rate = self.vicon_client.get_frame_rate()
            self.vicon_dt = (1.0 / self.vicon_frame_rate)
        else:
            raise RuntimeError(f"Failed to connect to Vicon at {self.vicon_ip}")
        
        self.high_state = unitree_go_msg_dds__SportModeState_()
        self.high_state_puber = ChannelPublisher(TOPIC_HIGHSTATE, SportModeState_)
        self.high_state_puber.Init()
        self.HighStateThread = RecurrentThread(
            interval=self.hight_state_dt, target=self.PublishHighState, name="base_highstate"
        )
        self.ViconClientThread = RecurrentThread(
            interval=self.vicon_dt, target=self.UpdateViconClient, name="vicon_client"
        )
        self.frame_number = -1
        self.t, self.prev_t, self.prev_prev_t = 0., 0., 0.
        self.p, self.prev_p, self.prev_prev_p = np.zeros(3), np.zeros(3), np.zeros(3)
        self.q, self.prev_q = np.zeros(4), np.zeros(4), np.zeros(4) # w, x, y, z
        self.v = np.zeros(3)
        self.w = np.zeros(3)
        
        self.start()
        
    def start(self):
        self.HighStateThread.start()
        self.ViconClientThread.start()
        
        timout = 3
        t = 0
        sleep = 0.1
        
        while t < timout:
            if self.t > 0:
                print("Vicon HighState Publisher started")
                return
            time.sleep(sleep)
            t += sleep
            
        raise TimeoutError(f"Vicon HighState Publisher failed to start. No messages received.")       
        
    def connect(self, ip=None):
        if ip is not None: # set
            print(f"Changing IP of Vicon Host to: {ip}")
            self.ip = ip
        
        ret = self.vicon_client.connect(self.ip)
        if ret != pv.Result.Success:
            print(f"Connection to {self.ip} failed")
            self.is_connected = False
        else:
            print(f"Connection to {self.ip} successful")
            self.is_connected = True
        return self.is_connected
    
    def _update_angular_velocity(self):
        # local angular velocity
        dt = self.prev_t - self.t
        # first order finite difference in lie space
        self.w = (2 / dt) * np.array([
            self.prev_q[0]*self.q[1] - self.prev_q[1]*self.q[0] - self.prev_q[2]*self.q[3] + self.prev_q[3]*self.q[2],
            self.prev_q[0]*self.q[2] + self.prev_q[1]*self.q[3] - self.prev_q[2]*self.q[0] - self.prev_q[3]*self.q[1],
            self.prev_q[0]*self.q[3] - self.prev_q[1]*self.q[2] + self.prev_q[2]*self.q[1] - self.prev_q[3]*self.q[0]
            ])
        
    def _update_linear_velocity(self):
        # global linear velocity
        avg_dt = (self.t - self.prev_prev_t) / 2.
        # Second order finite difference
        self.v = (3 * self.p - 4 * self.prev_p + self.prev_prev_p) / (2 * avg_dt)
    
    def _update_p_q_t(self, new_p : np.ndarray, new_q : np.ndarray, new_t : float):
        # time
        self.prev_prev_t = self.prev_t
        self.prev_t = self.t
        self.t = new_t
        # position
        self.prev_prev_p = self.prev_p
        self.prev_p = self.p
        self.p = new_p
        # quaternion
        self.prev_q = self.q
        self.q = new_q
        
    def UpdateViconClient(self):
        
        segment_count = self.vicon_client.get_segment_count(self.object_name)
        
        for seg_idx in range(segment_count):
            segment_name = self.vicon_client.get_segment_name(self.object_name, seg_idx)
            segment_global_translation = self.vicon_client.get_segment_global_translation(self.object_name, segment_name)
            segment_global_quaternion     = self.vicon_client.get_segment_global_quaternion(self.object_name, segment_name)
            frame_number = self.vicon_client.get_frame_number()
            
            if segment_global_translation is not None and segment_global_quaternion is not None:
                t = self.vicon_client.get_time_code()
                # if new frame
                if frame_number > self.frame_number:
                    self.frame_number = frame_number
                    self._update_p_q_t(segment_global_translation, segment_global_quaternion, t)

                    if self.prev_t > 0.:
                        self._update_angular_velocity()
                    if self.prev_prev_t > 0.:
                        self._update_linear_velocity()
                        
    def PublishHighState(self):
        
        self.high_state.position[0] = self.p[0]
        self.high_state.position[1] = self.p[1]
        self.high_state.position[2] = self.p[2]
        
        self.high_state.velocity[0] = self.v[0]
        self.high_state.velocity[1] = self.v[1]
        self.high_state.velocity[2] = self.v[2]
        
        self.high_state.imu_state.quaternion[0] = self.q[0]
        self.high_state.imu_state.quaternion[1] = self.q[1]
        self.high_state.imu_state.quaternion[2] = self.q[2]
        self.high_state.imu_state.quaternion[3] = self.q[3]
        
        self.high_state.imu_state.gyroscope[0] = self.w[0]
        self.high_state.imu_state.gyroscope[1] = self.w[1]
        self.high_state.imu_state.gyroscope[2] = self.w[2]
        
        self.high_state_puber.Write(self.high_state)