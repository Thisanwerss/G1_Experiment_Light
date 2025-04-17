import pygame
import sys
import time
import numpy as np

from sdk_controller.topics import TOPIC_WIRELESS_CONTROLLER
from unitree_sdk2py.core.channel import ChannelPublisher

from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from unitree_sdk2py.utils.thread import RecurrentThread

KEY_MAP = {
    "R1": 0,
    "L1": 1,
    "start": 2,
    "select": 3,
    "R2": 4,
    "L2": 5,
    "F1": 6,
    "F2": 7,
    "A": 8,
    "B": 9,
    "X": 10,
    "Y": 11,
    "up": 12,
    "right": 13,
    "down": 14,
    "left": 15,
}

class JoystickPublisher:
    def __init__(self,
                 device_id = 0,
                 js_type="xbox",
                 publish_freq=100
                 ):
        self.device_id = device_id
        self.js_type = js_type
        self.publish_freq = publish_freq
        
        self.joystick = None
        self.setup_joystick(device_id, js_type)
        
        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.wireless_controller_puber = ChannelPublisher(
            TOPIC_WIRELESS_CONTROLLER, WirelessController_
        )
        self.wireless_controller_puber.Init()
        self.WirelessControllerThread = RecurrentThread(
            interval=1/self.publish_freq,
            target=self.PublishWirelessController,
            name="sim_wireless_controller",
        )
        
        self.start()
        
    def PublishWirelessController(self):
        if self.joystick != None:
            pygame.event.get()
            key_state = [0] * 16
            key_state[KEY_MAP["R1"]] = self.joystick.get_button(
                self.button_id["RB"]
            )
            key_state[KEY_MAP["L1"]] = self.joystick.get_button(
                self.button_id["LB"]
            )
            key_state[KEY_MAP["start"]] = self.joystick.get_button(
                self.button_id["START"]
            )
            key_state[KEY_MAP["select"]] = self.joystick.get_button(
                self.button_id["SELECT"]
            )
            key_state[KEY_MAP["R2"]] = (
                self.joystick.get_axis(self.axis_id["RT"]) > 0
            )
            key_state[KEY_MAP["L2"]] = (
                self.joystick.get_axis(self.axis_id["LT"]) > 0
            )
            key_state[KEY_MAP["F1"]] = 0
            key_state[KEY_MAP["F2"]] = 0
            key_state[KEY_MAP["A"]] = self.joystick.get_button(self.button_id["A"])
            key_state[KEY_MAP["B"]] = self.joystick.get_button(self.button_id["B"])
            key_state[KEY_MAP["X"]] = self.joystick.get_button(self.button_id["X"])
            key_state[KEY_MAP["Y"]] = self.joystick.get_button(self.button_id["Y"])
            key_state[KEY_MAP["up"]] = self.joystick.get_hat(0)[1] > 0
            key_state[KEY_MAP["right"]] = self.joystick.get_hat(0)[0] > 0
            key_state[KEY_MAP["down"]] = self.joystick.get_hat(0)[1] < 0
            key_state[KEY_MAP["left"]] = self.joystick.get_hat(0)[0] < 0

            key_value = 0
            for i in range(16):
                key_value += key_state[i] << i

            self.wireless_controller.keys = key_value
            self.wireless_controller.lx = self.joystick.get_axis(self.axis_id["LX"])
            self.wireless_controller.ly = -self.joystick.get_axis(self.axis_id["LY"])
            self.wireless_controller.rx = self.joystick.get_axis(self.axis_id["RX"])
            self.wireless_controller.ry = -self.joystick.get_axis(self.axis_id["RY"])

            self.wireless_controller_puber.Write(self.wireless_controller)

    def setup_joystick(self, device_id=0, js_type="xbox"):
        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(device_id)
            self.joystick.init()
        else:
            print("No gamepad detected.")
            sys.exit()

        if js_type == "xbox":
            self.axis_id = {
                "LX": 0,  # Left stick axis x
                "LY": 1,  # Left stick axis y
                "RX": 3,  # Right stick axis x
                "RY": 4,  # Right stick axis y
                "LT": 2,  # Left trigger
                "RT": 5,  # Right trigger
                "DX": 6,  # Directional pad x
                "DY": 7,  # Directional pad y
            }

            self.button_id = {
                "X": 2,
                "Y": 3,
                "B": 1,
                "A": 0,
                "LB": 4,
                "RB": 5,
                "SELECT": 6,
                "START": 7,
            }

        elif js_type == "switch":
            self.axis_id = {
                "LX": 0,  # Left stick axis x
                "LY": 1,  # Left stick axis y
                "RX": 2,  # Right stick axis x
                "RY": 3,  # Right stick axis y
                "LT": 5,  # Left trigger
                "RT": 4,  # Right trigger
                "DX": 6,  # Directional pad x
                "DY": 7,  # Directional pad y
            }

            self.button_id = {
                "X": 3,
                "Y": 4,
                "B": 1,
                "A": 0,
                "LB": 6,
                "RB": 7,
                "SELECT": 10,
                "START": 11,
            }
        else:
            print("Unsupported gamepad. ")
            
    def start(self):
        self.WirelessControllerThread.Start()
        
        print("Press start with the joystick.")
        timout = 5
        t = 0
        sleep = 0.05
        
        start_button_id = KEY_MAP["start"]
        
        while t < timout:
            if self.wireless_controller.keys != 0:
                key_id = int(np.log2(self.wireless_controller.keys))
                if key_id == start_button_id:
                    print("Joystic started!")
                    return
            
            time.sleep(sleep)
            t += sleep
        
        raise TimeoutError(f"Joystick failed to start. No messages received.")