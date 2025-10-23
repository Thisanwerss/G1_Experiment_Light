"""
dex3_hand_state_sub.py
Dual Dex3-1 HandState_ subscription + diagnostics + summary

Usage:
  python3 dex3_hand_state_sub.py --iface enp7s0
Options:
  --domain 0             DDS DomainId (default 0)
  --freq auto            {auto, lf, hf}, default auto subscribes to both lf/hf
  --rate 5               Print frequency Hz
  --fresh-ms 1000        Data freshness threshold ms, used to detect disconnection/freeze

Description:
- Low frequency topic: rt/lf/dex3/<left|right>/state
- High frequency topic: rt/dex3/<left|right>/state
- Some firmware doesn't populate IMU/Power in low frequency stream, 0 is normal; try --freq hf
- PressSensor: Vendor indicates 30000 as no value, >=100000 valid. Here we simply do: 30000->None, other values scaled by /10000.
"""

import sys
import os
import time
import math
import argparse
import signal
from typing import Optional, List, Dict
import threading
import json

# --- Unitree Python SDK ---
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_
except Exception as e:
    print(f"❌ Unable to import unitree_sdk2py: {e}", file=sys.stderr)
    sys.exit(1)

# --- Constants and Global Config ---
# Get the script's directory to reliably locate global_config.json
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, 'global_config.json')

def load_default_interface_from_config():
    """Reads the default network interface from the global config file."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            return config.get('network_interface', 'enp7s0') # Fallback if key is missing
    except (FileNotFoundError, json.JSONDecodeError):
        return 'enp7s0' # Fallback if file is missing or corrupt

# --- Constants ---
MOTOR_MAX = 7
PRESS_CNT = 12

# Motor error bit decoding dictionary (example, may not cover all)
MOTOR_ERROR_BITS: Dict[int, str] = {
    0x01: "Overcurrent",
    0x02: "Transient overvoltage",
    0x04: "Sustained overvoltage",
    0x08: "Transient undervoltage",
    0x10: "Chip overheat",
    0x20: "MOS overheat/overcool",
    0x40: "MOS temperature abnormal",
    0x80: "Shell overheat/overcool",
    0x100: "Shell temperature abnormal",
    0x200: "Winding overheat",
    0x400: "Rotor encoder 1 error",
    0x800: "Rotor encoder 2 error",
    0x1000: "Output encoder error",
    0x2000: "Calibration/BOOT data error",
    0x4000: "Abnormal reset",
    0x8000: "Motor lock/authentication error",
    0x10000: "Chip verification error",
    0x20000: "Calibration mode warning",
    0x40000: "Communication checksum error",
    0x80000: "Driver version too low",
    0x40000000: "PC connection timeout (motor side)",
    0x80000000: "Motor disconnection timeout (PC side)",
}

def decode_motor_error(code: int) -> List[str]:
    msgs = []
    for bit, name in MOTOR_ERROR_BITS.items():
        if code & bit:
            msgs.append(f"0x{bit:X}:{name}")
    if not msgs and code != 0:
        msgs.append(f"Unknown error bit:0x{code:X}")
    return msgs

class HandSideMonitor:
    """Subscribe and cache single side hand (left/right)"""
    def __init__(self, side: str, freq: str):
        self.side = side  # 'left' or 'right'
        self.freq = freq  # 'auto' | 'lf' | 'hf'
        self.last_msg: Optional[HandState_] = None
        self.last_t: float = 0.0
        self._subs: List[ChannelSubscriber] = []

    def subscribe(self):
        topics = []
        if self.freq == "lf":
            topics = [f"rt/lf/dex3/{self.side}/state"]
        elif self.freq == "hf":
            topics = [f"rt/dex3/{self.side}/state"]
        else:
            topics = [f"rt/lf/dex3/{self.side}/state", f"rt/dex3/{self.side}/state"]
        for tp in topics:
            sub = ChannelSubscriber(tp, HandState_)
            sub.Init(self._cb, 8)
            self._subs.append(sub)
        print(f"✅ {self.side.upper()} subscribed: {', '.join(topics)}")

    def _cb(self, msg: HandState_):
        # Use whichever comes first (auto mode may mix lf/hf)
        self.last_msg = msg
        self.last_t = time.time()

    # Data printing & diagnostics
    def summarize(self, fresh_ms: int) -> Dict:
        out = {
            "has_data": self.last_msg is not None,
            "fresh_ms": None,
            "device_error": 0,
            "device_error_str": "N/A",
            "motor_errors": {},  # i -> [msg...]
            "motors_ok": True,
            "conclusion": "No data",
            "imu": None,
            "power": None,
            "q": None,
            "dq": None,
            "tau": None,
            "pressure": None,
            "temperature": None,
        }
        if not self.last_msg:
            return out

        age_ms = int((time.time() - self.last_t) * 1000)
        out["fresh_ms"] = age_ms
        fresh_ok = age_ms <= fresh_ms

        msg = self.last_msg

        # Device level error
        try:
            e0 = int(msg.error[0])
            e1 = int(msg.error[1])
            dev_code = (e1 << 32) | (e0 & 0xFFFFFFFF)
            out["device_error"] = dev_code
            out["device_error_str"] = f"0x{dev_code:X}" if dev_code else "OK"
        except Exception:
            pass

        # Motor status
        ms = msg.motor_state
        q = []
        dq = []
        tau = []
        motor_err_any = False
        motor_err_detail = {}

        for i in range(min(MOTOR_MAX, len(ms))):
            q.append(ms[i].q)
            dq.append(ms[i].dq)
            tau.append(ms[i].tau_est)
            code = getattr(ms[i], "motorstate", 0)
            if code:
                motor_err_any = True
                motor_err_detail[i] = decode_motor_error(int(code))

        out["q"], out["dq"], out["tau"] = q, dq, tau
        out["motor_errors"] = motor_err_detail
        out["motors_ok"] = not motor_err_any

        # Tactile
        ps = msg.press_sensor_state
        if len(ps) > 0:
            raw_p = list(ps[0].pressure)
            raw_t = list(ps[0].temperature)
            scaled_p = []
            for v in raw_p:
                if v == 30000.0:
                    scaled_p.append(None)  # No value
                else:
                    # Scale by /10000.0 (may vary slightly with different firmware)
                    scaled_p.append(round(v / 10000.0, 4))
            out["pressure"] = scaled_p
            out["temperature"] = raw_t

        # IMU & Power
        imu = msg.imu_state
        out["imu"] = {
            "quat": list(imu.quaternion),
            "gyro": list(imu.gyroscope),
            "acc":  list(imu.accelerometer),
        }
        out["power"] = {"V": msg.power_v, "A": msg.power_a}

        # Conclusion
        if not fresh_ok:
            out["conclusion"] = f"Data not fresh ({age_ms}ms) ⚠️"
        elif out["device_error"] != 0:
            out["conclusion"] = f"Device error {out['device_error_str']} ❌"
        elif not out["motors_ok"]:
            # List motor indices with errors
            idxs = ", ".join(str(k) for k in out["motor_errors"].keys())
            out["conclusion"] = f"Motor error: {idxs} ❌"
        else:
            out["conclusion"] = "OK ✅"
        return out


def print_side(side: str, info: Dict):
    title = f"[{side.upper()}]"
    print("=" * (len(title) + 2))
    print(title)
    print("=" * (len(title) + 2))
    if not info["has_data"]:
        print("… Waiting for data …")
        return
    print(f"Age: {info['fresh_ms']} ms")
    # IMU
    imu = info["imu"] or {}
    print("IMU:")
    q = imu.get("quat", [0,0,0,0]); g = imu.get("gyro", [0,0,0]); a = imu.get("acc", [0,0,0])
    print(f"  quat (wxyz): [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]")
    print(f"  gyro  (rad/s): [{g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f}]")
    print(f"  accel (m/s^2): [{a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f}]")

    # Power
    if info["power"]:
        print(f"Power:  V={info['power']['V']:.2f} V   I={info['power']['A']:.2f} A")

    # Motors
    if info["q"] is not None:
        def fmt(vs, p=3): return "[" + ", ".join(f"{x:.{p}f}" for x in vs) + "]"
        print("Motors:")
        print("  q   :", fmt(info["q"], 3))
        print("  dq  :", fmt(info["dq"], 3))
        print("  tau :", fmt(info["tau"], 3))
        # Error bits
        if info["motor_errors"]:
            print("Motor errors detail:")
            for i, msgs in info["motor_errors"].items():
                if msgs:
                    print(f"  M{i}: " + "; ".join(msgs))
        else:
            print("Motor errors: OK")

    # Tactile
    if info["pressure"] is not None:
        def fmt_ps(vs):
            def f(x):
                return "—" if x is None else f"{x:.4f}"
            return "[" + ", ".join(f(v) for v in vs) + "]"
        print("Pressure[12] (scaled):", fmt_ps(info["pressure"]))
        print("Temp    [12]:", info["temperature"])

    # Device error
    print("Device error:", info.get("device_error_str", "N/A"))

    # Conclusion
    print("Conclusion:", info["conclusion"])
    print()


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    parser = argparse.ArgumentParser(description="Dex3 dual hand state subscription + diagnostics + summary")
    parser.add_argument("--iface", default=load_default_interface_from_config(), 
                        help=f"Network interface name (default read from {CONFIG_FILE})")
    parser.add_argument("--domain", type=int, default=0, help="DDS DomainId, default 0")
    parser.add_argument("--freq", choices=["auto", "lf", "hf"], default="auto",
                        help="Subscribe to low/high frequency/auto (default auto)")
    parser.add_argument("--rate", type=float, default=5.0, help="Print frequency Hz")
    parser.add_argument("--fresh-ms", type=int, default=1000, help="Freshness threshold ms")
    args = parser.parse_args()

    # Initialize DDS (external PC needs to disable shared memory)
    try:
        ChannelFactoryInitialize(args.domain, args.iface, False)
    except TypeError:
        ChannelFactoryInitialize(args.domain, args.iface)

    left = HandSideMonitor("left", args.freq)
    right = HandSideMonitor("right", args.freq)
    left.subscribe()
    right.subscribe()

    dt = 1.0 / max(0.1, args.rate)
    while True:
        os.system("clear")
        li = left.summarize(args.fresh_ms)
        ri = right.summarize(args.fresh_ms)

        print_side("left", li)
        print_side("right", ri)

        # Summary
        print("# Summary")
        print(f"LEFT : {li['conclusion']}   |   RIGHT: {ri['conclusion']}")
        print("(Ctrl+C to exit)")
        time.sleep(dt)


if __name__ == "__main__":
    main()
