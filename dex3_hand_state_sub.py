# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# dex3_hand_state_sub.py
# 订阅并展示 Dex3 手的 HandState_（仅显示，不发命令）

# 用法示例：
#   python3 dex3_hand_state_sub.py --hand left  --iface enp7s0
#   python3 dex3_hand_state_sub.py --hand right --iface enp7s0 --freq auto
# 参数：
#   --hand {left,right}   选择手
#   --iface IFACE         网络网卡名（如 enp7s0）
#   --domain DOMAIN       DDS DomainId，默认 0
#   --freq {lf,hf,auto}   订阅低频/高频/自动（默认 auto：同时订阅lf与hf，谁来用谁）
# """

# import sys
# import time
# import argparse
# import os
# import signal

# from typing import Optional

# try:
#     from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
#     from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, IMUState_
# except Exception as e:
#     print(f"❌ 无法导入 unitree_sdk2py：{e}", file=sys.stderr)
#     sys.exit(1)

# MOTOR_MAX = 7
# PRESS_CNT = 12  # 每个触觉板 12 个通道（如固件未挂载则可能为空）

# class Dex3HandStateSub:
#     def __init__(self, hand: str, iface: str, domain: int = 0, freq: str = "auto"):
#         self.hand = hand.lower()
#         self.iface = iface
#         self.domain = domain
#         self.freq = freq.lower()
#         self.last_msg: Optional[HandState_] = None
#         self._subscribers = []

#     def init_dds(self):
#         # 外部 PC 强烈建议关闭共享内存（第三参数 False）
#         try:
#             ChannelFactoryInitialize(self.domain, self.iface, False)
#         except TypeError:
#             # 旧版 SDK 只有两个参数也能用
#             ChannelFactoryInitialize(self.domain, self.iface)

#     def _cb(self, msg: HandState_):
#         # 谁先到就用谁（auto 模式会同时订阅 lf 与 hf）
#         self.last_msg = msg

#     def subscribe(self):
#         side = "left" if self.hand == "left" else "right"
#         topics = []
#         if self.freq == "lf":
#             topics = [f"rt/lf/dex3/{side}/state"]
#         elif self.freq == "hf":
#             topics = [f"rt/dex3/{side}/state"]
#         else:  # auto：lf + hf 都订阅
#             topics = [f"rt/lf/dex3/{side}/state", f"rt/dex3/{side}/state"]

#         for tp in topics:
#             sub = ChannelSubscriber(tp, HandState_)
#             sub.Init(self._cb, 8)  # 开个小队列，避免阻塞
#             self._subscribers.append(sub)
#         print(f"✅ 已订阅：{', '.join(topics)}")

#     @staticmethod
#     def _fmt_vec(v, prec=3):
#         try:
#             return "[" + ", ".join(f"{x:.{prec}f}" for x in v) + "]"
#         except Exception:
#             return str(v)

#     def print_once(self):
#         msg = self.last_msg
#         if msg is None:
#             print("… 等待 HandState_ …")
#             return

#         # IMU
#         imu = msg.imu_state
#         print("IMU:")
#         print("  quat (wxyz):", self._fmt_vec(imu.quaternion, 3))
#         print("  gyro  (rad/s):", self._fmt_vec(imu.gyroscope, 3))
#         print("  accel (m/s^2):", self._fmt_vec(imu.accelerometer, 3))
#         # 电源
#         print(f"Power:  V={msg.power_v:.2f} V   I={msg.power_a:.2f} A")
#         # 7 个电机（位置/速度/估计力矩）
#         ms = msg.motor_state
#         if len(ms) >= MOTOR_MAX:
#             q = [ms[i].q for i in range(MOTOR_MAX)]
#             dq = [ms[i].dq for i in range(MOTOR_MAX)]
#             tau = [ms[i].tau_est for i in range(MOTOR_MAX)]
#             print("Motors:")
#             print("  q   :", self._fmt_vec(q, 3))
#             print("  dq  :", self._fmt_vec(dq, 3))
#             print("  tau :", self._fmt_vec(tau, 3))
#         else:
#             print(f"Motors: 收到 {len(ms)} 个（预期 {MOTOR_MAX}）")

#         # 触觉（若固件挂载）
#         ps = msg.press_sensor_state
#         if len(ps) > 0:
#             p = list(ps[0].pressure)
#             t = list(ps[0].temperature)
#             # 厂商建议：pressure >= 100000 为有效，=30000 表示无值（如需可在此做缩放/掩码）
#             print("Pressure[12]:", self._fmt_vec(p, 1))
#             print("Temp    [12]:", self._fmt_vec(t, 1))
#         else:
#             print("Pressure: 无")

#     def spin(self, rate_hz=10.0):
#         print(f"📡 监听 {self.hand.upper()} 手状态（iface={self.iface}, domain={self.domain}, freq={self.freq}）")
#         try:
#             dt = 1.0 / rate_hz
#             while True:
#                 os.system("clear")
#                 self.print_once()
#                 print("\n(CTRL+C 退出)")
#                 time.sleep(dt)
#         except KeyboardInterrupt:
#             print("\n👋 已退出。")


# def main():
#     parser = argparse.ArgumentParser(description="Dex3 Hand State subscriber (Python)")
#     parser.add_argument("--hand", choices=["left", "right"], required=True, help="选择左/右手")
#     parser.add_argument("--iface", required=True, help="网络接口名，如 enp7s0")
#     parser.add_argument("--domain", type=int, default=0, help="DDS DomainId，默认 0")
#     parser.add_argument("--freq", choices=["lf", "hf", "auto"], default="auto",
#                         help="订阅低频/高频/自动（默认 auto）")
#     parser.add_argument("--rate", type=float, default=10.0, help="打印频率 Hz")
#     args = parser.parse_args()

#     sub = Dex3HandStateSub(args.hand, args.iface, args.domain, args.freq)
#     sub.init_dds()
#     sub.subscribe()
#     sub.spin(rate_hz=args.rate)

# if __name__ == "__main__":
#     # 使 Ctrl+C 正常退出
#     signal.signal(signal.SIGINT, signal.SIG_DFL)
#     main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dex3_hand_state_sub.py
双手 Dex3-1 HandState_ 订阅 + 自检 + 结论

用法：
  python3 dex3_hand_state_sub.py --iface enp7s0
可选：
  --domain 0             DDS DomainId（默认0）
  --freq auto            {auto, lf, hf}，默认auto同时订阅lf/hf
  --rate 5               打印频率Hz
  --fresh-ms 1000        数据新鲜阈值ms，用于判断是否掉线/卡住

说明：
- 低频话题：rt/lf/dex3/<left|right>/state
- 高频话题：rt/dex3/<left|right>/state
- 某些固件在低频流中不填 IMU/Power，为0属正常；可切到 --freq hf 试试
- PressSensor：厂商示意 30000为无值，>=100000有效。此处简单做：30000->None，其他值/10000缩放显示。
"""

import sys
import os
import time
import math
import argparse
import signal
from typing import Optional, List, Dict

# --- Unitree Python SDK ---
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_
except Exception as e:
    print(f"❌ 无法导入 unitree_sdk2py：{e}", file=sys.stderr)
    sys.exit(1)

# --- 常量 ---
MOTOR_MAX = 7
PRESS_CNT = 12

# 仅作示例的电机错误位解码字典（不一定覆盖全部）
MOTOR_ERROR_BITS: Dict[int, str] = {
    0x01: "过流",
    0x02: "瞬态过压",
    0x04: "持续过压",
    0x08: "瞬态欠压",
    0x10: "芯片过热",
    0x20: "MOS过热/过冷",
    0x40: "MOS温度异常",
    0x80: "壳体过热/过冷",
    0x100: "壳体温度异常",
    0x200: "绕组过热",
    0x400: "转子编码器1错误",
    0x800: "转子编码器2错误",
    0x1000: "输出编码器错误",
    0x2000: "标定/BOOT数据错误",
    0x4000: "异常复位",
    0x8000: "电机锁定/认证错误",
    0x10000: "芯片验证错误",
    0x20000: "标定模式警告",
    0x40000: "通信校验错误",
    0x80000: "驱动版本过低",
    0x40000000: "PC连接超时(电机端判定)",
    0x80000000: "电机断联超时(PC端判定)",
}

def decode_motor_error(code: int) -> List[str]:
    msgs = []
    for bit, name in MOTOR_ERROR_BITS.items():
        if code & bit:
            msgs.append(f"0x{bit:X}:{name}")
    if not msgs and code != 0:
        msgs.append(f"未知错误位:0x{code:X}")
    return msgs

class HandSideMonitor:
    """订阅并缓存单侧手（left/right）"""
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
        print(f"✅ {self.side.upper()} 订阅：{', '.join(topics)}")

    def _cb(self, msg: HandState_):
        # 谁先来用谁（auto 情况下可能lf/hf混合）
        self.last_msg = msg
        self.last_t = time.time()

    # 数据打印 & 自检
    def summarize(self, fresh_ms: int) -> Dict:
        out = {
            "has_data": self.last_msg is not None,
            "fresh_ms": None,
            "device_error": 0,
            "device_error_str": "N/A",
            "motor_errors": {},  # i -> [msg...]
            "motors_ok": True,
            "conclusion": "无数据",
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

        # 设备级错误
        try:
            e0 = int(msg.error[0])
            e1 = int(msg.error[1])
            dev_code = (e1 << 32) | (e0 & 0xFFFFFFFF)
            out["device_error"] = dev_code
            out["device_error_str"] = f"0x{dev_code:X}" if dev_code else "OK"
        except Exception:
            pass

        # 电机状态
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

        # 触觉
        ps = msg.press_sensor_state
        if len(ps) > 0:
            raw_p = list(ps[0].pressure)
            raw_t = list(ps[0].temperature)
            scaled_p = []
            for v in raw_p:
                if v == 30000.0:
                    scaled_p.append(None)  # 无值
                else:
                    # 直接/10000.0 缩放（不同固件可能略有不同）
                    scaled_p.append(round(v / 10000.0, 4))
            out["pressure"] = scaled_p
            out["temperature"] = raw_t

        # IMU & 电源
        imu = msg.imu_state
        out["imu"] = {
            "quat": list(imu.quaternion),
            "gyro": list(imu.gyroscope),
            "acc":  list(imu.accelerometer),
        }
        out["power"] = {"V": msg.power_v, "A": msg.power_a}

        # 结论
        if not fresh_ok:
            out["conclusion"] = f"数据不新鲜({age_ms}ms) ⚠️"
        elif out["device_error"] != 0:
            out["conclusion"] = f"设备错误 {out['device_error_str']} ❌"
        elif not out["motors_ok"]:
            # 列出出现错误的电机编号
            idxs = ", ".join(str(k) for k in out["motor_errors"].keys())
            out["conclusion"] = f"电机错误: {idxs} ❌"
        else:
            out["conclusion"] = "OK ✅"
        return out


def print_side(side: str, info: Dict):
    title = f"[{side.upper()}]"
    print("=" * (len(title) + 2))
    print(title)
    print("=" * (len(title) + 2))
    if not info["has_data"]:
        print("… 等待数据 …")
        return
    print(f"Age: {info['fresh_ms']} ms")
    # IMU
    imu = info["imu"] or {}
    print("IMU:")
    q = imu.get("quat", [0,0,0,0]); g = imu.get("gyro", [0,0,0]); a = imu.get("acc", [0,0,0])
    print(f"  quat (wxyz): [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]")
    print(f"  gyro  (rad/s): [{g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f}]")
    print(f"  accel (m/s^2): [{a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f}]")

    # 电源
    if info["power"]:
        print(f"Power:  V={info['power']['V']:.2f} V   I={info['power']['A']:.2f} A")

    # 电机
    if info["q"] is not None:
        def fmt(vs, p=3): return "[" + ", ".join(f"{x:.{p}f}" for x in vs) + "]"
        print("Motors:")
        print("  q   :", fmt(info["q"], 3))
        print("  dq  :", fmt(info["dq"], 3))
        print("  tau :", fmt(info["tau"], 3))
        # 错误位
        if info["motor_errors"]:
            print("Motor errors detail:")
            for i, msgs in info["motor_errors"].items():
                if msgs:
                    print(f"  M{i}: " + "; ".join(msgs))
        else:
            print("Motor errors: OK")

    # 触觉
    if info["pressure"] is not None:
        def fmt_ps(vs):
            def f(x):
                return "—" if x is None else f"{x:.4f}"
            return "[" + ", ".join(f(v) for v in vs) + "]"
        print("Pressure[12] (scaled):", fmt_ps(info["pressure"]))
        print("Temp    [12]:", info["temperature"])

    # 设备错误
    print("Device error:", info.get("device_error_str", "N/A"))

    # 结论
    print("Conclusion:", info["conclusion"])
    print()


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    parser = argparse.ArgumentParser(description="Dex3 双手状态订阅 + 自检 + 结论")
    parser.add_argument("--iface", required=True, help="网络接口名，如 enp7s0")
    parser.add_argument("--domain", type=int, default=0, help="DDS DomainId，默认0")
    parser.add_argument("--freq", choices=["auto", "lf", "hf"], default="auto",
                        help="订阅低频/高频/自动（默认auto）")
    parser.add_argument("--rate", type=float, default=5.0, help="打印频率Hz")
    parser.add_argument("--fresh-ms", type=int, default=1000, help="新鲜阈值ms")
    args = parser.parse_args()

    # 初始化 DDS（外部PC需关闭共享内存）
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

        # 总结一句
        print("# 总结")
        print(f"LEFT : {li['conclusion']}   |   RIGHT: {ri['conclusion']}")
        print("(Ctrl+C 退出)")
        time.sleep(dt)


if __name__ == "__main__":
    main()
