import subprocess
import time
import os
import psutil
# Using go-series DDS, designed for Go2 robot
def kill_other_python3():
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if (
                proc.info['pid'] != current_pid and
                'python3' in proc.info['name'].lower() and
                'python3' in ' '.join(proc.info['cmdline'])
            ):
                proc.terminate()  # or proc.kill() for force
                print(f"Killed: PID {proc.pid} CMD: {' '.join(proc.info['cmdline'])}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        
def run_unitree_mujoco_simulation():
    path = "external_deps/unitree_mujoco/simulate_python/"
    script = "unitree_mujoco.py"

    try:
        kill_other_python3()
        time.sleep(0.1)
        subprocess.run(["python3", script], cwd=path, check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running the simulation script: {e}")
            
if __name__ == "__main__":
    run_unitree_mujoco_simulation()