import time
import mujoco
import pinocchio as pin
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import wraps
from datetime import datetime
import multiprocessing as mp
from tqdm import tqdm
from .ext.keyboard import KBHit
from .utils import mj_2_pin_qv, mj_joint_name2dof, pin_joint_name2dof, quat_from_euler

class Colors():
    RED =     (1.0, 0.0, 0.0, 1.)
    GREEN =   (0.0, 1.0, 0.0, 1.)
    BLUE =    (0.0, 0.0, 1.0, 1.)
    YELLOW =  (1.0, 1.0, 0.0, 1.)
    WHITE =   (0.9, 0.9, 0.9, 1.)
    BLACK =   (0.1, 0.1, 0.1, 1.)

    COLOR_ID_MAP = {
            0 : RED,
            1 : GREEN,
            2 : BLUE,
            3 : YELLOW,
            4 : WHITE,
            5 : BLACK,
        }

    COLOR_NAME_MAP = {
        "red" : RED,
        "green" : GREEN,
        "blue" : BLUE,
        "yellow" : YELLOW,
        "white" : WHITE,
        "black" : BLACK,
    }

    @staticmethod
    def id(id : int) -> List[str]:
        return dict.get(Colors.COLOR_ID_MAP, id, Colors.WHITE)
    
    @staticmethod
    def name(name : str) -> List[str]:
        return dict.get(Colors.COLOR_NAME_MAP, name, Colors.WHITE)

def call_every(func):
    """
    Decorator to call the decorated function only every `self.call_every` steps.
    """
    @wraps(func)
    def wrapper(self, sim_step: int, *args, **kwargs):
        if sim_step % self._call_every == 0:
            return func(self, sim_step, *args, **kwargs)
    return wrapper

class Controller(ABC):
    def __init__(self) -> None:
        # Stop simulation if diverged
        self.diverged : bool = False
        self.joint_name2dof = {}
        self.torques_dof = None
        
    def get_torque_map(self) -> Dict[str, float]:
        torque_map = {
            j_name : self.torques_dof[dof_id]
            for j_name, dof_id
            in self.joint_name2dof.items()
        }
        return torque_map
        
    @abstractmethod
    def compute_torques_dof(self, mj_data) -> None:
        "Update torques_dof"
        
    def get_torques(
        self,
        sim_step : int,
        mj_data,
    ) -> Dict[str, float]:
        self.compute_torques_dof(mj_data)
        return self.get_torque_map()

class MjController(Controller):
    def __init__(self, xml_path : str) -> None:
        super().__init__()
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.nu = mj_model.nu
        self.joint_name2dof = mj_joint_name2dof(mj_model)
        self.torques_dof = np.zeros(mj_model.nv) 

    def get_state(self, mj_data) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get state in mujoco format from mujoco data.
        q : [
             x, y, z,
             qw, qx, qy, qz,
             j1, ..., je,
            ]
        
        v : [
             vx, vy, vz, (global frame)
             wx, wy, wz, (local frame)
             vj1, ..., vje,
            ]

        Returns:
            Tuple[np.ndarray, np.ndarray]: q [nq], v [nv]
        """
        q = mj_data.qpos.copy()
        v = mj_data.qvel.copy()

        return q, v

class PinController(Controller):

    def __init__(self,
                 urdf_path : Optional[str] = "",
                 pin_model: Optional[pin.Model] = None,
                 floating_base_quat : bool = False,
                 floating_base_euler : bool = False):
        super().__init__()

        if not urdf_path and pin_model is None:
            raise ValueError("PinController: Provide at least a pinocchio model or URDF path.")
        
        if pin_model is not None:
            self.pin_model = pin_model
        else:
            self.urdf_path = urdf_path

            if floating_base_quat:
                root = pin.JointModelFreeFlyer()
                self.pin_model = pin.buildModelFromUrdf(urdf_path, root_joint=root)
            elif floating_base_euler:
                root = pin.JointModelComposite(2)
                root.addJoint(pin.JointModelTranslation())
                root.addJoint(pin.JointModelSphericalZYX())
                self.pin_model = pin.buildModelFromUrdf(urdf_path, root_joint=root)
            else:
                self.pin_model = pin.buildModelFromUrdf(urdf_path)

        self.pin_data = pin.Data(self.pin_model)

        self.nq = self.pin_model.nq
        self.nv = self.pin_model.nv
        self.nu = len([j for j in self.pin_model.joints 
                       if j.id < self.nv and j.nv == 1])

        self.joint_name2dof = pin_joint_name2dof(self.pin_model)
        self.torques_dof = np.zeros(self.nv)
        
    def get_state(self, mj_data) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get state in pinocchio format from mujoco data.

        q : [
             x, y, z,
             qx, qy, qz, qw
             j1, ..., je,
            ]
        
        v : [
             vx, vy, vz, (local frame)
             wx, wy, wz, (local frame)
             vj1, ..., vje,
            ]

        Returns:
            Tuple[np.ndarray, np.ndarray]: q [nq], v [nv]
        """
        q_pin, v_pin = mj_2_pin_qv(mj_data.qpos.copy(), mj_data.qvel.copy())

        return q_pin, v_pin
    
class Keyboard(ABC):
    KEYBOARD_UPDATE_FREQ = 50

    def __init__(self) -> None:
        super().__init__()

        self.keyboard = KBHit()
        self.last_key: str = ""
        self.stop = False
        self.update_thread = None
        self._start_update_thread()

    def _keyboard_thread(self):
        """
        Update base goal location based on keyboard events.
        """
        while not self.stop:
            if self.last_key == '\n':  # ENTER
                break
            if self.keyboard.kbhit():
                self.last_key = self.keyboard.getch()
            else:
                self.last_key = ""

            self.on_key()
            time.sleep(1. / Keyboard.KEYBOARD_UPDATE_FREQ)

    def _start_update_thread(self):
        if self.update_thread is None or not self.update_thread.is_alive():
            self.update_thread = threading.Thread(target=self._keyboard_thread)
            self.update_thread.start()

    def _stop_update_thread(self):
        self.last_key = '\n'
        self.stop = True
        if self.update_thread is not None and self.update_thread.is_alive():
            self.update_thread.join()
        self.update_thread = None

    def __del__(self):
        self._stop_update_thread()

    @abstractmethod
    def on_key(self, **kwargs) -> str:
        pass

class DataRecorder(ABC):
    def __init__(
        self,
        record_dir : str = "",
        record_step : int = 1,
        ) -> None:
        self.record_dir = record_dir
        self._call_every = record_step

    @staticmethod
    def get_date_time_str() -> str:
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        return date_time

    @call_every
    def _record(
        self,
        sim_step : int,
        mj_data,
        **kwargs,
    ) -> None:
        self.record(mj_data, **kwargs)
        
    @abstractmethod
    def record(
        self,
        mj_data,
        **kwargs,
    ) -> None:
        pass
    
    @abstractmethod
    def reset(self) -> None:
        pass
    
    @abstractmethod
    def save(self) -> None:
        pass

class VisualCallback(ABC):

    def __init__(self, update_step: int = 1):
        """
        Abstract base class for a MuJoCo viewer callback.

        Args:
            update_step (int): Number of simulation steps between each call to `render`.
        """
        super().__init__()
        self._call_every = update_step
        self.i_geom: int = 0
        self._geom_args = {}
        self.colors = Colors()

    def _add_geom(self, geom_type, pos, rot, size, rgba):
        """
        Add a geometry to the viewer's scene.

        Args:
            viewer: MuJoCo viewer instance.
            geom_type: Geometry type (e.g., `mujoco.mjtGeom.mjGEOM_SPHERE`).
            pos: Position of the geometry in world coordinates.
            rot: Rotation matrix (3x3).
            size: Size of the geometry (array-like).
            rgba: RGBA rgba of the geometry.
        """
        self._geom_args[self.i_geom] = [
            geom_type,
            size,
            pos,
            rot.flatten(),
            rgba,
        ]
        self.i_geom += 1

    def add_sphere(self, pos, radius, rgba):
        """
        Add a sphere to the viewer's scene.

        Args:
            viewer: MuJoCo viewer instance.
            pos: Position of the sphere in world coordinates.
            size: Radius of the sphere.
            rgba: RGBA rgba of the sphere.
        """
        self._add_geom(
            geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=pos,
            rot=np.eye(3),
            size=[radius, 0, 0],
            rgba=rgba,
        )

    def add_box(self, pos, rot_euler, size, rgba):
        """
        Add a box to the viewer's scene.

        Args:
            viewer: MuJoCo viewer instance.
            pos: Position of the box in world coordinates.
            rot: Rotation euler rpy.
            size: Dimensions of the box (length, width, height).
            rgba: RGBA rgba of the box.
        """
        quat = quat_from_euler(*rot_euler)
        rot_matrix = np.zeros(9)
        mujoco.mju_quat2Mat(rot_matrix, quat)
        self._add_geom(
            geom_type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            rot=rot_matrix,
            size=size,
            rgba=rgba,
        )

    @call_every
    def render(self, sim_step, viewer, mj_data):
        """
        Render the scene by calling `_render`.

        Args:
            viewer: MuJoCo viewer instance.
            sim_step: Current simulation step.
            mj_data: MuJoCo data instance.
        """
        self.i_geom = 0
        self.add_visuals(mj_data)

        for i, geom_args in self._geom_args.items():
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                *geom_args
            )

        viewer.user_scn.ngeom = self.i_geom

    @abstractmethod
    def add_visuals(self, viewer, sim_step, mj_data):
        """
        Abstract method to define rendering logic.

        Args:
            viewer: MuJoCo viewer instance.
            sim_step: Current simulation step.
            mj_data: MuJoCo data instance.
        """
        pass

class ParallelExecutorBase(ABC):
    SLEEP_TIME = 0.1
    def __init__(self, verbose: bool = False):
        self.verbose = verbose        
        
    def setup(self):
        self.stop_processes = mp.Value('b', False)
        self.job_submitted = mp.Value('i', 0)
        self.jobs_done = mp.Value('i', 0)  # Shared counter
        self.jobs_success = mp.Value('i', 0)  # Shared counter
        self.processes = []

    @abstractmethod
    def create_job(self, job_id : int) -> dict:
        """
        Producer.
        Creates arguments for the run_job method.
        """
        pass
    
    @abstractmethod
    def run_job(self, job_id : int, **kwargs) -> bool:
        """
        Consumer.
        Processes the job with the arguments provided.
        Returns True if jobs succeeded.
        """
        pass
    
    def _add_job(self, job_queue) -> None:
        """
        Add a job to the workers queue.
        """
        with self.job_submitted.get_lock():
            job_id = self.job_submitted.value
        
        try:
            kwargs = self.create_job(job_id)  # Get job arguments
        except Exception as e:
            print("Create job error.")
            print(e)
            return
           
        job_queue.put((job_id, kwargs))
        
        with self.job_submitted.get_lock():
            self.job_submitted.value = job_id + 1
    
    def _run_job(self, worker_id: int, job_queue) -> None:
        """
        Worker method that runs jobs from the queue.
        """       
        while not self.stop_processes.value:
            success = False
            try:
                # Get job args in the queue
                job_id, kwargs = job_queue.get(block=True, timeout=ParallelExecutorBase.SLEEP_TIME)
                
                # Run jobs
                success = self.run_job(job_id, **kwargs)

            except mp.queues.Empty:
                continue  # Queue is empty, retry

            except Exception as e:
                if self.verbose: print(f"Worker {worker_id} encountered an error: {e}")
                break
            
            if self.verbose:
                s = "succeded" if success else "failed"
                print(f"Job {job_id} {s} (worker {worker_id}).")
            
            # Increment counters
            with self.jobs_done.get_lock():
                self.jobs_done.value += 1
            if success:
                with self.jobs_success.get_lock():  
                    self.jobs_success.value += 1

    def run(self,
            n_cores : int = 1,
            n_jobs: Optional[int] = None,
            n_success: Optional[int] = None):
        """
        Run N iterations of the producer-consumer process in parallel with a progress bar.
        """
        self.setup()
        job_queue = mp.Queue(maxsize=n_cores)

        if n_jobs is None and n_success is None:
            raise ValueError("Provide one argument N_jobs or N_success.")
        
        # Start worker processes
        for i in range(n_cores):
            p = mp.Process(target=self._run_job, args=(i, job_queue))
            p.start()
            self.processes.append(p)

        # Create tqdm progress bar
        N_bar = n_success if n_jobs is None else n_jobs
        desc = "Success" if n_jobs is None else "Jobs"
        i = self.jobs_success if n_jobs is None else self.job_submitted
        
        try:
            with tqdm(total=N_bar, desc=f"{desc} progress", position=0, leave=True) as pbar:
                # Submit n_jobs
                while i.value < N_bar:
                    # Add job, blocking if queue is full
                    self._add_job(job_queue)
                    # Update tqdm bar as jobs are completed
                    pbar.n = self.jobs_done.value  # Update progress count
                    pbar.set_postfix_str(f"submitted: {self.job_submitted.value}, success: {self.jobs_success.value}")
                    pbar.refresh()
                    
                # Finishin all the jobs
                while self.jobs_done.value <= N_bar:
                    pbar.n = self.jobs_done.value  # Update progress count
                    pbar.set_postfix_str(f"submitted: {self.job_submitted.value}, success: {self.jobs_success.value}")
                    pbar.refresh()
                    if self.jobs_done.value == N_bar: break
                    time.sleep(ParallelExecutorBase.SLEEP_TIME)

        except KeyboardInterrupt:
            if self.verbose: print("Stop requested.")

        self.stop()

    def stop(self):
        """
        Stop all worker processes.
        """
        self.stop_processes.value = True
        if self.processes:
            print("Stopping all processes...")
            for i, p in enumerate(self.processes):
                p.join()
                if self.verbose: print(f"Worker {i} stopped.")
            print("All processes stopped.")
        self.processes = []

    def __del__(self):
        self.stop()