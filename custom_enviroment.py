
import numpy as np
from typing import Any, Dict, Union
from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda

from typing import Optional
from gym import spaces
from panda_gym.envs.core import PyBulletRobot
from panda_gym.envs.tasks.push import Push

class PickAndPlace(Task):
    def __init__(
        self,
        sim: PyBullet,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.3,
        goal_z_range: float = 0,
        obj_xy_range: float = 0.3) -> None:
        
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.goal_xy_range = goal_xy_range
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, goal_z_range])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.5, width=0.7, height=0.4, x_offset=-0.3)

        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        goal = np.array([0, 0, self.object_size / 2])  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        noise[2] = 0.0
        goal += noise
        #goal = np.array([-0.1, -0.1, self.object_size / 2])  # z offset for the cube center
        self.goal = goal
        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0, 0, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        noise[2] = 0.0
        object_position += noise
        theta = np.pi*np.random.rand(1)
        object_position[0] = self.goal[0] + (self.goal_xy_range/2)*np.sin(theta)
        object_position[1] = self.goal[1] + (self.goal_xy_range/2)*np.cos(theta)
        while abs(object_position[0]) > (self.goal_xy_range/2) + 0.025 or abs(object_position[1]) > (self.goal_xy_range/2) + 0.025:
            theta = np.pi*np.random.rand(1)
            object_position[0] = self.goal[0] + (self.goal_xy_range/2)*np.sin(theta)
            object_position[1] = self.goal[1] + (self.goal_xy_range/2)*np.cos(theta)
            #print('We be looping')
        #object_position = np.array([0.1, 0.1, self.object_size / 2])
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
#         hand_pos = self.sim.get_link_position("panda", 11)
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float64)
        elif self.reward_type == "dense":
            return -d


class OldCustomEnv(RobotTaskEnv):
    """Pick and Place task wih Panda robot.
    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        sim = PyBullet(render=render)
        self.robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = PickAndPlace(sim, reward_type=reward_type)
        super().__init__(self.robot, task)
        
class CustomEnv(RobotTaskEnv):
    """Pick and Place task wih Panda robot.
    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        sim = PyBullet(render=render)
        self.robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Push(sim, reward_type=reward_type)
        super().__init__(self.robot, task)
        