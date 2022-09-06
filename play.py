import torch
import os
import gym
import panda_gym
import pickle
from time import sleep
from agent import Agent
from copy import deepcopy as dc
import numpy as np
import matplotlib.pyplot as plt

from custom_stack_env import CustomStackEnv

from panda_gym.envs.panda_tasks.panda_pick_and_place import PandaPickAndPlaceEnv


# --------------------------------------------------------
# ---------------- DEFINE HYPERPARAMETERS ----------------
# --------------------------------------------------------

DIMENSION_METHOD = 'NONE'   # Options: ['PCA', 'tSNE', 'AE', 'NONE']

set_start = {'object':np.array([0.1, 0.1, 0.02]), 'target':np.array([-0.1, -0.1, 0.12])}
set_start = None

RENDER = True
INTRO = False
Train = True
Play_FLAG = False
EPISODE_STEPS = 50
memory_size = 1e6  # 7e+5 // 50
batch_size = 256
actor_lr = 1e-3
critic_lr = 1e-3
gamma = 0.98
tau = 0.05  
k_future = 0               # Determines what % of the sampled transitions are HER vs ER (k_future = 4 results in 80% HER)
weight_path = "pick_and_place_agent_weights.pth"
weight_path = "C:\\Users\\14127\\Abraham\\Experiments\\Data\\Stack - Humam25p - E200, C20, EP16, ES100, np4, ns0.3, rs61036\\agent_weights.pth"

# env = PandaPickAndPlaceEnv(render = RENDER)
env = CustomStackEnv(render = RENDER)

state_shape = env.observation_space.spaces["observation"].shape 
n_actions = env.action_space.shape[0]
n_goals = env.observation_space.spaces["desired_goal"].shape[0]
action_bounds = [env.action_space.low[0], env.action_space.high[0]]

print("\nState Shape: ", state_shape)
print("N Actions: ", n_actions)
print("N Goals: ", n_goals)
print("Action Bounds: ", action_bounds)

# directory for storing data
path = '/Experiments/Play_Data'
base = os.path.dirname(os.path.realpath(__file__))
path = base + path
print("Path: ", path)
load_path = os.path.join(path, 'run0')


# Create the agent
agent = Agent(n_states=state_shape,
              n_actions=n_actions,
              n_goals=n_goals,
              action_bounds=action_bounds,
              capacity=memory_size,
              action_size=n_actions,
              batch_size=batch_size,
              actor_lr=actor_lr,
              critic_lr=critic_lr,
              gamma=gamma,
              tau=tau,
              k_future=k_future,
              episode_length = EPISODE_STEPS,
              path=load_path,
              human_file_path = None,
              env=dc(env),
              action_penalty = 0.2)

# load the agent_weights.pth
agent.load_weights(weight_path)
agent.set_to_eval_mode()


# iterate 10 times to see a few different times
recordings = []
for n in range(20):
    obs = env.reset()
    sleep(1)
    if set_start is not None:
        env.sim.set_base_pose("target", set_start['target'], np.array([0.0, 0.0, 0.0, 1.0]))
        env.sim.set_base_pose("object", set_start['object'], np.array([0.0, 0.0, 0.0, 1.0]))
        env.task.goal = set_start['target']
        obs = env._get_obs()

    recording = {
        'ee_pos': [],
        'gripper': [],
        'object_pos': [],
        'reward': [],
        'is_gripped': [],
        'goal':[]}

    for i in range(EPISODE_STEPS):
        state = obs["observation"]
        achieved_goal = obs["achieved_goal"]
        desired_goal = obs["desired_goal"]
        ee_pos = state[:3]
        object_pos = state[-12:-9]
        if abs(object_pos[0] - ee_pos[0]) < 0.02 and \
            abs(object_pos[1] - ee_pos[1]) < 0.01 and \
            abs(object_pos[2] - ee_pos[2]) < 0.02:
            recording['is_gripped'].append(True)
        else:
            recording['is_gripped'].append(False)
        gripper = state[6]
        fingure_center = (env.sim.get_link_position('panda', 9) + \
            env.sim.get_link_position('panda', 10))/2
        recording['ee_pos'].append(fingure_center - np.array([0, 0, 0.015]))
        recording['gripper'].append(gripper)
        recording['object_pos'].append(object_pos)
        recording['goal'].append(desired_goal)
        # recording['reward'].append(reward)
        if RENDER:
            sleep(0.1)
        action = agent.choose_action(state, desired_goal)
        recording['state'] = state
        recording['action'] = action
        obs, reward, done, info = env.step(np.array([1, 1, 1, 0.3])*action)
    for key in recording:
        recording[key] = np.array(recording[key])
    recordings.append(recording)
env.close()

plt.figure()
for recording in recordings:
    distance_to_gripper = np.linalg.norm(recording['ee_pos'] - recording['object_pos'], axis = 1)
    distance_to_goal = np.linalg.norm(recording['goal'] - recording['object_pos'], axis = 1)
    # plt.plot(distance_to_gripper)
    # plt.plot(distance_to_goal)
    # plt.plot(recording['gripper'])
    # plt.scatter(distance_to_goal, recording['is_gripped'])
    # plt.legend()
    # plt.plot(distance_to_gripper, recording['gripper'])
    # plt.ylabel('gripper')
    # plt.xlabel('distance_to_gripper')
plt.show()