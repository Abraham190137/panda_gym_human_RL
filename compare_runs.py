import torch
from custom_enviroment import CustomEnv
import os
import gym
import panda_gym
import pickle
from time import sleep
from agent import Agent
from copy import deepcopy as dc
import numpy as np
import matplotlib.pyplot as plt
from human_agent import PickAndPlaceHumanAgent
from human_agent import PushHumanAgent

from panda_gym.envs.panda_tasks.panda_pick_and_place import PandaPickAndPlaceEnv


# --------------------------------------------------------
# ---------------- DEFINE HYPERPARAMETERS ----------------
# --------------------------------------------------------

DIMENSION_METHOD = 'NONE'   # Options: ['PCA', 'tSNE', 'AE', 'NONE']

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
# weight_path2 = "C:\\Users\\14127\\Abraham\\Experiments\\Data\\Pick and Place - Humam25p - E100, C20, EP16, np4, ns0.3, rs33060" + "\\agent_weights.pth"
# weight_path1 = "C:\\Users\\14127\\Abraham\\Experiments\\Data\\Pick and Place - Humam25p - E100, C20, EP16, np4, ns0.3, rs50467" + "\\agent_weights.pth"

weight_path1 = "C:\\Users\\14127\\Abraham\\Experiments\\Data\\Humam25p - E40, C20, EP16, np4, drag (fixed), rs7545" + "\\agent_weights.pth"
weight_path2 = "C:\\Users\\14127\\Abraham\\Experiments\\Data\\Human25p - E40, C20, EP16, np4, rs18731" + "\\agent_weights.pth"

action_multiplier = np.array([1, 1, 1, 0.3])

# env = PandaPickAndPlaceEnv(render = RENDER)
env = CustomEnv(render = RENDER)

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


# # Create the agent
# recording_file_name = "Oculus_Data\\test59.txt" # Oculus output file to read in
# agent = PickAndPlaceHumanAgent(env, recording_file_name, 8, EPISODE_STEPS, start=205, z_start = 260,
#                    end=-1)

# recording_file_name = "Oculus_Data\\test49.txt" # Oculus output file to read in
# agent = PushHumanAgent(env, recording_file_name, 5, EPISODE_STEPS, start=200,
#                    end=650)

recording_file_name = "Oculus_Data\\test56.txt" # Oculus output file to read in
agent = PushHumanAgent(env, recording_file_name, 0, EPISODE_STEPS, start=110,
                   end=-1, offset = [0, 0, -0.02, 0])

agent1 = Agent(n_states=state_shape,
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
              env=dc(env))

agent2 = Agent(n_states=state_shape,
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
              env=dc(env))

# load the agent_weights.pth
agent1.load_weights(weight_path1)
agent1.set_to_eval_mode()

agent2.load_weights(weight_path2)
agent2.set_to_eval_mode()

def find_angle(a, b):
    c = np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))

# iterate 10 times to see a few different times
recordings = []
for n in range(200):
    obs = env.reset()
    agent.reset(obs['achieved_goal'], obs['desired_goal'][0:3])
    recording = {
        'state':[],
        'action':[],
        'action1':[],
        'action2':[],
        'goal':[],
        'ee_pos':[],
        'object_pos':[],
        'angle':[],
        'cross':[],
        'is_gripped':[],
        'gripper':[],
        'velocity':[]}
    for i in range(EPISODE_STEPS):
        state = obs["observation"]
        desired_goal = obs["desired_goal"]
        # recording['reward'].append(reward)
        if RENDER:
            sleep(0.1)
        action = agent.get_action(i)/action_multiplier
        action1 = agent1.choose_action(state, desired_goal)
        action2 = agent2.choose_action(state, desired_goal)
        angle = find_angle(action[:3], action1[:3])

        object_pos = state[-12:-9]
        ee_pos = state[:3]
        if abs(object_pos[0] - ee_pos[0]) < 0.02 and \
            abs(object_pos[1] - ee_pos[1]) < 0.01 and \
            abs(object_pos[2] - ee_pos[2]) < 0.02 and \
            state[6] < 0.042:
            recording['is_gripped'].append(1)
        else:
            recording['is_gripped'].append(0)

        recording['object_pos'].append(state[-12:-9])
        recording['gripper'].append(state[6])
        recording['ee_pos'].append(ee_pos)
        achieved_goal = obs["achieved_goal"]
        recording['goal'] = desired_goal
        recording['state'].append(state)
        recording['action'].append(action)
        recording['action1'].append(action1)
        recording['action2'].append(action2)
        recording['angle'].append(angle)
        recording['cross'].append(np.linalg.norm(np.cross(action[:3], action1[:3])))
        recording['velocity'].append(state[3:6])
        obs, reward, done, info = env.step(action_multiplier*action)
    for key in recording:
        recording[key] = np.array(recording[key])
    if reward == 0:
        recordings.append(recording)
env.close()

def normalize_actions(action_array):
    norm_array = np.zeros(np.shape(action_array))
    norm_array[:, :3] = action_array[:, :3]/np.mean(np.linalg.norm(action_array[:, :3], axis = 1))
    norm_array[:, 3] = action_array[:, 3]/np.mean(abs(action_array[:, 3]))
    return norm_array

move_diff = []
move_diff_h = []


plt.figure()
for recording in recordings:

    distance_to_gripper = np.linalg.norm(recording['ee_pos'] - recording['object_pos'], axis = 1)
    distance_to_goal = np.linalg.norm(recording['goal'] - recording['object_pos'], axis = 1)

    # # plt.plot(np.linalg.norm(recording['action'] - recording['action1'], axis = 1))
    # plt.plot(3*distance_to_goal, label = 'dist to goal')
    # plt.plot(3*distance_to_gripper, label = 'dist to grip')
    # plt.plot(recording['is_gripped'], label = 'is gripped')
    # plt.plot(10*recording['gripper'], label = 'gripper width')
    # plt.plot(recording['action'][:, 3]/np.mean(abs(recording['action'][:, 3])), label = 'gripper act')
    # plt.plot(recording['action1'][:, 3]/np.mean(abs(recording['action1'][:, 3])), label = 'gripper act alt')
    # plt.plot(recording['action'][:, 0])
    # plt.plot(recording['action1'][:, 0])
    # plt.legend(['human', 'gen']
    # plt.legend(['angle', 'goal dist', 'grip dist', 'is grip', 'grip', 'grip act', 'grip act2'])
    action_norm = normalize_actions(recording['action'])
    action1_norm = normalize_actions(recording['action1'])
    action2_norm = normalize_actions(recording['action2'])

    move_diff_h.append(np.linalg.norm((action_norm[:, :3] - action2_norm[:, :3]), axis = 1))
    move_diff.append(np.linalg.norm((action2_norm[:, :3] - action1_norm[:, :3]), axis = 1))
    # plt.plot(np.linalg.norm((action_norm[:, :3] - action1_norm[:, :3]), axis = 1), label = 'move')

    # plt.plot(abs(action_norm[:, 3] - action1_norm[:, 3]), label = 'grip')
    # plt.legend()

    # plt.figure()
    # plt.plot(action_norm[:, 0])
    # plt.title('x')
    # plt.plot(action1_norm[:, 0])
    # plt.legend(['human', 'gen'])

    # plt.figure()
    # plt.title('y')
    # plt.plot(action_norm[:, 1])
    # plt.plot(action1_norm[:, 1])
    # plt.legend(['human', 'gen'])

    # plt.figure()
    # plt.title('z')
    # plt.plot(action_norm[:, 2])
    # plt.plot(action1_norm[:, 2])
    # plt.legend(['human', 'gen'])

    # plt.figure()
    # plt.title('g')
    # plt.plot(action_norm[:, 3])
    # plt.plot(action1_norm[:, 3])
    # plt.legend(['human', 'gen'])

    # # plt.figure()
    # # plt.plot(recording['cross'], label = 'cross')
    # # plt.plot(np.linalg.norm(recording['velocity'], axis = 1), label = 'velocity')

    # # plt.figure()
    # # plt.scatter(np.linalg.norm(recording['velocity'], axis = 1), recording['cross'])
plt.legend()

# pickle.dump(move_diff, open('non_human_compare.pkl', 'wb'))
pickle.dump(move_diff_h, open('pick_human.pkl', 'wb'))

mean_cross_run = np.array([np.mean(i['cross']) for i in recordings])
mean_cross_run = np.mean(np.array(move_diff), axis = 1)

plt.figure()
plt.hist([mean_cross_run, np.mean(np.array(move_diff_h), axis = 1)])


cross_data = np.array([i['cross'] for i in recordings])
cross_data = np.array(move_diff)
cross_avg = np.mean(cross_data, axis = 0)
cross_std = np.std(cross_data, axis = 0)

vel_data = np.array([np.linalg.norm(i['velocity'], axis = 1) for i in recordings])
vel_mean = np.mean(vel_data, axis = 0)
vel_std = np.std(vel_data, axis = 0)

fig, ax = plt.subplots()
ax.plot(range(EPISODE_STEPS), cross_avg, '-', color='black')
ax.fill_between(range(EPISODE_STEPS), cross_avg+cross_std, cross_avg-cross_std, 
                alpha=0.2, color='black', label = '_nolegend_')
ax.plot(range(EPISODE_STEPS), vel_mean, '-', color='red')
ax.fill_between(range(EPISODE_STEPS), vel_mean+vel_std, vel_mean-vel_std, 
                alpha=0.2, color='red', label = '_nolegend_')
plt.show()