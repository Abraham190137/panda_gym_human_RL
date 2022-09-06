import os
import gym
import panda_gym
import pickle
from time import sleep
from agent import Agent
from copy import deepcopy as dc
import numpy as np
import matplotlib.pyplot as plt

from panda_gym.envs.panda_tasks.panda_pick_and_place import PandaPickAndPlaceEnv


# --------------------------------------------------------
# ---------------- DEFINE HYPERPARAMETERS ----------------
# --------------------------------------------------------
DIMENSION_METHOD = 'NONE'   # Options: ['PCA', 'tSNE', 'AE', 'NONE']
RANDOM_SEED = 682           # Past Success for Push: [1008, 543, 1009] other good [682]

set_start = {'object':np.array([0.1, 0.1, 0.02]), 'target':np.array([-0.1, -0.1, 0.12])}

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

env = PandaPickAndPlaceEnv(render = False)
env.seed(RANDOM_SEED)

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
for n in range(1):

    obs = env.reset()
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
        'is_gripped': []}
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
        # recording['reward'].append(reward)
        sleep(0.025)
        action = agent.choose_action(state, desired_goal)
        obs, reward, done, info = env.step(np.array([1, 1, 1, 0.3])*action)
    for key in recording:
        recording[key] = np.array(recording[key])
    with open('test_rerun_buffer2.pkl', 'wb') as file:
            pickle.dump(recording, file)
env.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', computed_zorder = False)
# ax = plt.axes(projection='3d')
# gripper_n = 5
# gripper_points = {'x':[], 'y':[], 'z':[]}
# for i, gripper_pos in enumerate(recording['gripper']):
#     gripper_y = np.linspace(-gripper_pos, gripper_pos, gripper_n) + np.ones(gripper_n)*recording['ee_pos'][i,1]
#     gripper_x = np.ones(gripper_n)*recording['ee_pos'][i,0]
#     gripper_z = np.ones(gripper_n)*recording['ee_pos'][i,2]
#     gripper_points['x'].append(gripper_x.tolist)
#     gripper_points['y'].append(gripper_y.tolist)
#     gripper_points['z'].append(gripper_z.tolist)
c_array =  np.linspace(0, 1, np.shape(recording['ee_pos'])[0])
for i, c in enumerate(c_array[:-1]):
    x = recording['ee_pos'][:, 0]
    y = recording['ee_pos'][:, 1]
    z = recording['ee_pos'][:, 2]
    g = recording['gripper']
    ax.plot3D(x[i:i+2], y[i:i+2], z[i:i+2], color = plt.cm.jet(c), zorder = 2)
    # ax.plot3D(recording['object_pos'][i:i+2, 0], recording['object_pos'][i:i+2, 1], recording['object_pos'][i:i+2, 2], color = plt.cm.jet(c))
    x_surf = [x[i], x[i], x[i+1], x[i+1]]
    z_surf = [z[i], z[i], z[i+1], z[i+1]]
    y_surf = [y[i]-g[i]/2, y[i] + g[i]/2, y[i+1]-g[i+1]/2, y[i+1] + g[i+1]/2]
    if recording['is_gripped'][i+1] and recording['is_gripped'][i]:
        ax.plot_trisurf(x_surf, y_surf, z_surf, color = 'green', alpha = 0.5, zorder = 1)
    else:
        ax.plot_trisurf(x_surf, y_surf, z_surf, color = 'red', alpha = 0.5, zorder = 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=40))
# fake up the array of the scalar mappable. Urgh…
sm._A = []
plt.colorbar(sm)

# block_size = 0.04
# alpha = 0.9
# colors = np.empty([1, 1, 1, 4], dtype=np.float32)
# colors[:] = [1, 0, 0, alpha]  # red

# x, y, z = np.indices((2,2,2))*block_size - block_size/2
# x += recording['object_pos'][0, 0]
# y += recording['object_pos'][0, 1]
# z += recording['object_pos'][0, 2]
# ax.voxels(x, y, z, np.ones([1, 1, 1], dtype = bool), facecolors=colors)

# x, y, z = np.indices((2,2,2))*block_size - block_size/2
# x += recording['object_pos'][-1, 0]
# y += recording['object_pos'][-1, 1]
# z += recording['object_pos'][-1, 2]
# ax.voxels(x, y, z, np.ones([1, 1, 1], dtype = bool), facecolors=colors)

# ax.scatter(recording['object_pos'][0, 0], recording['object_pos'][0, 1], recording['object_pos'][0, 2], color = plt.cm.jet(c_array[0]))
# ax.scatter(recording['object_pos'][-1, 0], recording['object_pos'][-1, 1], recording['object_pos'][-1, 2], color = plt.cm.jet(c_array[-1]))
# print(c)
# ax.plot_trisurf(gripper_points['x'], gripper_points['y'], gripper_points['z'])
# ax.plot3D(recording['ee_pos'][:, 0], recording['ee_pos'][:, 1], recording['ee_pos'][:, 2], c=plt.cm.hot(c))

block_size = 0.04
r = [-block_size/2, block_size/2]
X, Y = np.meshgrid(r, r)
Z = ((block_size/2)*np.ones(4)).reshape(2, 2)
alpha = 0.3

color = 'red'
cen = recording['object_pos'][0]
ax.plot_surface(X + cen[0],Y + cen[1],Z + cen[2], alpha=alpha, color = color)
ax.plot_surface(X + cen[0],Y + cen[1],-Z + cen[2], alpha=alpha, color = color)
ax.plot_surface(X + cen[0],-Z + cen[1],Y + cen[2], alpha=alpha, color = color)
ax.plot_surface(X + cen[0],Z + cen[1],Y + cen[2], alpha=alpha, color = color)
ax.plot_surface(Z + cen[0],X + cen[1],Y + cen[2], alpha=alpha, color = color)
ax.plot_surface(-Z + cen[0],X + cen[1],Y + cen[2], alpha=alpha, color = color)

color = 'green'
cen = desired_goal
ax.plot_surface(X + cen[0],Y + cen[1],Z + cen[2], alpha=alpha, color = color)
ax.plot_surface(X + cen[0],Y + cen[1],-Z + cen[2], alpha=alpha, color = color)
ax.plot_surface(X + cen[0],-Z + cen[1],Y + cen[2], alpha=alpha, color = color)
ax.plot_surface(X + cen[0],Z + cen[1],Y + cen[2], alpha=alpha, color = color)
ax.plot_surface(Z + cen[0],X + cen[1],Y + cen[2], alpha=alpha, color = color)
ax.plot_surface(-Z + cen[0],X + cen[1],Y + cen[2], alpha=alpha, color = color)


axis_limits = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
print(axis_limits)
scaling = max([axis_limits[i][1]-axis_limits[i][0] for i in range(3)])
print(scaling)
new_axis_limits = np.array([[np.mean(axis_limits[i]) - scaling/2, np.mean(axis_limits[i]) + scaling/2] for i in range(3)])
print(new_axis_limits)
ax.auto_scale_xyz(new_axis_limits[0], new_axis_limits[1], new_axis_limits[2])
ax.set_box_aspect((1,1,1))

block_size = scaling
r = [-block_size/2, block_size/2]
X, Y = np.meshgrid(r, r)
Z = ((block_size/2)*np.ones(4)).reshape(2, 2)
alpha = 0.3

cen = np.mean(axis_limits, axis = 1)
cen[2] = 0 - block_size/20
color = [0.7, 0.7, 0.7]
ax.plot_surface(X + cen[0],Y + cen[1],Z/10 + cen[2], alpha=alpha, color = color, zorder=0)
ax.plot_surface(X + cen[0],Y + cen[1],-Z/10 + cen[2], alpha=alpha, color = color, zorder=0)
ax.plot_surface(X + cen[0],-Z + cen[1],Y/10 + cen[2], alpha=alpha, color = color, zorder=0)
ax.plot_surface(X + cen[0],Z + cen[1],Y/10 + cen[2], alpha=alpha, color = color, zorder=0)
ax.plot_surface(Z + cen[0],X + cen[1],Y/10 + cen[2], alpha=alpha, color = color, zorder=0)
ax.plot_surface(-Z + cen[0],X + cen[1],Y/10 + cen[2], alpha=alpha, color = color, zorder=0)
ax.plot_wireframe(X + cen[0],Y + cen[1],Z/10 + cen[2], color = color, zorder=-1)
# ax.plot_wireframe(X + cen[0],Y + cen[1],-Z/10 + cen[2], color = color, zorder=0)
# ax.plot_wireframe(X + cen[0],-Z + cen[1],Y/10 + cen[2], color = color, zorder=0)
# ax.plot_wireframe(X + cen[0],Z + cen[1],Y/10 + cen[2], color = color, zorder=0)
# ax.plot_wireframe(Z + cen[0],X + cen[1],Y/10 + cen[2], color = color, zorder=0)
# ax.plot_wireframe(-Z + cen[0],X + cen[1],Y/10 + cen[2], color = color, zorder=0)
plt.show()

# dist_grip_to_block = np.linalg.norm(recording['ee_pos'] - recording['object_pos'], axis = 1)
# dist_block_to_goal = np.linalg.norm(desired_goal - recording['object_pos'], axis = 1)
# plt.figure()
# plt.plot(dist_block_to_goal)
# plt.plot(dist_grip_to_block)
# plt.plot(recording['gripper'])
# plt.show()