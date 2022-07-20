import os
import gym
import panda_gym
from time import sleep
from agent import Agent
from copy import deepcopy as dc

from panda_gym.envs.panda_tasks.panda_stack import PandaStackEnV


# --------------------------------------------------------
# ---------------- DEFINE HYPERPARAMETERS ----------------
# --------------------------------------------------------
DIMENSION_METHOD = 'NONE'   # Options: ['PCA', 'tSNE', 'AE', 'NONE']
RANDOM_SEED = 682           # Past Success for Push: [1008, 543, 1009] other good [682]

INTRO = False
Train = True
Play_FLAG = False
memory_size = 1e6  # 7e+5 // 50
batch_size = 256
actor_lr = 1e-3
critic_lr = 1e-3
gamma = 0.98
tau = 0.05  
k_future = 0               # Determines what % of the sampled transitions are HER vs ER (k_future = 4 results in 80% HER)
weight_path = "pick_and_place_agent_weights.pth"

env = PandaStackEnV(render = True)
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
              human_file_path = human_buffer_file,
              env=dc(env),
              action_penalty = 0.2)

# load the agent_weights.pth
agent.load_weights(weight_path)
agent.set_to_eval_mode()

# iterate 10 times to see a few different times
for _ in range(10):

    obs = env.reset()
    for _ in range(EPISODE_STEPS)
        state = obs["observation"]
        achieved_goal = obs["achieved_goal"]
        desired_goal = obs["desired_goal"]

        action = agent.choose_action(state, desired_goal)
        # print("Action: ", action)
        obs, reward, done, info = env.step(action)
        sleep(0.01)
    
env.close()