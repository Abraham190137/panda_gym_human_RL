
from custom_enviroment import CustomEnv
from custom_stack_env import CustomStackEnv
from panda_gym.envs.panda_tasks.panda_pick_and_place import PandaPickAndPlaceEnv
# from panda_gym.envs.panda_tasks.panda_stack import PandaStackEnv
from noise import UniformNoise, OUNoise, UniformNoiseDecay
from agent import Agent
from human_buffer import HumanBuffer
import math
import gym
import itertools
import panda_gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from mpi4py import MPI
import psutil
import time
from copy import deepcopy as dc
import os
import torch
import csv
import time

# Name of the folder to save the run data in:
run_label = "Stack - Humam25p - E200, C20, EP16, ES100, np4, ns0.3, HER0, rs"
# run_label = 'Delete me'

# Name of the human_buffer_file to use. If no human buffer is desired, put None
human_buffer_file = "Human_Buffers/Custom Stack (semi-sparse), ns0.3, s5000, 7-21.pkl"
RunEnv = CustomStackEnv # Panda_Gym environment for the run.
env_name = "CustomStackEnv"
    
# --------------------------------------------------------
# ---------------- DEFINE HYPERPARAMETERS ----------------
# --------------------------------------------------------
# SAMPLING_METHOD = 'RANDOM'  # Options: ['CONVEX', 'CLUSTER', 'LSH', 'RANDOM', 'COSINE', 'ICM', 'EC']
# DIMENSION_METHOD = 'VAESIM' # Options: ['PCA', 'tSNE', 'AE', 'VAE', 'VAESIM', NONE', 'LSH']
RANDOM_SEED = np.random.randint(0, 100000)             # Past Success for Push: [1008, 543, 1009] other good [682]

# NOTE: random seeds to use = [5, 117, 243, 682, 905]
RENDER = True
INTRO = False
Train = True
Play_FLAG = False
MAX_EPOCHS = 200             # NOTE: fetch push should need 12-13 epochs to achieve baseline performance
MAX_CYCLES = 20
num_updates = 40
MAX_EPISODES = 16           # NOTE: matching HER paper that has 800 episodes per epoch (50 cycles * 16 episodes = 800)
EPISODE_STEPS = 100 
memory_size = 1e6  # 7e+5 // 50
batch_size = 256
actor_lr = 1e-3
critic_lr = 1e-3
gamma = 0.98
tau = 0.05  
k_future = 0                # Determines what % of the sampled transitions are HER vs ER (k_future = 4 results in 80% HER)
human_portion = 0.25 # Portion of training done with the human buffer
action_penalty = 1
action_multiplier = np.array([1, 1, 1, 0.3]) # Adjust the action. Between 0.1 and 0.3 seems good.
# resume_file = "pick_and_place_agent_weights.pth"
# resume_epoch = 100


# store the hyper parameters in pandas data frame to save later on
n_timesteps = MAX_EPOCHS * MAX_CYCLES * MAX_EPISODES * EPISODE_STEPS
hyperparams = {'col1': ["Random Seed", "Max Epochs", "Max Cycles", "Max Episodes",
        "Total Env Steps", "Number of Updates", "Memory Size", "Batch Size", 
        "Actor LR", "Critic LR", "Gamma", "Tau", "K Future", "Human Portion", 
        "Env", "Run Label", "Episode Steps", "action_penalty", 'action mult'], 
    'col2': [RANDOM_SEED, MAX_EPOCHS, MAX_CYCLES, MAX_EPISODES, n_timesteps, 
        num_updates, memory_size, batch_size, actor_lr, critic_lr, gamma, tau, 
        k_future, human_portion, env_name, run_label, EPISODE_STEPS, 
        action_penalty, action_multiplier]}

hyp_df = pd.DataFrame(data=hyperparams)



# --------------------------------------------------------
# ------------------ CREATE ENVIRONMENT ------------------
# --------------------------------------------------------

test_env = RunEnv(render = False)
state_shape = test_env.observation_space.spaces["observation"].shape
n_actions = test_env.action_space.shape[0]
n_goals = test_env.observation_space.spaces["desired_goal"].shape[0]
action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['IN_MPI'] = '1'

# --------------------------------------------------------
# --------------- CREATE EXPERIMENT FOLDER ---------------
# --------------------------------------------------------
# path = '/Experiments'
path = '/Experiments/Data'

# directory for storing data
#base = os.path.dirname(os.path.realpath(__file__))
base = os.getcwd()
path = base + path

# # check the existing experiment runs to choose the new filename
# directory_name = 'run27'
#directory_name = run_label + str(RANDOM_SEED)
directory_name = run_label + str(RANDOM_SEED)
load_path = os.path.join(path, 'run1')
path = os.path.join(path, directory_name)

def eval_agent(env_, agent_, delay = 0.01):
    """
    Evaluate the current RL agent for 10 episodes in the simulation environment.
    :param env_:    the environment
    :param agent_:  the RL agent
    :return:        success rate, running reward, episode reward
    """

    total_success_rate = []
    running_r = []
    
    # iterate through 10 episodes 
    for ep in range(10):
        per_success_rate = []
        env_dictionary = env_.reset()
        s = env_dictionary["observation"]
        ag = env_dictionary["achieved_goal"]
        g = env_dictionary["desired_goal"]

        # check if the achieved goal and goal are the same (if so, reset until not the case)
        while env_.compute_reward(ag, g, None) == 0:
            env_dictionary = env_.reset()
            s = env_dictionary["observation"]
            ag = env_dictionary["achieved_goal"]
            g = env_dictionary["desired_goal"]
        ep_r = 0

        # iterate through the 50 environment steps (max number of environment steps allowed)
        for t in range(EPISODE_STEPS):
            with torch.no_grad():
                # have the agent choose an action given the observation & goal (training turned off)
                a = agent_.choose_action(s, g)

            # step the environment & get new observation and reward
            observation_new, r, _, info_ = env_.step(a*action_multiplier)
            s = observation_new['observation']
            g = observation_new['desired_goal']
            per_success_rate.append(info_['is_success'])
            ep_r += r 
            if delay != 0:
                time.sleep(delay)
            
        total_success_rate.append(per_success_rate)
        if ep == 0:
            running_r.append(ep_r)
        else:
            running_r.append(running_r[-1] * 0.99 + 0.01 * ep_r)

    total_success_rate = np.array(total_success_rate)
    local_success_rate = np.mean(total_success_rate[:, -1])
    global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
    return global_success_rate / MPI.COMM_WORLD.Get_size(), running_r, ep_r


# --------------------------------------------------------
# ---------- INTRO THAT PRINTS ENVIRONMENT INFO ----------
# --------------------------------------------------------
if INTRO:
    print(f"state_shape:{state_shape[0]}\n"
          f"number of actions:{n_actions}\n"
          f"action boundaries:{action_bounds}\n"
          f"max timesteps:{test_env._max_episode_steps}")
    for _ in range(3):
        done = False
        test_env.reset()
        while not done:
            action = test_env.action_space.sample()
            test_state, test_reward, test_done, test_info = test_env.step(action*action_multiplier)
            # substitute_goal = test_state["achieved_goal"].copy()
            # substitute_reward = test_env.compute_reward(
            #     test_state["achieved_goal"], substitute_goal, test_info)
            # print("r is {}, substitute_reward is {}".format(r, substitute_reward))
            test_env.render()
    exit(0)


# --------------------------------------------------------
# ------------- CREATE ENVIRONMENT & AGENT ---------------
# --------------------------------------------------------
env = RunEnv(render = False)

# Define the random seeds
rank_seed = RANDOM_SEED + 1000000 * MPI.COMM_WORLD.Get_rank()
env.reset(seed = rank_seed)
random.seed(rank_seed)
np.random.seed(rank_seed)
torch.manual_seed(rank_seed)

if MPI.COMM_WORLD.Get_rank() == 0:
    if not os.path.exists(path):
        os.mkdir(path)

    print_file = path + "/printout.txt"    
    with open(print_file, 'a') as file:
        file.write('Start Main \n\n')

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
              action_penalty = action_penalty)


# --------------------------------------------------------
# -------------------- TRAINING LOOP ---------------------
# --------------------------------------------------------
# if resume_file is None:
#     total_episodes = 0
#     resume_epoch = 0
# else:
#     agent.load_weights(resume_file)
#     total_episodes = resume_epoch*MAX_CYCLES*MAX_EPISODES
#     if RENDER and MPI.COMM_WORLD.Get_rank() == 0:
#             env.close()
#             env = RunEnv(render = True)
#             success_rate, running_reward, episode_reward = eval_agent(env, agent)
#             env.close()
#             env = RunEnv(render = False)
#         else:
#             success_rate, running_reward, episode_reward = eval_agent(env, agent, delay=0)
total_episodes = 0
noise = UniformNoise(env.action_space, amplitude = 0.2, pure_rand = 0.3)
noise.reset()

if Train:
    t_success_rate = []
    t_reward = []
    ep_success = []
    ep_reward = []
    total_ac_loss = []
    total_cr_loss = []

    # iterate though the epochs
    for epoch in range(MAX_EPOCHS):
        start_time = time.time()
        epoch_actor_loss = 0
        epoch_critic_loss = 0
        
        # iterate through the cycles
        for cycle in range(0, MAX_CYCLES):
            mb = []

            cycle_actor_loss = 0
            cycle_critic_loss = 0

            # iterate through episodes
            for episode in range(MAX_EPISODES):
                total_episodes += 1
                episode_dict = {
                    "state": [],
                    "action": [],
                    "info": [],
                    "achieved_goal": [],
                    "desired_goal": [],
                    "next_state": [],
                    "next_achieved_goal": []}
                env_dict = env.reset()
                state = env_dict["observation"]
                achieved_goal = env_dict["achieved_goal"]
                desired_goal = env_dict["desired_goal"]

                # make sure that the achieved goal and desired goal aren't the same!
                while env.compute_reward(achieved_goal, desired_goal, None) == 0:
                    env_dict = env.reset()
                    state = env_dict["observation"]
                    achieved_goal = env_dict["achieved_goal"]
                    desired_goal = env_dict["desired_goal"]

                rew = 0
                noise.reset()

                # maximum episode length of 50 ---- MAYBE TURN THIS INTO ANOTHER HYPERPARAMETER?????
                for t in range(EPISODE_STEPS):

                    action_noiseless = agent.choose_action(state, desired_goal)
                    action = noise.get_action(action_noiseless, total_episodes)
                    
                    next_env_dict, reward, done, info = env.step(action*action_multiplier)
                    rew+=reward

                    next_state = next_env_dict["observation"]
                    next_achieved_goal = next_env_dict["achieved_goal"]
                    next_desired_goal = next_env_dict["desired_goal"]

                    episode_dict["state"].append(state.copy())
                    episode_dict["action"].append(action.copy())
                    episode_dict["achieved_goal"].append(achieved_goal.copy())
                    episode_dict["desired_goal"].append(desired_goal.copy())

                    state = next_state.copy()
                    achieved_goal = next_achieved_goal.copy()
                    desired_goal = next_desired_goal.copy()

                episode_dict["state"].append(state.copy())
                episode_dict["achieved_goal"].append(achieved_goal.copy())
                episode_dict["desired_goal"].append(desired_goal.copy())
                episode_dict["next_state"] = episode_dict["state"][1:]
                episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]
                mb.append(dc(episode_dict))
                ep_reward.append(rew)

            # add transitions to the replay buffer (after each cycle ---> MAX_EPISODES * MAX_EP_LENGTH)
            agent.store(mb)

            # iterate through the number of updates (hyperparameter)
            for n_update in range(num_updates):
                actor_loss, critic_loss = agent.train(human_portion = human_portion)
                cycle_actor_loss += actor_loss
                cycle_critic_loss += critic_loss

            epoch_actor_loss += cycle_actor_loss / num_updates
            epoch_critic_loss += cycle_critic_loss /num_updates
            agent.update_networks()

        ram = psutil.virtual_memory()
        
        if RENDER and MPI.COMM_WORLD.Get_rank() == 0:
            env.close()
            env = RunEnv(render = True)
            success_rate, running_reward, episode_reward = eval_agent(env, agent)
            env.close()
            env = RunEnv(render = False)
        else:
            success_rate, running_reward, episode_reward = eval_agent(env, agent, delay=0)
        
        total_ac_loss.append(epoch_actor_loss)
        total_cr_loss.append(epoch_critic_loss)

        # perform these actions at each epoch
        if MPI.COMM_WORLD.Get_rank() == 0:
            t_success_rate.append(success_rate)
            t_reward.append(running_reward[-1])

            print(f"Epoch:{epoch}| "
                  f"Running_reward:{running_reward[-1]:.3f}| "
                  f"EP_reward:{episode_reward:.3f}| "
                  f"Memory_length:{len(agent.memory)}| "
                  f"Duration:{time.time() - start_time:.3f}| "
                  f"Actor_Loss:{actor_loss:.3f}| "
                  f"Critic_Loss:{critic_loss:.3f}| "
                  f"Success rate:{success_rate:.3f}| "
                  f"{to_gb(ram.used):.1f}/{to_gb(ram.total):.1f} GB RAM")
            
            with open(print_file, 'a') as file:
                file.write(f"Epoch:{epoch}| "
                  f"Running_reward:{running_reward[-1]:.3f}| "
                  f"EP_reward:{episode_reward:.3f}| "
                  f"Memory_length:{len(agent.memory)}| "
                  f"Duration:{time.time() - start_time:.3f}| "
                  f"Actor_Loss:{actor_loss:.3f}| "
                  f"Critic_Loss:{critic_loss:.3f}| "
                  f"Success rate:{success_rate:.3f}| "
                  f"{to_gb(ram.used):.1f}/{to_gb(ram.total):.1f} GB RAM \n\n")

            # check if this is the first time saving the weights & create experiment run folder
            if not os.path.exists(path):
                os.mkdir(path)
            
            file = path + "/agent_weights.pth" # "FetchPickAndPlace-50-50-16.pth"
            agent.save_weights(file)

    if MPI.COMM_WORLD.Get_rank() == 0:

        for i, success_rate in enumerate(t_success_rate):
            print("Success_rate", success_rate, i)

        # -------------------------------------------------------------
        # -------------------- SAVE DATA TO CSV -----------------------
        # -------------------------------------------------------------
        
        # want to have columns: epochs, reward, success (NOTE: maybe add loss too?)
        titles = ['Epochs', 'Reward', 'Success Rate']
        data = np.zeros((len(t_success_rate), 3))
        epochs = np.arange(0, MAX_EPOCHS)

        # iterate through the lists of epochs, loss, etc.
        for i in range(len(t_success_rate)):
            row = [epochs[i], t_reward[i], t_success_rate[i]]
            data[i,:] = row

        pd_data = pd.DataFrame(data, columns = titles)

        # write to the csv file
        data_csv = path + '/data.csv'
        pd_data.to_csv(data_csv, index=None) 


        # -------------------------------------------------------------
        # --------------- SAVE HYPERPARAMETERS TO CSV -----------------
        # -------------------------------------------------------------     
        hyp_csv = path + '/hyperparameters.csv'
        hyp_df.to_csv(hyp_csv, index=None) 


        # -------------------------------------------------------------
        # ------------------- PLOT SUCCESS/REWARD ---------------------
        # -------------------------------------------------------------
        plt.style.use('ggplot')

        # plot epoch vs. success rate
        plt.figure()
        plt.plot(np.arange(0, MAX_EPOCHS), t_success_rate)
        plt.xlabel("Epoch")
        plt.ylabel("Success Rate")
        plt.savefig(path + "/success_rate_epoch.png")
        # plt.show()

        # plot epoch vs. reward
        plt.figure()
        plt.plot(np.arange(0, MAX_EPOCHS), t_reward)
        plt.xlabel("Epoch")
        plt.ylabel("Reward")
        plt.savefig(path + "/reward_epoch.png")
        # plt.show()

        # plot environment steps vs. success rate
        plt.figure()
        plt.plot(np.arange(0, MAX_EPOCHS*MAX_CYCLES*MAX_EPISODES*50, MAX_CYCLES*MAX_EPISODES*50), t_success_rate)
        plt.xlabel("Environment Steps")
        plt.ylabel("Success Rate")
        plt.savefig(path + "/success_rate_env_steps.png")
        # plt.show()

        # plot environment steps vs episode reward
        plt.figure()
        plt.plot(np.arange(0, len(ep_reward)), ep_reward)
        plt.xlabel("Environment Steps")
        plt.ylabel("Reward")
        plt.savefig(path + "/reward_env_steps.png")
        # plt.show()

        # plot environment steps vs avg reward per cycle
        # get average of ep_reward for every MAX_CYCLES
        cyc_r = []
        for i in range(int(len(ep_reward) / MAX_CYCLES)):
            average = np.mean(ep_reward[(i*MAX_CYCLES):((i+1)*MAX_CYCLES)])
            cyc_r.append(average)
        plt.figure()
        plt.plot(np.arange(0, len(cyc_r)), cyc_r)
        plt.xlabel("Cycle Steps")
        plt.ylabel("Reward")
        plt.savefig(path + "/avg_reward_each_cycle.png")
        # plt.show()

        titles = ['Episodes', 'Reward']
        rdata = np.zeros((len(ep_reward), 2))
        episodes = np.arange(0, len(ep_reward))

        for i in range(len(ep_reward)):
            row = [episodes[i], ep_reward[i]]
            rdata[i,:] = row
        pd_rdata = pd.DataFrame(rdata, columns = titles)

        # write to the csv file
        rdata_csv = path + '/reward_data.csv'
        pd_rdata.to_csv(rdata_csv, index=None)


# --------------------------------------------------------
# -------------- LOOP PLAYING TRAIN AGENT ----------------
# --------------------------------------------------------
elif Play_FLAG:
    weight_file = "pick_and_place_agent_weights.pth"
    player = Play(env, agent, weight_file, max_episode=100)
    player.evaluate()
