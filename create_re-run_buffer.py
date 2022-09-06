
### --- Create Human Buffer --- ###
import numpy as np
import time

from custom_enviroment import CustomEnv
from custom_stack_env import CustomStackEnv
from copy import deepcopy as dc
from rerun_agent import *
from panda_gym.envs.panda_tasks.panda_pick_and_place import PandaPickAndPlaceEnv
    
size = 100 # Total number of sims to save into the human buffer 
render = False
use_noise = False
save_file_name = "Re-Run_Buffers\\Test1.pkl" # File to pickle the buffer into
MAX_EPISODES = 10 # Number of Episodes (how many runs per minibatch - how often the runs are saved into memory)
MAX_CYCLES = int(size/MAX_EPISODES)
save_file_name = 'Picka and Place Re-Run Buffer Test 1' 
save_file_name = 'delete me'


# Create the human agent
recording_file_name = "Oculus_Data\\test59.txt" # Oculus output file to read in
recording_file_name = "test_rerun_buffer2.pkl"
env = PandaPickAndPlaceEnv(render = render) # task enviroment
EPISODE_LENGTH = 50 # Number of sim steps per simulation
# agent = PickAndPlaceHumanAgent(env, recording_file_name, 8, EPISODE_LENGTH, start=205, z_start = 260,
#                    end=-1, size = size)
agent = PickAndPlaceReRunAgent(env, recording_file_name, 50, end_steps = 5, size = size)
action_multiplier = np.array([1, 1, 1, 0.3])


# recording_file_name = "Oculus_Data\\test69.txt" # Oculus output file to read in
# env = CustomStackEnv(render = render) # task enviroment
# EPISODE_LENGTH = 100
# agent = StackHumanAgent(env, recording_file_name, 0, EPISODE_LENGTH, start=0, switch_blocks = 340,
#                    end=-1, size = size)
# action_multiplier = np.array([1, 1, 1, 0.3])


count = 0 # Count the total number of simulations
for cycle in range(0, MAX_CYCLES):
    mb = [] # Initilize the minibatch container
    
    # iterate through episodes
    print(MAX_EPISODES*cycle, '\t', count)
    episode = 0 # Keep track of the number of sucessful sims
    while episode < MAX_EPISODES: # keep runing sims untill MAX_EPISODES sucesfull ones are generated
        # Create an empty dict to store the simulation info.
        episode_dict = {
            "state": [],
            "action": [],
            "info": [],
            "achieved_goal": [],
            "desired_goal": [],
            "next_state": [],
            "next_achieved_goal": []}
        
        # Restart the enviroment and get intial conditions
        env_dict = env.reset()
        state = env_dict["observation"] 
        achieved_goal = env_dict["achieved_goal"]
        desired_goal = env_dict["desired_goal"]

        # make sure that the achieved goal and desired goal aren't the same
        while env.compute_reward(achieved_goal, desired_goal, None) == 0:
            if render:
                time.sleep(1)
            env_dict = env.reset()
            state = env_dict["observation"]
            achieved_goal = env_dict["achieved_goal"]
            desired_goal = env_dict["desired_goal"]

        #rest the human agent
        agent.reset(env_dict['achieved_goal'], env_dict['desired_goal'][0:3])
        # agent.reset(env_dict['achieved_goal'][0:3], env_dict['achieved_goal'][3:6], env_dict['desired_goal'][0:3])
        
        # generate random sigma and theta values for the OUNoise 
        sigma = 0.3*np.random.uniform()
        theta = 0.4*np.random.uniform() + 0.1
        noise = 0 # Noise value to add to the generated actions
        
        # Initialize the OU noise by running it for 50 steps 
        for _ in range(50):
            noise += np.random.uniform(-sigma, sigma, 4)
            noise += -noise*theta
        
        # Run the simulation for EPISODE_LENGTH steps
        for t in range(EPISODE_LENGTH):
            # Update OUNoise
            noise += np.random.uniform(-sigma, sigma, 4)
            noise += -noise*theta

            if not use_noise:
                noise = 0
            
            # Get the human action and pass it into the sim
            action = agent.get_action(t)/action_multiplier + noise
            next_env_dict, reward, done, info = env.step(action*action_multiplier)
            
            # Store the state data
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
            if render: # if render, pause so that the rendering is human viewable 
                time.sleep(0.05)

        # Check to see if the task was done sucessfully. If it was, add 1 to the episode
        # counter and save the episode_dict to the minibatch list
        if reward == 0:
            episode += 1
            episode_dict["state"].append(state.copy())
            episode_dict["achieved_goal"].append(achieved_goal.copy())
            episode_dict["desired_goal"].append(desired_goal.copy())
            episode_dict["next_state"] = episode_dict["state"][1:]
            episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]
            mb.append(dc(episode_dict))
        #     print('success')
        # else:
        #     print('failure')
        count += 1
        

    # add transitions to the replay buffer (after each cycle ---> MAX_EPISODES * MAX_EP_LENGTH)
    agent.store(mb)

print(MAX_EPISODES*MAX_CYCLES, '\t', count)
agent.memory.save(save_file_name) # pickle the memory

env.close() # close the sim
