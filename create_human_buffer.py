
### --- Create Human Buffer --- ###
import numpy as np
import time

from enviroment import CoustomEnv
from copy import deepcopy as dc
from human_agent import StackHumanAgent
from panda_gym.envs.panda_tasks.panda_stack import PandaStackEnv
create_human_buffer = False
if not create_human_buffer:
    exit(0)
    
size = 1000 # Total number of sims to save into the human buffer 
render = False 
recording_file_name = "Oculus_Data\\test69.txt" # Oculus output file to read in
save_file_name = "Human_Buffer_Stack_ns2_s1000_7-18.pkl" # File to pickle the buffer into
env = PandaStackEnv(render = render) # task enviroment
MAX_EPISODES = 100 # Number of Episodes (how many runs per minibatch - how often the runs are saved into memory)
MAX_CYCLES = int(size/MAX_EPISODES)
EPISODE_LENGTH = 100 # Number of sim steps per simulation

# Create the human agent
agent = StackHumanAgent(env, recording_file_name, 0, EPISODE_LENGTH, start=0, switch_blocks = 340,
                   end=-1, size = size, offset=[0, 0, -0.02, 0])

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
        while env.compute_reward(achieved_goal, desired_goal, None) != 0:
            env_dict = env.reset()
            state = env_dict["observation"]
            achieved_goal = env_dict["achieved_goal"]
            desired_goal = env_dict["desired_goal"]

        #rest the envirmonet and the human agent
        obs = env.reset()
        agent.reset(obs['achieved_goal'][0:3], obs['achieved_goal'][3:6], obs['desired_goal'][0:3])
        
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
            
            # Get the human action and pass it into the sim
            action = agent.get_action(t)*np.array([1, 1, 1, 5]) + noise
            next_env_dict, reward, done, info = env.step(action*np.array([1, 1, 1, 0.2]))
            
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
                time.sleep(0.01)

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
        count += 1
        

    # add transitions to the replay buffer (after each cycle ---> MAX_EPISODES * MAX_EP_LENGTH)
    agent.store(mb)

agent.memory.save(save_file_name) # pickle the memory

env.close() # close the sim
