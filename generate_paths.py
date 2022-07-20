
import numpy as np
import time

### --- Create Human Buffer --- ###
from enviroment import CoustomEnv
from copy import deepcopy as dc
from human_agent import StackHumanAgent
from panda_gym.envs.panda_tasks.panda_stack import PandaStackEnv
create_human_buffer = False
if not create_human_buffer:
    exit(0)
    
size = 10
render = True
recording_file_name = "Oculus_Data\\test69.txt"
save_file_name = "Human_Buffer_Stack_7-18.pkl"
env = PandaStackEnv(render = render)
MAX_EPISODES = 10
MAX_CYCLES = int(size/MAX_EPISODES)


agent = StackHumanAgent(env, recording_file_name, 0, 100, start=0, switch_blocks = 340,
                   end=-1, size = size, offset=[0, 0, -0.02, 0])

print('made agent')
count = 0
for cycle in range(0, MAX_CYCLES):
    mb = []
    
    # iterate through episodes
    print(MAX_EPISODES*cycle, '\t', count)
    episode = 0
    while episode < MAX_EPISODES:
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
        while np.linalg.norm(achieved_goal - desired_goal) <= 0.05:
            env_dict = env.reset()
            state = env_dict["observation"]
            achieved_goal = env_dict["achieved_goal"]
            desired_goal = env_dict["desired_goal"]

        # maximum episode length of 50 ---- MAYBE TURN THIS INTO ANOTHER HYPERPARAMETER?????
        obs = env.reset()
        agent.reset(obs['achieved_goal'], obs['desired_goal'])
        
        sigma = 0.3*np.random.uniform()
        theta = 0.4*np.random.uniform() + 0.1
        noise = 0
        for _ in range(50):
            noise += np.random.uniform(-sigma, sigma, 4)
            noise += -noise*theta
        
        for t in range(50):
            noise += np.random.uniform(-sigma, sigma, 4)
            noise += -noise*theta
            
            action = agent.get_action(t)*np.array([1, 1, 1, 5]) + noise
            next_env_dict, reward, done, info = env.step(action*np.array([1, 1, 1, 0.2]))

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
            if render:
                time.sleep(0.01)

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
    

agent.memory.save(save_file_name)

env.close()
