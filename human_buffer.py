
### --- Human Buffer --- ###
import numpy as np
from copy import deepcopy as dc
import pickle

class HumanBuffer:
    def __init__(self, env, file_path, episode_length):
        """
        The Human Buffer is used to store tranistions generated from the human example of the task
        :param env:         training enviroment
        :param file_path:         file path of the pickle file the human transitions are stored in
        :param episode_length:    total number of timesteps per episode
        """
        # load in the pickled memory
        with open(file_path, 'rb') as pickle_file:
            self.memory = pickle.load(pickle_file)
        self.episode_length = episode_length
        self.env = env
        self.length = len(self.memory)
        
    def sample(self, batch_size):
        """
        Sample transitions from the human buffer to be used for training. The human buffer is in the same
        format as the 
        of a list where each element is a dictionary corresponding to a list of the 50 timesteps of a 
        single episode. E.g. self.memory[0]["state"][0] is the state of the agent at a single instance.
        :param batch_size:  the batch sizes
        :return:            states, actions, rewards, next_states, desired_goals                
        """
        # Get a random sampling of episode indicies and corresponding
        # time indicies
        
        episode_indices = np.random.choice(self.length, batch_size)
        time_indices = np.random.choice(self.episode_length, batch_size)
        
        states = []
        actions = []
        desired_goals = []
        next_states = []
        next_achieved_goals = []
        # The actor needs to know both the observation (which included the achieved goal)
        # and the desired_goal. So rather than storing observations desired_goals, and
        # acheived_goals, states (which include all pieces of information) are stored. 

        # Loop through each episode timestep pair, recording transition info.
        for episode, timestep in zip(episode_indices, time_indices):
            states.append(dc(self.memory[episode]["state"][timestep]))
            actions.append(dc(self.memory[episode]["action"][timestep]))
            desired_goals.append(dc(self.memory[episode]["desired_goal"][timestep]))
            next_achieved_goals.append(dc(self.memory[episode]["next_achieved_goal"][timestep]))
            next_states.append(dc(self.memory[episode]["next_state"][timestep]))

        states = np.vstack(states)
        actions = np.vstack(actions)
        desired_goals = np.vstack(desired_goals)
        next_achieved_goals = np.vstack(next_achieved_goals)
        next_states = np.vstack(next_states)
        
        # Recalculate the rewards
        rewards = np.expand_dims(self.env.compute_reward(next_achieved_goals, desired_goals, None), 1)
        return states, actions, rewards, next_states, desired_goals
