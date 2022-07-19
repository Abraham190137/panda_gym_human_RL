
### --- memory.py --- ###

import numpy as np
from copy import deepcopy as dc
import random
import pickle

# create the ER + HER 
class Memory:
    def __init__(self, capacity, k_future, env):
        self.capacity = capacity
        self.memory = []
        self.memory_counter = 0
        self.memory_length = 0
        self.env = env

        self.future_p = 1 - (1. / (1 + k_future))   # 80 percent (that means 0.8*(1 - .125) + 0.125 is HER)

    def sample(self, batch_size):
        """
        Sample transitions from the replay buffer (memory) to be used for training. The percentage
        of samples from the ER vs HER are determined entirely by self.future_p, which is equivalent 
        to this percentage. For example, with self.future_p = 0.8, then 80% of all transitions sampled
        from the replay buffer will be from HER (the desired goal modified to be the future achieved
        goal), whereas the remaining 20% of all transitions are sampled from the ER (the desired goal
        is whatever was desired while the agent performed that action). The ER buffer is in the format 
        of a list where each element is a dictionary corresponding to a list of the 50 timesteps of a 
        single episode. E.g. self.memory[0]["state"][0] is the state of the agent at a single instance.
        :param batch_size:  the batch sizes
        :return:            states, actions, rewards, next_states, desired_goals                
        """

        # get random episode and time indices from the memory (total number = batch_size)
        ep_indices = np.random.choice(len(self.memory), batch_size)
        time_indices = np.random.choice(len(self.memory[0]["next_state"]), batch_size)
        states = []
        actions = []
        desired_goals = []
        next_states = []
        next_achieved_goals = []

        # iterate through episodes and timesteps to collect states, actions, desired_goals, next_achieved goal, next_states
        for episode, timestep in zip(ep_indices, time_indices):
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
        
        # get the her indices --- random uniform selection of numbers below threshold self.future_p
        her_indices = np.where(np.random.uniform(size=batch_size) < self.future_p)

        if np.size(her_indices) != 0:
            # get the future_offset which is array of random uniform values times the length of the next state values minus time_indices
            future_offset = np.random.uniform(size=batch_size) * (len(self.memory[0]["next_state"]) - time_indices)
            future_offset = future_offset.astype(int)
            future_t = (time_indices + 1 + future_offset)[her_indices]

            # iterate through the episodes and times for HER selection based on HER indices to update future_ag as achieved goal
            future_ag = []
            for episode, f_offset in zip(ep_indices[her_indices], future_t):
                future_ag.append(dc(self.memory[episode]["achieved_goal"][f_offset]))   # NOTE: f_offset is how far into the future the goal is achieved I think!
            future_ag = np.vstack(future_ag)

            desired_goals[her_indices] = future_ag
            
        rewards = np.expand_dims(self.env.compute_reward(next_achieved_goals, desired_goals, None), 1)
        return self.clip_obs(states), actions, rewards, self.clip_obs(next_states), self.clip_obs(desired_goals)

    def add(self, transition):
        """
        Adds the transition to the memory (replay buffer)
        :param transition:  transition
        """

        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
        assert len(self.memory) <= self.capacity

    def reformat(self):
        """
        Reformats the memory buffer to be the same format as the HV buffer,
        specifically a list where each element is a dictionary corresponding
        to the state, action, achieved_goal, etc. of a single step of the
        environment.
        :return:    ER_buffer (the reformatted buffer)
        """
        ER_buffer = []

        for i in range(len(self.memory)):
            for j in range(len(self.memory[i]["next_state"])):   # originally "next_state"
                er_dict = {
                            "state": self.memory[i]["state"][j],
                            "action": self.memory[i]["action"][j],
                            "achieved_goal": self.memory[i]["achieved_goal"][j],
                            "desired_goal": self.memory[i]["desired_goal"][j],
                            "next_state": self.memory[i]["next_state"][j],
                            "next_achieved_goal": self.memory[i]["next_achieved_goal"][j]
                        }
                ER_buffer.append(er_dict)
        return ER_buffer

    def return_states(self):
        """
        Given the ER buffer in the original format, return a list of the lists of states
        in each episode.
        """

        states = []
        for episode in range(len(self.memory)):
            # if len(self.memory[episode]["state"] > len(self.memory[episode]["next_state"])):
            # ep_states = self.memory[episode]["state"]
            ep_states = self.memory[episode]["state"][0:len(self.memory[episode]["next_state"])]
            states.append(ep_states)

        return states

    def save(self, file):
        """
        Save the ER buffer to the filepath provided
        :param file:    File path and name to save the buffer
        """

        with open(file, 'wb') as f:
            pickle.dump(self.memory, f)

    def get_length(self):
        print("\nOther Length: ", len(self.memory[0]["next_state"]))
        print("Current Length: ", len(self.memory[0]["state"]))
        return len(self.memory)*len(self.memory[0]["next_state"])

    def load(self, file):
        """
        Load the saved HVB from the filepath provided
        :param file:    File path and name of file to load
        """

        with open(file, 'rb') as f:
            buffer = pickle.load(f)
            self.memory = buffer

    def __len__(self):
        return len(self.memory)

    @staticmethod
    def clip_obs(x):
        return np.clip(x, -200, 200)

    def sample_for_normalization(self, batch):
        """
        Sample from the batch of transitions for normalization
        :param batch:       batch of transitions 
        :return:            states, desired_goals
        """

        size = len(batch[0]["next_state"])
        ep_indices = np.random.randint(0, len(batch), size)
        time_indices = np.random.randint(0, len(batch[0]["next_state"]), size)
        states = []
        desired_goals = []

        for episode, timestep in zip(ep_indices, time_indices):
            states.append(dc(batch[episode]["state"][timestep]))
            desired_goals.append(dc(batch[episode]["desired_goal"][timestep]))

        states = np.vstack(states)
        desired_goals = np.vstack(desired_goals)

        if self.future_p != 0:
            her_indices = np.where(np.random.uniform(size=size) < self.future_p)
            future_offset = np.random.uniform(size=size) * (len(batch[0]["next_state"]) - time_indices)
            future_offset = future_offset.astype(int)
            future_t = (time_indices + 1 + future_offset)[her_indices]

            future_ag = []
            for episode, f_offset in zip(ep_indices[her_indices], future_t):
                future_ag.append(dc(batch[episode]["achieved_goal"][f_offset]))
            future_ag = np.vstack(future_ag)

            desired_goals[her_indices] = future_ag

        return self.clip_obs(states), self.clip_obs(desired_goals)
