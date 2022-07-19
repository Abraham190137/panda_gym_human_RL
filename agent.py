
### --- agent.py --- ###
from human_buffer import HumanBuffer
from models import Actor, Critic
from memory import Memory
from normalizer import Normalizer
import time
import torch
import pickle
from torch import from_numpy, device
import numpy as np
#from models import Actor, Critic
#from memory import Memory
from torch.optim import Adam
from mpi4py import MPI
#from normalizer import Normalizer
from copy import deepcopy as dc

# create the DDPG + HER agent
class Agent:
    def __init__(self, n_states, n_actions, n_goals, action_bounds, capacity, env,
                 k_future,
                 batch_size,
                 episode_length,
                 path,
                 human_file_path = None,
                 action_size=1,
                 tau=0.05,
                 actor_lr=1e-3,
                 critic_lr=1e-3,
                 gamma=0.98,
                 action_penalty = 1): # NOTE: originally had hvb_capacity=15000
        """
        DDPG:   An off-policy method that learns a Q-function and a policy to iterate over actions.
                It consist of two models: the actor and critic. The actor is a policy network that
                takes the state as input and outputs the exact action (continuous). The critic is
                a Q-value network that takes in state and action as input and outputs the Q-value.
        HER:    A buffer of past experiences is used to stabilize training by decorrelating the
                training examples ni each batch used to update the neural network.
        """

        # NOTE: can try to get this running on Desktop's GPU (I failed)
        self.device = device("cpu")
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_goals = n_goals
        self.k_future = k_future
        self.action_bounds = action_bounds
        self.action_size = action_size
        self.env = env

        self.actor = Actor(self.n_states, n_actions=self.n_actions, n_goals=self.n_goals).to(self.device)
        self.critic = Critic(self.n_states, action_size=self.action_size, n_goals=self.n_goals).to(self.device)
        self.sync_networks(self.actor)
        self.sync_networks(self.critic)
        self.actor_target = Actor(self.n_states, n_actions=self.n_actions, n_goals=self.n_goals).to(self.device)
        self.critic_target = Critic(self.n_states, action_size=self.action_size, n_goals=self.n_goals).to(self.device)
        self.init_target_networks()
        self.tau = tau
        self.gamma = gamma

        if human_file_path is not None:
            self.human_buffer = HumanBuffer(env, human_file_path, episode_length = episode_length)
        else:
            self.human_buffer = None

        self.capacity = capacity
        self.path = path
        self.memory = Memory(self.capacity, self.k_future, self.env)

        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_optim = Adam(self.actor.parameters(), self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), self.critic_lr)

        self.state_normalizer = Normalizer(self.n_states[0], default_clip_range=5)
        self.goal_normalizer = Normalizer(self.n_goals, default_clip_range=5)
        self.action_penalty = action_penalty

    def choose_action(self, state, goal):
        """
        Agent chooses the action based on the actor network (with some injected randomness)
        :param state:       the state of the environment
        :param goal:        the goal
        ##:param train_mode:  True/False defining if the agent is training or not
        :return:            action
        """

        state = self.state_normalizer.normalize(state)
        goal = self.goal_normalizer.normalize(goal)
        state = np.expand_dims(state, axis=0)
        goal = np.expand_dims(goal, axis=0)

        with torch.no_grad():
            x = np.concatenate([state, goal], axis=1)
            x = from_numpy(x).float().to(self.device)
            action = self.actor(x)[0].cpu().data.numpy()

        return action

    def store(self, mini_batch):
        """
        Store the mini batch to the replay buffer (HER memory).
        :param mini_batch:       mini batch of transitions (all the transitions over the 50 episodes)
        """

        for batch in mini_batch:
            self.memory.add(batch)
        self._update_normalizer(mini_batch)

    def init_target_networks(self):
        """
        Intitialize the actor and critic networks.
        """

        self.hard_update_networks(self.actor, self.actor_target)
        self.hard_update_networks(self.critic, self.critic_target)

    @staticmethod
    def hard_update_networks(local_model, target_model):
        target_model.load_state_dict(local_model.state_dict())

    @staticmethod
    def soft_update_networks(local_model, target_model, tau=0.05):
        for t_params, e_params in zip(target_model.parameters(), local_model.parameters()):
            t_params.data.copy_(tau * e_params.data + (1 - tau) * t_params.data)

    def train(self, human_portion = None):
        """
        Training loop for the DDPG + HER + HVB (optional) agent. 
        :return:    actor loss, critic loss
        """

        # Check if using the human buffer or not:
        if self.human_buffer is not None and human_portion is not None:
            if human_portion != 1:
                states, actions, rewards, next_states, goals = self.memory.sample(int(self.batch_size * (1 - human_portion)))

                # sample from the human buffer
                s, a, r, ns, g = self.human_buffer.sample(int(self.batch_size * human_portion))

                # combine transitions sampled from the different buffers
                states = np.vstack((states, s))
                actions = np.vstack((actions, a))
                rewards = np.vstack((rewards, r)) 
                next_states = np.vstack((next_states, ns)) 
                goals = np.vstack((goals, g))
                
            else:
                states, actions, rewards, next_states, goals = self.human_buffer.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, goals = self.memory.sample(self.batch_size)

        states = self.state_normalizer.normalize(states)
        next_states = self.state_normalizer.normalize(next_states)
        goals = self.goal_normalizer.normalize(goals)
        inputs = np.concatenate([states, goals], axis=1)
        next_inputs = np.concatenate([next_states, goals], axis=1)

        inputs = torch.Tensor(inputs).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        next_inputs = torch.Tensor(next_inputs).to(self.device)
        actions = torch.Tensor(actions).to(self.device)

        with torch.no_grad():
            target_q = self.critic_target(next_inputs, self.actor_target(next_inputs))
            target_returns = rewards + self.gamma * target_q.detach()
            target_returns = torch.clamp(target_returns, -1 / (1 - self.gamma), 0)

        q_eval = self.critic(inputs, actions)
        critic_loss = (target_returns - q_eval).pow(2).mean()

        a = self.actor(inputs)
        actor_loss = -self.critic(inputs, a).mean()
        # Add a penalty for movement to encourage smooth operation
        actor_loss += self.action_penalty*a.pow(2).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.sync_grads(self.actor)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.sync_grads(self.critic)
        self.critic_optim.step()

        return actor_loss.item(), critic_loss.item()

    def save_weights(self, file):
        """
        Save the network weights with a specific filename.
        :param file:    the filename for that specific training run
        """

        torch.save({"actor_state_dict": self.actor.state_dict(),
                    "state_normalizer_mean": self.state_normalizer.mean,
                    "state_normalizer_std": self.state_normalizer.std,
                    "goal_normalizer_mean": self.goal_normalizer.mean,
                    "goal_normalizer_std": self.goal_normalizer.std}, file)
                    # "goal_normalizer_std": self.goal_normalizer.std}, "FetchPickAndPlace.pth")

    def save_buffers(self, hv_file, er_file):
        """
        Save ER buffers with specific filenames.
        :param er_file:     the ER filename
        """
        self.memory.save(er_file)

    def load_weights(self, weight_file):
        """
        Load the network weights from a specific training run given the filename of the weights.
        :param weight_file:    the filename for these weights
        """

        # checkpoint = torch.load("FetchPickAndPlace.pth")
        checkpoint = torch.load(weight_file)
        actor_state_dict = checkpoint["actor_state_dict"]
        self.actor.load_state_dict(actor_state_dict)
        state_normalizer_mean = checkpoint["state_normalizer_mean"]
        self.state_normalizer.mean = state_normalizer_mean
        state_normalizer_std = checkpoint["state_normalizer_std"]
        self.state_normalizer.std = state_normalizer_std
        goal_normalizer_mean = checkpoint["goal_normalizer_mean"]
        self.goal_normalizer.mean = goal_normalizer_mean
        goal_normalizer_std = checkpoint["goal_normalizer_std"]
        self.goal_normalizer.std = goal_normalizer_std

    def set_to_eval_mode(self):
        self.actor.eval()
        # self.critic.eval()

    def update_networks(self):
        self.soft_update_networks(self.actor, self.actor_target, self.tau)
        self.soft_update_networks(self.critic, self.critic_target, self.tau)

    def _update_normalizer(self, mini_batch):
        """
        Update the normalizer given the mini batch of transitions
        :param mini_batch:       mini batch of transitions (all the transitions over the 50 episodes)
        """

        states, goals = self.memory.sample_for_normalization(mini_batch)

        self.state_normalizer.update(states)
        self.goal_normalizer.update(goals)
        self.state_normalizer.recompute_stats()
        self.goal_normalizer.recompute_stats()

    @staticmethod
    def sync_networks(network):
        comm = MPI.COMM_WORLD
        flat_params = _get_flat_params_or_grads(network, mode='params')
        comm.Bcast(flat_params, root=0)
        _set_flat_params_or_grads(network, flat_params, mode='params')

    @staticmethod
    def sync_grads(network):
        flat_grads = _get_flat_params_or_grads(network, mode='grads')
        comm = MPI.COMM_WORLD
        global_grads = np.zeros_like(flat_grads)
        comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
        _set_flat_params_or_grads(network, global_grads, mode='grads')


def _get_flat_params_or_grads(network, mode='params'):
    attr = 'data' if mode == 'params' else 'grad'
    return np.concatenate([getattr(param, attr).cpu().numpy().flatten() for param in network.parameters()])


def _set_flat_params_or_grads(network, flat_params, mode='params'):
    attr = 'data' if mode == 'params' else 'grad'
    pointer = 0
    for param in network.parameters():
        getattr(param, attr).copy_(
            torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
        pointer += param.data.numel()
