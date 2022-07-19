
### --- Noise Generator --- ###
import numpy as np
import random
from copy import deepcopy as dc

class UniformNoise:
    def __init__(self, action_space, amplitude = 0.2, pure_rand = 0.3):
        """
        Create a uniform noise generator for to add to the robot actions
        :param action_space:    action space of the envirmont                      
        :param amplitude:       noise amplitude
        :param pure_rand:       chance the action will be replaced with 
                                pure random noise
        """
        self.amp = amplitude
        self.pure_rand = pure_rand
        self.action_bounds = [action_space.low[0], action_space.high[0]]
        self.n_actions = action_space.shape[0]
    
    def reset(self):
        """
        The reset method is used in other forms of noise, so this is just
        a space holder so that the main code doesn't have to be adjusted.
        """
        pass
    
    def get_action(self, action, t=0):
        """
        Get a noise value
        :param action:  noiseless action for noise to be added to.
        :param t:       Training timestep. Not used in Uniform noise
        :return:        noisy action
        """
        # Add uniform noise to the action and clip it to the action bounds.
        action += self.amp * np.random.randn(self.n_actions)
        action = np.clip(action, self.action_bounds[0], self.action_bounds[1])
        
        # Every self.pure_rand sample, return a purely random action
        random_actions = np.random.uniform(low=self.action_bounds[0], high=self.action_bounds[1],
                                           size=self.n_actions)  
        action += np.random.binomial(1, self.pure_rand, 1)[0] * (random_actions - action)
        return action


class UniformNoiseDecay:
    def __init__(self, action_space, amp_decay_period, max_amp, min_amp, rand_decay_period, max_rand, min_rand):
        """
        Create a uniform noise generator with exponential decay
        :param action_space:      action space of the envirmont                      
        :param amp_decay_period:  expontial decay period for the noise amplitude     
        :param max_amp:           max noise amplitude
        :param min_amp:           min noise amplitude
        :param rand_decay_period: expontial decay period for the random action chance     
        :param max_rand:          max random action chance 
        :param min_rand:          min random action chance
        """
        self.action_bounds = [action_space.low[0], action_space.high[0]]
        self.n_actions = action_space.shape[0]
        self.amp_decay_period = amp_decay_period
        self.max_amp = max_amp
        self.min_amp = min_amp
        self.rand_decay_period = rand_decay_period
        self.max_rand = max_rand
        self.min_rand = min_rand
    
    def reset(self):
        """
        The reset method is used in other forms of noise, so this is just
        a space holder so that the main code doesn't have to be adjusted.
        """
        pass
    
    def get_action(self, action, t=0):
        """
        Get a noise value
        :param action:  noiseless action for noise to be added to.
        :param t:       Training timestep. Used for the exponential decay
        :return:        noisy action
        """
        # calcuate the noise amplitdue and random chance using exponential decay.
        amp = self.min_amp + (self.max_amp-self.min_amp)*np.exp(-t/self.amp_decay_period)
        rand_chance = self.min_rand + (self.max_rand-self.min_rand)*np.exp(-t/self.rand_decay_period)
        
        # Add uniform noise to the action and clip it to the action bounds.
        action += amp*np.random.randn(self.n_actions)
        action = np.clip(action, self.action_bounds[0], self.action_bounds[1])

        # Every self.pure_rand sample, return a purely random action
        random_actions = np.random.uniform(low=self.action_bounds[0], high=self.action_bounds[1],
                                           size=self.n_actions)
        action += np.random.binomial(1, rand_chance, 1)[0] * (random_actions - action)

        return action


class OUNoise:
    # For more information on OU Noise, see https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, 
                 decay_period=100000):
        """
        Create a Ornstein Uhlenbeck (OU) noise generator with exponential decay
        :param action_space:action space of the envirmont    
        :param mu:          average noise  
        :param theta:       "spring contsant" for the OU noise
        :param max_sima:    max sigma value for the OU noise. Determines how
                                much the noise changes each time step
        :param max_sima:    min sigma value for the OU noise. Determines how
                                much the noise changes each time step     
        :param decay_period:expential decay period for sigma
        """
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def evolve_state(self):
        """
        Update the OU Noise
        :param return:  new noise value   
        """
        x  = self.state
        # Use a proportional controller to pull the noise back towards mean,
        # then add a random amount of noise.
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def reset(self):
        """
        ResEts the noise value
        """
        self.state = np.ones(self.action_dim) * self.mu
        
        # Run evolve_state 50 times to initalize the noise
        for _ in range(50):
            self.evolve_state()
    
    def get_action(self, action, t=0):
        """
        Get a noise value
        :param action:  noiseless action for noise to be added to.
        :param t:       Training timestep. Used for the exponential decay
        :return:        noisy action
        """
        ou_state = self.evolve_state()
        self.sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * np.exp(-t/self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)
