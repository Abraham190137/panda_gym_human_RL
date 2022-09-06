
### --- Human Agent --- ###
import numpy as np
from memory import Memory


# Human Agent for the push task
class PushHumanAgent:

    def __init__(self, env, filename, start_steps, total_steps, start=0, end=-1, 
                 size = 1000, offset = [0, 0, 0, 0]):
        """
        The Pick and Place Human Agent is used to gerate action from a recorded human demonstation. 
        :param env:         training enviroment
        :param file_path:   file path of the txt file the oculus recording data is stored in
        :param start_steps: The number of steps the human agent will spend at the begining of the 
                            simulation trying to reach its initital postiion (first recorded point)
        :param total_steps: total number of timesteps per episode
        :param start:       The index of the oculus recorded data to start at
        :param z_start:     The index at which scaling of the z-axis (vertical motion) begins.
        :param end:         The index of the oculus recording to stop at
        :param size:        Size of the memory buffer. How many total runs to be stored
        :param offset:      Offset to add to way points from the oculus 
        """
        
        self.memory = Memory(capacity = size, k_future = 0, env = env)
        self.env = env
        self.start_steps = start_steps
        self.total_steps = total_steps
        self.offset = np.array(offset)
        data = []
        
        # Read in the data from the oculus and organize into a list of lists.
        # Each element of the list corresponds to a single oculus time step.
        with open(filename) as f:
            lines = f.readlines()

        for line in lines:
            index, position, finger = line[:-1].split('\t')
            position = np.array(position[1:-1].split(', ')).astype(np.float)
            dataline = [float(index), position, float(finger)]
            data.append(dataline)
        
        # Clip the data to the start and end indices
        self.data = data[start:end]
        
    def reset(self, env_object_location, env_goal_location):
        """
        Resets the ajsutments for the start and goal locations.
        Needs to be run every time the enviroment is reset.
        :param object_location: The start location of the object (the block)
        :param goal_location:   The location of the goal                       
        """
        #Trim object and goal location to 2d (remove z if present)
        env_object_location = env_object_location[:2]
        env_goal_location = env_goal_location[:2]

        # The locations of the object and goal in the oculus sim.
        rec_obj_location = np.array([0.1, 0.1])
        rec_goal_location = np.array([-0.1, -0.1])
        
        # Find the adjustments for the simulation instance
        self.adjust_mult = (env_object_location - env_goal_location)/(rec_obj_location - rec_goal_location)
        self.adjust_bias = env_object_location - self.adjust_mult*rec_obj_location
          
    def get_action(self, step):
        """
        Returns an action from the human agent given a simulation step 
        :param step:  Pybullet simulation step
        :return:      Panda EE Action (4x1 np array)
        """
        # Find i, the index of the oculus recording that corresponds 
        # to the given simulation step
        
        # Durring the first start_steps, have the arm just travel to the 
        # starting location. Then, evenly distribute the remaining sim
        # steps amoung the oculus recording steps.
        if step < self.start_steps:
            i = 0
        else:
            i = round((step-self.start_steps)*len(self.data)/(self.total_steps - self.start_steps))
               
        # Adjust the recorded goal location to align the coordinate frames.
        goal_location = self.data[i][1]
        goal_location = np.array([goal_location[2], -goal_location[0], goal_location[1]])
        
        # Adjust the goal locations to scale and shift the EE motions to match
        # the sim's oject start and goal locations.  
        goal_location[0:2] = goal_location[0:2]*self.adjust_mult + self.adjust_bias
                
        # Combine the goal location and goal gripper location to get the goal_postion
        goal_position = np.zeros(4)
        goal_position[0:3] = goal_location
        goal_position[3] = 2*self.data[i][2]
        
        # Use a proportional controller to generate the action.
        position = np.zeros(4)
        position[0:3] = self.env.robot.get_ee_position()
        position[3] = self.env.robot.get_fingers_width()
        error = goal_position + self.offset - position
        return 10*np.array([1, 1, 1, 0.07])*error
    
    def store(self, mini_batch):
        """
        Store the mini batch to the replay buffer (memory).
        :param mini_batch:       mini batch of transitions
        """

        for batch in mini_batch:
            self.memory.add(batch)


# Human Agent for the pick and place task
class PickAndPlaceHumanAgent:

    def __init__(self, env, filename, start_steps, total_steps, start=0, z_start=0, end=-1, 
                 size = 1000, offset = [0, 0, 0, 0]):
        """
        The Pick and Place Human Agent is used to gerate action from a recorded human demonstation. 
        :param env:         training enviroment
        :param file_path:   file path of the txt file the oculus recording data is stored in
        :param start_steps: The number of steps the human agent will spend at the begining of the 
                            simulation trying to reach its initital postiion (first recorded point)
        :param total_steps: total number of timesteps per episode
        :param start:       The index of the oculus recorded data to start at
        :param z_start:     The index at which scaling of the z-axis (vertical motion) begins.
        :param end:         The index of the oculus recording to stop at
        :param size:        Size of the memory buffer. How many total runs to be stored
        :param offset:      Offset to add to way points from the oculus 
        """
        
        self.z_start = z_start - start # Adjust z_start wrt start
        self.memory = Memory(capacity = size, k_future = 0, env = env)
        self.env = env
        self.start_steps = start_steps
        self.total_steps = total_steps
        self.offset = np.array(offset)
        data = []
        
        # Read in the data from the oculus and organize into a list of lists.
        # Each element of the list corresponds to a single oculus time step.
        with open(filename) as f:
            lines = f.readlines()

        for line in lines:
            index, position, finger = line[:-1].split('\t')
            position = np.array(position[1:-1].split(', ')).astype(np.float)
            dataline = [float(index), position, float(finger)]
            data.append(dataline)
        
        # Clip the data to the start and end indices
        self.data = data[start:end]
        
    def reset(self, object_location, goal_location):
        """
        Resets the ajsutments for the start and goal locations.
        Needs to be run every time the enviroment is reset.
        :param object_location: The start location of the object (the block)
        :param goal_location:   The location of the goal                       
        """
        # The locations of the object and goal in the oculus sim.
        rec_obj_location = np.array([0.1, 0.1, 0.02])
        rec_goal_location = np.array([-0.1, -0.1, 0.12])
        
        # Find the adjustments for the simulation instance
        self.adjust_mult = (object_location - goal_location)/(rec_obj_location - rec_goal_location)
        self.adjust_bias = object_location - self.adjust_mult*rec_obj_location
          
    def get_action(self, step):
        """
        Returns an action from the human agent given a simulation step 
        :param step:  Pybullet simulation step
        :return:      Panda EE Action (4x1 np array)
        """
        # Find i, the index of the oculus recording that corresponds 
        # to the given simulation step
        
        # Durring the first start_steps, have the arm just travel to the 
        # starting location. Then, evenly distribute the remaining sim
        # steps amoung the oculus recording steps.
        if step < self.start_steps:
            i = 0
        else:
            i = round((step-self.start_steps)*len(self.data)/(self.total_steps - self.start_steps))
               
        # Adjust the recorded goal location to align the coordinate frames.
        goal_location = self.data[i][1]
        goal_location = np.array([goal_location[2], -goal_location[0], goal_location[1]])
        
        # Adjust the goal locations to scale and shift the EE motions to match
        # the sim's oject start and goal locations. 
        # If the time step is before z_start, only adjust the x and y corrdinates. 
        if i > self.z_start:
            goal_location = goal_location*self.adjust_mult + self.adjust_bias
        else:
            goal_location[0:2] = goal_location[0:2]*self.adjust_mult[0:2] + self.adjust_bias[0:2]
                
        # Combine the goal location and goal gripper location to get the goal_postion
        goal_position = np.zeros(4)
        goal_position[0:3] = goal_location
        goal_position[3] = 2*self.data[i][2]
        
        # Use a proportional controller to generate the action.
        position = np.zeros(4)
        position[0:3] = self.env.robot.get_ee_position()
        position[3] = self.env.robot.get_fingers_width()
        error = goal_position + self.offset - position
        return 10*np.array([1, 1, 1, 0.07])*error
    
    def store(self, mini_batch):
        """
        Store the mini batch to the replay buffer (memory).
        :param mini_batch:       mini batch of transitions
        """

        for batch in mini_batch:
            self.memory.add(batch)
            

class StackHumanAgent:

    def __init__(self, env, filename, start_steps, total_steps, start=0, switch_blocks=0, end=-1, size = 1000, offset = [0, 0, 0, 0]):
        """
        The Pick and Place Human Agent is used to gerate action from a recorded human demonstation. 
        :param env:         training enviroment
        :param file_path:   file path of the txt file the oculus recording data is stored in
        :param start_steps: The number of steps the human agent will spend at the begining of the 
                            simulation trying to reach its initital postiion (first recorded point)
        :param total_steps: total number of timesteps per episode
        :param start:       The index of the oculus recorded data to start at
        :param switch_blocks:The index at which the EE switches from moving one block to the other.
        :param end:         The index of the oculus recording to stop at
        :param size:        Size of the memory buffer. How many total runs to be stored
        :param offset:      Offset to add to way points from the oculus 
        """
        self.switch_blocks = switch_blocks - start
        self.memory = Memory(capacity = size, k_future = 0, env = env)
        self.env = env
        self.start_steps = start_steps
        self.total_steps = total_steps
        self.offset = np.array(offset)
        data = []
        
        # Read in the data from the oculus and organize into a list of lists.
        # Each element of the list corresponds to a single oculus time step.
        with open(filename) as f:
            lines = f.readlines()

        for line in lines:
            index, position, finger = line[:-1].split('\t')
            position = np.array(position[1:-1].split(', ')).astype(float)
            dataline = [float(index), position, float(finger)]
            data.append(dataline)
            
        # Clip the data to the start and end indices
        self.data = data[start:end]
        
    def reset(self, object1_location, object2_location, goal_location):
        """
        Resets the ajsutments for the start and goal locations.
        Needs to be run every time the enviroment is reset.
        :param object1_location: The start location of the first block
        :param object2_location: The start location of the second block
        :param goal_location:   The location of the goal                       
        """
        # The locations of the object and goal in the oculus sim.
        rec_obj2_location = np.array([0.1, 0.1])
        rec_obj1_location = np.array([0.1, -0.1])
        rec_goal_location = np.array([-0.1, 0])
        
        # Find the path adjustments given the goal location and the object locations. 
        # This is only an xy adjustment, and two different adjustments are created, one 
        # for each block
        adjust_mult1 = (object1_location[0:2] - goal_location[0:2])/(rec_obj1_location - rec_goal_location)
        adjust_bias1 = object1_location[0:2] - adjust_mult1*rec_obj1_location[0:2]
        
        adjust_mult2 = (object2_location[0:2] - goal_location[0:2])/(rec_obj2_location - rec_goal_location)
        adjust_bias2 = object2_location[0:2] - adjust_mult2*rec_obj2_location[0:2]
        
        # Save the adjustment values
        self.adjust_mult1 = adjust_mult1
        self.adjust_bias1 = adjust_bias1
        self.adjust_mult2 = adjust_mult2
        self.adjust_bias2 = adjust_bias2
        
    
    def get_action(self, step):
        """
        Returns an action from the human agent given a simulation step 
        :param step:  Pybullet simulation step
        :return:      Panda EE Action (4x1 np array)
        """
        # Find i, the index of the oculus recording that corresponds 
        # to the given simulation step
        
        # Durring the first start_steps, have the arm just travel to the 
        # starting location. Then, evenly distribute the remaining sim
        # steps amoung the oculus recording steps.
        if step < self.start_steps:
            i = 0
        else:
            i = round((step-self.start_steps)*len(self.data)/(self.total_steps - self.start_steps))
        
        # Adjust the recorded goal location to align the coordinate frames.
        goal_location = self.data[i][1]
        goal_location = np.array([goal_location[2], -goal_location[0], goal_location[1]])
        
        # Adjust the goal locations to scale and shift the EE motions to match
        # the sim's oject start and goal locations. 
        # If the time step is before switch_blocks, use the first block's adjustments
        # and if its after switch_blocks, use the second block's adjustments
        if i < self.switch_blocks:
            goal_location[0:2] = goal_location[0:2]*self.adjust_mult1 + self.adjust_bias1
        else:
            goal_location[0:2] = goal_location[0:2]*self.adjust_mult2+ self.adjust_bias2
                
        # Combine the goal location and goal gripper location to get the goal_postion
        goal_position = np.zeros(4)
        goal_position[0:3] = goal_location
        goal_position[3] = 2*self.data[i][2]
                
        # Use a proportional controller to generate the action.
        position = np.zeros(4)
        position[0:3] = self.env.robot.get_ee_position()
        position[3] = self.env.robot.get_fingers_width()
        error = goal_position + self.offset - position
        return 10*np.array([1, 1, 1, 0.07])*error
    
    def store(self, mini_batch):
        """
        Store the mini batch to the replay buffer.
        :param mini_batch:       mini batch of transitions
        """

        for batch in mini_batch:
            self.memory.add(batch)
