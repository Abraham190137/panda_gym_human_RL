
### --- Human Agent --- ###
import numpy as np
from memory import Memory
import pickle
from scipy import interpolate

# Human Agent for the pick and place task
class PickAndPlaceReRunAgent:

    def __init__(self, env, filename, sim_steps, end_steps = 5, size = 1000):
        """
        The Pick and Place Human Agent is used to gerate action from a recorded human demonstation. 
        :param env:         training enviroment
        :param file_path:   file path of the pkl file from the sucessful play
        :param size:        Size of the memory buffer. How many total runs to be stored
        """
        
        self.memory = Memory(capacity = size, k_future = 0, env = env)
        self.env = env
        
        # Read in the data from the oculus and organize into a list of lists.
        # Each element of the list corresponds to a single oculus time step.
        with open(filename, 'rb') as file:
            raw_data = pickle.load(file)

        end_index = len(raw_data['reward']) - end_steps
        for i in range(len(raw_data['reward'])):
            if raw_data['reward'][i] == 0 and end_index == len(raw_data['reward']) - end_steps:
                end_index = i

        if end_index >= len(raw_data['reward']) - end_steps:
            raise Exception('Task not solved')

        end_index += end_steps
        x = np.arange(0.5, end_index, 1)
        xnew = np.arange(0, end_index, end_index/sim_steps)
        data = {}
        
        for key in raw_data:
            if raw_data[key].ndim > 1:
                entry = np.zeros((sim_steps, np.shape(raw_data[key])[1]))
                for i in range(np.shape(raw_data[key])[1]):
                    tck = interpolate.splrep(x, raw_data[key][:end_index, i], s=0)
                    entry[:, i] = interpolate.splev(xnew, tck, der=0)
            else:
                tck = interpolate.splrep(x, raw_data[key][:end_index], s=0)
                entry = interpolate.splev(xnew, tck, der=0)
            data[key] = entry
        
        self.data = data
    def reset(self, object_location, goal_location):
        """
        Resets the ajsutments for the start and goal locations.
        Needs to be run every time the enviroment is reset.
        :param object_location: The start location of the object (the block)
        :param goal_location:   The location of the goal                       
        """
        # The locations of the object and goal in the oculus sim.
        rec_obj_location = self.data['object_pos'][0]
        rec_goal_location = self.data['object_pos'][-1]
        
        # Find the adjustments for the simulation instance
        self.adjust_mult = (object_location - goal_location)/(rec_obj_location - rec_goal_location)
        self.adjust_bias = object_location - self.adjust_mult*rec_obj_location
          
    def get_action(self, step):
        """
        Returns an action from the human agent given a simulation step 
        :param step:  Pybullet simulation step
        :return:      Panda EE Action (4x1 np array)
        """
        goal_location = self.data['ee_pos'][step]
        # Adjust the goal locations to scale and shift the EE motions to match
        # the sim's oject start and goal locations. 
        # If the time step is before z_start, only adjust the x and y corrdinates. 
        goal_location = goal_location*self.adjust_mult + self.adjust_bias
                
        # Combine the goal location and goal gripper location to get the goal_postion
        goal_position = np.zeros(4)
        goal_position[0:3] = goal_location
        goal_position[3] = self.data['gripper'][step]
        # print(self.data[step][2])
        
        # Use a proportional controller to generate the action.
        position = np.zeros(4)
        position[0:3] = self.env.robot.get_ee_position()
        position[3] = self.env.robot.get_fingers_width()
        error = goal_position - position
        return 15*np.array([1, 1, 1, 0.2])*error
    
    def store(self, mini_batch):
        """
        Store the mini batch to the replay buffer (memory).
        :param mini_batch:       mini batch of transitions
        """

        for batch in mini_batch:
            self.memory.add(batch)

    def interpolate(self, x, y, xnew):
        tck = interpolate.splrep(x, y, s=0)
        ynew = interpolate.splev(xnew, tck, der=0)
            

# class StackHumanAgent:

#     def __init__(self, env, filename, start_steps, total_steps, start=0, switch_blocks=0, end=-1, size = 1000, offset = [0, 0, 0, 0]):
#         """
#         The Pick and Place Human Agent is used to gerate action from a recorded human demonstation. 
#         :param env:         training enviroment
#         :param file_path:   file path of the txt file the oculus recording data is stored in
#         :param start_steps: The number of steps the human agent will spend at the begining of the 
#                             simulation trying to reach its initital postiion (first recorded point)
#         :param total_steps: total number of timesteps per episode
#         :param start:       The index of the oculus recorded data to start at
#         :param switch_blocks:The index at which the EE switches from moving one block to the other.
#         :param end:         The index of the oculus recording to stop at
#         :param size:        Size of the memory buffer. How many total runs to be stored
#         :param offset:      Offset to add to way points from the oculus 
#         """
#         self.switch_blocks = switch_blocks - start
#         self.memory = Memory(capacity = size, k_future = 0, env = env)
#         self.env = env
#         self.start_steps = start_steps
#         self.total_steps = total_steps
#         self.offset = np.array(offset)
#         data = []
        
#         # Read in the data from the oculus and organize into a list of lists.
#         # Each element of the list corresponds to a single oculus time step.
#         with open(filename) as f:
#             lines = f.readlines()

#         for line in lines:
#             index, position, finger = line[:-1].split('\t')
#             position = np.array(position[1:-1].split(', ')).astype(float)
#             dataline = [float(index), position, float(finger)]
#             data.append(dataline)
            
#         # Clip the data to the start and end indices
#         self.data = data[start:end]
        
#     def reset(self, object1_location, object2_location, goal_location):
#         """
#         Resets the ajsutments for the start and goal locations.
#         Needs to be run every time the enviroment is reset.
#         :param object1_location: The start location of the first block
#         :param object2_location: The start location of the second block
#         :param goal_location:   The location of the goal                       
#         """
#         # The locations of the object and goal in the oculus sim.
#         rec_obj2_location = np.array([0.1, 0.1])
#         rec_obj1_location = np.array([0.1, -0.1])
#         rec_goal_location = np.array([-0.1, 0])
        
#         # Find the path adjustments given the goal location and the object locations. 
#         # This is only an xy adjustment, and two different adjustments are created, one 
#         # for each block
#         adjust_mult1 = (object1_location[0:2] - goal_location[0:2])/(rec_obj1_location - rec_goal_location)
#         adjust_bias1 = object1_location[0:2] - adjust_mult1*rec_obj1_location[0:2]
        
#         adjust_mult2 = (object2_location[0:2] - goal_location[0:2])/(rec_obj2_location - rec_goal_location)
#         adjust_bias2 = object2_location[0:2] - adjust_mult2*rec_obj2_location[0:2]
        
#         # Save the adjustment values
#         self.adjust_mult1 = adjust_mult1
#         self.adjust_bias1 = adjust_bias1
#         self.adjust_mult2 = adjust_mult2
#         self.adjust_bias2 = adjust_bias2
        
    
#     def get_action(self, step):
#         """
#         Returns an action from the human agent given a simulation step 
#         :param step:  Pybullet simulation step
#         :return:      Panda EE Action (4x1 np array)
#         """
#         # Find i, the index of the oculus recording that corresponds 
#         # to the given simulation step
        
#         # Durring the first start_steps, have the arm just travel to the 
#         # starting location. Then, evenly distribute the remaining sim
#         # steps amoung the oculus recording steps.
#         if step < self.start_steps:
#             i = 0
#         else:
#             i = round((step-self.start_steps)*len(self.data)/(self.total_steps - self.start_steps))
        
#         # Adjust the recorded goal location to align the coordinate frames.
#         goal_location = self.data[i][1]
#         goal_location = np.array([goal_location[2], -goal_location[0], goal_location[1]])
        
#         # Adjust the goal locations to scale and shift the EE motions to match
#         # the sim's oject start and goal locations. 
#         # If the time step is before switch_blocks, use the first block's adjustments
#         # and if its after switch_blocks, use the second block's adjustments
#         if i < self.switch_blocks:
#             goal_location[0:2] = goal_location[0:2]*self.adjust_mult1 + self.adjust_bias1
#         else:
#             goal_location[0:2] = goal_location[0:2]*self.adjust_mult2+ self.adjust_bias2
                
#         # Combine the goal location and goal gripper location to get the goal_postion
#         goal_position = np.zeros(4)
#         goal_position[0:3] = goal_location
#         goal_position[3] = 2*self.data[i][2]
                
#         # Use a proportional controller to generate the action.
#         position = np.zeros(4)
#         position[0:3] = self.env.robot.get_ee_position()
#         position[3] = self.env.robot.get_fingers_width()
#         error = goal_position + self.offset - position
#         return 10*np.array([1, 1, 1, 0.07])*error
    
#     def store(self, mini_batch):
#         """
#         Store the mini batch to the replay buffer.
#         :param mini_batch:       mini batch of transitions
#         """

#         for batch in mini_batch:
#             self.memory.add(batch)
