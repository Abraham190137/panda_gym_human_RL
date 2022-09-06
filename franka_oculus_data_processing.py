import numpy as np
import pickle

oculus_file = "Oculus_Data\\test57.txt"
output_file = "franka_full_oculus_data_test.pkl"
start = 160
end = -1
preview = True
# Read in the data from the oculus and organize into a list of lists.
# Each element of the list corresponds to a single oculus time step.
data = []
with open(oculus_file) as f:
    lines = f.readlines()

for line in lines:
    index, position, finger = line[:-1].split('\t')
    position = np.array(position[1:-1].split(', ')).astype(np.float)
    dataline = [float(index), position, float(finger)]
    data.append(dataline)

data = data[start:end]

output = np.zeros((len(data), 4))

for i, entry in enumerate(output):
    entry[0:3] = np.array([data[i][1][2], -data[i][1][0], data[i][1][1]])
    entry[0:3] += np.array([0.6, 0, 0])
    entry[3] = 2*data[i][2]

print('The trajectory contains', np.shape(output)[0], 'entries')

if preview:
    from panda_gym.envs.panda_tasks.panda_pick_and_place import PandaPickAndPlaceEnv
    import time
    env = PandaPickAndPlaceEnv(render = True)
    env.reset()
    env.sim.set_base_pose("target", [0, 0, -0.2], np.array([0.0, 0.0, 0.0, 1.0]))
    env.sim.set_base_pose("object", [0, 0, -0.2], np.array([0.0, 0.0, 0.0, 1.0]))
    for entry in output:
        position = np.zeros(4)
        position[0:3] = env.robot.get_ee_position() + np.array([0.6, 0, 0])
        position[3] = env.robot.get_fingers_width()
        env.step(10*np.array([1, 1, 1, 0.07])*(entry - position ))
        time.sleep(0.01)

env.close()

proceed = input('Proceed? y/n  ')

if proceed == 'y' or proceed == 'Y':
    print('file saved')
    with open(output_file, 'wb') as file:
        pickle.dump(output, file)
else:
    print('not saved')
