# Panda-Gym
Customization of the original [panda-gym environment](github.com/qgallouedec/panda-gym). To run this codebase, follow the instructions to install standard panda-gym, and then clone this code. Additionally, the human buffer will need to be generated (ADD THIS CODE), or one of the buffers from the repository will need to be used.


## DDPG + HER + Human Example
Implementation of the Deep Deterministic Policy Gradient with Hindsight Experience Replay Extension on the panda-gym environments. In addition, we modify this standard DDPG + HER method with a secondary replay buffer called the Human Buffer. This secondary buffer is composed of recorded motions from a human completing the task in VR. Durring training, a set portion of examples that would have been pulled from the replay buffer are instead pulled from the human buffer.


## Usage
To train on multiple CPU cores, need to use one of the following commands.

For Windows:
```shell
mpiexec -np $(nproc) python3 -u main.py
```

For Linux:
```shell
mpirun -np $(nproc) python3 -u main.py
```

```shell
mpirun -np $(nproc) --use-hwthread-cpus python3 -u main.py
```

## Method of Generating Human Buffer
EXPLAIN THIS


## Guiding Questions:
INSERT

## Results
INSERT
<!-- <p align="center">
  <img src="Result/Fetch_PickandPlace.png" height=400>
</p> -->

## References
EDIT THESE
1. [_Continuous control with deep reinforcement learning_, Lillicrap et al., 2015](https://arxiv.org/abs/1509.02971)  
2. [_Hindsight Experience Replay_, Andrychowicz et al., 2017](https://arxiv.org/abs/1707.01495)  
3. [_Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research_, Plappert et al., 2018](https://arxiv.org/abs/1802.09464)
4. [_Prioritized Experience Replay_, Schaul et al., 2016](https://arxiv.org/pdf/1511.05952.pdf)

## Acknowledgement
The architecture of the DDPG + HER implementation is taken from [@AlisonBartsch](https://github.com/alison-bartsch)'s [modified implementation](https://github.com/alison-bartsch/panda-gym) of [@TianhongDai](https://github.com/TianhongDai)'s [simplified implementation](https://github.com/TianhongDai/hindsight-experience-replay) of [the original OpenAI's code](https://github.com/openai/baselines/tree/master/baselines/her).  

