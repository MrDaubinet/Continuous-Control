from unityagents import UnityEnvironment
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from info import Info

from agent import Agent
from baseline import Baseline
from ddpg import DDPG

#  Replace with your location of the reacher
env = UnityEnvironment(file_name='C:/Udacity/Deep Reinforcement Learning/deep-reinforcement-learning/p2_continuous-control/Reacher_Windows_x86_64/Reacher.exe')

# Info
info = Info(env) # create the info object
info.print_info() # print out information

# set action and state
action_size, state_size = info.getInfo()

# baseline = Baseline(env, action_size, state_size)
# baseline.run()
# Create the agent
agent = Agent(state_size=state_size, action_size=action_size, random_seed=6)

random.seed(6)
torch.manual_seed(6)

# train agent
# scores, avgs = DDPG.train(env, agent)
# info.plotResults(scores, avgs) # plot the scores

# test best agent
DDPG.test(agent, env)

