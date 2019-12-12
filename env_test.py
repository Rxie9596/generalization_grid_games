import gym
import generalization_grid_games
import matplotlib.pyplot as plt
import time
import numpy as np


env_name = "{}{}-v0".format("MazeNavigation", 2)
# env_name = "{}{}-v0".format("TwoPileNim", 0)
env = gym.make(env_name)

obs = env.reset()

imgplot = plt.imshow(env.render())
plt.show()
time.sleep(1)
# actions = np.array([[13, 10],
#                     [13, 10],
#                     [13, 10],
#                     [13, 8],
#                     [13, 8]])
actions = np.array([[11, 10],
                    [11, 10],
                    [11, 10],
                    [11, 10],
                    [11, 10],
                    [11, 8],
                    [11, 8],
                    [11, 8],
                    [11, 8],
                    [11, 10],
                    [11, 10],
                    [11, 10],])
for i in range(50):
    # action = env.action_space.sample()
    action = actions[i]
    obs, reward, done, debug_info = env.step(action)

    imgplot = plt.imshow(env.render())
    plt.show()

    if done:
        break