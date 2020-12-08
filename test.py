import gym
import numpy as np
import matplotlib.pyplot as plt
import gym.wrappers as w


if __name__ == '__main__':

    env = w.FrameStack(w.GrayScaleObservation((gym.make('MsPacman-v0'))), 4) #this is important
    observation = env.reset()
    array = np.array(observation) 
    imgplot = plt.imshow(array[0])
    plt.show()
