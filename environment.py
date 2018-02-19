import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from scipy.stats import multivariate_normal

class MountainCar(object):
    """The environment in the reinforcement learning framework.
    """

    def __init__(self, t_step=0.3):
      self.t_step = t_step
      self.target = [0.6, 0.0]
      self.reward_length_scale = 0.05

    def next_state(self, current_state, action, visualise=False):
        def diff(state, t, action):
            return [state[1], self.acceleration(state[0], action)]

        t = np.linspace(0, self.t_step, 101)
        start = [current_state[0], current_state[1]]

        sol = odeint(diff, start, t, args=(action,))
        
        if visualise:
            plt.plot(sol[:,0], sol[:,1], "r-")

        final_x, final_x_dot = sol[-1, 0], sol[-1, 1]
        return final_x, final_x_dot

    def acceleration(self, current_x, action):
        def gradient(x):
            if x >= 0:
                return math.pow((1 + 5 * x ** 2), -1.5)
            else:
                return 2 * x + 1
        G = 9.81
        return action - G * math.sin(math.atan(gradient(current_x)))

    def reward(self, next_state):
        return multivariate_normal.pdf(next_state, self.target, self.reward_length_scale**2)