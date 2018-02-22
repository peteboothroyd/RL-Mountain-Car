# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud

A merge between two sources:

* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia

* the OpenAI/gym MountainCar environment
itself from 
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy.integrate import odeint
from scipy.stats import multivariate_normal

class Continuous_MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, gaussian_reward=False, t_step=0.3):
        self.min_action = -4.0              # measured in Nm
        self.max_action = 4.0               # measured in Nm
        self.last_action = 0.0              # measured in Nm (used for render)
        self.min_position = -1.0            # measured in m
        self.max_position = 1.0             # measured in m
        self.min_velocity = -2.0            # measured in m/s
        self.max_velocity = 2.0             # measured in m/s

        self.goal_position = 0.6            # measured in m
        self.goal_velocity = 0.0            # measured in m/s
        self.goal_position_threshold = 0.1  # measured in m
        self.goal_velocity_threshold = 0.1  # measured in m/s

        self.t_step= t_step                 # measured in s

        self.gaussian_reward = gaussian_reward
        if self.gaussian_reward:
            self.gaussian_reward_length_scale = 0.05

        self.low_state = np.array([self.min_position, self.min_velocity])
        self.high_state = np.array([self.max_position, self.max_velocity])

        self.viewer = None

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, dtype=np.float32, shape=(1,))
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.last_action = action
        def acceleration(current_x, action):
            G = 9.81
            return action - G * math.sin(math.atan(self._gradient(current_x)))
        
        def diff(state, t, action):
            return [state[1], acceleration(state[0], action)]

        action = np.clip(action, self.min_action, self.max_action)

        t = np.linspace(0, self.t_step, 101)
        sol = odeint(diff, self.state, t, args=(action,))
        position = np.clip(sol[-1, 0], self.min_position, self.max_position)
        velocity  = np.clip(sol[-1, 1], self.min_velocity, self.max_velocity)

        reward, done = self._reward([position, velocity], action)
        self.state = np.array([position, velocity])

        return self.state, reward, done, {}

    def reset(self, state=None):
        if state is None:
            self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        else:
            self.state = state
        return np.array(self.state)
    
    def _reward(self, next_state, action):
        reward = 0
        done = False

        position, velocity = next_state
        # near_goal_position = self.goal_position - self.goal_position_threshold <= position \
        #                     and position <= self.goal_position + self.goal_position_threshold
        # near_goal_velocity = self.goal_velocity - self.goal_velocity_threshold <= velocity \
        #                     and velocity <= self.goal_velocity + self.goal_velocity_threshold
        # done = near_goal_position and near_goal_velocity
        done = position >= self.goal_position

        if self.gaussian_reward:
            reward = multivariate_normal.pdf(next_state,
                                            [self.goal_position, self.goal_velocity],
                                            self.gaussian_reward_length_scale**2)
        elif done:
            reward = 100

        reward = reward - self.t_step #- 0.1*action**2

        return reward, done

    def get_state(self):
        return self.state

    def _height(self, xs):
        def height(x):
            if x <= 0:
                return x**2+x
            else:
                return x*(1+5*x**2)**-0.5

        height = np.vectorize(height)
        return height(xs)

    def _gradient(self, x):
        if x >= 0:
            return math.pow((1 + 5 * x ** 2), -1.5)
        else:
            return 2 * x + 1

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20

        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        min_y = np.min(ys)

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xys = list(zip((xs-self.min_position)*scale, (ys-min_y)*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = (self._height(self.goal_position)-min_y)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)
            
            action = rendering.Line((0, 0), (0, 50))
            self.action_trans = rendering.Transform()
            action.add_attr(self.action_trans)
            self.viewer.add_geom(action)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, (self._height(pos)-min_y)*scale)
        self.cartrans.set_rotation(self._gradient(pos))
        self.action_trans.set_translation(50, 50)
        self.action_trans.set_rotation(np.sin(-self.last_action*np.pi/(2*self.max_action)))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()