# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud

A merge between two sources:

* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia

* the OpenAI/gym MountainCar environment
itself from http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
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

  def __init__(self, gaussian_reward_scale=None, t_step=0.3, 
               terminating=False, hard=False):
    self._min_action = -4.0              #  measured in Nm
    self._max_action = 4.0               #  measured in Nm
    self._last_action = 0.0              #  measured in Nm (used for render)
    self._min_position = -1.0            #  measured in m
    self._max_position = 1.0             #  measured in m
    self._min_velocity = -2.0            #  measured in m/s
    self._max_velocity = 2.0             #  measured in m/s

    self._goal_position = 0.6            #  measured in m
    self._goal_velocity = 0.0            #  measured in m/s
    self._goal_position_threshold = 0.1  #  measured in m
    self._goal_velocity_threshold = 0.1  #  measured in m/s

    self._t_step = t_step                #  measured in s

    if gaussian_reward_scale is not None:
      self._gaussian_reward = True
      self._gaussian_reward_length_scale = gaussian_reward_scale
    else:
      self._gaussian_reward = False

    self._terminating = terminating
    self._hard = hard

    self._low_state = np.array([self._min_position, self._min_velocity])
    self._high_state = np.array([self._max_position, self._max_velocity])

    self._viewer = None

    self.action_space = spaces.Box(
        low=self._min_action, high=self._max_action, dtype=np.float32, shape=(1,))
    self.observation_space = spaces.Box(
        low=self._low_state, high=self._high_state, dtype=np.float32)

    self.seed()
    self.reset()

  def seed(self, seed=None):
    self._np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    def acceleration(current_x, action):
      G = 9.81
      return action - G * math.sin(math.atan(self._gradient(current_x)))

    def diff(state, t, action):
      speed = state[1]
      accel = acceleration(state[0], action)
      return [speed, accel]

    action = np.clip(action, self._min_action, self._max_action)
    self._last_action = action

    t = np.linspace(0, self._t_step, 101)
    sol = odeint(diff, self._state, t, args=(action,))
    
    position = sol[-1, 0]
    velocity = sol[-1, 1]

    if position <= self._min_position:
      position = self._min_position
      velocity = 0
    
    if position >= self._max_position:
      position = self._max_position
      velocity = 0
    
    # if velocity >= self._max_velocity:
    #   velocity = self._max_velocity
    
    # if velocity <= self._min_velocity:
    #   velocity = self._min_velocity

    done = self._done([position, velocity])

    reward = self._reward([position, velocity], done)
    self._state = np.array([position, velocity])

    return self._state, reward, done, {}

  def _done(self, next_state):
    if self._terminating:
      position, velocity = next_state
      if self._hard:
        near_goal_position = abs(position - self._goal_position) <= self._goal_position_threshold
        near_goal_velocity = abs(velocity - self._goal_velocity) <= self._goal_velocity_threshold
        return near_goal_position and near_goal_velocity
      else:
        return position >= self._goal_position
    else:
      return False

  def reset(self, state=None):
    if state is None:
      self._state = np.array([self._np_random.uniform(low=-0.6, high=-0.4), 0])
    else:
      self._state = np.array(state)
    return self._state

  def _reward(self, next_state, done):
    reward = 0
    if self._gaussian_reward:
      reward = multivariate_normal.pdf(
          next_state, [self._goal_position, self._goal_velocity],
          self._gaussian_reward_length_scale**2)
    elif done:
      reward = 100

    # reward = reward #- self._t_step

    return reward

  def get_state(self):
    return self._state

  def _height(self, xs):
    def height(x):
      if x >= 0:
        return x*(1+5*x**2)**-0.5 
      else:
        return x**2+x

    height = np.vectorize(height)
    return height(xs)

  def _gradient(self, x):
    if x >= 0:
      return (1 + 5 * x ** 2)**-1.5
    else:
      return 2 * x + 1

  def render(self, mode='human'):
    screen_width = 600
    screen_height = 400

    world_width = self._max_position - self._min_position
    scale = screen_width/world_width
    carwidth = 40
    carheight = 20

    xs = np.linspace(self._min_position, self._max_position, 100)
    ys = self._height(xs)
    min_y = np.min(ys)

    if self._viewer is None:
      from gym.envs.classic_control import rendering
      self._viewer = rendering.Viewer(screen_width, screen_height)
      xys = list(zip((xs-self._min_position)*scale, (ys-min_y)*scale))

      self._track = rendering.make_polyline(xys)
      self._track.set_linewidth(4)
      self._viewer.add_geom(self._track)

      clearance = 10

      l, r, t, b = -carwidth/2, carwidth/2, carheight, 0
      car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
      car.add_attr(rendering.Transform(translation=(0, clearance)))
      self._cartrans = rendering.Transform()
      car.add_attr(self._cartrans)
      self._viewer.add_geom(car)
      frontwheel = rendering.make_circle(carheight/2.5)
      frontwheel.set_color(.5, .5, .5)
      frontwheel.add_attr(rendering.Transform(
          translation=(carwidth/4, clearance)))
      frontwheel.add_attr(self._cartrans)
      self._viewer.add_geom(frontwheel)
      backwheel = rendering.make_circle(carheight/2.5)
      backwheel.add_attr(rendering.Transform(
          translation=(-carwidth/4, clearance)))
      backwheel.add_attr(self._cartrans)
      backwheel.set_color(.5, .5, .5)
      self._viewer.add_geom(backwheel)
      flagx = (self._goal_position-self._min_position)*scale
      flagy1 = (self._height(self._goal_position)-min_y)*scale
      flagy2 = flagy1 + 50
      flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
      self._viewer.add_geom(flagpole)
      flag = rendering.FilledPolygon(
          [(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
      flag.set_color(.8, .8, 0)
      self._viewer.add_geom(flag)

      action = rendering.Line((0, 0), (0, 50))
      self._action_trans = rendering.Transform()
      action.add_attr(self._action_trans)
      self._viewer.add_geom(action)

    pos = self._state[0]
    self._cartrans.set_translation(
        (pos-self._min_position)*scale, (self._height(pos)-min_y)*scale)
    self._cartrans.set_rotation(self._gradient(pos))
    self._action_trans.set_translation(50, 50)
    self._action_trans.set_rotation(
        np.sin(-self._last_action*np.pi/(2*self._max_action)))

    return self._viewer.render(return_rgb_array=mode == 'rgb_array')

  def close(self):
    if self._viewer:
      self._viewer.close()
