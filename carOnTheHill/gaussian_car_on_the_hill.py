import math
import datetime
import pickle
import numpy as np
import itertools

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from scipy.stats import multivariate_normal
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error

from scipy.integrate import odeint

def cartesian(arrays, out=None):
  arrays = [np.asarray(x) for x in arrays]
  dtype = arrays[0].dtype

  n = np.prod([x.size for x in arrays])
  if out is None:
    out = np.zeros([n, len(arrays)], dtype=dtype)

  m = int(n / arrays[0].size)
  out[:,0] = np.repeat(arrays[0], m)
  if arrays[1:]:
    cartesian(arrays[1:], out=out[0:m,1:])
    for j in range(1, arrays[0].size):
      out[j*m:(j+1)*m,1:] = out[0:m,1:]
  return out

class Agent(object):
  X_SUPPORT_MIN, X_SUPPORT_MAX = -1, 1
  X_SUPPORT_POINTS = 21
  X_SUPPORT = np.linspace(X_SUPPORT_MIN, X_SUPPORT_MAX, num=X_SUPPORT_POINTS)

  X_DOT_SUPPORT_MIN, X_DOT_SUPPORT_MAX = -2, 2
  X_DOT_SUPPORT_POINTS = 21
  X_DOT_SUPPORT = np.linspace(X_DOT_SUPPORT_MAX, X_DOT_SUPPORT_MIN, num=X_DOT_SUPPORT_POINTS)

  X_SUPPORT_FINE = np.linspace(X_SUPPORT_MIN, X_SUPPORT_MAX, num=250)
  X_DOT_SUPPORT_FINE = np.linspace(X_DOT_SUPPORT_MAX, X_DOT_SUPPORT_MIN, num=250)
  SUPPORT_STATES_FINE = np.zeros((250 * 250, 2))

  A_SUPPORT_MIN, A_SUPPORT_MAX = -4, 4
  A_SUPPORT_POINTS = 21
  A_SUPPORT = np.linspace(A_SUPPORT_MIN, A_SUPPORT_MAX, num=A_SUPPORT_POINTS)

  GAMMA = 0.9

  def __init__(self, environment):
    self.support_states = np.zeros((self.X_SUPPORT_POINTS * self.X_DOT_SUPPORT_POINTS, 2))
    
    for i in range(self.X_DOT_SUPPORT_POINTS):
      for j in range(self.X_SUPPORT_POINTS):
        self.support_states[i * self.X_SUPPORT_POINTS + j][0] = self.X_SUPPORT[j]
        self.support_states[i * self.X_SUPPORT_POINTS + j][1] = self.X_DOT_SUPPORT[i]

    for i in range(250):
      for j in range(250):
        self.SUPPORT_STATES_FINE[i * 250 + j][0] = self.X_SUPPORT_FINE[j]
        self.SUPPORT_STATES_FINE[i * 250 + j][1] = self.X_DOT_SUPPORT_FINE[i]

    self.support_values = np.zeros((self.X_SUPPORT_POINTS * self.X_DOT_SUPPORT_POINTS, 1))
    self.environment = environment

  def learn(self):
    try:
      with open ('support_values', 'rb') as fp:
        print("loaded support_values file")
        self.support_values = pickle.load(fp)
        self.gp_x, self.gp_x_dot = self.learn_dynamics(50)
        self.learn_value_function()
        self.visualise_value_function()
        return
    except FileNotFoundError:
      print("support_values file not found")
      self.initialise_values()

    self.learn_value_function()
    self.visualise_value_function()
    self.gp_x, self.gp_x_dot = self.learn_dynamics(50)

    for l in range(5):
      k_v_inv_v = self.gp_val.alpha_
      k_v_chol = self.gp_val.L_.dot(self.gp_val.L_.T)
      k_v_inv = np.linalg.inv(k_v_chol)

      v_squared, l1, l2 = np.exp(self.gp_val.kernel_.theta)
      print("v_squared: %s, l1: %s, l2: %s" % (v_squared, l1, l2))
      lengths = np.array([l1, l2]).reshape((-1,1))
      lengths_squared = np.square(lengths)

      R = np.zeros((self.X_SUPPORT_POINTS * self.X_DOT_SUPPORT_POINTS, 1))
      W = np.zeros((self.X_SUPPORT_POINTS * self.X_DOT_SUPPORT_POINTS, self.X_SUPPORT_POINTS * self.X_DOT_SUPPORT_POINTS))
      maximising_actions = np.zeros((self.X_SUPPORT_POINTS * self.X_DOT_SUPPORT_POINTS, 1))

      target = np.array([0.6, 0.0]).reshape((-1,1))
      
      for state_index in range(len(self.support_states)):
        x, x_dot = self.support_states[state_index]
        max_action_val = None

        state_actions = np.array(list(itertools.product([x], [x_dot], self.A_SUPPORT)))

        mu_x, std_dev_x = self.gp_x.predict(state_actions, return_std=True)
        mu_x_dot, std_dev_x_dot = self.gp_x_dot.predict(state_actions, return_std=True)

        means = np.array([mu_x, mu_x_dot])
        var = np.square(np.array([std_dev_x, std_dev_x_dot]))
        length_squared_plus_var = np.add(var, lengths_squared)
        state_diffs = np.subtract(self.support_states[:,:, np.newaxis], means)
        state_diffs_squared = np.square(state_diffs)
        state_diffs_squared_divided_length_plus_var = np.divide(state_diffs_squared, length_squared_plus_var)
        summed = -0.5 * np.sum(state_diffs_squared_divided_length_plus_var, axis=1)
        exponentiated = np.exp(summed)
        product = np.prod(length_squared_plus_var, axis=0)
        square_root = np.sqrt(product)

        w = np.prod(lengths) * v_squared * np.divide(exponentiated, square_root)

        target_minus_mean = np.subtract(target, means)
        squared_target_minus_mean = np.square(target_minus_mean)
        var_plus = var + 0.05 ** 2
        divided = np.divide(squared_target_minus_mean, var_plus)
        summed = -0.5 * np.sum(divided, axis=0)
        exponentiated = np.exp(summed)
        product = np.prod(var_plus, axis=0)
        square_root = np.sqrt(product)
        r = np.divide(exponentiated, square_root) / (2 * math.pi * 63.66)

        v = self.GAMMA * w.T.dot(k_v_inv_v)
        val_i = r + v.T

        max_val_index = np.argmax(val_i)
        maximising_actions[state_index][0] = self.A_SUPPORT[max_val_index]
        R[state_index][0] = r[max_val_index]
        W[state_index] = w[:,max_val_index]
    
      intermediate1 = np.eye(self.X_SUPPORT_POINTS * self.X_DOT_SUPPORT_POINTS) - self.GAMMA * W.dot(k_v_inv)
      intermediate2 = np.linalg.inv(intermediate1)
      new_v = intermediate2.dot(R)

      change_in_val = mean_squared_error(self.support_values, new_v)
      print("rms change in support point values: %s" % (change_in_val))

      self.support_values = new_v

      self.learn_value_function()
      self.visualise_value_function(maximising_actions=maximising_actions)
    
      with open('support_values%s' % (datetime.datetime.now()), 'wb') as fp:
        pickle.dump(self.support_values, fp)
    
    self.visualise_value_function(show_fig=True)

  def initialise_values(self):
    for i in range(self.X_SUPPORT_POINTS * self.X_DOT_SUPPORT_POINTS):
      state = self.support_states[i]
      reward = env.reward(state)
      self.support_values[i][0] = reward
  
  def learn_value_function(self):
    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 100)) * RBF(length_scale=[0.1, 0.1], length_scale_bounds=(0.05, 10.0))
    gp_val = GaussianProcessRegressor(kernel=kernel, alpha=0.01, n_restarts_optimizer=9)
    gp_val = gp_val.fit(self.support_states, self.support_values)

    predicted_vals = gp_val.predict(self.support_states)
    rms = math.sqrt(mean_squared_error(predicted_vals, self.support_values))
    print("val function rms: %s" % rms)
    self.gp_val = gp_val

  def visualise_value_function(self, maximising_actions=None, R=None, V=None, show_fig=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X, X_DOT = np.meshgrid(self.X_SUPPORT_FINE, self.X_DOT_SUPPORT_FINE)
    predicted_vals = self.gp_val.predict(self.SUPPORT_STATES_FINE).reshape((250, 250))
    vals = self.support_values.reshape((self.X_SUPPORT_POINTS, self.X_DOT_SUPPORT_POINTS))
    surf = ax.plot_surface(X, X_DOT, predicted_vals, cmap=cm.rainbow, antialiased=True, linewidth=0.001)

    # Customize the z axis.
    ax.set_zlim(np.amin(vals), np.amax(predicted_vals))
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig("gp%s.png" % datetime.datetime.now())
    
    if maximising_actions is not None:
      plt.clf()
      X, X_DOT = np.meshgrid(self.X_SUPPORT, self.X_DOT_SUPPORT)
      maximising_actions = maximising_actions.reshape((self.X_SUPPORT_POINTS, self.X_DOT_SUPPORT_POINTS))
      contour = plt.contourf(X, X_DOT, maximising_actions)
      plt.colorbar(contour, shrink=0.5)
      plt.savefig("actions%s.png" % datetime.datetime.now())
    
    if R is not None:
      fig = plt.figure()
      ax = fig.gca(projection='3d')
      X, X_DOT = np.meshgrid(self.X_SUPPORT, self.X_DOT_SUPPORT)
      R = R.reshape((self.X_SUPPORT_POINTS, self.X_DOT_SUPPORT_POINTS))
      surf = ax.plot_surface(X, X_DOT, R, cmap=cm.plasma, antialiased=True, linewidth=0.001)
      ax.set_zlim(np.amin(R), np.amax(R))
      ax.zaxis.set_major_locator(LinearLocator(10))
      ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
      fig.colorbar(surf, shrink=0.5, aspect=5)
      plt.savefig("reward%s.png" % datetime.datetime.now())
    
    if V is not None:
      fig = plt.figure()
      ax = fig.gca(projection='3d')
      X, X_DOT = np.meshgrid(self.X_SUPPORT, self.X_DOT_SUPPORT)
      V = V.reshape((self.X_SUPPORT_POINTS, self.X_DOT_SUPPORT_POINTS))
      surf = ax.plot_surface(X, X_DOT, V, cmap=cm.coolwarm, antialiased=False, linewidth=0)
      ax.set_zlim(np.amin(V), np.amax(V))
      ax.zaxis.set_major_locator(LinearLocator(10))
      ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
      fig.colorbar(surf, shrink=0.5, aspect=5)
      plt.savefig("value%s.png" % datetime.datetime.now())

    if show_fig:
      plt.show()
  
  def learn_dynamics(self, num_dynamics_examples):
    x = np.random.uniform(low=self.X_SUPPORT_MIN, high=self.X_SUPPORT_MAX, size=num_dynamics_examples)
    x_dot = np.random.uniform(low=self.X_DOT_SUPPORT_MIN, high=self.X_DOT_SUPPORT_MAX, size=num_dynamics_examples)
    a = np.random.uniform(low=self.A_SUPPORT_MIN, high=self.A_SUPPORT_MAX, size=num_dynamics_examples)

    start_states = list(zip(x, x_dot, a))
    next_xs, next_x_dots = [], []

    for state in start_states:
      x, x_dot, a = state
      next_x, next_x_dot = env.next_state((x, x_dot), a)
      next_xs.append(next_x)
      next_x_dots.append(next_x_dot)
    
    kernel1 = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=[0.25, 0.25, 0.25], length_scale_bounds=(1e-3, 20)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 10.0))
    gp_x = GaussianProcessRegressor(kernel=kernel1, n_restarts_optimizer=9)
    gp_x.fit(start_states, next_xs)

    kernel2 = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=[0.25, 0.25, 0.25], length_scale_bounds=(1e-3, 20)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 10.0))
    gp_x_dot = GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer=9)
    gp_x_dot.fit(start_states, next_x_dots)

    # estimated_next_xs = gp_x.predict(start_states)
    # estimated_next_x_dots = gp_x_dot.predict(start_states)
    # x_rms = math.sqrt(mean_squared_error(next_xs, estimated_next_xs))
    # x_dot_rms = math.sqrt(mean_squared_error(next_x_dots, estimated_next_x_dots))
    
    # print("Learning dynamics x_rms: %s, x_dot_rms: %s" % (x_rms, x_dot_rms))

    return gp_x, gp_x_dot

  def test_learn_dynamics(self):
    num_training_examples = [5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 200, 250, 300]
    num_validation_examples = 1000

    x = np.random.uniform(low=self.X_SUPPORT_MIN, high=self.X_SUPPORT_MAX, size=num_validation_examples)
    x_dot = np.random.uniform(low=self.X_DOT_SUPPORT_MIN, high=self.X_DOT_SUPPORT_MAX, size=num_validation_examples)
    a = np.random.uniform(low=self.A_SUPPORT_MIN, high=self.A_SUPPORT_MAX, size=num_validation_examples)

    start_states = list(zip(x, x_dot, a))
    next_xs, next_x_dots = [], []
    x_rmss, x_dot_rmss = [], []
    percent_x_outside_ranges, percent_x_dot_outside_ranges = [], []

    for state in start_states:
      x, x_dot, a = state
      next_x, next_x_dot = env.next_state((x, x_dot), a)
      next_xs.append(next_x)
      next_x_dots.append(next_x_dot)

    for num_examples in num_training_examples:
      gp_x, gp_x_dot = self.learn_dynamics(num_examples)
      estimated_next_xs, estimated_next_x_stddevs = gp_x.predict(start_states, return_std=True)
      estimated_next_x_dots, estimated_next_x_dot_stddevs = gp_x_dot.predict(start_states, return_std=True)
      x_rms = math.sqrt(mean_squared_error(next_xs, estimated_next_xs))
      x_rmss.append(x_rms)
      x_dot_rms = math.sqrt(mean_squared_error(next_x_dots, estimated_next_x_dots))
      x_dot_rmss.append(x_dot_rms)

      num_x_outside_estimate_range, num_x_dot_outside_estimate_range = 0, 0 
      for i in range(len(next_xs)):
        if abs(next_xs[i] - estimated_next_xs[i]) > 2 * estimated_next_x_stddevs[i]:
          num_x_outside_estimate_range += 1
        if abs(next_x_dots[i] - estimated_next_x_dots[i]) > 2 * estimated_next_x_dot_stddevs[i]:
          num_x_dot_outside_estimate_range += 1

      percent_x_outside_range = 100 * num_x_outside_estimate_range / len(next_xs)
      percent_x_outside_ranges.append(percent_x_outside_range)
      percent_x_dot_outside_range = 100 * num_x_dot_outside_estimate_range / len(next_xs)
      percent_x_dot_outside_ranges.append(percent_x_dot_outside_range)
      print("Number of training samples: %s, x_rms: %s, x_dot_rms: %s, percent_x_outside_range: %s, percent_x_dot_outside_range: %s " % (num_examples, x_rms, x_dot_rms, percent_x_outside_range, percent_x_dot_outside_range))
    
    x_rms_handle, = plt.plot(num_training_examples, x_rmss, label="X RMS")
    x_dot_rms_handle, = plt.plot(num_training_examples, x_dot_rmss, label="X_DOT RMS")
    plt.legend(handles=[x_rms_handle, x_dot_rms_handle])
    plt.show()
    x_percent_handle, = plt.plot(num_training_examples, percent_x_outside_ranges, label="X % Outside Bound")
    x_dot_percent_handle, = plt.plot(num_training_examples, percent_x_dot_outside_ranges, label="X_DOT % Outside Bound")
    plt.legend(handles=[x_percent_handle, x_dot_percent_handle])
    plt.show()

  
  def act(self, start_state):
    plt.close()
    current_x, current_x_dot = start_state
    target_x, target_x_dot = 0.6, 0.0
    initial_val = self.gp_val.predict(np.array((current_x, current_x_dot)).reshape(1, -1))
    path_x, path_x_dot, val = [current_x], [current_x_dot], [initial_val]
    predicted_xs, predicted_x_dots = [current_x], [current_x_dot]

    for i in range(15):
      max_v, argmax_v = None, None

      k_v_inv_v = self.gp_val.alpha_
      k_v_chol = self.gp_val.L_.dot(self.gp_val.L_.T)
      k_v_inv = np.linalg.inv(k_v_chol)
      v_squared, l1, l2 = np.exp(self.gp_val.kernel_.theta)
      l1, l2 = 1/l1, 1/l2

      for k in range(self.A_SUPPORT_POINTS):
        action = self.A_SUPPORT[k]
        state_action = np.array((current_x, current_x_dot, action)).reshape(1, -1)
        mu_x, std_dev_x = self.gp_x.predict(state_action, return_std=True)
        mu_x_dot, std_dev_x_dot = self.gp_x_dot.predict(state_action, return_std=True)

        ri = math.exp(-0.5 * (pow(0.6-mu_x[0], 2)/(pow(std_dev_x[0], 2) + pow(0.05, 2)) + pow(mu_x_dot[0], 2)/(pow(std_dev_x_dot[0], 2) + pow(0.05, 2)))) / (pow((pow(std_dev_x[0], 2) + pow(0.05, 2))*(pow(std_dev_x_dot[0], 2) + pow(0.05, 2)), 0.5) * 2 * math.pi * 63.66)
        wi = np.zeros((1, self.X_SUPPORT_POINTS * self.X_DOT_SUPPORT_POINTS))
        
        i = 0
        for statej in self.support_states:
          wi[0][i] = v_squared * math.exp(-0.5 * (pow(l1 * (statej[0] - mu_x[0]), 2) / (1 + pow(l1 * std_dev_x[0], 2)) + pow(l2 * (statej[1] - mu_x_dot[0]), 2) / (1 + pow(l2 * std_dev_x_dot[0], 2)))) / pow((pow(l1 * std_dev_x[0], 2) + 1) * (pow(l2 * std_dev_x_dot[0], 2) + 1), 0.5)
          i += 1

        vi = self.GAMMA * np.dot(wi, k_v_inv_v)
        val_i = ri + vi[0][0]

        if max_v is None or val_i > max_v:
          max_v = val_i
          argmax_v = action
          predicted_x, predicted_x_dot = mu_x, mu_x_dot
          
      current_x, current_x_dot = self.environment.next_state((current_x, current_x_dot), argmax_v, visualise=True)

      path_x.append(current_x)
      path_x_dot.append(current_x_dot)
      val.append(max_v)
      predicted_xs.append(predicted_x)
      predicted_x_dots.append(predicted_x_dot)

      if abs(current_x - target_x) < 0.05 and abs(current_x_dot - target_x_dot) < 0.05:
        break
    
    # plt.plot(path_x, path_x_dot, )
    plt.plot(predicted_xs, predicted_x_dots)
    plt.show()

class Environment(object):
  T_STEP = 0.3

  def next_state(self, current_state, action, visualise=False):
    def diff(state, t, action):
      return [state[1], self.acceleration(state[0], action)]

    current_x, current_x_dot = current_state
    t = np.linspace(0, self.T_STEP, 101)
    start = [current_state[0], current_state[1]]

    sol = odeint(diff, start, t, args=(action,))
    
    if visualise:
      plt.plot(sol[:,0], sol[:,1], "r-")
      print("Visualising...")

    final_x, final_x_dot = sol[-1, 0], sol[-1, 1]
    return final_x, final_x_dot

    # accel = self.acceleration(current_x, action)
    # next_x = current_x + current_x_dot * self.T_STEP + 0.5 * accel * self.T_STEP ** 2
    # next_x_dot = current_x_dot + accel * self.T_STEP
    # return next_x, next_x_dot
    # n_steps = 100
    # for _ in range(n_steps):
    #   x_dot, x_double_dot = self.step((current_x, current_x_dot), action)
    #   current_x += (self.T_STEP / n_steps) * x_dot
    #   current_x_dot += (self.T_STEP / n_steps) * x_double_dot
    # return current_x, current_x_dot

  def step(self, current_state, action):
    current_x, current_x_dot = current_state
    accel = self.acceleration(current_x, action)
    return [current_x_dot, accel]

  def acceleration(self, current_x, action):
    def gradient(x):
      if x >= 0:
        return math.pow((1 + 5 * x ** 2), -1.5)
      else:
        return 2 * x + 1
    G = 9.81
    return action - G * math.sin(math.atan(gradient(current_x)))

  def reward(self, next_state):
    return multivariate_normal.pdf(next_state, [0.6, 0.0], 0.05**2)

if __name__ == "__main__":
  env = Environment()
  agent = Agent(env)
  # agent.test_learn_dynamics()
  agent.learn()
  agent.act((-0.5, 0))