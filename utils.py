import tensorflow as tf
import numpy as np
import random

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt


def set_global_seeds(i):
  tf.set_random_seed(i)
  np.random.seed(i)
  random.seed(i)


def plot(mean_rewards,
         std_dev_rewards,
         episode_lengths,
         summary_every,
         dir_path):
  def plot_fig(series, name):
    mean = np.mean(series, axis=0)
    lower = np.percentile(series, 5, axis=0)
    upper = np.percentile(series, 95, axis=0)
    n = len(mean)
    x = np.arange(0, n * summary_every, summary_every)
    plt.plot(x, mean)
    plt.fill_between(x, lower, upper, color='b', alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel(name)
    plt.title(name + ' vs. Episode Number')
    save_name = dir_path + '_'.join(name.lower().split(' ')) + '.png'
    plt.savefig(save_name, dpi=300)
    plt.clf()

  plot_fig(mean_rewards, 'Mean Reward')
  plot_fig(std_dev_rewards, 'Standard Deviation Rewards')
  plot_fig(episode_lengths, 'Episode Lengths')

def plot_value_func(estimator, episode, ob_space):
  plt.clf()
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  num_points = 250
  x_support = np.linspace(ob_space.low[0],
                          ob_space.high[0],
                          num=num_points)
  x_dot_support = np.linspace(ob_space.low[1],
                              ob_space.high[1],
                              num=num_points)
  predicted_vals = []
  for i in range(num_points):
    for j in range(num_points):
      state = np.array([x_support[i], x_dot_support[j]]).reshape((-1, 2))
      val = estimator.critic(state)
      predicted_vals.append(np.squeeze(val))
  predicted_vals = np.array(predicted_vals).reshape((num_points, num_points))
  x_grid, x_dot_grid = np.meshgrid(x_support, x_dot_support)
  surf = ax.plot_surface(x_grid, x_dot_grid, predicted_vals,
                        cmap=cm.rainbow, antialiased=True, linewidth=0.001)

  # Customize the z axis.
  ax.set_zlim(np.amin(predicted_vals), np.amax(predicted_vals))
  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

  # Add a color bar which maps values to colors.
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.savefig("value_surface_{0}.png".format(episode), dpi=300)

  plt.clf()
  contour = plt.contourf(x_grid, x_dot_grid, predicted_vals)
  plt.colorbar(contour, shrink=0.5)
  plt.savefig("value_contour_{0}.png".format(episode), dpi=300)