import math
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# from scipy.stats import multivariate_normal
# import pandas as pd


def gaussian_reward():
  X, X_DOT = np.meshgrid(np.linspace(-1, 1, num=250),
                         np.linspace(-2, 2, num=250))
  states = np.array([X, X_DOT]).T
  r = multivariate_normal.pdf(states, [0.6, 0.0], 0.05**2) / 66.84
  r = np.rot90(r)
  contour = plt.contourf(X, X_DOT, r)
  plt.colorbar(contour, shrink=0.5)
  plt.xlabel("x")
  plt.ylabel("dx")
  plt.xlim([-1, 1])
  plt.ylim([-2, 2])
  plt.title("Reward Function")
  plt.savefig("gaussian_reward.png", dpi=300)


def hill():
  x1 = np.linspace(-1, 0, num=150)
  x2 = np.linspace(0, 1, num=150)
  y1 = x1 * x1 + x1
  y2 = x2 / np.sqrt(1 + 5 * x2**2)
  x = np.concatenate([x1, x2])
  y = np.concatenate([y1, y2])

  end_height = 0.6/math.sqrt(1+5*0.6**2)

  start, = plt.plot([-0.5], [-0.25], marker='*',
                    markersize=10, color="red", label="Start")
  end, = plt.plot([0.6], [end_height], marker='o',
                  markersize=10, color="green", label="Finish")

  plt.plot(x, y)
  plt.legend(handles=[start, end])
  plt.xlabel("x")
  plt.ylabel("height")
  plt.xlim([-1, 1])
  plt.ylim([-0.3, 0.5])
  plt.show()


def entropy_plot():
  entropy_csv = pd.read_csv('/Users/peterboothroyd/Downloads/entropy.csv')
  data = entropy_csv.values

  vals = data[:, 2]
  t_steps = data[:, 1]*80

  N = 10
  averaged_vals = np.convolve(vals, np.ones((N,))/N, mode='same')

  fig, ax = plt.subplots()
  ax.plot(t_steps, averaged_vals)
  ax.set_xlabel("Training Step")
  ax.set_ylabel("Entropy")
  ax.set_ylim(0, 1.5)
  ax.set_title('Collapsing Entropy during A2C Learning')
  ax.get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%0.00e'))
  plt.show()


def pg_plot():
  pg_loss_csv = pd.read_csv('/Users/peterboothroyd/Downloads/pg_loss.csv')
  data = pg_loss_csv.values

  vals = data[:, 2]
  t_steps = data[:, 1]*80

  N = 10
  averaged_vals = np.convolve(vals, np.ones((N,))/N, mode='same')

  fig, ax = plt.subplots()
  ax.plot(t_steps, averaged_vals)
  ax.set_xlabel("Training Step")
  ax.set_ylabel("Policy Loss")
  ax.set_title('Destabilised Policy Loss during Entropy Collapse')
  ax.get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%0.00e'))
  plt.show()


def return_plot():
  import json
  with open('/Users/peterboothroyd/Desktop/returns.json', 'r') as f:
    jsn = json.load(f)
    rewards = np.array(jsn['episode_rewards'])

    N = 100
    averaged_rewards = np.convolve(rewards, np.ones((N,))/N, mode='valid')
    t_steps = np.arange(0, len(averaged_rewards))

    fig, ax = plt.subplots()
    ax.plot(t_steps, averaged_rewards)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Episode Return")
    ax.set_title('Destabilised Episode Return during Entropy Collapse')
    ax.get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%0.00e'))
    plt.show()

def value_rollout_plot(values):
  t_steps = np.arange(0, len(values))
  fig, ax = plt.subplots()
  ax.plot(t_steps, values)
  ax.set_xlabel("Training Step")
  ax.set_ylabel("Value")
  ax.set_title('Value During Rollout')
  # ax.get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%0.00e'))
  plt.savefig("/Users/peterboothroyd/Desktop/values.png", dpi=300)

def conv_filters_plot():
  """
  Plots convolutional filters
  :param weights: numpy array of rank 4
  :param channels_all: boolean, optional
  """
  with tf.Session() as sess:
    saver = tf.train.import_meta_graph('/Users/peterboothroyd/Documents/IIB/Project/Code/car_on_the_hill/archived_model_out/good_adamlr7e-4/model/model-62200.meta')
    saver.restore(sess, tf.train.latest_checkpoint('/Users/peterboothroyd/Documents/IIB/Project/Code/car_on_the_hill/archived_model_out/good_adamlr7e-4/model/'))
    graph = tf.get_default_graph()
    
    conv_weights_tensor = graph.get_tensor_by_name('hidden/conv_1/kernel:0')
    conv_weights = sess.run(conv_weights_tensor)
    
    max_vals = np.amax(conv_weights, axis=2)

    print('conv_weights.shape', conv_weights.shape)
    print('max_vals.shape', max_vals.shape)

    w_min = np.min(conv_weights)
    w_max = np.max(conv_weights)

    # channels = [0]
    # make a list of channels if all are plotted
    channels = range(conv_weights.shape[2])

    # get number of convolutional filters
    num_filters = conv_weights.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = 4, 4 #utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                              max([grid_r, grid_c]))

    # iterate channels
    for channel in channels:
      # iterate filters inside every channel
      for l, ax in enumerate(axes.flat):
        # get a single filter
        img = conv_weights[:, :, channel, l]
        # put it on the grid
        ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
        # remove any labels from the axes
        ax.set_xticks([])
        ax.set_yticks([])
      # save figure
      plt.savefig('./fig_out/{}-{}.png'.format('conv_weights', channel), bbox_inches='tight')


if __name__ == "__main__":
  conv_filters_plot()
