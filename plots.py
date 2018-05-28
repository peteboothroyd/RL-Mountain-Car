import math
import tensorflow as tf
import numpy as np
import matplotlib
# matplotlib.use('AGG')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import matplotlib.animation as animation


def gaussian_reward():
  from scipy.stats import multivariate_normal
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
  import pandas as pd
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
  import pandas as pd
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
    print(rewards)
    N = 100
    averaged_rewards = np.convolve(rewards, np.ones((N,))/N, mode='valid')
    t_steps = np.arange(0, len(averaged_rewards))

    fig, ax = plt.subplots()
    ax.plot(t_steps, averaged_rewards)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Episode Return")
    ax.set_title('Returns')
    ax.get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%0.00e'))
    plt.show()


def value_rollout_plot(values, probs, i):
  np_vals = np.array(values)

  num_frames = len(np_vals)

  vid_length = 24
  t_steps = np.linspace(0, 21, num=num_frames)

  # PLOT VALUE FUNCTION
  # fig, ax = plt.subplots()
  # ax.plot(t_steps, values)
  # ax.set_xlabel("Training Step")
  # ax.set_ylabel("Value")
  # ax.set_title('Value During Rollout')
  # # ax.get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%0.00e'))
  # plt.savefig("/Users/peterboothroyd/Desktop/values{}.png".format(i), dpi=300)

  # ANIMATE VALUES
  fig, ax = plt.subplots()
  line, = ax.plot(t_steps, values)
  ax.set_xlabel("Time (s)")
  ax.set_ylabel("Critic Output")
  ax.set_title('Critic Output During Rollout')

  def animate_val(i, x, y):
    line.set_data(x[:i], y[:i])  # update the data
    return line,

  # line, = ax.plot([], [], 'o-', lw=2)
  # time_template = 'time = %.1fs'
  # time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

  Writer = animation.writers['ffmpeg']
  writer = Writer(fps=30, metadata=dict(
      artist='Peter Boothroyd'), bitrate=3600)

  ani = animation.FuncAnimation(
      fig, animate_val, num_frames, fargs=[t_steps, np_vals], interval=34.2,
      blit=False)  # , init_func=init
  ani.save("/Users/peterboothroyd/Desktop/value_animation{}.mp4".format(i), )

  # ANIMATE ACTIONS
  np_probs = np.squeeze(np.array(probs))
  print('np_probs.shape', np_probs.shape)

  num_actions = np_probs.shape[1]
  plt.clf()
  fig, ax = plt.subplots()
  ax.set_xlabel("Action")
  ax.set_ylabel("Probability")
  ax.set_ylim(0, 1)
  ax.set_title('Actor Action Probabilities During Rollout')
  ax.set_xticklabels(('', '', 'Noop', '', 'Fire', '', 'Right', '', 'Left', ''))

  # def barlist(n):
  #   return [1.0/(n*k) for k in range(1, 4)]

  actions = range(0, num_actions)
  barcollection = plt.bar(actions, np_probs[0])

  def animate_act(i):
    y = np_probs[i]
    for n, b in enumerate(barcollection):
      b.set_height(y[n])

  ani = animation.FuncAnimation(
      fig, animate_act, repeat=False, blit=False, frames=num_frames, interval=34.2)

  ani.save("/Users/peterboothroyd/Desktop/action_animation{}.mp4".format(i))


def conv_filters_plot():
  """
  Plots convolutional filters
  :param weights: numpy array of rank 4
  :param channels_all: boolean, optional
  """
  with tf.Session() as sess:
    saver = tf.train.import_meta_graph(
        '/Users/peterboothroyd/Documents/IIB/Project/Code/car_on_the_hill/archived_model_out/good_adamlr7e-4/model/model-62200.meta')
    saver.restore(sess, tf.train.latest_checkpoint(
        '/Users/peterboothroyd/Documents/IIB/Project/Code/car_on_the_hill/archived_model_out/good_adamlr7e-4/model/'))
    graph = tf.get_default_graph()

    conv_weights_tensor = graph.get_tensor_by_name('hidden/conv_1/kernel:0')
    conv_weights = sess.run(conv_weights_tensor)

    max_vals = np.sum(conv_weights, axis=2)

    print('conv_weights.shape', conv_weights.shape)
    print('max_vals.shape', max_vals.shape)

    w_min = np.min(max_vals)
    w_max = np.max(max_vals)

    # get number of convolutional filters
    num_filters = max_vals.shape[2]

    # get number of grid rows and columns
    grid_r, grid_c = 4, 4  # utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate filters inside every channel
    for l, ax in enumerate(axes.flat):
      # get a single filter
      img = max_vals[:, :, l]
      # put it on the grid
      ax.imshow(img, vmin=w_min, vmax=w_max,
                interpolation='nearest', cmap='seismic')
      # remove any labels from the axes
      ax.set_xticks([])
      ax.set_yticks([])
    # save figure
    plt.savefig('./fig_out/{}.png'.format('conv_weights'), bbox_inches='tight')


def embeddings_saver(embeddings, obs, sess):
  from tensorboard.plugins import projector

  # NUM_TO_VISUALISE = 1000

  np_embeddings = np.squeeze(np.array(embeddings))#[:NUM_TO_VISUALISE]
  np_obs = np.squeeze(np.array(obs, dtype=np.float32))
  
  print('embeddings.shape', np_embeddings.shape)
  print('obs.shape', np_obs.shape, np_obs.dtype, np.amin(np_obs), np.amax(np_obs))

  mult = np.array([0.25, 0.5, 0.75, 1.0])
  np_obs = np.multiply(np_obs, mult)
  print('obs.shape', np_obs.shape, np_obs.dtype, np.amin(np_obs), np.amax(np_obs))
  np_obs_flattened = np.amax(np_obs, axis=3)#[:NUM_TO_VISUALISE]
  print('np_obs_flattened.shape', np_obs_flattened.shape)

  # TENSORBOARD VISUALISATION

  # embedding_name = 'embedding'
  out_path = './embed_out/'
  image_name = 'sprite.png'

  # embedding_var = tf.Variable(np_embeddings, name=embedding_name)
  # sess.run(embedding_var.initializer)

  # config = projector.ProjectorConfig()
  # embedding = config.embeddings.add()
  # embedding.tensor_name = embedding_name
  # embedding.sprite.image_path = image_name
  # embedding.sprite.single_image_dim.extend([np_obs.shape[1], np_obs.shape[2]])

  # projector.visualize_embeddings(tf.summary.FileWriter(out_path), config)

  # saver = tf.train.Saver({embedding_name: embedding_var})
  # saver.save(sess, out_path+'model.ckpt')

  # # Save obs to sprite
  def create_sprite_image(images):
    """ Returns a sprite image consisting of images passed as argument.
        Images should be count x width x height
    """
    if isinstance(images, list):
      images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

    for i in range(n_plots):
      for j in range(n_plots):
        this_filter = i * n_plots + j
        if this_filter < images.shape[0]:
          this_img = images[this_filter]
          this_img = this_img / (np.amax(this_img) - np.amin(this_img))
          plt.imsave(out_path+str(i * n_plots + j)+image_name, this_img, cmap='gray')
          spriteimage[i * img_h:(i + 1) * img_h,
                      j * img_w:(j + 1) * img_w] = this_img

    return spriteimage

  sprite = create_sprite_image(np_obs_flattened)
  # plt.imsave(out_path+image_name, sprite, cmap='gray')

  # SKLEARN PLOT
  def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    plt.axis('off')
    for i, label in enumerate(labels):
      x, y = low_dim_embs[i, :]
      plt.scatter(x, y)
      plt.annotate(
          label,
          xy=(x, y),
          xytext=(5, 2),
          textcoords='offset points',
          ha='right',
          va='bottom')
    plt.savefig(filename)

  try:
    from sklearn.manifold import TSNE

    tsne = TSNE(
        perplexity=15, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = 500
    low_dim_embs = tsne.fit_transform(np_embeddings[:plot_only, :])
    plot_with_labels(low_dim_embs, np.arange(plot_only), './fig_out/tsne.png')

  except ImportError as ex:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
    print(ex)

if __name__ == "__main__":
  return_plot()
