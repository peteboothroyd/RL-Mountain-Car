import time
from collections import deque
import os

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

import logger
import utils
from mlp_policy import MlpPolicy


class PolicyGradientAgent(object):
  def __init__(self, env, visualise, model_dir, max_episode_steps,
               debug, summary_every=100, full_reward=False, critic=False,
               normalize_adv=True, beta=1e-4, exploration=0.01, gamma=1.0):

    self._debug = debug
    self._sess = tf.Session()
    self._episode = 0

    # Wrap the session in a CLI debugger
    if self._debug:
      self._sess = tf_debug.LocalCLIDebugWrapperSession(self._sess)

    self._summary_every = summary_every
    self._visualise = visualise
    self._max_episode_steps = max_episode_steps
    self._critic = critic
    self._full_reward = full_reward
    self._normalize_adv = normalize_adv
    self._env = env
    self._model_dir = model_dir
    self._gamma = gamma

    self._policy = MlpPolicy(self._sess, env.observation_space, env.action_space,
                             beta=beta, exploration=exploration)

    self._summary_writer = tf.summary.FileWriter(model_dir)
    self._summary_writer.add_graph(self._sess.graph, global_step=self._episode)

    # Check that the graph has been successfully initialised
    uninitialized_vars = self._sess.run(tf.report_uninitialized_variables())
    if uninitialized_vars.shape[0] != 0:
      print('Not all variables have been initialized!')

  def act(self, observation):
    ''' Given an observation of the state return an action
        according to the current policy parameterisation.
    '''
    return self._policy.act(observation[np.newaxis, :])

  def learn(self, n_steps=int(1e4)):
    ''' Learn an optimal policy parameterisation by 
        interacting with the environment.
    '''
    # Track learning progress using mean reward and episode length
    mean_rewards_progress = []
    std_rewards_progress = []
    episode_length_progress = []

    total_timesteps = 0

    # Store mean rewards between summaries to report progress
    reward_history = deque(maxlen=self._summary_every)
    episode_lengths = deque(maxlen=self._summary_every)

    num_actions = self._env.action_space.shape[0]
    ob_dim = self._env.observation_space.shape[0]

    try:
      for self._episode in range(n_steps):
        visualise = self._episode % self._summary_every == 0 and self._visualise

        rewards, actions, states, mean_episode_length = self._generate_rollouts(visualise)
        states = np.concatenate(
            [np.array(state).reshape((-1, ob_dim)) for state in states])
        actions = np.concatenate([np.array(action).reshape(
            (-1, num_actions)) for action in actions])
        episode_lengths.append(mean_episode_length)

        # TODO: Try with both
        # Use future rewards or total episode rewards
        if self._full_reward:
          q = self._compute_returns(rewards, gamma=self._gamma)
        else:
          q = self._compute_future_returns(rewards, gamma=self._gamma)

        q = np.reshape(q, (-1,))
        reward_history.append(np.mean(q))

        total_timesteps += q.shape[0]

        if self._critic:
          val = self._policy.predict_val(states).reshape((-1,))

          # Change statistics of predicted values to match current rollout
          val = val - np.mean(val) + np.mean(q)
          val = (np.std(q)+1e-4) * val / (np.std(val)+1e-4)

          adv = q - val

          # Train critic
          val_target = (q - np.mean(q)) / (np.std(q) + 1e-4)
          self._policy.train_val(val_target, states)
        else:
          adv = q

        # Normalising returns can aid learning
        if self._normalize_adv:
          adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-4)

        self._policy.train_pol(adv, actions, states)

        if self._episode % self._summary_every == 0:
          mean_returns = np.mean(reward_history)
          mean_rewards_progress.append(mean_returns)

          std_dev_returns = np.std(reward_history)
          std_rewards_progress.append(std_dev_returns)

          mean_ep_length = np.mean(episode_lengths)
          episode_length_progress.append(mean_ep_length)

          self._print_stats('actions', actions)
          self._print_stats('returns', adv)

          logger.record_tabular('episode', self._episode)
          logger.record_tabular('mean_returns', mean_returns)
          logger.record_tabular('std_dev_returns', std_dev_returns)
          logger.record_tabular('total_timesteps', total_timesteps)
          logger.record_tabular('mean_ep_length', mean_ep_length)
          logger.dump_tabular()

          summary, run_metadata = self._policy.summarize(
              adv, actions, states)
          self._summary_writer.add_run_metadata(
              run_metadata, 'step%d' % self._episode)
          self._summary_writer.add_summary(summary, self._episode)
    except:
      # Reraise to allow early termination of learning, and display 
      # of learned optimal behaviour.
      raise
    finally:
      # Save out necessary checkpoints & diagnostics
      save_model_path = os.path.join(self._model_dir, 'model.ckpt')
      self._save_model(save_model_path)
      self._summary_writer.flush()
      self._summary_writer.close()
      self._plot(mean_rewards_progress, std_rewards_progress,
                 episode_length_progress, self._env, self._policy)

  def _save_model(self, save_path):
    saver = tf.train.Saver()
    saver.save(self._sess, save_path=save_path)

  def _print_stats(self, name, x):
    print(name, 'stats, mean: {0:.2f}'.format(np.mean(x)),
          'std_dev: {0:.2f}'.format(np.std(x)),
          'max: {0:.2f}'.format(np.amax(x)),
          'min: {0:.2f}'.format(np.amin(x)),
          'shape: ', str(x.shape))

  def _compute_returns(self, rollout_rewards, gamma=1.0):
    ''' Compute the returns given a rollout of immediate rewards.
        Used for REINFORCE algorithm.

    # Params:
      rewards ([float]): List of immediate rewards at different time steps

    # Returns:
      returns ([float]): List of returns for each trajectory
                              $\sum_{t=0}^{T-1}r_t$
    '''
    gamma_factor = 1
    returns = []

    for rollout in rollout_rewards:
      ret = 0

      for reward in rollout:
        ret += reward * gamma_factor
        gamma_factor *= gamma

      returns.extend(np.ones_like(rollout) * ret)

    return np.array(returns).reshape((-1))

  def _compute_future_returns(self, rollout_rewards, gamma=1.0):
    ''' Compute the returns for a given rollout of immediate rewards.
        Used for GMDP algorithm

    # Params:
      rollout_rewards ([[float]]): List of immediate rewards 
      at different time steps for multiple rollouts

    # Returns:
      returns ([tf.float32]): List of returns from that timestep onwards
                              $\sum_{h=t}^{T-1}r_h$
    '''
    returns = []

    for rollout in rollout_rewards:
      return_sum = 0
      rollout_returns = []
      for reward in reversed(rollout):
        return_sum = reward + gamma * return_sum
        rollout_returns.append(return_sum)

      rollout_returns = list(reversed(rollout_returns))
      returns.extend(rollout_returns)

    return returns

  def _generate_rollouts(self, render=False):
    '''Generate rollouts for learning.

    # Params:
      render (bool): Whether to render the rollouts

    # Returns:
      rollout_rewards ([tf.float32]): List of lists of returns for the rollouts
      rollout_actions ([tf.float32]): List of lists of actions for the rollouts
      rollout_states ([tf.float32]): List of lists of states for the rollouts
    '''
    def rollout(render=False):
      '''Generate a trajectory using current policy parameterisation.'''
      state = self._env.reset()

      rewards, actions, states = [], [], []
      for _ in range(self._max_episode_steps):
        state = np.squeeze(state)

        if render:
          self._env.render()

        action = self._policy.act(state[np.newaxis, :])
        state, reward, done, _ = self._env.step(action)
        rewards.append(reward)
        states.append(state)
        actions.append(action)

        if done:
          break

      rewards = np.squeeze(np.array(rewards)).tolist()
      actions = np.squeeze(np.array(actions)).tolist()
      states = np.squeeze(np.array(states)).tolist()

      return rewards, actions, states

    t_steps = 0
    rollout_rewards, rollout_actions, rollout_states = [], [], []
    counter = 0

    while t_steps < self._max_episode_steps:
      rewards, actions, states = rollout(render=render)
      rollout_rewards.append(rewards)
      rollout_actions.append(actions)
      rollout_states.append(states)
      t_steps += len(rewards)
      counter += 1

    mean_episode_length = t_steps*1.0/counter

    # print('Generated rollout with {0} trajectories, totalling {1} timesteps.'.format(counter, t_steps))
    # print('Average trajectory length = {0}'.format(mean_episode_length))
    # print('Average reward {0}'.format(np.mean(rollout_rewards)))

    return rollout_rewards, rollout_actions, rollout_states, mean_episode_length
  

  def _plot(self, mean_rewards, std_dev_rewards, episode_lengths, env, estimator):
    def plot_fig(series, name):
      plt.plot(series)
      plt.xlabel("Episode")
      plt.ylabel(name)
      plt.title(name + ' vs. Episode Number')
      save_name = '_'.join(name.lower().split(' ')) + '.png'
      plt.savefig(save_name, dpi=300)
      plt.clf()

    plot_fig(mean_rewards, 'Mean Reward')
    plot_fig(std_dev_rewards, 'Standard Deviation Rewards')
    plot_fig(episode_lengths, 'Episode Lengths')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    num_points = 250
    x_support = np.linspace(env.observation_space.low[0],
                            env.observation_space.high[0],
                            num=num_points)
    x_dot_support = np.linspace(env.observation_space.low[1],
                                env.observation_space.high[1],
                                num=num_points)
    predicted_vals = []
    for i in range(num_points):
        for j in range(num_points):
            state = np.array([x_support[i], x_dot_support[j]]).reshape((1,2))
            val = estimator.predict_val(state)
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
    plt.savefig("value_surface.png", dpi=300)

    plt.clf()
    contour = plt.contourf(x_grid, x_dot_grid, predicted_vals)
    plt.colorbar(contour, shrink=0.5)
    plt.savefig("value_contour.png", dpi=300)
