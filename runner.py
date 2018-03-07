import numpy as np
import time
from stats import Stats

class EpisodicRunner(object):
  ''' Handles executing the policy and returning trajectories of states,
      returns, and actions for episodic tasks.
  '''
  def __init__(self,
               policy,
               env,
               max_episode_steps,
               gamma,
               summary_every,
               future_returns=True):
    self._policy = policy
    self._env = env
    self._max_episode_steps = max_episode_steps
    self._future_returns = future_returns
    self._gamma = gamma
    self._stats = Stats(summary_every)

    self._num_actions = self._env.action_space.shape[0]
    self._ob_dim = self._env.observation_space.shape[0]

  def generate_rollouts(self, render=False, summarise=False):
    '''Generate rollouts for learning.

    # Params:
      render (bool): Whether to render the rollouts

    # Returns:
      rollout_returns (np.array([float])): List of returns for the rollouts, (dimension [-1, 1])
      rollout_actions (np.array([[float]]): List of actions for the rollouts, (dimension [-1, ac_dim])
      rollout_states ([[tf.float32]]): List of states for the rollouts, (dimension [-1, ob_dim])
    '''
    t_steps = 0
    rollout_rewards, rollout_actions, rollout_states = [], [], []

    while t_steps < self._max_episode_steps:
      rewards, actions, states = self.rollout(render=render)
      rollout_rewards.append(rewards)
      rollout_actions.append(actions)
      rollout_states.append(states)
      t_steps += len(rewards)

    states = np.concatenate([np.array(state).reshape(
        (-1, self._ob_dim)) for state in rollout_states])
    actions = np.concatenate([np.array(action).reshape(
        (-1, self._num_actions)) for action in rollout_actions])

    self._stats.store_episode_stats(rollout_rewards, summarise)

    if self._future_returns:
      returns = self._compute_future_returns(rollout_rewards)
    else:
      returns = self._compute_returns(rollout_rewards)

    return returns, actions, states

  def rollout(self, render=False, t_sleep=0.0):
    '''Generate a trajectory using current policy parameterisation.'''
    state = self._env.reset()

    rewards, actions, states = [], [], []
    for _ in range(self._max_episode_steps):
      state = np.squeeze(state)

      if render:
        self._env.render()
        time.sleep(t_sleep)

      action = self._policy.actor(state[np.newaxis, :])
      state, reward, done, _ = self._env.step(action)
      rewards.append(reward)
      states.append(np.squeeze(state))
      actions.append(action)

      if done:
        break

    rewards = np.squeeze(np.array(rewards)).tolist()
    actions = np.squeeze(np.array(actions)).tolist()
    states = np.squeeze(np.array(states))
    states = states.tolist()

    return rewards, actions, states
  
  def report_stats(self):
    return self._stats.report_stats()

  def _compute_returns(self, rollout_rewards):
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
        gamma_factor *= self._gamma

      returns.extend(np.ones_like(rollout) * ret)

    return np.array(returns).reshape((-1))

  def _compute_future_returns(self, rollout_rewards):
    ''' Compute the future returns for a given rollout of immediate rewards.
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
        return_sum = reward + self._gamma * return_sum
        rollout_returns.append(return_sum)

      rollout_returns = list(reversed(rollout_returns))
      returns.extend(rollout_returns)

    return returns
