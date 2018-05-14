import numpy as np

class A2CRunner(object):
  ''' Handles executing the policy and returning trajectories of states,
      returns, and actions for the a2c algorithm.

      # Note
        env must be descended from the Vector
  '''
  def __init__(self, policy, env, n_steps, gamma):
    self._policy = policy
    self._env = env
    self._n_steps = n_steps
    self._gamma = gamma

    self._batch_obs_shape = (self._n_steps*self._env.num_envs,) \
        + self._env.observation_space.shape

    self._obs = self._env.reset()

  def generate_rollouts(self):
    '''Generate rollouts for learning.

    # Returns:
      rollout_returns (np.array([float])): List of returns for the rollouts,
          (dimension [-1, 1])
      rollout_actions (np.array([[float]]): List of actions for the rollouts,
          (dimension [-1, act_dim])
      rollout_observations ([[tf.float32]]): List of observations for the
          rollouts, (dimension [-1, obs_dim])
    '''

    rollout_rewards, rollout_actions, rollout_observations = [], [], []
    rollout_dones, rollout_values = [], []

    for _ in range(self._n_steps):
      actions = self._policy.actor(self._obs)
      values = self._policy.critic(self._obs)
      observations, rewards, dones, _ = self._env.step(actions)
      rollout_rewards.append(rewards)
      rollout_values.append(values)
      rollout_observations.append(np.copy(self._obs))
      rollout_actions.append(actions)
      rollout_dones.append(dones)
      self._obs = observations

    # Switch lists of [n_envs, n_steps] to [n_steps, n_envs]
    rollout_observations = np.array(rollout_observations).swapaxes(0, 1)\
        .reshape(self._batch_obs_shape)
    rollout_actions = np.array(rollout_actions).swapaxes(0, 1)
    rollout_values = np.array(rollout_values).swapaxes(0, 1)
    rollout_dones = np.array(rollout_dones).swapaxes(0, 1)
    rollout_rewards = np.array(rollout_rewards).swapaxes(0, 1)

    rollout_returns = self._compute_future_returns(
        rollout_rewards, rollout_dones, rollout_values)

    rollout_actions = rollout_actions.flatten()
    rollout_values = rollout_values.flatten()
    rollout_returns = rollout_returns.flatten()

    # Change statistics of predicted values to match current rollout
    values = values - np.mean(values) + np.mean(rollout_returns)
    values = (np.std(rollout_returns)+1e-4) * values / (np.std(values)+1e-4)

    return rollout_returns, rollout_actions, \
        rollout_observations, rollout_values

  def _compute_future_returns(self, rewards, dones, values):
    ''' Compute the future returns for a given rollout of immediate rewards.
        Used for GMDP algorithm. Each row contains the data for each
        environment.

    # Params:
      rewards ([[float]]): List of immediate rewards at different time steps for
          multiple rollouts (dimension [n_steps, n_envs])
      dones: ([[bool]]): List of done flags for each step
          (dimension [n_steps, n_envs])
      values: ([[float]]): List of values (dimension [n_steps, n_envs])

    # Returns:
      returns ([tf.float32]): List of returns from that timestep onwards
          $\sum_{h=t}^{T-1}r_h$ (dimension [n_steps, n_envs])
    '''
    returns = []

    for _, (rollout_rewards, rollout_dones, rollout_values) in enumerate(
        zip(rewards, dones, values)):
      rollout_rewards = rollout_rewards.tolist()
      rollout_values = rollout_values.tolist()
      rollout_dones = rollout_dones.tolist()

      running_sum = 0
      rollout_returns = []

      for i, (reward, done, val) in enumerate(zip(reversed(rollout_rewards),
                                                  reversed(rollout_dones),
                                                  reversed(rollout_values))):
        # If the last step in the rollout is not terminal, the unbiased estimate
        # of the return is the state value.
        if i == 0 and not done:
          running_sum = val[0]
        running_sum = reward + self._gamma * running_sum if not done else 0
        rollout_returns.append(running_sum)

      rollout_returns = list(reversed(rollout_returns))
      returns.append(rollout_returns)

    returns = np.array(returns)

    return returns
