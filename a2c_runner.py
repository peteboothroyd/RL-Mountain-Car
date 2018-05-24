import numpy as np

class A2CRunner(object):
  ''' Handles executing the policy and returning trajectories of states,
      returns, and actions for the a2c algorithm.

      # Note
        env must be descended from the Vector
  '''
  def __init__(self, policy, env, n_steps, gamma, discrete):
    n_envs = env.num_envs
    self._policy = policy
    self._env = env
    self._n_steps = n_steps
    self._gamma = gamma
    self._discrete = discrete

    self._batch_obs_shape = (n_steps*n_envs,) \
        + self._env.observation_space.shape
    self._dones = [False for _ in range(n_envs)]

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
      # NOTE: In each iteration we are saving:
      # $x_t, v(x_t), a_t, r_{t+1}, done(x_{t+1})$
      rollout_observations.append(np.copy(self._obs))
      actions, values = self._policy.step(self._obs)
      rollout_values.append(np.squeeze(values))
      rollout_actions.append(actions)
      self._obs, rewards, self._dones, _ = self._env.step(actions)
      rollout_rewards.append(rewards)
      rollout_dones.append(self._dones)

    last_values = self._policy.critic(self._obs)

    obs_dtype = self._env.observation_space.dtype
    act_dtype = self._env.action_space.dtype

    # Switch lists of [n_steps, n_envs] to [n_envs, n_steps]
    rollout_observations = np.array(rollout_observations, dtype=obs_dtype)\
        .swapaxes(1, 0).reshape(self._batch_obs_shape)
    rollout_actions = np.array(rollout_actions, dtype=act_dtype)
    rollout_actions = np.array(rollout_actions, dtype=act_dtype).swapaxes(1, 0)
    rollout_values = np.array(rollout_values).swapaxes(1, 0)
    rollout_dones = np.array(rollout_dones).swapaxes(1, 0)
    rollout_rewards = np.array(rollout_rewards).swapaxes(1, 0)

    rollout_returns = self._compute_future_returns(
        rollout_rewards, rollout_dones, last_values)

    rollout_actions = rollout_actions.flatten()
    rollout_values = rollout_values.reshape(-1, 1)
    rollout_returns = rollout_returns.reshape(-1, 1)

    return rollout_returns, rollout_actions, \
        rollout_observations, rollout_values

  def _compute_future_returns(self, rewards, dones, last_values):
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
      returns ([float]): List of returns from that timestep onwards
          $\sum_{h=t}^{T-1}r_h$ (dimension [n_steps, n_envs])
    '''
    returns = []

    for _, (rollout_rewards, rollout_dones, last_val) in enumerate(
        zip(rewards, dones, last_values)):
      rollout_rewards = rollout_rewards.tolist()
      rollout_dones = rollout_dones.tolist()

      running_sum = 0
      rollout_returns = []

      for i, (reward, done) in enumerate(
          zip(reversed(rollout_rewards), reversed(rollout_dones))):
        if done:
          running_sum = reward
        else:
          if i == 0:
            # If the last step in the rollout is not terminal, the unbiased
            # estimate of the return is the reward + discounted value of the
            # next state
            running_sum = reward + self._gamma * last_val
          else:
            running_sum = reward + self._gamma * running_sum

        rollout_returns.append(running_sum)

      rollout_returns = list(reversed(rollout_returns))
      returns.append(rollout_returns)

    returns = np.array(returns)

    return returns
