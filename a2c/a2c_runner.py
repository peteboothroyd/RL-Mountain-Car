import numpy as np

class A2CRunner(object):
  ''' Handles executing the policy and returning trajectories of states,
      returns, and actions for the a2c algorithm.

      # Params
        actor_critic: A parameterised actor_critic which given an observation
            will return a value estimate and sampled actions.
        env: An environment descended from the VectorEnv defined in OpenAI
            baselines.
        n_steps (int): The number of steps to rollout at each interval
        gamma (float): The discounting factor.
        discrete (bool): Flag set for discrete action spaces.
  '''
  def __init__(self, actor_critic, env, n_steps, gamma, discrete):
    self._actor_critic = actor_critic
    self._env = env
    self._n_steps = n_steps
    self._gamma = gamma
    self._discrete = discrete
    self._obs_dtype = self._env.observation_space.dtype
    self._act_dtype = self._env.action_space.dtype

    self._act_dim = 1 if discrete else env.action_space.shape[0]
    self._batch_obs_shape = (-1,) \
        + self._env.observation_space.shape

    # We must store the observations between calls to generate_rollouts
    self._obs = self._env.reset()

  def generate_rollouts(self):
    '''Generate rollouts for learning.

    # Note
      Only a finite number of steps are rolled out for each environment. Returns
      are bootstrapped using the critic value predictions.

    # Returns:
      rollout_returns (np.array([np.float32])): List of returns for the
          rollouts: dimension (-1,)
      rollout_actions (np.array([env.action_space.dtype]): List of actions for
          the rollouts: dimension (-1, act_dim)
      rollout_observations ([env.observation_space.dtype]): List of observations
          for the rollouts: dimension (-1, obs_dim)
      rollout_values ([np.float32]): List of observations for the rollouts:
          dimension (-1,)
    '''
    rewards, actions, observations, dones, values = [], [], [], [], []

    for _ in range(self._n_steps):
      # NOTE: In each iteration we are saving:
      # $x_t, v(x_t), a_t, r_{t+1}, done(x_{t+1})$
      observations.append(np.copy(self._obs))
      act, val = self._actor_critic.step(self._obs)
      values.append(val)
      actions.append(act)
      self._obs, rew, ds, _ = self._env.step(act)
      rewards.append(rew)
      dones.append(ds)

    # Store last values, $v(x_{n_steps+1})$ for bootstrapping the returns
    _, last_values = self._actor_critic.step(self._obs)

    # Switch lists of [n_steps, n_envs] to [n_envs, n_steps]
    observations = np.array(observations, dtype=self._obs_dtype).swapaxes(1, 0)\
        .reshape(self._batch_obs_shape)
    actions = np.array(actions, dtype=self._act_dtype).swapaxes(1, 0)
    values = np.array(values, dtype=np.float32).swapaxes(1, 0)
    dones = np.array(dones, dtype=np.bool).swapaxes(1, 0)
    rewards = np.array(rewards, dtype=np.float32).swapaxes(1, 0)

    returns = self._compute_future_returns(rewards, dones, last_values)

    actions = actions.reshape(-1, self._act_dim)
    values = values.flatten()
    returns = returns.flatten()

    return returns, actions, observations, values

  def _compute_future_returns(self, rewards, dones, last_values):
    r'''Compute the future returns for a given rollout of immediate rewards.
        Used for GMDP algorithm. Each row contains the data for each
        environment.

    # Params:
      rewards ([[float]]): Immediate rewards at different time steps for
          multiple rollouts. Each row represents a single environment.
          (dimension [n_envs, n_steps])
      dones: ([[bool]]): Done flags for each step. Each row represents a single
          environment. (dimension [n_envs, n_steps])
      last_values: ([[float]]): Values of last states in rollout. Each row
          represents a single environment. (dimension [n_envs, n_steps])

    # Returns:
      returns ([float]): Returns from that timestep onwards
          $\sum_{h=t}^{T-1}\gamma^{h-t}r_h$ (dimension [n_envs, n_steps]). Each
          row represents a single environment.
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

    returns = np.array(returns, dtype=np.float32)

    return returns
