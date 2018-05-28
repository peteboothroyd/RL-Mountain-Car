import tensorflow as tf
import numpy as np
import gym

MAX_PIXEL_VALUE = 255.0


class ActorCritic(object):
  '''The policy for the A2C algorithm. Can deal with discrete and continuous
     action spaces.

     Note abbreviations:
      act - actions
      obs - observations
      adv - advantages
      val - values
      lrt - learning rate
      ret - returns
  '''

  def __init__(self, sess, obs_space, act_space, cnn, num_policy_updates,
               initial_ent_coeff=0.01, initial_learning_rate=7e-4,
               decay_ent=True):
    lrt_scheduler = Scheduler(initial_learning_rate, 0, num_policy_updates)
    if decay_ent:
      ent_scheduler = Scheduler(
          initial_ent_coeff, 0.005, num_policy_updates//2)
    else:
      ent_scheduler = Scheduler(initial_ent_coeff, initial_ent_coeff, 1)

    discrete = isinstance(act_space, gym.spaces.Discrete)

    act_dim = act_space.n if discrete else act_space.shape[0]

    with tf.name_scope('inputs'):
      adv, obs, act, ret, lrt, ent_coeff = _create_input_placeholders(
          act_dim, obs_space, discrete, cnn)

    with tf.name_scope('hidden'):
      if cnn:
        hidden = _build_cnn(obs)
      else:
        hidden = _build_mlp(obs)
      tf.summary.histogram('hidden_output', hidden)

    with tf.name_scope('actor'):
      if discrete:
        act_logits = tf.layers.dense(
            inputs=hidden, units=act_dim, activation=tf.nn.relu,
            use_bias=True, bias_initializer=tf.zeros_initializer(),
            kernel_initializer=tf.glorot_normal_initializer(),
            kernel_regularizer=tf.nn.l2_loss)
        tf.summary.histogram('act_logits', act_logits)
      else:
        act_mean = tf.layers.dense(
            inputs=hidden, units=act_dim, activation=None,
            kernel_initializer=tf.glorot_normal_initializer(),
            bias_initializer=tf.zeros_initializer(), name='mean')
        tf.summary.histogram('mean', act_mean)

        # Standard deviation of the normal distribution over actions
        act_std_dev = tf.get_variable(
            name="act_std_dev", shape=[act_dim],
            dtype=tf.float32, initializer=tf.ones_initializer())
        act_std_dev = tf.nn.softplus(act_std_dev) + 1e-4
        tf.summary.scalar('act_std_dev', tf.squeeze(act_std_dev))

      with tf.name_scope('sample_action'):
        if discrete:
          dist = tf.distributions.Categorical(
              logits=act_logits, name='categorical_dist')
          sample_act = dist.sample()
        else:
          dist = tf.contrib.distributions.MultivariateNormalDiag(
              loc=act_mean, scale_diag=act_std_dev,
              name='multivariate_gaussian_dist')
          sample_act = _generate_bounded_continuous_sample_action(
              dist, act_space)

        tf.summary.histogram('sample_action', sample_act)

      with tf.name_scope('log_prob'):
        log_prob = dist.log_prob(act, name='log_prob')
        tf.summary.histogram('log_prob', log_prob)

      with tf.name_scope('entropy'):
        ent = tf.reduce_mean(dist.entropy())
        tf.summary.scalar('actor_entropy', ent)

    with tf.name_scope('critic'):
      critic_prediction = tf.squeeze(tf.layers.dense(
          inputs=hidden, units=1, kernel_regularizer=tf.nn.l2_loss,
          kernel_initializer=tf.glorot_normal_initializer()))
      tf.summary.histogram('critic_prediction', critic_prediction)

    with tf.name_scope('loss'):
      # Minimising negative equivalent to maximising
      actor_pg_loss = tf.reduce_mean(-log_prob * adv, name='loss')
      tf.summary.scalar('actor_pg_loss', actor_pg_loss)

      actor_explore_loss = ent * ent_coeff
      tf.summary.scalar('actor_explore_loss', actor_explore_loss)

      actor_total_loss = actor_pg_loss - actor_explore_loss
      tf.summary.scalar('actor_total_loss', actor_total_loss)

      critic_loss = tf.reduce_mean(
          tf.square(critic_prediction - ret))/2
      tf.summary.scalar('critic_loss', critic_loss)

      total_loss = actor_total_loss + 0.5 * critic_loss
      tf.summary.scalar('total_loss', total_loss)

    with tf.name_scope('train'):
      optimizer = tf.train.AdamOptimizer(learning_rate=lrt, epsilon=1e-5)
      grads_and_vars = optimizer.compute_gradients(total_loss)
      grads_and_vars = _clip_by_global_norm(grads_and_vars)
      train_op = optimizer.apply_gradients(
          grads_and_vars, global_step=tf.train.get_or_create_global_step())

    summaries = tf.summary.merge_all()

    def step(observations):
      ''' Output actions and values for observations.

      # Params
        observations: List of observed states

      # Returns
        values: The predicted values of the states
      '''
      feed_dict = {obs: observations}
      actions, values = sess.run(
          [sample_act, critic_prediction], feed_dict=feed_dict)
      return actions, values

    def train(observations, returns, actions, values):
      ''' Train the value function and policy.

      # Params
        observations:   List of observed states
        returns:        List of observed returns
        actions:        List of actions taken
        values:         List of values

      # Returns
        pg_loss:              The policy gradient loss
        val_loss:             The critic loss
        explore_loss:         The actor exploration loss
        regularization_loss:  The regularization loss
        ent:                  The policy entropy
      '''
      advantages = returns - values

      global_step = tf.train.get_or_create_global_step().eval(session=sess)
      current_lrt = lrt_scheduler.current_value(global_step)
      current_ent_coeff = ent_scheduler.current_value(global_step)

      feed_dict = {
          obs: observations,
          ret: returns,
          adv: advantages,
          act: actions,
          lrt: current_lrt,
          ent_coeff: current_ent_coeff
      }

      _, pg_loss, val_loss, expl_loss, entropy, summary = \
          sess.run([train_op, actor_pg_loss, critic_loss,
                    actor_explore_loss, ent, summaries],
                   feed_dict=feed_dict)

      return pg_loss, val_loss, expl_loss, entropy, summary

    def reset():
      ''' Reset the policy. '''
      sess.run(tf.global_variables_initializer())

    self.reset = reset
    self.train = train
    self.step = step

    self.reset()


def _generate_bounded_continuous_sample_action(dist, ac_space):
  # TODO: This currently is only suitable for scalar actions. Generalise for
  #      arbitrary dimensions.
  bounded_sample = tf.nn.tanh(dist.sample(), name='sample_action')
  scaled_shifted_sample = bounded_sample \
      * (ac_space.high[0]-ac_space.low[0]) * 0.5 \
      + (ac_space.high[0]+ac_space.low[0]) * 0.5
  tf.summary.histogram('sample_action', scaled_shifted_sample)
  return scaled_shifted_sample


def _build_mlp(input_placeholder, n_layers=2,
               size=64, activation=tf.nn.relu):
  hidden = input_placeholder
  for i in range(n_layers):
    hidden = tf.layers.dense(
        inputs=hidden, units=size, activation=activation,
        name="dense_{}".format(i), use_bias=True,
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.glorot_normal_initializer(),
        kernel_regularizer=tf.nn.l2_loss)

    tf.summary.histogram('dense{0}_activation'.format(i), hidden)

  return hidden


def _build_cnn(input_placeholder):
  # The CNN architecture as described in the A3C Paper
  scaled_obs = tf.cast(input_placeholder, tf.float32) / MAX_PIXEL_VALUE

  conv1 = tf.layers.conv2d(
      inputs=scaled_obs,
      filters=16,
      kernel_size=[8, 8],
      strides=[4, 4],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.glorot_normal_initializer(),
      use_bias=True,
      bias_initializer=tf.zeros_initializer(),
      data_format='channels_last',
      name='conv_1',
      kernel_regularizer=tf.nn.l2_loss)
  tf.summary.histogram('conv_1', conv1)

  conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=32,
      kernel_size=[4, 4],
      strides=[2, 2],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.glorot_normal_initializer(),
      use_bias=True,
      bias_initializer=tf.zeros_initializer(),
      data_format='channels_last',
      name='conv_2',
      kernel_regularizer=tf.nn.l2_loss)
  tf.summary.histogram('conv_2', conv2)

  flattened = tf.layers.flatten(conv2)

  dense = tf.layers.dense(
      inputs=flattened, units=256, activation=tf.nn.relu, name="conv_fc",
      kernel_initializer=tf.glorot_normal_initializer(), use_bias=True,
      bias_initializer=tf.zeros_initializer(),
      kernel_regularizer=tf.nn.l2_loss)

  return dense


def _clip_by_global_norm(grads_and_vars, norm=1.0):
  grads, variables = zip(*grads_and_vars)
  clipped_grads, _ = tf.clip_by_global_norm(grads, norm)
  return zip(clipped_grads, variables)


def _create_input_placeholders(act_dim, obs_space, discrete, cnn):
  if discrete:
    act = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='act')
  else:
    act = tf.placeholder(
        dtype=tf.float32, shape=[None, act_dim], name='act')

  adv = tf.placeholder(
      dtype=tf.float32, shape=[None], name='adv')

  obs = tf.placeholder(
      dtype=obs_space.dtype, shape=(None,)+obs_space.shape, name='obs')
  ret = tf.placeholder(
      dtype=tf.float32, shape=[None], name='returns')

  lrt = tf.placeholder(tf.float32, shape=[], name='learning_rate')
  ent_coeff = tf.placeholder(tf.float32, shape=[], name='entropy_coefficient')

  tf.summary.scalar('lrt', lrt)
  tf.summary.scalar('ent_coeff', ent_coeff)

  tf.summary.histogram('adv', adv)
  tf.summary.histogram('ret', ret)
  tf.summary.histogram('act', act)

  if cnn:
    tf.summary.image('obs', obs)
  else:
    tf.summary.histogram('obs', obs)

  return adv, obs, act, ret, lrt, ent_coeff


class Scheduler(object):
  # Creates linear decay schedule from init_val to final_val in n_steps
  def __init__(self, init_val, final_val, n_steps):
    self._init_val = init_val
    self._final_val = final_val
    self._n_steps = n_steps

  def current_value(self, train_step):
    return self._init_val - min(train_step, self._n_steps) \
        * (self._init_val-self._final_val) / self._n_steps
