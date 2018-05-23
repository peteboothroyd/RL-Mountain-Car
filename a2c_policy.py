import tensorflow as tf
import numpy as np
import gym


class A2CPolicy(object):
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

  ENT_THRESHOL = 0.4

  def __init__(self, sess, obs_space, act_space, cnn, reg_coeff=0,
               initial_ent_coeff=0.05, initial_learning_rate=7e-4,
               batch_norm=False, decay_ent=True, adam=False):
    #TODO: Remove hardcoded decay_steps from decays
    lrt = tf.train.polynomial_decay(
        learning_rate=initial_learning_rate, end_learning_rate=1e-5,
        decay_steps=int(1e5),
        global_step=tf.train.get_or_create_global_step())
    tf.summary.scalar('lrt', lrt)

    if decay_ent:
      ent_coeff = tf.train.polynomial_decay(
          learning_rate=initial_ent_coeff, decay_steps=int(3e4),
          global_step=tf.train.get_or_create_global_step(),
          end_learning_rate=0.001)
    else:
      ent_coeff = initial_ent_coeff
    tf.summary.scalar('ent_coeff', ent_coeff)

    discrete = isinstance(act_space, gym.spaces.Discrete)

    act_dim = act_space.n if discrete else act_space.shape[0]
    print('act_dim', act_dim)

    with tf.name_scope('inputs'):
      adv, obs, is_training, act, ret = _create_input_placeholders(
          act_dim, obs_space, discrete, cnn)

      if batch_norm:
        normalised_adv = tf.layers.batch_normalization(
            inputs=adv, training=is_training, renorm=True)

        ret_batch_norm = tf.layers.BatchNormalization(renorm=True)
        normalised_ret = ret_batch_norm(ret, training=is_training)

        tf.summary.histogram('normalised_ret', normalised_ret)
        tf.summary.histogram('normalised_adv', normalised_adv)
      else:
        normalised_ret = ret
        normalised_adv = adv

    with tf.name_scope('action'):
      if cnn:
        hidden = _build_cnn(obs, is_training, 'hidden', batch_norm)
      else:
        hidden = _build_mlp(
            obs, is_training, 'hidden', activation=tf.nn.tanh, size=64)
      tf.summary.histogram('hidden_output', hidden)

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

      with tf.name_scope('generate_sample_action'):
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

    with tf.name_scope('critic'):
      critic_pred = tf.layers.dense(
          inputs=hidden, units=1, kernel_regularizer=tf.nn.l2_loss,
          kernel_initializer=tf.glorot_normal_initializer())

      # Rescale value predictions based on moments of returns
      if batch_norm:
        critic_pred_mean, critic_pred_var = tf.nn.moments(
            critic_pred, axes=[0])
        critic_pred = critic_pred - critic_pred_mean + \
            ret_batch_norm.moving_mean
        critic_pred = tf.sqrt(ret_batch_norm.moving_variance) * critic_pred \
            / (tf.sqrt(critic_pred_var)+1e-4)

        tf.summary.scalar('critic_pred_mean', tf.squeeze(critic_pred_mean))
        tf.summary.scalar('critic_pred_var', tf.squeeze(critic_pred_var))
        tf.summary.scalar('returns_mean', tf.squeeze(
            ret_batch_norm.moving_mean))
        tf.summary.scalar('returns_var', tf.squeeze(
            ret_batch_norm.moving_variance))
      tf.summary.histogram('critic_pred', critic_pred)

    with tf.name_scope('log_prob'):
      log_prob = dist.log_prob(act, name='log_prob')
      tf.summary.histogram('log_prob', log_prob)

    with tf.name_scope('entropy'):
      ent = tf.reduce_mean(dist.entropy())
      tf.summary.scalar('actor_entropy', ent)

    with tf.name_scope('loss'):
      # Minimising negative equivalent to maximising
      actor_pg_loss = tf.reduce_mean(-log_prob * normalised_adv, name='loss')
      tf.summary.scalar('actor_pg_loss', actor_pg_loss)

      actor_expl_loss = ent * ent_coeff
      tf.summary.scalar('actor_expl_loss', actor_expl_loss)
      # # Trouble seems to occur when the entropy of the distribution drops too low
      # # add a large loss if the entropy drops too low
      # ent_too_low_loss = tf.nn.relu(self.ENT_THRESHOL - ent) * 10.0
      # tf.summary.scalar('actor_entropy', ent)
      # actor_expl_loss += ent_too_low_loss

      actor_total_loss = actor_pg_loss - actor_expl_loss
      tf.summary.scalar('actor_total_loss', actor_total_loss)

      critic_loss = tf.reduce_mean(
          tf.square(critic_pred - normalised_ret))/2
      tf.summary.scalar('critic_loss', critic_loss)

      reg_loss = tf.losses.get_regularization_loss() * reg_coeff
      tf.summary.scalar('reg_loss', reg_loss)

      total_loss = actor_total_loss + 0.5 * critic_loss + reg_loss
      tf.summary.scalar('total_loss', total_loss)

    with tf.name_scope('train_network'):
      if adam:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lrt, epsilon=1e-5, beta2=0.99) #decay=0.99, epsilon=1e-5
      else:
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=lrt, decay=0.99, epsilon=1e-5)
      grads_and_vars = optimizer.compute_gradients(total_loss)
      grads_and_vars = _clip_by_global_norm(grads_and_vars)
      train_op = _train_with_batch_norm_update(optimizer, grads_and_vars)

    summaries = tf.summary.merge_all()

    def actor(observation):
      ''' Return the action output by the policy given the current
          parameterisation.

      # Params:
        observation: The observation input to the policy

      # Returns:
        actions: Sample actions
      '''

      feed_dict = {
          obs: observation,
          is_training: False
      }

      actions = sess.run(sample_act, feed_dict=feed_dict)

      return actions

    def critic(observations):
      ''' Predict the value for given observations.

      # Params
        observations: List of observed states

      # Returns
        values: The predicted values of the states
      '''

      feed_dict = {
          obs: observations,
          is_training: False
      }

      values = sess.run(critic_pred, feed_dict=feed_dict)

      return values

    def step(observations):
      ''' Output actions and values for observations

      # Params
        observations: List of observed states

      # Returns
        values: The predicted values of the states
      '''

      feed_dict = {
          obs: observations,
          is_training: False
      }

      actions, values = sess.run(
          [sample_act, critic_pred], feed_dict=feed_dict)

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
        expl_loss:            The actor exploration loss
        regularization_loss:  The regularization loss
        ent:                  The policy entropy
      '''
      advantages = returns - values

      feed_dict = {
          obs: observations,
          ret: returns,
          adv: advantages,
          act: actions,
          is_training: True
      }

      _, pg_loss, val_loss, exploration_loss, regularization_loss, entropy, summary = \
          sess.run([train_op, actor_pg_loss, critic_loss,
                    actor_expl_loss, reg_loss, ent, summaries],
                   feed_dict=feed_dict)

      return pg_loss, val_loss, exploration_loss, regularization_loss, entropy, summary


    def reset():
      ''' Reset the policy. '''
      sess.run(tf.global_variables_initializer())

    self.reset = reset
    self.actor = actor
    self.critic = critic
    self.train = train
    self.step = step

    self.reset()


def _generate_bounded_continuous_sample_action(dist, ac_space):
  bounded_sample = tf.nn.tanh(dist.sample(), name='sample_action')
  scaled_shifted_sample = bounded_sample \
      * (ac_space.high[0]-ac_space.low[0]) * 0.5 \
      + (ac_space.high[0]+ac_space.low[0]) * 0.5
  tf.summary.histogram('sample_action', scaled_shifted_sample)
  return scaled_shifted_sample


def _build_mlp(input_placeholder, is_training, scope, n_layers=2,
               size=64, activation=tf.nn.relu):
  with tf.variable_scope(scope):
    hidden = input_placeholder
    for i in range(n_layers):
      hidden = tf.layers.dense(
          inputs=hidden, units=size, activation=activation,
          name="dense_{}".format(i), use_bias=True,
          bias_initializer=tf.zeros_initializer(),
          kernel_initializer=tf.glorot_normal_initializer(),
          kernel_regularizer=tf.nn.l2_loss)

      tf.summary.histogram('dense{0}_activation'.format(i), hidden)

      hidden = tf.layers.batch_normalization(
          hidden, training=is_training, renorm=True)
      tf.summary.histogram('dense{0}_batch_norm'.format(i), hidden)

  return hidden


def _build_cnn(input_placeholder, is_training, scope, batch_norm):
  # The CNN architecture as described in the A3C Paper
  with tf.variable_scope(scope):
    if batch_norm:
      batch_norm_in = tf.layers.batch_normalization(
          inputs=input_placeholder, training=is_training, renorm=True)
    else:
      batch_norm_in = input_placeholder

    conv1 = tf.layers.conv2d(
        inputs=batch_norm_in,
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

    if batch_norm:
      batch_norm1 = tf.layers.batch_normalization(
          inputs=conv1, training=is_training, renorm=True)
      tf.summary.histogram('conv_batch_norm1', batch_norm1)
    else:
      batch_norm1 = conv1

    conv2 = tf.layers.conv2d(
        inputs=batch_norm1,
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

    if batch_norm:
      batch_norm2 = tf.layers.batch_normalization(
          inputs=conv2, training=is_training, renorm=True)
      tf.summary.histogram('conv_batch_norm2', batch_norm2)
    else:
      batch_norm2 = conv2

    flattened = tf.layers.flatten(batch_norm2)

    dense = tf.layers.dense(
        inputs=flattened, units=512, activation=tf.nn.relu, name="conv_fc",
        kernel_initializer=tf.glorot_normal_initializer(), use_bias=True,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=tf.nn.l2_loss)

    if batch_norm:
      out = tf.layers.batch_normalization(
          inputs=dense, training=is_training, renorm=True)
    else:
      out = dense

    return out


def _train_with_batch_norm_update(optimizer, grads_and_vars):
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = optimizer.apply_gradients(
        grads_and_vars,
        global_step=tf.train.get_or_create_global_step())
  return train_op


def _clip_by_global_norm(grads_and_vars, norm=0.5):
  grads, variables = zip(*grads_and_vars)
  clipped_grads, _ = tf.clip_by_global_norm(grads, norm)
  return zip(clipped_grads, variables)


def _create_input_placeholders(act_dim, obs_space, discrete, cnn):
  if discrete:
    act = tf.placeholder(dtype=tf.int32, shape=[None], name='act')
  else:
    act = tf.placeholder(
        dtype=tf.float32, shape=[None, act_dim], name='act')

  adv = tf.placeholder(
      dtype=tf.float32, shape=[None, 1], name='adv')

  obs = tf.placeholder(
      dtype=obs_space.dtype, shape=(None,)+obs_space.shape, name='obs')
  obs = tf.cast(obs, tf.float32) / 255.0
  ret = tf.placeholder(
      dtype=tf.float32, shape=[None, 1], name='returns')
  is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

  tf.summary.histogram('adv', adv)
  tf.summary.histogram('ret', ret)
  tf.summary.histogram('act', act)
  if cnn:
    tf.summary.image('obs', obs)
  else:
    tf.summary.histogram('obs', obs)

  return adv, obs, is_training, act, ret
