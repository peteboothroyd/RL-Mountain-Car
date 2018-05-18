import tensorflow as tf
import numpy as np
import gym

ACTOR_SCOPE = 'actor'
CRITIC_SCOPE = 'critic'


class A2CPolicy(object):
  '''The policy for the A2C algorithm. Can deal with discrete and continuous
     action spaces.

     Note abbreviations:
      act - actions
      obs - observations
      adv - advantages
      val - values
  '''

  def __init__(self, sess, obs_space, act_space, reg_coeff, ent_coeff,
               critic_learning_rate, actor_learning_rate,
               use_actor_reg_loss, use_actor_expl_loss, cnn):

    discrete = isinstance(act_space, gym.spaces.Discrete)

    obs_shape = obs_space.shape
    act_dim = act_space.n if discrete else act_space.shape[0]

    with tf.name_scope('inputs'):
      adv, obs, is_training, act = _create_input_placeholders(
          act_dim, obs_shape, discrete)

      normalised_adv = tf.layers.batch_normalization(
          inputs=adv, training=is_training, renorm=True)
      tf.summary.histogram('advantages', normalised_adv)

    with tf.variable_scope(ACTOR_SCOPE):
      with tf.name_scope('action'):
        if cnn:
          actor_hidden = _build_cnn(obs, is_training, 'actor_hidden')
        else:
          actor_hidden = _build_mlp(
              obs, is_training, 'actor_hidden', activation=tf.nn.tanh, size=64)

        if discrete:
          act_logits = tf.layers.dense(
              inputs=actor_hidden, units=act_dim, activation=tf.nn.relu,
              use_bias=True, bias_initializer=tf.zeros_initializer(),
              kernel_initializer=tf.glorot_normal_initializer(),
              kernel_regularizer=tf.nn.l2_loss)
          tf.summary.histogram('act_logits', act_logits)
        else:
          act_mean = tf.layers.dense(
              inputs=actor_hidden, units=act_dim, activation=None,
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
                logits=act_logits, validate_args=True, name='categorical_dist')
            sample_act = dist.sample()
          else:
            dist = tf.contrib.distributions.MultivariateNormalDiag(
                loc=act_mean, scale_diag=act_std_dev,
                name='multivariate_gaussian_dist')
            sample_act = _generate_bounded_continuous_sample_action(
                dist, act_space)

          tf.summary.histogram('sample_action', sample_act)

      with tf.name_scope('log_prob'):
        if discrete:
          log_prob = dist.log_prob(act, name='categorical_log_prob')
        else:
          log_prob = dist.log_prob(act, name='multivariate_gaussian_log_prob')

      with tf.name_scope('loss'):
        # Minimising negative equivalent to maximising
        actor_loss = tf.reduce_mean(-log_prob * normalised_adv, name='loss')
        tf.summary.scalar('actor_loss', actor_loss)

        actor_reg_loss = tf.losses.get_regularization_loss(scope=ACTOR_SCOPE) \
            * reg_coeff
        tf.summary.scalar('actor_reg_loss', actor_reg_loss)

        actor_expl_loss = tf.reduce_mean(dist.entropy()) * ent_coeff
        tf.summary.scalar('actor_expl_loss', actor_expl_loss)

        actor_total_loss = actor_loss

        if use_actor_expl_loss:
          actor_total_loss -= actor_expl_loss
        if use_actor_reg_loss:
          actor_total_loss += actor_reg_loss

        tf.summary.scalar('actor_total_loss', actor_total_loss)

      with tf.name_scope('train_network'):
        actor_optimizer = tf.train.AdamOptimizer(actor_learning_rate)
        actor_grads_and_vars = actor_optimizer.compute_gradients(
            actor_total_loss)
        actor_grads_and_vars = _clip_by_global_norm(actor_grads_and_vars)
        train_actor_op = _train_with_batch_norm_update(
            actor_optimizer, actor_grads_and_vars)

    with tf.variable_scope(CRITIC_SCOPE):
      with tf.name_scope('predict'):
        val = tf.placeholder(
            dtype=tf.float32, shape=[None, 1], name='values_placeholder')

        val_batch_norm = tf.layers.BatchNormalization(renorm=True)
        normalised_val = val_batch_norm(val, training=is_training)

        if cnn:
          critic_hidden = _build_cnn(obs, is_training, 'critic_hidden')
        else:
          critic_hidden = _build_mlp(
              obs, is_training, 'critic_hidden', activation=tf.nn.tanh, size=64)

        critic_pred = tf.layers.dense(
            inputs=critic_hidden, units=1, kernel_regularizer=tf.nn.l2_loss,
            kernel_initializer=tf.glorot_normal_initializer())

        # Rescale value predictions based on calculated moments of returns
        critic_pred_mu, critic_pred_var = tf.nn.moments(critic_pred, axes=[0])
        critic_pred = critic_pred - critic_pred_mu + val_batch_norm.moving_mean
        critic_pred = tf.sqrt(val_batch_norm.moving_variance) * critic_pred \
            / (tf.sqrt(critic_pred_var)+1e-4)

        tf.summary.histogram('critic_pred', critic_pred)

      with tf.name_scope('critic_loss'):
        critic_loss = tf.reduce_mean(tf.squared_difference(
            critic_pred, normalised_val))
        tf.summary.scalar('critic_loss', critic_loss)

        critic_reg_loss = tf.losses.get_regularization_loss(scope=CRITIC_SCOPE)\
            * reg_coeff
        tf.summary.scalar('critic_reg_loss', critic_reg_loss)

        critic_total_loss = critic_loss + critic_reg_loss
        tf.summary.scalar('critic_total_loss', critic_total_loss)

      with tf.name_scope('critic_gradients'):
        critic_optimizer = tf.train.AdamOptimizer(critic_learning_rate)
        critic_grads_and_vars = critic_optimizer.compute_gradients(
            critic_total_loss)
        critic_grads_and_vars = _clip_by_global_norm(critic_grads_and_vars)
        train_critic_op = _train_with_batch_norm_update(
            critic_optimizer, critic_grads_and_vars)

    summaries = tf.summary.merge_all()

    def actor(observation):
      ''' Return the action output by the policy given the current
          parameterisation.

      # Params:
        observation: The observation input to the policy

      # Returns:
        act: Action
      '''

      feed_dict = {
          obs: observation,
          is_training: False
      }

      act = sess.run(sample_act, feed_dict=feed_dict)

      return act

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

    def train(value_targets, observations, advantages, actions):
      ''' Train the value function and policy.

      # Params
        value_targets: List of observed returns
        observations:  List of observed states
        advantages:   List of advantages
        actions:      List of actions taken

      # Returns
        critic_loss: The loss for the critic predictions
        actor_loss: The loss for the actor
      '''

      feed_dict = {
          obs: observations,
          val: value_targets,
          adv: advantages,
          act: actions,
          is_training: True
      }

      _, critic_loss, _, actor_loss = sess.run(
          [train_critic_op, critic_total_loss,
           train_actor_op, actor_total_loss], feed_dict=feed_dict)

      return critic_loss, actor_loss

    def summarize(value_targets, observations, advantages, actions):
      ''' Summarize key stats for TensorBoard.

      # Params:
        advantages:     List of advantages from a rollout
        actions:        List of executed actions from a rollout
        observations:   List of observed states from a rollout
        value_targets:  List of returns from a rollout
      '''

      feed_dict = {
          obs: observations,
          val: value_targets,
          adv: advantages,
          act: actions,
          is_training: False
      }

      _, _, summary = sess.run(
          [train_critic_op, train_actor_op, summaries], feed_dict=feed_dict)

      return summary

    def reset():
      ''' Reset the policy. '''
      sess.run(tf.global_variables_initializer())

    self.reset = reset
    self.actor = actor
    self.summarize = summarize
    self.critic = critic
    self.train = train

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


def _build_cnn(input_placeholder, is_training, scope, n_layers=3):
  with tf.variable_scope(scope):
    batch_norm = tf.layers.batch_normalization(
        inputs=input_placeholder, training=is_training, renorm=True)

    for i in range(n_layers):
      conv = tf.layers.conv2d(
          inputs=batch_norm,
          filters=64,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.glorot_normal_initializer(),
          use_bias=True,
          bias_initializer=tf.zeros_initializer(),
          data_format='channels_last',
          name='conv_{0}'.format(i),
          kernel_regularizer=tf.nn.l2_loss)
      pool = tf.layers.max_pooling2d(
          inputs=conv, pool_size=[2, 2], strides=2,
          name='conv_maxpool{0}'.format(i))
      batch_norm = tf.layers.batch_normalization(
          inputs=pool, training=is_training, renorm=True)

    flattened = tf.layers.flatten(batch_norm)
    dense = tf.layers.dense(
        inputs=flattened, units=512, activation=tf.nn.relu, name="conv_fc",
        kernel_initializer=tf.glorot_normal_initializer(), use_bias=True,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=tf.nn.l2_loss)
    out = tf.layers.batch_normalization(
        inputs=dense, training=is_training, renorm=True)

    return out


def _train_with_batch_norm_update(optimizer, grads_and_vars):
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = optimizer.apply_gradients(
        grads_and_vars,
        global_step=tf.train.get_or_create_global_step())
  return train_op


def _clip_by_global_norm(grads_and_vars, norm=5.0):
  grads, variables = zip(*grads_and_vars)
  clipped_grads, _ = tf.clip_by_global_norm(grads, norm)
  return zip(clipped_grads, variables)


def _create_input_placeholders(act_dim, obs_shape, discrete):
  if discrete:
    act = tf.placeholder(dtype=tf.int32, shape=[None], name='act')
  else:
    act = tf.placeholder(
        dtype=tf.float32, shape=[None, act_dim], name='act')

  adv = tf.placeholder(
      dtype=tf.float32, shape=[None, 1], name='adv')
  obs = tf.placeholder(
      dtype=tf.float32, shape=(None,)+obs_shape, name='obs')
  is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

  return adv, obs, is_training, act
