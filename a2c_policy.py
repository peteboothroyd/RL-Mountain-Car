import tensorflow as tf
import numpy as np
import gym

ACTOR_NETWORK_REGULARIZATION_COLLECTION = 'actor_network_regularization'
CRITIC_NETWORK_REGULARIZATION_COLLECTION = 'critic_network_regularization'


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

      tf.summary.histogram('advantages', adv)
      adv = _normalise(adv)
      tf.summary.histogram('normalised_advantages', adv)

    with tf.name_scope('actor'):
      with tf.name_scope('action'):
        if cnn:
          actor_hidden = _build_cnn(
              obs,
              'actor_hidden',
              var_collection=ACTOR_NETWORK_REGULARIZATION_COLLECTION)
        else:
          actor_hidden = _build_mlp(
              obs,
              is_training,
              'actor_hidden',
              activation=tf.nn.tanh,
              var_collection=ACTOR_NETWORK_REGULARIZATION_COLLECTION,
              size=64)

        if discrete:
          act_layer = tf.layers.Dense(
              units=act_dim, activation=tf.nn.relu, use_bias=True,
              bias_initializer=tf.zeros_initializer(),
              kernel_initializer=tf.glorot_normal_initializer())
          act_logits = act_layer(actor_hidden)
          tf.summary.histogram('mean', act_logits)
        else:
          act_mean_layer = tf.layers.Dense(
              units=act_dim,
              activation=None,
              kernel_initializer=tf.glorot_normal_initializer(),
              bias_initializer=tf.zeros_initializer(),
              name='mean')
          act_mean = act_mean_layer(actor_hidden)
          tf.summary.histogram('mean', act_mean)

          _add_weights_to_regularisation_collection(
              act_mean_layer, ACTOR_NETWORK_REGULARIZATION_COLLECTION)

          # Standard deviation of the normal distribution over actions
          act_std_dev = tf.get_variable(
              name="act_std_dev", shape=[act_dim],
              dtype=tf.float32, initializer=tf.ones_initializer())
          act_std_dev = tf.nn.softplus(act_std_dev) + 1e-4
          tf.summary.scalar('act_std_dev', tf.squeeze(act_std_dev))

        with tf.name_scope('generate_sample_action'):
          if discrete:
            dist = tf.distributions.Categorical(logits=act_logits)
            sample_act = dist.sample()
          else:
            sample_act = _generate_continuous_sample_action(
                obs, act_dim, act_std_dev, act_mean, act_space)
          tf.summary.histogram('sample_action', sample_act)

      with tf.name_scope('log_prob'):
        if discrete:
          log_prob = dist.log_prob(act, name='categorical_log_prob')
        else:
          dist = tf.contrib.distributions.MultivariateNormalDiag(
              loc=act_mean, scale_diag=act_std_dev)
          log_prob = dist.log_prob(act, name='multivariate_gaussian_log_prob')

      with tf.name_scope('loss'):
        # Minimising negative equivalent to maximising
        actor_loss = tf.reduce_mean(-log_prob * adv, name='loss')
        tf.summary.scalar('loss', actor_loss)

        actor_weights = tf.get_collection(
            ACTOR_NETWORK_REGULARIZATION_COLLECTION)
        actor_reg_loss = tf.identity(
            sum(tf.nn.l2_loss(weight)
                for weight in actor_weights) * reg_coeff,
            name='actor_reg_loss')
        tf.summary.scalar('actor_reg_loss', actor_reg_loss)

        actor_expl_loss = tf.identity(
            tf.reduce_mean(dist.entropy()) * ent_coeff, name='actor_expl_loss')
        tf.summary.scalar('actor_expl_loss', actor_expl_loss)

        actor_total_loss = actor_loss

        # TODO: Experiment with including these losses
        if use_actor_expl_loss:
          actor_total_loss -= actor_expl_loss
        if use_actor_reg_loss:
          actor_total_loss += actor_reg_loss

        tf.summary.scalar('total_loss', actor_total_loss)

      with tf.name_scope('train_network'):
        actor_optimizer = tf.train.AdamOptimizer(actor_learning_rate)
        actor_grads_and_vars = actor_optimizer.compute_gradients(
            actor_total_loss)
        actor_grads_and_vars = _clip_by_global_norm(actor_grads_and_vars)
        train_actor_op = _train_with_batch_norm_update(
            actor_optimizer, actor_grads_and_vars)

    with tf.name_scope('critic'):
      with tf.name_scope('predict'):
        val = tf.placeholder(
            dtype=tf.float32, shape=[None], name='values_placeholder')
        val = _normalise(val)

        if cnn:
          critic_hidden = _build_cnn(
              obs,
              'critic_hidden',
              var_collection=CRITIC_NETWORK_REGULARIZATION_COLLECTION)
        else:
          critic_hidden = _build_mlp(
              obs, is_training, 'critic_hidden', activation=tf.nn.tanh,
              var_collection=CRITIC_NETWORK_REGULARIZATION_COLLECTION, size=64)

        dense_layer = tf.layers.Dense(
            units=1, kernel_initializer=tf.glorot_normal_initializer())
        critic_pred = dense_layer(critic_hidden)
        tf.summary.histogram('critic_pred', critic_pred)

        _add_weights_to_regularisation_collection(
            dense_layer, CRITIC_NETWORK_REGULARIZATION_COLLECTION)

      with tf.name_scope('critic_loss'):
        critic_loss = tf.nn.l2_loss(critic_pred - val)
        tf.summary.scalar('critic_loss', critic_loss)

        critic_weights = tf.get_collection(
            CRITIC_NETWORK_REGULARIZATION_COLLECTION)
        critic_reg_loss = tf.identity(
            sum(tf.nn.l2_loss(weight)
                for weight in critic_weights) * reg_coeff,
            name='critic_reg_loss')
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

      # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      # run_metadata = tf.RunMetadata()
      feed_dict = {
          obs: observations,
          val: value_targets,
          adv: advantages,
          act: actions,
          is_training: False
      }

      _, _, summary = sess.run(
          [train_critic_op, train_actor_op, summaries], feed_dict=feed_dict)
      # options=run_options, run_metadata=run_metadata

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


def _generate_continuous_sample_action(obs, ac_dim, std_dev, mean, ac_space):
  standard_normal = tf.random_normal(
      shape=(tf.shape(obs)[0], ac_dim),
      name='standard_normal')
  bounded_sample = tf.nn.tanh(
      std_dev * standard_normal + mean, name='sample_action')
  scaled_shifted_sample = bounded_sample \
      * (ac_space.high[0]-ac_space.low[0]) * 0.5 \
      + (ac_space.high[0]+ac_space.low[0]) * 0.5
  tf.summary.histogram('sample_action', scaled_shifted_sample)
  return scaled_shifted_sample


def _build_mlp(input_placeholder, is_training, scope, n_layers=2,
               size=64, activation=tf.nn.relu, var_collection=None):
  with tf.variable_scope(scope):
    output = input_placeholder
    for i in range(n_layers):
      layer = tf.layers.Dense(
          units=size, activation=activation, name="dense_{}".format(i),
          kernel_initializer=tf.glorot_normal_initializer(), use_bias=True,
          bias_initializer=tf.zeros_initializer())
      output = layer(output)

      tf.summary.histogram('dense{0}_activation'.format(i), output)

      output = tf.layers.batch_normalization(output, training=is_training)
      tf.summary.histogram('dense{0}_batch_norm'.format(i), output)

      if var_collection is not None:
        _add_weights_to_regularisation_collection(layer, var_collection)

  return output


def _build_cnn(input_placeholder, scope, n_layers=3, var_collection=None):
  with tf.variable_scope(scope):
    hidden = input_placeholder

    for i in range(n_layers):
      conv_layer = tf.layers.Conv2D(
          filters=64,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.glorot_normal_initializer(),
          use_bias=True,
          bias_initializer=tf.zeros_initializer(),
          data_format='channels_last',
          name='conv_conv{0}'.format(i))
      hidden = conv_layer(hidden)
      hidden = tf.layers.max_pooling2d(
          inputs=hidden, pool_size=[2, 2], strides=2,
          name='conv_maxpool{0}'.format(i))

      if var_collection is not None:
        _add_weights_to_regularisation_collection(conv_layer,
                                                  var_collection)
    hidden = flatten_conv(hidden)
    fc_layer = tf.layers.Dense(
        units=256, activation=tf.nn.relu, name="conv_fc",
        kernel_initializer=tf.glorot_normal_initializer(), use_bias=True,
        bias_initializer=tf.zeros_initializer())
    hidden = fc_layer(hidden)

    if var_collection is not None:
      _add_weights_to_regularisation_collection(fc_layer,
                                                var_collection)

    return hidden


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


def _normalise(x):
  mean, var = tf.nn.moments(x, axes=[0])
  normalised_x = tf.nn.batch_normalization(x, mean=mean, variance=var,
                                           offset=None, scale=None,
                                           variance_epsilon=1e-4)
  return normalised_x


def _create_input_placeholders(act_dim, obs_shape, discrete):
  if discrete:
    act = tf.placeholder(dtype=tf.int32, shape=[None], name='act')
  else:
    act = tf.placeholder(
        dtype=tf.float32, shape=[None, act_dim], name='act')

  adv = tf.placeholder(
      dtype=tf.float32, shape=[None], name='adv')
  obs = tf.placeholder(
      dtype=tf.float32, shape=(None,)+obs_shape, name='obs')
  is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

  return adv, obs, is_training, act


def _add_weights_to_regularisation_collection(layer, reg_collection):
  weights = layer.trainable_variables
  for weight in weights:
    # Generally don't want to regularise the biases
    if 'bias' not in weight.name:
      tf.add_to_collection(reg_collection, weight)


def flatten_conv(conv_output):
  ''' Flattens the output of a convolutional layer to feed into a
      dense layer.
  '''
  num_elems = np.prod([dim.value for dim in conv_output.get_shape()[1:]])
  flattened = tf.reshape(conv_output, [-1, num_elems])

  return flattened
