import tensorflow as tf


class A2CPolicy(object):
  def __init__(self, sess, ob_space, ac_space, reg_coeff, ent_coeff,
               critic_learning_rate, actor_learning_rate,
               use_actor_reg_loss, use_actor_expl_loss):
    '''Define the tensorflow graph to be used for the policy.'''
    ob_shape = (None,) + ob_space.shape
    ac_dim = ac_space.shape[0]

    with tf.name_scope('inputs'):
      returns, states, is_training, actions = self._create_input_placeholders(
          ac_dim, ob_shape)
      tf.summary.histogram('returns', returns)
      normalised_returns = self._normalise(returns)
      tf.summary.histogram('normalised_returns', normalised_returns)

    with tf.name_scope('actor'):
      with tf.name_scope('action_statistics'):
        actor_network_reg_collection = 'actor_network_regularization'
        hidden = self._build_mlp(states,
                                 is_training,
                                 'hidden_policy_network',
                                 activation=tf.nn.tanh,
                                 var_collection=actor_network_reg_collection)

        mean_layer = tf.layers.Dense(
            units=ac_dim,
            activation=None,
            kernel_initializer=tf.glorot_normal_initializer(),
            bias_initializer=tf.zeros_initializer(),
            name='mean')
        mean = mean_layer(hidden)
        tf.summary.histogram('mean', mean)

        self._add_weights_to_regularisation_collection(
            mean_layer, actor_network_reg_collection)

        # Standard deviation of the normal distribution over actions
        std_dev = tf.get_variable(
            name="std_dev", shape=[ac_dim],
            dtype=tf.float32, initializer=tf.ones_initializer())
        std_dev = tf.nn.softplus(std_dev) + 1e-4
        tf.summary.histogram('std_dev', std_dev)

      with tf.name_scope('generate_sample_action'):
        sample_action = self._generate_sample_action(states,
                                                     ac_dim,
                                                     std_dev,
                                                     mean,
                                                     ac_space)
        tf.summary.histogram('sample_action', sample_action)

      with tf.name_scope('loss'):
        dist = tf.contrib.distributions.MultivariateNormalDiag(
            loc=mean, scale_diag=std_dev)
        log_prob = dist.log_prob(actions, name='log_prob')
        # Minimising negative equivalent to maximising
        actor_loss = tf.reduce_mean(-log_prob * normalised_returns,
                                    name='loss')
        tf.summary.scalar('loss', actor_loss)

        actor_reg_losses = tf.get_collection(actor_network_reg_collection)
        actor_reg_loss = tf.identity(
            sum(tf.nn.l2_loss(reg_loss) for reg_loss in actor_reg_losses) * reg_coeff,
            name='reg_loss')
        tf.summary.scalar('reg_loss', actor_reg_loss)

        actor_expl_loss = tf.identity(tf.reduce_mean(dist.entropy()) * ent_coeff,
                                    name='expl_loss')
        tf.summary.scalar('expl_loss', actor_expl_loss)

        actor_total_loss = actor_loss

        # TODO: Experiment with including these losses
        if use_actor_expl_loss:
          actor_total_loss -= actor_expl_loss
        if use_actor_reg_loss:
          actor_total_loss += actor_reg_loss
        
        tf.summary.scalar('total_loss', actor_total_loss)

      with tf.name_scope('train_network'):
        actor_optimizer = tf.train.AdamOptimizer(actor_learning_rate)
        actor_grads_and_vars = actor_optimizer.compute_gradients(actor_total_loss)
        actor_grads_and_vars = self._clip_by_global_norm(actor_grads_and_vars)
        train_actor_op = self._train_with_batch_norm_update(
            actor_optimizer, actor_grads_and_vars)

    with tf.name_scope('critic'):
      with tf.name_scope('predict'):
        critic_network_reg_collection = 'critic_network_regularization'
        critic_hidden = self._build_mlp(
            states,
            is_training,
            'critic_hidden',
            activation=tf.nn.tanh,
            var_collection=critic_network_reg_collection)

        dense_layer = tf.layers.Dense(
            units=1,
            kernel_initializer=tf.glorot_normal_initializer())
        critic_pred = dense_layer(critic_hidden)
        tf.summary.histogram('critic_pred', critic_pred)

        self._add_weights_to_regularisation_collection(
            dense_layer, critic_network_reg_collection)

      with tf.name_scope('loss'):
        critic_loss = tf.nn.l2_loss(critic_pred - normalised_returns)
        tf.summary.scalar('loss', critic_loss)

        critic_reg_losses = tf.get_collection(critic_network_reg_collection)
        critic_reg_loss = tf.identity(
            sum(tf.nn.l2_loss(reg_loss) for reg_loss in critic_reg_losses) * reg_coeff,
            name='critic_reg_loss')
        tf.summary.scalar('critic_reg_loss', critic_reg_loss)

        critic_total_loss = critic_loss + critic_reg_loss
        tf.summary.scalar('critic_total_loss', critic_total_loss)

      with tf.name_scope('gradients'):
        critic_optimizer = tf.train.AdamOptimizer(critic_learning_rate)
        critic_grads_and_vars = critic_optimizer.compute_gradients(critic_total_loss)
        critic_grads_and_vars = self._clip_by_global_norm(critic_grads_and_vars)
        train_critic_op = self._train_with_batch_norm_update(
            critic_optimizer, critic_grads_and_vars)

    summaries = tf.summary.merge_all()

    def actor(obs):
      ''' Return the action output by the policy given the current
          parameterisation.

      # Params:
        observation: The observation input to the policy

      # Returns:
        a: Action
      '''
      feed_dict = {states: obs, is_training: False}
      return sess.run(sample_action, feed_dict=feed_dict)

    def critic(obs):
      ''' Predict the value for given states. '''
      feed_dict = {states: obs, is_training: False}
      return sess.run(critic_pred, feed_dict=feed_dict)

    def train_actor(advs, acs, obs):
      ''' Train the policy. Return loss. '''
      feed_dict = {
          states: obs,
          returns: advs,
          actions: acs,
          is_training: True
      }

      _, actor_loss = sess.run(
          [train_actor_op, actor_total_loss], feed_dict=feed_dict)
      return actor_loss

    def train_critic(advs, obs):
      ''' Train the value function. Return loss. '''
      feed_dict = {
          states: obs,
          returns: advs,
          is_training: False
      }

      _, critic_loss = sess.run(
          [train_critic_op, critic_total_loss], feed_dict=feed_dict)
      return critic_loss

    def summarize(advs, acs, obs):
      '''Summarize key stats for TensorBoard. '''
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()

      feed_dict = {
          states: obs,
          returns: advs,
          actions: acs,
          is_training: False
      }

      _, _, summary = sess.run(
          [train_critic_op, train_actor_op, summaries],
          feed_dict=feed_dict,
          options=run_options, run_metadata=run_metadata)

      return summary, run_metadata

    def reset():
      ''' Reset the policy. '''
      sess.run(tf.global_variables_initializer())

    self.reset = reset
    self.actor = actor
    self.summarize = summarize
    self.critic = critic
    self.train_critic = train_critic
    self.train_actor = train_actor

    self.reset()
  
  def _generate_sample_action(self, states, ac_dim, std_dev, mean, ac_space):
    standard_normal = tf.random_normal(
        shape=(tf.shape(states)[0], ac_dim),
        name='standard_normal')
    bounded_sample = tf.nn.tanh(
        std_dev * standard_normal + mean, name='sample_action')
    scaled_shifted_sample = bounded_sample \
        * (ac_space.high[0]-ac_space.low[0]) * 0.5 \
        + (ac_space.high[0]+ac_space.low[0]) * 0.5
    tf.summary.histogram('sample_action', scaled_shifted_sample)
    return scaled_shifted_sample

  def _build_mlp(
      self,
      input_placeholder,
      is_training,
      scope,
      n_layers=2,
      size=64,
      activation=tf.tanh,
      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
      var_collection=None
  ):

    with tf.variable_scope(scope):
      output = input_placeholder
      for i in range(n_layers):
        layer = tf.layers.Dense(
            units=size, activation=activation,
            kernel_initializer=tf.glorot_normal_initializer(),
            kernel_regularizer=kernel_regularizer,
            name="dense_{}".format(i))
        output = layer(output)
        tf.summary.histogram('dense{0}_activation'.format(i), output)
        output = tf.layers.batch_normalization(output, training=is_training)
        tf.summary.histogram('dense{0}_batch_norm'.format(i), output)

        if var_collection is not None:
          self._add_weights_to_regularisation_collection(layer, var_collection)

    return output

  def _train_with_batch_norm_update(self, optimizer, grads_and_vars):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.apply_gradients(
          grads_and_vars,
          global_step=tf.train.get_or_create_global_step())
    return train_op

  def _clip_by_global_norm(self, grads_and_vars, norm=5.0):
    grads, variables = zip(*grads_and_vars)
    clipped_grads, _ = tf.clip_by_global_norm(grads, norm)
    return zip(clipped_grads, variables)

  def _normalise(self, x):
    mean, var = tf.nn.moments(x, axes=[0])
    normalised_x = tf.nn.batch_normalization(
        x,
        mean=mean,
        variance=var,
        offset=None,
        scale=None,
        variance_epsilon=1e-4)
    return normalised_x

  def _create_input_placeholders(self, ac_dim, ob_shape):
    actions = tf.placeholder(
        tf.float32, shape=[None, ac_dim], name='actions')
    returns = tf.placeholder(
        tf.float32, shape=[None], name='returns')
    states = tf.placeholder(
        tf.float32, shape=ob_shape, name='obs')
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    return returns, states, is_training, actions

  def _add_weights_to_regularisation_collection(self,
                                                layer,
                                                reg_collection):
    weights = layer.trainable_variables
    for weight in weights:
      if 'bias' not in weight.name:
        # Don't want to regularise the biases
        tf.add_to_collection(reg_collection, weight)

# Eligibility trace
# for grad, var in pol_grads_and_vars:
#   if grad is not None:
#     trace_name = var.name + '/trace'
#     trace = tf.get_variable(name=trace_name,
#                             trainable=False,
#                             shape=grad.get_shape(),
#                             initializer=tf.zeros_initializer())
#     grad = gamma * lmbda * trace + grad
#     tf.summary.histogram(trace_name, trace)
