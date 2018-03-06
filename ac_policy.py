import tensorflow as tf


class AcPolicy(object):
  def __init__(self, sess, ob_space, ac_space, beta, entropy_weight,
               val_learning_rate=1e-3, pol_learning_rate=1e-4,
               use_pol_reg_loss=False, use_pol_expl_loss=False):
    '''Define the tensorflow graph to be used for the policy.'''
    ob_shape = (None,) + ob_space.shape
    ac_dim = ac_space.shape[0]

    with tf.name_scope('inputs'):
      returns, states, is_training, actions = self._create_input_placeholders(
          ac_dim, ob_shape)
      normalised_returns = self._normalise(returns)

    with tf.name_scope('actor'):
      with tf.name_scope('action_statistics'):
        pol_network_reg_collection = 'pol_network_regularization'
        hidden = self._build_mlp(states,
                                 'hidden_policy_network',
                                 activation=tf.nn.tanh,
                                 var_collection=pol_network_reg_collection,
                                 is_training=is_training)

        mean_layer = tf.layers.Dense(
            units=ac_dim,
            activation=None,
            kernel_initializer=tf.glorot_normal_initializer(),
            bias_initializer=tf.zeros_initializer(),
            name='mean')
        mean = mean_layer(hidden)

        self._add_weights_to_regularisation_collection(
            mean_layer, pol_network_reg_collection)

        # Standard deviation of the normal distribution over actions
        std_dev = tf.get_variable(
            name="std_dev", shape=[ac_dim],
            dtype=tf.float32, initializer=tf.ones_initializer())
        std_dev = tf.nn.softplus(std_dev) + 1e-4

        tf.summary.histogram('mean', mean)
        tf.summary.histogram('std_dev', std_dev)

      with tf.name_scope('generate_sample_action'):
        sample_action = self._generate_sample_action(states,
                                                     ac_dim,
                                                     std_dev,
                                                     mean,
                                                     ac_space)

      with tf.name_scope('loss'):
        dist = tf.contrib.distributions.MultivariateNormalDiag(
            loc=mean, scale_diag=std_dev)
        log_prob = dist.log_prob(actions, name='log_prob')
        pol_loss = tf.reduce_mean(-log_prob * normalised_returns,
                                  name='loss')

        pol_reg_losses = tf.get_collection(pol_network_reg_collection)
        pol_reg_loss = tf.identity(
            sum(tf.nn.l2_loss(reg_loss) for reg_loss in pol_reg_losses) * beta,
            name='pol_reg_loss')

        pol_expl_loss = tf.identity(tf.reduce_mean(
            dist.entropy() * entropy_weight), name='pol_expl_loss')

        pol_total_loss = pol_loss

        # TODO: Experiment with including these losses
        if use_pol_expl_loss:
          pol_total_loss -= pol_expl_loss
        if use_pol_reg_loss:
          pol_total_loss += pol_reg_loss

        tf.summary.scalar('loss', pol_loss)
        tf.summary.scalar('expl_loss', pol_expl_loss)
        tf.summary.scalar('reg_loss', pol_reg_loss)
        tf.summary.scalar('total_loss', pol_total_loss)

      # policy training update
      with tf.name_scope('train_policy_network'):
        pol_optimizer = tf.train.AdamOptimizer(pol_learning_rate)
        pol_grads_and_vars = pol_optimizer.compute_gradients(pol_total_loss)
        pol_grads_and_vars = self._clip_by_global_norm(pol_grads_and_vars)
        train_pol_op = self._train_with_batch_norm_update(
            pol_optimizer, pol_grads_and_vars)

    with tf.name_scope('critic'):
      with tf.name_scope('predict'):
        val_network_reg_collection = 'val_network_regularization'
        val_pred_hidden = self._build_mlp(
            states,
            'hidden_value_network',
            activation=tf.nn.tanh,
            var_collection=val_network_reg_collection,
            is_training=is_training)

        layer = tf.layers.Dense(
            units=1,
            kernel_initializer=tf.glorot_normal_initializer())
        val_pred = layer(val_pred_hidden)
        tf.summary.histogram('val_pred', val_pred)

        # Add weights to collection for regularization
        self._add_weights_to_regularisation_collection(
            layer, val_network_reg_collection)

      with tf.name_scope('loss'):
        val_loss = tf.nn.l2_loss(val_pred - normalised_returns)
        tf.summary.scalar('val_loss', val_loss)

        val_reg_losses = tf.get_collection(val_network_reg_collection)
        val_reg_loss = tf.identity(
            sum(tf.nn.l2_loss(reg_loss) for reg_loss in val_reg_losses) * beta,
            name='val_reg_loss')
        tf.summary.scalar('val_reg_loss', val_reg_loss)

        val_total_loss = val_loss + val_reg_loss
        tf.summary.scalar('val_total_loss', val_total_loss)

      with tf.name_scope('gradients'):
        val_optimizer = tf.train.AdamOptimizer(val_learning_rate)
        val_grads_and_vars = val_optimizer.compute_gradients(val_total_loss)
        val_grads_and_vars = self._clip_by_global_norm(val_grads_and_vars)
        train_val_op = self._train_with_batch_norm_update(
            val_optimizer, val_grads_and_vars)

    self._summaries = tf.summary.merge_all()

    self._sess = sess
    self._is_training = is_training
    self._actions = actions
    self._returns = returns
    self._states = states
    self._sample_action = sample_action
    self._pol_total_loss = pol_total_loss
    self._train_pol_op = train_pol_op
    self._val_pred = val_pred
    self._val_total_loss = val_total_loss
    self._train_val_op = train_val_op

    self.reset()

  def act(self, observation):
    ''' Return the action output by the policy given the current
        parameterisation.

    # Params:
      observation: The observation input to the policy

    # Returns:
      a: Action
    '''
    feed_dict = {self._states: observation, self._is_training: False}
    return self._sess.run(self._sample_action, feed_dict=feed_dict)

  def predict_val(self, states):
    ''' Predict the value for given states. '''
    feed_dict = {self._states: states, self._is_training: False}
    return self._sess.run(self._val_pred, feed_dict=feed_dict)

  def train_pol(self, returns, actions, states):
    ''' Train the policy. Return loss. '''
    feed_dict = {
        self._states: states,
        self._returns: returns,
        self._actions: actions,
        self._is_training: True
    }

    _, pol_loss = self._sess.run(
        [self._train_pol_op, self._pol_total_loss], feed_dict=feed_dict)
    return pol_loss

  def train_val(self, returns, states):
    ''' Train the value function. Return loss. '''
    feed_dict = {
        self._states: states,
        self._returns: returns,
        self._is_training: False
    }

    _, val_loss = self._sess.run(
        [self._train_val_op, self._val_total_loss], feed_dict=feed_dict)
    return val_loss

  def summarize(self, returns, actions, states):
    '''Summarize key stats for TensorBoard. '''
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    feed_dict = {
        self._states: states,
        self._returns: returns,
        self._actions: actions,
        self._is_training: False
    }

    _, summary = self._sess.run([self._train_pol_op, self._summaries],
                                feed_dict=feed_dict,
                                options=run_options, run_metadata=run_metadata)

    return summary, run_metadata

  def reset(self):
    ''' Reset the policy. '''
    self._sess.run(tf.global_variables_initializer())

  def _generate_sample_action(self, states, ac_dim, std_dev, mean, ac_space):
    standard_normal = tf.random_normal(
        shape=(tf.shape(states)[0], ac_dim),
        name='z')
    sample_action = tf.nn.tanh(
        std_dev * standard_normal + mean, name='sample_action')
    sample_action = sample_action * (ac_space.high[0]-ac_space.low[0]) * 0.5 \
        + (ac_space.high[0]+ac_space.low[0]) * 0.5
    tf.summary.histogram('sample_action', sample_action)
    return sample_action

  def _build_mlp(
      self,
      input_placeholder,
      scope,
      n_layers=2,
      size=64,
      activation=tf.tanh,
      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
      var_collection=None,
      is_training=True
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
        output = tf.layers.batch_normalization(output, training=is_training)

        if var_collection is not None:
          weights = layer.trainable_weights
          for weight in weights:
            tf.add_to_collection(var_collection, weight)

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
    normalised_x = tf.nn.batch_normalization(x,
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
                                                val_network_reg_collection):
    weights = layer.trainable_variables
    for weight in weights:
      if 'bias' not in weight.name:
        # Don't want to regularise the biases
        tf.add_to_collection(val_network_reg_collection, weight)

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
