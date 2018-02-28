import tensorflow as tf
import utils

class MlpPolicy(object):
  def __init__(self, sess, ob_space, action_space, beta, exploration,
               val_learning_rate=1e-2, pol_learning_rate=1e-3):
    '''Define the tensorflow graph to be used for the policy.'''
    self._sess = sess
    self._ob_space = ob_space
    self._action_space = action_space

    ob_shape = (None,) + ob_space.shape
    ac_dim = action_space.shape[0]

    with tf.name_scope('inputs'):
      self._actions = tf.placeholder(
          tf.float32, shape=[None, ac_dim], name='actions')
      self._returns = tf.placeholder(
          tf.float32, shape=[None], name='returns')
      # tf.summary.histogram('unnormalised_returns', self._returns)

      # Whiten returns
      # mean_returns, var_returns = tf.nn.moments(self._returns, axes=[0])
      # normalised_returns = tf.nn.batch_normalization(self._returns,
      #              mean=mean_returns,
      #              variance=var_returns,
      #              offset=None,
      #              scale=None,
      #              variance_epsilon=1e-4)
      # tf.summary.histogram('normalised_returns', normalised_returns)

    with tf.name_scope('policy'):
      with tf.name_scope('obs_input'):
        self._states = tf.placeholder(
            tf.float32, shape=ob_shape, name='obs')

      with tf.name_scope('action_statistics'):
        pol_network_reg_collection = 'pol_network_regularization'
        hidden = utils.build_mlp(self._states,
                                 'hidden_policy_network',
                                 activation=tf.nn.tanh,
                                 var_collection=pol_network_reg_collection)

        mean_layer = tf.layers.Dense(
            units=ac_dim,
            activation=None,
            kernel_initializer=tf.glorot_normal_initializer(),
            bias_initializer=tf.zeros_initializer(),
            name='mean')
        mean = mean_layer(hidden)

        # Add weights to collection for regularization
        mean_weights = mean_layer.trainable_variables
        for weight in mean_weights:
          if 'bias' not in weight.name:
            tf.add_to_collection(
                pol_network_reg_collection, weight)

        # mean = mean * (action_space.high[0]-action_space.low[0]) * 0.5 \
        #       + (action_space.high[0]+action_space.low[0]) * 0.5

        std_dev = tf.get_variable(name="std_dev", shape=[
            ac_dim], dtype=tf.float32, initializer=tf.ones_initializer())
        std_dev = tf.nn.softplus(std_dev) + 1e-4
        tf.add_to_collection(pol_network_reg_collection, std_dev)

        tf.summary.histogram('mean', mean)
        tf.summary.histogram('std_dev', std_dev)

      with tf.name_scope('predict_actions'):
        standard_normal = tf.random_normal(
            shape=(tf.shape(self._states)[0], ac_dim),
            name='z')
        self._sample_action = tf.identity(
            std_dev * standard_normal + mean, name='sample_action')
        self._sample_action = tf.clip_by_value(
            self._sample_action, action_space.low[0], action_space.high[0])
        tf.summary.histogram('sample_action', self._sample_action)

      with tf.name_scope('loss'):
        dist = tf.contrib.distributions.MultivariateNormalDiag(
            loc=mean, scale_diag=std_dev)
        log_prob = dist.log_prob(self._actions, name='log_prob')
        pol_loss = tf.reduce_sum(-log_prob *
                                 self._returns, name='loss')

        pol_reg_losses = tf.get_collection(pol_network_reg_collection)
        pol_reg_loss = tf.identity(sum(tf.nn.l2_loss(
            reg_loss) for reg_loss in pol_reg_losses) * beta,
            name='pol_reg_loss')

        pol_expl_loss = tf.identity(tf.reduce_mean(
            dist.entropy()) * exploration, name='pol_expl_loss')
        self._pol_total_loss = pol_loss  # + pol_reg_loss #- pol_expl_loss

        tf.summary.scalar('loss', pol_loss)
        tf.summary.scalar('expl_loss', pol_expl_loss)
        tf.summary.scalar('reg_loss', pol_reg_loss)
        tf.summary.scalar('total_loss', self._pol_total_loss)

      # policy training update
      with tf.name_scope('train_policy_network'):
        # Compute gradients
        pol_optimizer = tf.train.AdamOptimizer(pol_learning_rate)
        pol_gradients = pol_optimizer.compute_gradients(self._pol_total_loss)

        # # Clip gradient norm and summarize gradients for Tensorboard
        for grad, _ in pol_gradients:
          # tf.summary.histogram(var.name, var)
          if grad is not None:
            grad = tf.clip_by_norm(grad, 5.0)
            # tf.summary.histogram(var.name + '/gradient', grad)

        # apply gradients to update policy network
        self._train_pol_op = pol_optimizer.apply_gradients(
            pol_gradients, global_step=tf.train.get_or_create_global_step())

    with tf.name_scope('val'):
      # TODO: Implement eligibility traces
      with tf.name_scope('predict'):
        val_network_reg_collection = 'val_network_regularization'
        val_pred_hidden = utils.build_mlp(self._states,
                                          'hidden_value_network',
                                          activation=tf.nn.tanh,
                                          var_collection=val_network_reg_collection)

        layer = tf.layers.Dense(units=1,
                                kernel_initializer=tf.glorot_normal_initializer())
        self._val_pred = layer(val_pred_hidden)
        tf.summary.histogram('val_pred', self._val_pred)

        # Add weights to collection for regularization
        weights = layer.trainable_variables
        for weight in weights:
          if 'bias' not in weight.name:
            tf.add_to_collection(
                val_network_reg_collection, weight)

      with tf.name_scope('loss'):
        val_loss = tf.nn.l2_loss(self._val_pred - self._returns)
        tf.summary.scalar('val_loss', val_loss)

        val_reg_losses = tf.get_collection(val_network_reg_collection)
        val_reg_loss = tf.identity(sum(tf.nn.l2_loss(
            reg_loss) for reg_loss in val_reg_losses) * beta,
            name='val_reg_loss')
        tf.summary.scalar('val_reg_loss', val_reg_loss)

        self._val_total_loss = val_loss + val_reg_loss
        tf.summary.scalar('val_total_loss', self._val_total_loss)

      with tf.name_scope('gradients'):
        val_optimizer = tf.train.AdamOptimizer(val_learning_rate)
        val_gradients = val_optimizer.compute_gradients(self._val_total_loss)

        # Clip gradient norm and summarize gradients for Tensorboard
        for grad, _ in val_gradients:
          if grad is not None:
            grad = tf.clip_by_norm(grad, 5.0)
            # tf.summary.histogram(var.name + '/gradient', grad)

        self._train_val_op = val_optimizer.apply_gradients(
            val_gradients, global_step=tf.train.get_or_create_global_step())

    self._summaries = tf.summary.merge_all()
    self._reset()

  def act(self, observation):
    ''' Return the action

    # Params:
      observation: The observation input to the policy

    # Returns:
      a: Predicted action
    '''
    return self._sess.run(self._sample_action, feed_dict={self._states: observation})

  def predict_val(self, states):
    ''' Predict the value for a given state. '''
    feed_dict = {self._states: states}
    return self._sess.run(self._val_pred, feed_dict=feed_dict)

  def train_pol(self, returns, actions, states):
    ''' Train the policy. Return loss. '''
    feed_dict = {self._states: states,
                 self._returns: returns, self._actions: actions}
    _, pol_loss = self._sess.run([self._train_pol_op, self._pol_total_loss], feed_dict=feed_dict)
    return pol_loss

  def train_val(self, returns, states):
    ''' Train the value function. Return loss. '''
    feed_dict = {self._states: states, self._returns: returns}
    _, val_loss = self._sess.run([self._train_val_op, self._val_total_loss], feed_dict=feed_dict)
    return val_loss

  def summarize(self, returns, actions, states):
    '''Summarize key stats for TensorBoard. '''
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    feed_dict = {
        self._states: states,
        self._returns: returns,
        self._actions: actions
    }

    _, summary = self._sess.run([self._train_pol_op, self._summaries],
                                feed_dict=feed_dict,
                                options=run_options, run_metadata=run_metadata)
    return summary, run_metadata

  def _reset(self):
    ''' Reset the policy. '''
    self._sess.run(tf.global_variables_initializer())
