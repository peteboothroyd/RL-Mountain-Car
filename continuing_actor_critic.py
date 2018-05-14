import os
import argparse
import time
import gym

import tensorflow as tf
import numpy as np

import logger
from gym_environment import Continuous_MountainCarEnv
import utils


class ContinuingActorCritic(object):
  def __init__(self, sess, ob_space, ac_space, lmbda=0.5):
    '''Define the tensorflow graph to be used for the actor critic.'''
    self.mean_reward = 0
    tf.summary.scalar('mean_reward', tf.convert_to_tensor(self.mean_reward))

    actor_learning_rate = 0.001
    critic_learning_rate = 0.01

    ob_shape = (1,) + ob_space.shape
    ac_dim = ac_space.shape[0]
    ac_shape = (1, ac_dim)

    with tf.name_scope('inputs'):
      delta, state, action = self._create_input_placeholders(ac_shape, ob_shape)
      tf.summary.histogram('delta', delta)

    with tf.name_scope('actor'):
      with tf.name_scope('action_statistics'):
        actor_network_reg_collection = 'actor_network_regularization'
        hidden = self._build_mlp(state,
                                 'actor_hidden',
                                 activation=tf.nn.tanh,
                                 var_collection=actor_network_reg_collection,
                                 size=32)

        mean_layer = tf.layers.Dense(
            units=ac_dim,
            activation=None,
            kernel_initializer=tf.glorot_normal_initializer(),
            bias_initializer=tf.zeros_initializer(),
            name='mean')
        mean = tf.squeeze(mean_layer(hidden))
        tf.summary.histogram('mean', mean)

        self._add_weights_to_regularisation_collection(
            mean_layer, actor_network_reg_collection)

        # Standard deviation of the normal distribution over actions
        std_dev = tf.get_variable(
            name="std_dev", shape=ac_shape,
            dtype=tf.float32, initializer=tf.ones_initializer())
        std_dev = tf.squeeze(tf.nn.softplus(std_dev) + 1e-4)
        tf.summary.scalar('std_dev', tf.squeeze(std_dev))

      with tf.name_scope('generate_sample_action'):
        sample_action = self._generate_sample_action(ac_shape,
                                                     std_dev,
                                                     mean,
                                                     ac_space)

      with tf.name_scope('loss'):
        dist = tf.distributions.Normal(loc=mean, scale=std_dev)
        log_prob = dist.log_prob(action, name='log_prob')
        # loss = log_prob + 0.1 * dist.entropy()

      with tf.name_scope('train_network'):
        actor_optimizer = tf.train.AdamOptimizer(actor_learning_rate)

        ########################## Optimizer version ###########################
        # actor_grads_and_vars = actor_optimizer.compute_gradients(-log_prob)
        ########################################################################

        ######################## Manual gradient version #######################
        tvars = tf.trainable_variables()
        actor_grads = tf.gradients(log_prob, tvars)
        actor_grads_and_vars = list(zip(actor_grads, tvars))
        ########################################################################

        actor_grads_and_vars = self._eligibility_trace(actor_grads_and_vars,
                                                       lmbda, delta, 'actor')
        actor_grads_and_vars = self._clip_by_global_norm(actor_grads_and_vars)

        train_actor_op = actor_optimizer.apply_gradients(
            actor_grads_and_vars,
            global_step=tf.train.get_or_create_global_step())

    with tf.name_scope('critic'):
      with tf.name_scope('predict'):
        critic_network_reg_collection = 'critic_network_regularization'
        critic_hidden = self._build_mlp(
            state,
            'critic_hidden',
            activation=tf.nn.tanh,
            var_collection=critic_network_reg_collection,
            size=32)

        dense_layer = tf.layers.Dense(
            units=1,
            kernel_initializer=tf.glorot_normal_initializer(), 
            name='critic_out')
        critic_pred = dense_layer(critic_hidden)
        tf.summary.histogram('critic_pred', critic_pred)

        self._add_weights_to_regularisation_collection(
            dense_layer, critic_network_reg_collection)

      with tf.name_scope('gradients'):
        critic_optimizer = tf.train.AdamOptimizer(critic_learning_rate)

        ########################## Optimizer version ###########################
        # critic_grads_and_vars = critic_optimizer.compute_gradients(-critic_pred)
        ########################################################################

        ######################## Manual gradient version #######################
        tvars = tf.trainable_variables()
        critic_grads = tf.gradients(critic_pred, tvars)
        critic_grads_and_vars = list(zip(critic_grads, tvars))
        ########################################################################

        critic_grads_and_vars = self._eligibility_trace(critic_grads_and_vars,
                                                        lmbda, delta, 'critic')
        critic_grads_and_vars = self._clip_by_global_norm(critic_grads_and_vars)

        train_critic_op = critic_optimizer.apply_gradients(
            critic_grads_and_vars,
            global_step=tf.train.get_or_create_global_step())

    summaries = tf.summary.merge_all()

    def actor(obs):
      ''' Return the action output by the policy given the current
          parameterisation.

      # Params:
        observation: The observation input to the policy

      # advantage:
        a: Action
      '''
      obs = obs.reshape(1, -1)
      feed_dict = {state: obs}
      return sess.run(sample_action, feed_dict=feed_dict)

    def critic(obs):
      ''' Predict the value for given states. '''
      obs = obs.reshape(1, -1)
      feed_dict = {state: obs}
      return sess.run(critic_pred, feed_dict=feed_dict)

    def train(reward, obs, next_obs, acs):
      ''' Train the policy and critic '''
      obs = obs.reshape(1, -1)
      next_obs = next_obs.reshape(1, -1)
      acs = acs.reshape(1, -1)

      val_state = self.critic(obs)
      val_next_state = self.critic(next_obs)

      dlta = reward - self.mean_reward + val_next_state - val_state
      dlta = np.squeeze(dlta)

      self.mean_reward += 0.01 * dlta

      feed_dict = {
          state: obs,
          delta: dlta,
          action: acs
      }

      _, _, summary = sess.run([train_actor_op, train_critic_op, summaries],
                               feed_dict=feed_dict)
      
      return summary

    def reset():
      ''' Reset the policy. '''
      sess.run(tf.global_variables_initializer())

    self.reset = reset
    self.actor = actor
    self.critic = critic
    self.train = train

    self.reset()
  
  def _generate_sample_action(self, ac_shape, std_dev, mean, ac_space):
    standard_normal = tf.random_normal(shape=ac_shape, name='standard_normal')
    bounded_sample = tf.nn.tanh(std_dev * standard_normal + mean,
                                name='sample_action')
    scaled_shifted_sample = bounded_sample \
        * (ac_space.high[0]-ac_space.low[0]) * 0.5 \
        + (ac_space.high[0]+ac_space.low[0]) * 0.5
    tf.summary.histogram('sample_action', scaled_shifted_sample)
    return scaled_shifted_sample

  def _build_mlp(self, input_placeholder, scope, n_layers=2, size=32,
                 activation=tf.tanh, var_collection=None):
    with tf.variable_scope(scope):
      output = input_placeholder
      for i in range(n_layers):
        layer = tf.layers.Dense(
            units=size, activation=activation,
            kernel_initializer=tf.glorot_normal_initializer(),
            name="dense_{}".format(i))
        output = layer(output)

        tf.summary.histogram('dense{0}_activation'.format(i), output)

        if var_collection is not None:
          self._add_weights_to_regularisation_collection(layer, var_collection)

    return output


  def _clip_by_global_norm(self, grads_and_vars, norm=5.0):
    grads, variables = zip(*grads_and_vars)
    clipped_grads, _ = tf.clip_by_global_norm(grads, norm)
    return zip(clipped_grads, variables)


  def _create_input_placeholders(self, ac_shape, ob_shape):
    action = tf.placeholder(tf.float32, shape=ac_shape, name='action')
    delta = tf.placeholder(tf.float32, name='delta', shape=[])
    state = tf.placeholder(tf.float32, shape=ob_shape, name='state')
    print('action expected shape ', action.get_shape())
    print('delta expected shape ', delta.get_shape())
    print('state expected shape ', state.get_shape())
    return delta, state, action

  def _add_weights_to_regularisation_collection(self, layer, reg_collection):
    weights = layer.trainable_variables
    for weight in weights:
      # Don't want to regularise the biases
      if 'bias' not in weight.name:
        tf.add_to_collection(reg_collection, weight)
  
  def _eligibility_trace(self, grads_and_vars, lmbda, delta, name):
    for grad, var in grads_and_vars:
      if grad is not None:
        with tf.variable_scope('trace'):
          trace_name = var.op.name +  '/' + name + '/trace'
          trace = tf.get_variable(name=trace_name,
                                  trainable=False,
                                  shape=grad.get_shape(),
                                  initializer=tf.zeros_initializer())
          trace = lmbda * trace + grad
          grad = trace * delta
          tf.summary.histogram(trace_name, trace)
    
    return grads_and_vars

class ContinuingAgent(object):
  def __init__(self, visualise=False, exp_name="exp"):

    self._sess = tf.Session()
    self._total_timesteps = 0
    self._summary_every = 1000
    self._visualise = visualise
    self._directory = './tmp/continuing/'+exp_name
    self._t_step = 0.1

    self._env = Continuous_MountainCarEnv(gaussian_reward_scale=0.1, t_step=self._t_step)
    self._policy = ContinuingActorCritic(self._sess,
                                         ob_space=self._env.observation_space,
                                         ac_space=self._env.action_space)

    self._summary_writer = tf.summary.FileWriter(self._directory)
    self._summary_writer.add_graph(self._sess.graph,
                                   global_step=self._total_timesteps)

    if not self._graph_initialized():
      raise Exception('Graph not initialised')

  def learn(self):
    ''' Learn an optimal policy parameterisation by
        interacting with the environment.
    '''
    try:
      state = self._env.reset()

      while True:
        if self._visualise:
          self._env.render()
        self._total_timesteps += 1

        summarise = self._total_timesteps % self._summary_every == 0

        action = self._policy.actor(state)
        next_state, reward, _, _ = self._env.step(action)
        summary = self._policy.train(reward, state, next_state, action)
        state = next_state

        if summarise:
          mean_reward = self._policy.mean_reward
          logger.record_tabular('total_timesteps', self._total_timesteps)
          logger.record_tabular('mean_reward', mean_reward)
          logger.dump_tabular()

          self._summary_writer.add_summary(summary, self._total_timesteps)

          summary = tf.Summary(
              value=[tf.Summary.Value(tag='mean_reward',
              simple_value=mean_reward)])
          self._summary_writer.add_summary(summary, self._total_timesteps)
    except:
      # Reraise to allow early termination of learning, and display
      # of learned optimal behaviour.
      raise
    finally:
      # Save out necessary checkpoints & diagnostics
      self._save_model(self._directory + '/model.ckpt')
      self._summary_writer.close()


  def reset_policy(self):
    ''' Reset the policy. '''
    self._policy.reset()


  def close(self):
    ''' Close down. '''
    self._env.close()


  def rollout(self):
    ''' Rollout the learned agent and render. '''
    state = self._env.reset()

    for _ in range(500):
      time.sleep(self._t_step)
      self._env.render()
      action = self._policy.actor(state)
      state, _, _, _ = self._env.step(action)


  def _save_model(self, save_path):
    saver = tf.train.Saver()
    saver.save(self._sess, save_path=save_path)


  def _graph_initialized(self):
    uninitialized_vars = self._sess.run(tf.report_uninitialized_variables())
    return True if uninitialized_vars.shape[0] == 0 else False


def main():
  utils.set_global_seeds()
  parser = argparse.ArgumentParser(
      description='Uses advantage actor critic \
      techniques to solve the Mountain Car reinforcement learning task.')
  parser.add_argument(
      '--visualise', action='store_true',
      help='whether to visualise the graphs (default: False)')
  parser.add_argument(
      '--exp_name', type=str,
      help='the name of the experiment (default: exp)',
      default='exp')
  args = parser.parse_args()
  agent = ContinuingAgent(visualise=args.visualise, exp_name=args.exp_name)

  try:
    agent.learn()
  finally:
    agent.rollout()
    agent.close()


if __name__ == '__main__':
  main()
