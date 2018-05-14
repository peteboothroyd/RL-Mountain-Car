import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from baselines import logger
from baselines.common import set_global_seeds

from a2c_policy import A2CPolicy
from a2c_runner import A2CRunner


class A2CAgent(object):
  def __init__(self, env, model_dir, n_steps, debug, gamma=1.0,
               summary_every=25, reg_coeff=1e-3, ent_coeff=0.01,
               tensorboard_summaries=True, actor_learning_rate=5e-3,
               critic_learning_rate=1e-3, num_learning_steps=5000, seed=0,
               use_actor_reg_loss=False, use_actor_expl_loss=False, cnn=False):

    self._sess = tf.Session()
    self._episode = 0
    self._summary_every = summary_every
    self._seed = seed

    # Wrap the session in a CLI debugger
    if debug:
      self._sess = tf_debug.LocalCLIDebugWrapperSession(self._sess)

    self._model_dir = model_dir+'/model'
    self._tensorboard_summaries = tensorboard_summaries
    self._num_learning_steps = num_learning_steps
    self._env = env

    self._policy = A2CPolicy(
        sess=self._sess, obs_space=env.observation_space, cnn=cnn,
        act_space=env.action_space, use_actor_expl_loss=use_actor_expl_loss,
        actor_learning_rate=actor_learning_rate, reg_coeff=reg_coeff,
        critic_learning_rate=critic_learning_rate, ent_coeff=ent_coeff,
        use_actor_reg_loss=use_actor_reg_loss)
    self._runner = A2CRunner(
        policy=self._policy, env=env, n_steps=n_steps, gamma=gamma)

    if self._tensorboard_summaries:
      self._summary_writer = tf.summary.FileWriter(model_dir)
      self._summary_writer.add_graph(self._sess.graph,
                                     global_step=self._episode)

    if not self._graph_initialized():
      raise Exception('Graph not initialised')

  def act(self, observation):
    ''' Given an observation of the state return an action
        according to the current policy parameterisation.
    '''
    return self._policy.actor(observation[np.newaxis, :])

  def learn(self):
    ''' Learn an optimal policy parameterisation by
        interacting with the environment.
    '''
    set_global_seeds(self._seed)
    total_timesteps = 0
    start_time = time.time()

    try:
      for self._episode in range(self._num_learning_steps):
        summarise = self._episode % self._summary_every == 0

        returns, actions, observations, values = \
            self._runner.generate_rollouts()

        total_timesteps += returns.shape[0]
        n_seconds = time.time()-start_time

        advantages = returns - values
        
        val_loss, pol_loss = self._policy.train(
            returns, observations, advantages, actions)

        if summarise:
          logger.record_tabular('seconds', n_seconds)
          logger.record_tabular('episode', self._episode)
          logger.record_tabular('total_timesteps', total_timesteps)
          logger.record_tabular('pol_loss', pol_loss)
          logger.record_tabular('val_loss', val_loss)
          logger.dump_tabular()

          if self._tensorboard_summaries:
            summary = self._policy.summarize(
                returns, observations, advantages, actions)
            self._summary_writer.add_summary(summary, self._episode)
    except:
      # Reraise to allow early termination of learning, and display
      # of learned optimal behaviour.
      raise
    finally:
      # Save out necessary checkpoints & diagnostics
      self._close()

  def reset_policy(self):
    self._policy.reset()

  def _close(self):
    save_model_path = os.path.join(self._model_dir, 'model.ckpt')
    self._save_model(save_model_path)

    if self._tensorboard_summaries:
      self._summary_writer.close()

  def _save_model(self, save_path):
    saver = tf.train.Saver()
    saver.save(self._sess, save_path=save_path)

  def _graph_initialized(self):
    uninitialized_vars = self._sess.run(tf.report_uninitialized_variables())
    return True if uninitialized_vars.shape[0] == 0 else False
