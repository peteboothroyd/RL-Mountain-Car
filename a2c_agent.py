import os

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import logger
import utils
from a2c_policy import A2CPolicy
from runner import EpisodicRunner

class A2CAgent(object):
  def __init__(self, env, visualise, model_dir, max_episode_steps,
               debug, summary_every=100, future_returns=True,
               reg_coeff=1e-3, ent_coeff=0.01, gamma=1.0,
               tensorboard_summaries=True, learning_rate=5e-3, num_learning_steps=500,
               use_actor_reg_loss=False, use_actor_expl_loss=False):

    self._sess = tf.Session()
    self._episode = 0

    # Wrap the session in a CLI debugger
    if debug:
      self._sess = tf_debug.LocalCLIDebugWrapperSession(self._sess)

    self._summary_every = summary_every
    self._visualise = visualise
    self._model_dir = model_dir
    self._tensorboard_summaries = tensorboard_summaries
    self._num_learning_steps = num_learning_steps
    self._env = env

    self._policy = A2CPolicy(sess=self._sess,
                            ob_space=env.observation_space,
                            ac_space=env.action_space,
                            reg_coeff=reg_coeff,
                            ent_coeff=ent_coeff,
                            actor_learning_rate=learning_rate,
                            critic_learning_rate=learning_rate,
                            use_actor_reg_loss=use_actor_reg_loss,
                            use_actor_expl_loss=use_actor_expl_loss)
    self._runner = EpisodicRunner(policy=self._policy,
                                  env=env,
                                  max_episode_steps=max_episode_steps,
                                  gamma=gamma,
                                  summary_every=summary_every,
                                  future_returns=future_returns)

    if not self._no_tensorboard_summaries:
      self._summary_writer = tf.summary.FileWriter(model_dir)
      self._summary_writer.add_graph(self._sess.graph, global_step=self._episode)

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
    total_timesteps = 0

    try:
      for self._episode in range(self._num_learning_steps):
        summarise = self._episode % self._summary_every == 0
        visualise = summarise and self._visualise

        returns, actions, states = self._runner.generate_rollouts(render=visualise)
        q = np.reshape(returns, (-1,))

        val = self._policy.critic(states).reshape((-1,))

        # Change statistics of predicted values to match current rollout
        val = val - np.mean(val) + np.mean(q)
        val = (np.std(q)+1e-4) * val / (np.std(val)+1e-4)

        adv = q - val
        val_loss = self._policy.train_critic(q, states)
        pol_loss = self._policy.train_actor(adv, actions, states)

        if summarise:
          self._print_stats('actions', actions)
          self._print_stats('returns', adv)

          logger.record_tabular('episode', self._episode)
          logger.record_tabular('total_timesteps', total_timesteps)
          logger.record_tabular('pol_loss', pol_loss)
          logger.record_tabular('val_loss', val_loss)
          logger.dump_tabular()
          # get stats

          # utils.plot_value_func(self._policy,
          #                       self._episode,
          #                       self._env.observation_space)

          if not self._no_tensorboard_summaries:
            summary, run_metadata = self._policy.summarize(
                adv, actions, states)
            self._summary_writer.add_run_metadata(
                run_metadata, 'step%d' % self._episode)
            self._summary_writer.add_summary(summary, self._episode)
    except:
      # Reraise to allow early termination of learning, and display
      # of learned optimal behaviour.
      raise
    finally:
      # Save out necessary checkpoints & diagnostics
      save_model_path = os.path.join(self._model_dir, 'model.ckpt')
      self._save_model(save_model_path)

      if not self._no_tensorboard_summaries:
        self._summary_writer.flush()
        self._summary_writer.close()
  
  def rollout(self):
    self._runner.rollout(render=True, t_sleep=0.1)

  def rollout(self, t_sleep):
    self._runner.rollout(render=True, t_sleep=t_sleep)

  def reset_policy(self):
    self._policy.reset()

  def _save_model(self, save_path):
    saver = tf.train.Saver()
    saver.save(self._sess, save_path=save_path)

  def _print_stats(self, name, x):
    print(name, 'stats, mean: {0:.2f}'.format(np.mean(x)),
          'std_dev: {0:.2f}'.format(np.std(x)),
          'max: {0:.2f}'.format(np.amax(x)),
          'min: {0:.2f}'.format(np.amin(x)),
          'shape: ', str(x.shape))

  def _graph_initialized(self):
    uninitialized_vars = self._sess.run(tf.report_uninitialized_variables())
    return True if uninitialized_vars.shape[0] == 0 else False
  