import os
import time
import gym
import multiprocessing

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from baselines import logger
from baselines.common import set_global_seeds

from a2c import ActorCritic
from a2c_runner import A2CRunner

INFO_ALE_LIVES_KEY = 'ale.lives'


class A2CAgent(object):
  def __init__(self, train_envs, eval_env, model_dir, n_steps, debug, gamma, cnn,
               summary_every, num_learning_steps, seed, tensorboard_summaries,
               save_every, load_checkpoint, checkpoint_prefix):
    discrete = isinstance(train_envs.action_space, gym.spaces.Discrete)

    num_cpu = multiprocessing.cpu_count()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    self._sess = tf.Session(config=tf_config)
    self._saver = None

    self._step = 0
    self._summary_every = summary_every
    self._save_every = save_every
    self._seed = seed

    # Wrap the session in a CLI debugger
    if debug:
      self._sess = tf_debug.LocalCLIDebugWrapperSession(self._sess)

    self._model_dir = model_dir+'/model'
    self._tensorboard_summaries = tensorboard_summaries
    self._train_envs = train_envs
    self._eval_env = eval_env

    batch_size = n_steps*train_envs.num_envs
    self._num_policy_updates = num_learning_steps//batch_size

    self._actor_critic = ActorCritic(
        sess=self._sess, obs_space=train_envs.observation_space, cnn=cnn,
        act_space=train_envs.action_space,
        num_policy_updates=self._num_policy_updates)
    self._runner = A2CRunner(
        actor_critic=self._actor_critic, env=train_envs, n_steps=n_steps,
        gamma=gamma, discrete=discrete)

    if self._tensorboard_summaries:
      self._summary_writer = tf.summary.FileWriter(model_dir)
      self._summary_writer.add_graph(self._sess.graph,
                                     global_step=self._step)

    if not self._graph_initialized():
      raise Exception('Graph not initialised!')

    if load_checkpoint:
      self.load(checkpoint_prefix)

  def load(self, checkpoint_file_prefix):
    ''' Load a trained model from saved checkpoint files.

      # Params:
        checkpoint_file_prefix (str): The prefix of all checkpoint files to
            load. This is not an actual file. Example, if checkpoint files are
            stored in directory /tmp/checkpoint_files and all 3 files begin with
            'model' then provide '/tmp/checkpoint_files/model'.

    '''
    saver = tf.train.Saver()
    saver.restore(self._sess, checkpoint_file_prefix)

  def evaluate(self):
    ''' Evaluate a learned model by rolling out the policy. '''
    obs = self._eval_env.reset()

    while True:
      actions, _ = self._actor_critic.step(obs)
      obs, _, _, info = self._eval_env.step(actions)

      if info[0][INFO_ALE_LIVES_KEY] < 1:
        break

  def learn(self):
    ''' Learn an optimal policy parameterisation by
        interacting with the environment.
    '''
    set_global_seeds(self._seed)
    total_timesteps = 0
    start_time = time.time()

    try:
      for self._step in range(self._num_policy_updates):
        summarise = self._step % self._summary_every == 0

        returns, actions, observations, values = \
            self._runner.generate_rollouts()

        total_timesteps += returns.shape[0]
        n_seconds = time.time()-start_time

        pg_loss, val_loss, expl_loss, ent, summary = self._actor_critic.train(
            observations, returns, actions, values)

        if summarise:
          logger.record_tabular('seconds', n_seconds)
          logger.record_tabular('step', self._step)
          logger.record_tabular('total_timesteps', total_timesteps)
          logger.record_tabular('pg_loss', pg_loss)
          logger.record_tabular('expl_loss', expl_loss)
          logger.record_tabular('val_loss', val_loss)
          logger.record_tabular('entropy', ent)
          logger.dump_tabular()

          if self._tensorboard_summaries:
            self._summary_writer.add_summary(summary, self._step)

        if self._step % self._save_every == 0 and self._step > 0:
          self.evaluate()
          self.save_model()
    finally:
      # Save out necessary checkpoints & diagnostics
      self._close()

  def reset_actor_critic(self):
    self._actor_critic.reset()

  def _close(self):
    self._eval_env.close()
    self._train_envs.close()

    if self._tensorboard_summaries:
      self._summary_writer.close()

  def save_model(self):
    if self._saver is None:
      self._saver = tf.train.Saver()
    save_model_path = os.path.join(self._model_dir, 'model.ckpt')

    self._saver.save(
        self._sess, save_path=save_model_path, global_step=self._step)

  def _graph_initialized(self):
    uninitialized_vars = self._sess.run(tf.report_uninitialized_variables())
    return True if uninitialized_vars.shape[0] == 0 else False
