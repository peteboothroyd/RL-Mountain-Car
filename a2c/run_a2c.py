import argparse
import datetime
import os
import gym

import numpy as np
import tensorflow as tf

from gym.wrappers import Monitor as GymMonitor
from baselines.bench import Monitor as BenchMonitor

from a2c_agent import A2CAgent
from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

# Breakout actions = ['noop', 'fire', 'right', 'left']
BREAKOUT_ID = 'BreakoutNoFrameskip-v4'


def main():
  args = command_line_args()

  set_global_seeds(args.seed)
  model_dir = '{}/{}_{:%Y-%m-%d_%H:%M:%S}'.format(
      args.model_dir, args.exp_name, datetime.datetime.now())
  logger.configure(model_dir)

  num_env = args.num_env if not args.evaluate else 1

  train_envs, eval_env = make_atari_env(
      env_id=args.env_id, num_env=num_env, seed=args.seed) 
  train_envs = VecFrameStack(train_envs, 4)
  eval_env = VecFrameStack(eval_env, 4)

  if not args.use_mlp:
    cnn = True
  else:
    cnn = False

  agent = A2CAgent(
      train_envs,
      eval_env,
      model_dir=model_dir,
      n_steps=args.n_steps,
      num_learning_steps=args.num_learning_steps,
      debug=args.debug,
      summary_every=args.summary_every,
      gamma=args.gamma,
      tensorboard_summaries=args.tensorboard_summaries,
      cnn=cnn,
      seed=args.seed,
      save_every=args.save_every,
      load_checkpoint=args.load_checkpoint,
      checkpoint_prefix=args.checkpoint_prefix)

  if args.evaluate:
    agent.evaluate()
  else:
    agent.learn()


def command_line_args():
  ''' Setup command line interface. '''
  parser = argparse.ArgumentParser(
      description='Uses advantage actor critic techniques to solve a \
      reinforcement learning task.')
  parser.add_argument(
      '--model_dir', type=str, help='the output directory for summaries',
      default='./model_out')
  parser.add_argument(
      '--exp_name', type=str, help='the name of the experiment', default='')
  parser.add_argument(
      '--n_steps', type=int, help='the number of steps per update', default=5)
  parser.add_argument(
      '--num_learning_steps', type=int, default=int(10e6),
      help='the maximum number of steps per episode')
  parser.add_argument(
      '--debug', action='store_true', help='debug the application')
  parser.add_argument(
      '--summary_every', type=int, help='summary every n steps', default=25)
  parser.add_argument(
      '--save_every', type=int, help='save every n steps', default=100)
  parser.add_argument(
      '--gamma', type=float, default=0.99,
      help='value of gamma for Bellman equations')
  parser.add_argument(
      '--tensorboard_summaries', action='store_false',
      help='store diagnostics for tensorboard')
  parser.add_argument(
      '--evaluate', action='store_true', help='evaluate a prelearned agent')
  parser.add_argument(
      '--use_mlp', action='store_true',
      help='use a multilayer perceptron architecture')
  parser.add_argument(
      '--load_checkpoint', action='store_true',
      help='restore the model from a checkpoint')
  parser.add_argument(
      '--checkpoint_prefix', help='prefix of checkpoint files', default='')
  parser.add_argument(
      '--env_id', choices=[BREAKOUT_ID], default=BREAKOUT_ID,
      help='The environment to use for the A2C algorithm (default: {0})'
      .format(BREAKOUT_ID))
  parser.add_argument(
      '--num_env',
      help="The number of different environments", type=int, default=16)
  parser.add_argument(
      '--seed', help='The random number generator seed', default=1, type=int)
  return parser.parse_args()

# def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
#     """
#     Create a wrapped, monitored SubprocVecEnv for Atari.
#     """
#     if wrapper_kwargs is None: wrapper_kwargs = {}
#     # Make the training envs which use the BenchMonitor and periodically flush
#     # progress.
#     def make_env(rank): # pylint: disable=C0111
#         def _thunk():
#             env = make_atari(env_id)
#             env.seed(seed + rank)
#             env = BenchMonitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
#             return wrap_deepmind(env, **wrapper_kwargs)
#         return _thunk
#     set_global_seeds(seed)
#     return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
  """
  Create a wrapped, monitored SubprocVecEnv for Atari. Note this is altered from
  the OpenAI baselines.common.cmd_util
  This version changes the Monitors to the OpenAI gym monitor, for recording
  video and statistics, and provides an evaluation agent in addition to the
  SubprocVecEnv wrapped training environments.
  """
  if wrapper_kwargs is None:
    wrapper_kwargs = {}

  def make_bench_monitor_env(rank):
    def _thunk():
      env = make_atari(env_id)
      env.seed(seed + rank)
      env = BenchMonitor(env, logger.get_dir()
                         and os.path.join(logger.get_dir(), str(rank)))
      return wrap_deepmind(env, **wrapper_kwargs)
    return _thunk
  
  def make_gym_monitor_env(name):
    def _thunk():
      env = make_atari(env_id)
      env.seed(seed+num_env)
      env = GymMonitor(env, os.path.join(logger.get_dir(), name),
                       video_callable=lambda ep_id: True)
      return wrap_deepmind(env, **wrapper_kwargs)
    return _thunk

  # Make the training envs which use the BenchMonitor and periodically flush
  # progress.
  train_envs = SubprocVecEnv(
      [make_bench_monitor_env(i+start_index) for i in range(num_env)])

  # Create one evaluation environment with a GymMonitor which can record video
  eval_env = SubprocVecEnv([make_gym_monitor_env('eval')])

  return train_envs, eval_env


if __name__ == '__main__':
  main()
