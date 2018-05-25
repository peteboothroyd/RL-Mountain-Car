import argparse
import datetime
import os
import gym

import numpy as np
import tensorflow as tf
from gym.wrappers import Monitor

from a2c_agent import A2CAgent
from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

from gym_environment import Continuous_MountainCarEnv

# Environment ids
MOUNTAIN_CAR_ID = 'mountain_car'
BREAKOUT_ID = 'BreakoutNoFrameskip-v4' #: ACTION_MEANINGS=['noop', 'fire', 'left', 'right']


def main():
  args = command_line_args()
  model_dir = '{}/{}_{:%Y-%m-%d_%H:%M:%S}'.format(
      args.model_dir, args.exp_name, datetime.datetime.now())
  logger.configure(model_dir)

  cnn = False

  if not args.no_atari_env:
    env = VecFrameStack(make_atari_env(args.env_id, args.num_env, args.seed), 4)
    # Use a CNN for the ATARI games
    cnn = True
  elif args.env_id == MOUNTAIN_CAR_ID:
    def env_factory():
      return Continuous_MountainCarEnv(terminating=True, t_step=0.3)
    env = make_env(env_factory, args.num_env, args.seed, deepmind=False)

  agent = A2CAgent(
      env,
      model_dir=model_dir,
      n_steps=args.n_steps,
      num_learning_steps=args.num_learning_steps,
      debug=args.debug,
      summary_every=args.summary_every,
      gamma=args.gamma,
      tensorboard_summaries=args.tensorboard_summaries,
      cnn=cnn,
      seed=args.seed,
      save_every=args.save_every)

  # Teach the agent how to act optimally
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
      '--num_learning_steps', type=int, default=int(80e6),
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
      '--env_id', choices=[MOUNTAIN_CAR_ID, BREAKOUT_ID], default=BREAKOUT_ID,
      help='The environment to use for the A2C algorithm (default: {0})'\
          .format(BREAKOUT_ID))
  parser.add_argument(
      '--no_atari_env', action='store_true',
      help='Whether to make an ATARI environment from the ALE. This will \
        cause a CNN rather than an MLP to be used on the observation for \
        the policy.')
  parser.add_argument(
      '--num_env',
      help="The number of different environments", type=int, default=16)
  parser.add_argument(
      '--seed', help='The random number generator seed', default=1, type=int)
  return parser.parse_args()

def make_env(env_factory, num_env, seed, wrapper_kwargs=None, start_index=0, 
             deepmind=True):
  """
  Create a wrapped, monitored SubprocVecEnv for Atari. Note this is altered from
  the OpenAI baselines repo:
  https://github.com/openai/baselines/blob/master/baselines/common/cmd_util.py
  This version changes the Monitor to the OpenAI gym monitor, for recording
  video and statistics.
  """
  if wrapper_kwargs is None: wrapper_kwargs = {}
  def make_envs(rank, monitor):
    def render_video(episode_id):
      print('Monitor: current episode id: {0}'.format(episode_id))
      return episode_id % 250 == 0
    def _thunk():
      env = env_factory()
      env.seed(seed + rank)
      if monitor:
        env = Monitor(
            env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
            video_callable=render_video)
      if deepmind:
        return wrap_deepmind(env, **wrapper_kwargs)
      else:
        return env
    return _thunk
  set_global_seeds(seed)

  # Only monitor one of the environments
  env_list = []
  for i in range(num_env):
    if i == 0:
      env_list.append(make_envs(i+start_index, True))
    else:
      env_list.append(make_envs(i+start_index, False))

  return SubprocVecEnv(env_list)

def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0, 
                   deepmind=True):
  def env_factory():
    return make_atari(env_id)
  return make_env(env_factory, num_env, seed, wrapper_kwargs, start_index)

if __name__ == '__main__':
  main()
