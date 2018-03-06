import argparse

import gym
import numpy as np
import tensorflow as tf

import utils
from gym_environment import Continuous_MountainCarEnv
from a2c_agent import A2CAgent


def main():
  args = command_line_args()
  utils.set_global_seeds(1)

  # # Create environment and agent
  # if args.gaussian_env:
  #   print('Using Gaussian reward')
  #   env = gym.wrappers.TimeLimit(Continuous_MountainCarEnv(
  #       gaussian_reward_scale=0.05, t_step=0.1), max_episode_steps=1000)
  # else:
  env = gym.wrappers.TimeLimit(Continuous_MountainCarEnv(
      terminating=True, t_step=0.1), max_episode_steps=1000)

  if args.hyper_search:
    hyperparameter_search(env)
  print(args)
  agent = A2CAgent(
      env,
      visualise=args.visualise,
      model_dir=args.model_dir+'/'+args.exp_name,
      max_episode_steps=args.max_episode_steps,
      debug=args.debug,
      summary_every=args.summary_every,
      future_returns=args.future_returns,
      gamma=args.gamma,
      tensorboard_summaries=args.tensorboard_summaries,
      use_actor_expl_loss=args.actor_expl_loss,
      use_actor_reg_loss=args.actor_reg_loss
      )

  if not args.plot_learning:
    # Teach the agent how to act optimally
    try:
      agent.learn()
    finally:
      agent.rollout(0.1)
      env.close()
  else:
    generate_plot(agent, args.summary_every, args.exp_name)


def command_line_args():
  ''' Setup command line interface. '''
  parser = argparse.ArgumentParser(
      description='Uses advantage actor critic \
      techniques to solve the Mountain Car reinforcement learning task.')
  parser.add_argument(
      '--visualise', '-v', action='store_true',
      help='whether to visualise the graphs (default: False)')
  parser.add_argument(
      '--model_dir', type=str,
      help='the output directory for summaries (default: ./tmp)',
      default='./tmp')
  parser.add_argument(
      '--exp_name', type=str,
      help='the name of the experiment (default: exp)',
      default='exp')
  parser.add_argument(
      '--max_episode_steps', type=int,
      help='the maximum number of steps per episode (default: 500)',
      default=500)
  parser.add_argument(
      '--debug', action='store_true',
      help='debug the application (default: False)')
  parser.add_argument(
      '--hyper_search', action='store_true',
      help='search hyperparameter space (default: False)')
  parser.add_argument(
      '--summary_every', type=int,
      help='summary every n episodes (default: 50)',
      default=50)
  parser.add_argument(
      '--future_returns', action='store_false',
      help='use only future returns for the gradient (default: True)')
  # parser.add_argument(
  #     '--gaussian_env', action='store_true',
  #     help='use the gaussian reward (default: False)')
  parser.add_argument(
      '--gamma', type=float,
      help='value of gamma for Bellman equations (default: 1.0)',
      default=1.0)
  parser.add_argument(
      '--plot_learning', action='store_true',
      help='plot the learning curves with error bars (default: False)')
  parser.add_argument(
      '--tensorboard_summaries', action='store_true',
      help='do not store diagnostics for tensorboard (default: False)')
  parser.add_argument(
      '--actor_expl_loss', action='store_true',
      help='include exploration loss (default: False)')
  parser.add_argument(
      '--actor_reg_loss', action='store_true',
      help='include regularisation loss (default: False)')

  return parser.parse_args()


def hyperparameter_search(env):
  entropy_weights = np.logspace(-3, -1, num=3)
  reg_coeffs = np.logspace(-3, 0, num=3)
  learning_rates = np.logspace(-5, -2, num=3)

  for e_w in entropy_weights:
    for r_c in reg_coeffs:
      for l_r in learning_rates:
        for actor_expl_loss in [False, True]:
          for actor_reg_loss in [False, True]:
            env.reset()
            tf.reset_default_graph()
            directory = './tmp/ew_%s_rc_%s_lr_%s' % (e_w, r_c, l_r)
            print(directory)

            agent = A2CAgent(env,
                             visualise=False,
                             model_dir=directory,
                             max_episode_steps=500,
                             debug=False,
                             summary_every=25,
                             future_returns=True,
                             gamma=1.0,
                             reg_coeff=r_c,
                             learning_rate=l_r,
                             entropy_weight=e_w,
                             use_actor_expl_loss=actor_expl_loss,
                             use_actor_reg_loss=actor_reg_loss)
            agent.learn()


def generate_plot(agent, summary_every, exp_name):
  mean_rewards_progress = []
  std_rewards_progress = []
  episode_length_progress = []

  for i in range(50):
    print('********* Iteration {0} *********'.format(i))
    utils.set_global_seeds(i)
    mrp, srp, elp, = agent.learn()
    mean_rewards_progress.append(mrp)
    std_rewards_progress.append(srp)
    episode_length_progress.append(elp)
    agent.reset_policy()

  mean_rewards_progress = np.array(mean_rewards_progress)
  std_rewards_progress = np.array(std_rewards_progress)
  episode_length_progress = np.array(episode_length_progress)

  utils.plot(mean_rewards_progress,
             std_rewards_progress,
             episode_length_progress,
             summary_every, exp_name)


if __name__ == '__main__':
  main()
