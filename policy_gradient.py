import argparse

import gym
import time
import numpy as np
import tensorflow as tf

from gym_environment import Continuous_MountainCarEnv
from policy_gradient_agent import PolicyGradientAgent
import utils


def main():
  # Setup command line interface
  parser = argparse.ArgumentParser(description='Uses policy gradient \
    techniques to solve the Mountain Car reinforcement learning task.')
  parser.add_argument('--visualise', '-v', type=bool,
                      help='whether to visualise the graphs (default: False)', default=False)
  parser.add_argument('--model_dir', type=str,
                      help='the output directory for summaries (default: ./tmp)',
                      default='./tmp')
  parser.add_argument('--exp_name', type=str,
                      help='the name of the experiment (default: exp)',
                      default='exp')
  parser.add_argument('--max_episode_steps', type=int,
                      help='the maximum number of steps per episode (default: 500)',
                      default=500)
  parser.add_argument('--debug', type=bool,
                      help='debug the application (default: False)',
                      default=False)
  parser.add_argument('--hyper_search', type=bool,
                      help='search hyperparameter space (default: False)',
                      default=False)
  parser.add_argument('--summary_every', type=int,
                      help='summary every n episodes (default: 500)',
                      default=500)
  parser.add_argument('--critic', type=bool,
                      help='whether to use a critic (default: False)',
                      default=False)
  parser.add_argument('--full_reward', type=bool,
                      help='whether to use a the total return for gradient (default: False)',
                      default=False)
  parser.add_argument('--normalize_advantages', type=bool,
                      help='whether to use normalize the advantage (default: True)',
                      default=True)
  parser.add_argument('--gaussian_env', type=bool,
                      help='whether to use the gaussian reward (default: False)',
                      default=False)
  parser.add_argument('--gamma', type=float,
                      help='value of gamma for Bellman equations (default: 1.0)',
                      default=1.0)
  args = parser.parse_args()

  utils.set_global_seeds(1)

  # Create environment and agent
  print(args.gaussian_env)
  if args.gaussian_env:
    print('Gaussian continuing task')
    env = Continuous_MountainCarEnv(gaussian_reward_scale=0.5)
  else:
    print('Episodic task')
    env = gym.wrappers.TimeLimit(Continuous_MountainCarEnv(
        terminating=True), max_episode_steps=1000)

  if args.hyper_search:
    hyperparameter_search(env)

  agent = PolicyGradientAgent(env,
                              visualise=args.visualise,
                              model_dir=args.model_dir+'/'+args.exp_name,
                              max_episode_steps=args.max_episode_steps,
                              debug=args.debug,
                              summary_every=args.summary_every,
                              critic=args.critic,
                              full_reward=args.full_reward,
                              normalize_adv=args.normalize_advantages,
                              gamma=args.gamma)

  # Teach the agent how to act optimally
  try:
    agent.learn()
  finally:
    # Display learned optimal behaviour when exiting learn early

    # Reset environment
    obs = env.reset()

    # Run one rollout using trained agent
    for t_step in range(args.max_episode_steps):
      if args.visualise:
        env.render()
        time.sleep(0.1)

        action = agent.act(obs)
        obs, _, done, _ = env.step(action)

        if done:
          print('Episode finished after {} timesteps'.format(t_step+1))
          break
    
    env.close()


def hyperparameter_search(env):
  max_episode_steps = np.logspace(2, 3, num=3)
  entropy_weights = np.logspace(-3, -1, num=3)
  betas = np.logspace(-3, 0, num=3)

  for mas in max_episode_steps:
    mas = int(mas)
    for ew in entropy_weights:
      for b in betas:
        env.reset()
        tf.reset_default_graph()
        directory = './tmp/mas_%s_ew_%s_b_%s' % (mas, ew, b)
        print(directory)
        agent = PolicyGradientAgent(
            env=env,
            visualise=False,
            model_dir=directory,
            max_episode_steps=mas,
            debug=False,
            beta=b,
            exploration=ew)
        agent.learn()


if __name__ == '__main__':
  main()
