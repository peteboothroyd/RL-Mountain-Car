from gym_environment import Continuous_MountainCarEnv
from gaussian_process_agent import GaussianProcessAgent
from time import sleep

import argparse


def main():
  parser = argparse.ArgumentParser(
      description='Uses Gaussian Process techniques to solve the \
      Mountain Car reinforcement learning task.')
  parser.add_argument('--visualise', type=bool,
                      help='whether to visualise the graphs')

  args = parser.parse_args()

  env = Continuous_MountainCarEnv(gaussian_reward_scale=0.05)
  agent = GaussianProcessAgent(env, args.visualise)

  agent.learn()
  env.reset()

  for t_step in range(100):
    if args.visualise:
      env.render()

    action = agent.act(env.get_state())
    _, _, done, _ = env.step(action)

    sleep(0.05)

    if done:
      print("Episode finished after {} timesteps".format(t_step + 1))
      break


if __name__ == "__main__":
  main()
