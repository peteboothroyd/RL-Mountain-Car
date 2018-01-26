from environment import Environment
from gaussian_process_agent import GaussianProcessAgent

import argparse

def main():
    parser = argparse.ArgumentParser(description='Uses Gaussian Process techniques to solve the Mountain Car reinforcement learning task.')
    parser.add_argument('--visualise', type=bool,
                    help='whether to visualise the graphs')

    args = parser.parse_args()
    
    env = Environment()
    agent = GaussianProcessAgent(env, args.visualise)
    agent.learn()
    agent.act((-0.5, 0))

if __name__ == "__main__":
    main()