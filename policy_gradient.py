from gym_environment import Continuous_MountainCarEnv
from policy_gradient_agent import PolicyGradientAgent
from time import sleep
import argparse
import gym

def main():
    # Setup command line interface
    parser = argparse.ArgumentParser(description='Uses policy gradient \
                                    techniques to solve the Mountain Car reinforcement learning task.')
    parser.add_argument('--visualise', '-v', type=bool,
                        help='whether to visualise the graphs (default: False)', default=False)
    parser.add_argument('--model_dir', type=str,
                        help='the output directory for summaries (default: ./tmp)',
                        default='./tmp')
    parser.add_argument('--max_episode_steps', type=int,
                        help='the maximum number of steps per episode (default: 10000)',
                        default=500)
    parser.add_argument('--debug', type=bool,
                        help='debug the application (default: False)',
                        default=False)
    args = parser.parse_args()
    
    # Create environment and agent
    env = Continuous_MountainCarEnv()
    # env = gym.envs.make("MountainCarContinuous-v0")

    agent = PolicyGradientAgent(env, args.visualise, args.model_dir, args.max_episode_steps, args.debug)

    # Teach the agent how to act optimally
    agent.learn()

    # Reset environment
    env.reset()

    # Run one rollout using trained agent
    for t in range(args.max_episode_steps):

        if args.visualise:
            env.render()

        action = agent.act(env.get_state())
        _, _, done, _ = env.step(action)

        if done:
            print('Episode finished after {} timesteps'.format(t+1))
            break

if __name__ == '__main__':
    main()
        