from gym_environment import Continuous_MountainCarEnv
from policy_gradient_agent import PolicyGradientAgent
from time import sleep
import argparse

def main():
    parser = argparse.ArgumentParser(description='Uses policy gradient techniques to solve the Mountain Car reinforcement learning task.')
    parser.add_argument('--visualise', type=bool, help='whether to visualise the graphs')

    args = parser.parse_args()
    
    env = Continuous_MountainCarEnv()
    agent = PolicyGradientAgent(env, args.visualise)
    env.reset()

    for _ in range(1000):
        sleep(0.05)

        if args.visualise:
            env.render()

        action = agent.act(env.get_state())
        _, _, done, _ = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

if __name__ == "__main__":
    main()
        