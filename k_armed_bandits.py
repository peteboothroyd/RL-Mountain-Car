import numpy as np
import matplotlib.pyplot as plt

EPSILON = 0.01
K = 10
STEPS = 100
EPISODES = 2000

def main():
    mus = np.random.normal(0, 10, K)
    variances = np.random.uniform(1, 5, K)
    optimal_action = np.argmax(mus)
    print("max mean = %s at index %s" % (np.amax(mus), optimal_action))

    qs = np.zeros(K)
    ns = np.zeros(K)
    greedy = np.random.binomial(1, EPSILON, (EPISODES, STEPS))
    rs = np.zeros(EPISODES)
    optimal_actions = np.zeros(EPISODES)

    rs_first_episode = np.zeros(STEPS / K)
    optimal_actions_first_episode = np.zeros(STEPS / K)

    for i in xrange(EPISODES):
        for j in xrange(STEPS):
            if greedy[i][j] == 1: #take random move
                action = np.random.randint(0, K)
            else: #take greedy move
                action = np.argmax(qs)

            optimal_actions[i] += EPSILON / (K * STEPS)
            if action == optimal_action:
                optimal_actions[i] += (1 - EPSILON) / STEPS

            sampledReturn = np.random.normal(mus[action], variances[action])
            rs[i] += sampledReturn / STEPS
            ns[action] += 1
            qs[action] = qs[action] + (sampledReturn - qs[action])/ns[action]

            if i == 0:
                rs_first_episode[int(j/K)] += sampledReturn / K
                optimal_actions_first_episode[int(j/K)] +=  EPSILON / (K * K)
                if action == optimal_action:
                    optimal_actions_first_episode[int(j/K)] += (1 - EPSILON) / K
    
    print(mus, qs)
    plt.plot(rs)
    plt.show()
    plt.plot(optimal_actions)
    plt.show()
    plt.plot(rs_first_episode)
    plt.show()
    plt.plot(optimal_actions_first_episode)
    plt.show()

if __name__ == "__main__":
    main()
