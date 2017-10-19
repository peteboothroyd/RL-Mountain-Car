import math
import numpy as np
import matplotlib.pyplot as plt

class Agent(object):
    X_SUPPORT_MIN, X_SUPPORT_MAX = -1, 1
    X_SUPPORT_POINTS = 21
    X_SUPPORT = np.linspace(X_SUPPORT_MIN, X_SUPPORT_MAX, num=X_SUPPORT_POINTS)

    X_DOT_SUPPORT_MIN, X_DOT_SUPPORT_MAX = -2, 2
    X_DOT_SUPPORT_POINTS = 21
    X_DOT_SUPPORT = np.linspace(
        X_DOT_SUPPORT_MIN, X_DOT_SUPPORT_MAX, num=X_DOT_SUPPORT_POINTS)

    A_SUPPORT_MIN, A_SUPPORT_MAX = -4, 4
    A_SUPPORT_POINTS = 21
    A_SUPPORT = np.linspace(A_SUPPORT_MIN, A_SUPPORT_MAX, num=A_SUPPORT_POINTS)

    def __init__(self, environment):
        self.state_values = np.zeros((self.X_SUPPORT_POINTS, self.X_DOT_SUPPORT_POINTS))
        self.environment = environment

    def learn(self):
        not_converged = True
        i = 0

        while not_converged:
            i += 1
            any_value_changed = False
            for x in range(self.X_SUPPORT_POINTS):
                for x_dot in range(self.X_DOT_SUPPORT_POINTS):
                    current_x, current_x_dot = self.X_SUPPORT[x], self.X_DOT_SUPPORT[x_dot]
                    max_v_estimate = None
                    for a in range(self.A_SUPPORT_POINTS):
                        current_action = self.A_SUPPORT[a]
                        
                        next_x, next_x_dot = self.environment.next_state((current_x, current_x_dot), current_action)
                        next_x_index, next_x_dot_index = self.binary_search(self.X_SUPPORT, next_x), self.binary_search(self.X_DOT_SUPPORT, next_x_dot)
                        
                        instantaneous_reward = self.environment.reward((current_x, current_x_dot))
                        action_val_estimate = instantaneous_reward + self.state_values[next_x_index][next_x_dot_index]
                        
                        if max_v_estimate == None or action_val_estimate > max_v_estimate:
                            max_v_estimate = action_val_estimate

                    if self.state_values[x][x_dot] != max_v_estimate:
                        any_value_changed = True

                    self.state_values[x][x_dot] = max_v_estimate
            
            if not any_value_changed:
                not_converged = False
            
        print("Iterations %s" % i)
        print(self.state_values)
    
    def act(self, start_state):
        start_x, start_x_dot = start_state
        start_x_index, start_x_dot_index = self.binary_search(self.X_SUPPORT, start_x), self.binary_search(self.X_DOT_SUPPORT, start_x_dot)
        current_x, current_x_dot = self.X_SUPPORT[start_x_index], self.X_DOT_SUPPORT[start_x_dot_index]

        target_x_index, target_x_dot_index = self.binary_search(self.X_SUPPORT, 0.6), self.binary_search(self.X_DOT_SUPPORT, 0.0)
        target_x, target_x_dot = self.X_SUPPORT[target_x_index], self.X_DOT_SUPPORT[target_x_dot_index]

        path_x, path_x_dot = [start_x], [start_x_dot]

        reached_target = False

        while not reached_target:
            max_v, argmax_v = None, None

            for a in range(self.A_SUPPORT_POINTS):
                current_action = self.A_SUPPORT[a]
                
                next_x, next_x_dot = self.environment.next_state((current_x, current_x_dot), current_action)
                next_x_index, next_x_dot_index = self.binary_search(self.X_SUPPORT, next_x), self.binary_search(self.X_DOT_SUPPORT, next_x_dot)
 
                action_val_estimate = self.state_values[next_x_index][next_x_dot_index]
                
                if max_v == None or action_val_estimate > max_v:
                    max_v, argmax_v = action_val_estimate, a
            
            best_next_x, best_next_x_dot = self.environment.next_state((current_x, current_x_dot), self.A_SUPPORT[argmax_v])
            best_next_x_index, best_next_x_dot_index = self.binary_search(self.X_SUPPORT, best_next_x), self.binary_search(self.X_DOT_SUPPORT, best_next_x_dot)
            current_x, current_x_dot = self.X_SUPPORT[best_next_x_index], self.X_DOT_SUPPORT[best_next_x_dot_index]

            path_x.append(current_x)
            path_x_dot.append(current_x_dot)

            if abs(current_x - target_x) < 0.01 and abs(current_x_dot - target_x_dot) < 0.01:
                reached_target = True
        
        plt.plot(path_x, path_x_dot, '-o')
        plt.show()

    def binary_search(self, sorted_list, target):
        min = 0
        max = len(sorted_list) - 1
        if target < sorted_list[0]:
            return 0
        elif target > sorted_list[-1]:
            return max
        while True:
            if (max - min) == 1:
                if abs(sorted_list[max] - target) < abs(target - sorted_list[min]):
                    return max
                else:
                    return min

            mid = (min + max) // 2

            if sorted_list[mid] < target:
                min = mid
            elif sorted_list[mid] > target:
                max = mid
            else:
                return mid


class Environment(object):
    T_STEP = 0.3
    X_TARGET, X_DOT_TARGET = 0.6, 0

    def next_state(self, current_state, action):
        current_x, current_x_dot = current_state
        grad = self.gradient(current_x)
        accel = self.acceleration(action, grad)
        next_x = self.next_x(current_x_dot, self.T_STEP, accel, current_x)
        next_x_dot = self.next_x_dot(current_x_dot, self.T_STEP, accel)
        return next_x, next_x_dot

    def gradient(self, x):
        if x >= 0:
            return math.pow((1 + 5 * x * x), -1.5)
        else:
            return 2 * x + 1

    def acceleration(self, force, gradient):
        G = 9.81
        return force - G * math.sin(math.atan(gradient))

    def next_x(self, u, t, a, current_x):
        return current_x + u * t + 0.5 * a * t * t

    def next_x_dot(self, u, t, a):
        return u + a * t

    def reward(self, next_state):
        TOLERANCE = 0.001
        x, x_dot = next_state
        if abs(x - self.X_TARGET) < TOLERANCE  and abs(x_dot - self.X_DOT_TARGET) < TOLERANCE:
            return 0
        else:
            return -1


if __name__ == "__main__":
    env = Environment()
    agent = Agent(env)
    agent.learn()
    agent.act((-0.5, 0))