import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from gym_environment import Continuous_MountainCarEnv

class Agent(object):
  X_SUPPORT_MIN, X_SUPPORT_MAX = -1, 1
  X_SUPPORT_POINTS = 31
  X_SUPPORT = np.linspace(X_SUPPORT_MIN, X_SUPPORT_MAX, num=X_SUPPORT_POINTS)

  X_DOT_SUPPORT_MIN, X_DOT_SUPPORT_MAX = -2, 2
  X_DOT_SUPPORT_POINTS = 31
  X_DOT_SUPPORT = np.linspace(X_DOT_SUPPORT_MIN, X_DOT_SUPPORT_MAX, num=X_DOT_SUPPORT_POINTS)

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
          max_v_estimate = float('-inf')
          for a in range(self.A_SUPPORT_POINTS):
            current_action = self.A_SUPPORT[a]
            
            next_x, next_x_dot = self.environment.next_state((current_x, current_x_dot), current_action)
            next_x_index, next_x_dot_index = self.binary_search(self.X_SUPPORT, next_x), self.binary_search(self.X_DOT_SUPPORT, next_x_dot)
            
            instantaneous_reward = self.environment.reward((current_x, current_x_dot))
            action_val_estimate = instantaneous_reward + self.state_values[next_x_index][next_x_dot_index]
            max_v_estimate = max(max_v_estimate, action_val_estimate)

          if self.state_values[x][x_dot] != max_v_estimate:
            any_value_changed = True

          self.state_values[x][x_dot] = max_v_estimate
      
      if not any_value_changed:
        not_converged = False
      
    # print("Iterations %s" % i)
    # print(self.state_values)
  
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

      if abs(current_x - target_x) <= 0.1 and abs(current_x_dot - target_x_dot) <= 0.1:
        reached_target = True
    
    trajectory, = plt.plot(path_x, path_x_dot, '-o', label='Trajectory')
    start, = plt.plot([-0.5], [0.0], marker='*', markersize=10, color="red", label="Start")
    end, = plt.plot([0.6], [0.0], marker='o', markersize=10, color="green", label="Finish")
    plt.legend(handles=[trajectory, start, end], loc='lower right')
    plt.xlabel("x")
    plt.ylabel("dx")
    plt.xlim([-1.1,1.1])
    plt.ylim([-2.1,2.1])
    plt.title('Trajectory')
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

# class Environment(object):
#   T_STEP = 0.2
#   X_TARGET, X_DOT_TARGET = 0.6, 0

#   def next_state(self, current_state, action, visualise=False):
#     def diff(state, t, action):
#       return [state[1], self.acceleration(state[0], action)]

#     current_x, current_x_dot = current_state
#     t = np.linspace(0, self.T_STEP, 101)
#     start = [current_state[0], current_state[1]]

#     sol = odeint(diff, start, t, args=(action,))
    
#     if visualise:
#       plt.plot(sol[:,0], sol[:,1], "r-")

#     final_x, final_x_dot = sol[-1, 0], sol[-1, 1]
    
#     return final_x, final_x_dot

#   def step(self, current_state, action):
#     current_x, current_x_dot = current_state
#     accel = self.acceleration(current_x, action)
#     return [current_x_dot, accel]

#   def acceleration(self, current_x, action):
#     def gradient(x):
#       if x >= 0:
#         return math.pow((1 + 5 * x ** 2), -1.5)
#       else:
#         return 2 * x + 1
    
#     G = 9.81
    
#     return action - G * math.sin(math.atan(gradient(current_x)))

#   def reward(self, next_state):
#     TOLERANCE = 0.1
#     x, x_dot = next_state
#     if abs(x - self.X_TARGET) <= TOLERANCE  and abs(x_dot - self.X_DOT_TARGET) <= TOLERANCE:
#       return 0
#     else:
#       return -1


if __name__ == "__main__":
  # env = Environment()
  env = Continuous_MountainCarEnv()
  agent = Agent(env)
  print('learning')
  agent.learn()
  print('acting')
  agent.act((-0.5, 0))