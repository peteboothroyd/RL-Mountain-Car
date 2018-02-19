import math
import pickle
import numpy as np
import itertools

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.lines as mlines

from scipy.stats import multivariate_normal
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error

from scipy.integrate import odeint

class GaussianProcessAgent(object):
    """The agent in the reinforcement learning framework. The agent must first learn a value function before
    it can act optimally with respect to that value function.
    """

    def __init__(self, environment, visualise=None):
        self.x_min, self.x_max = -1, 1
        self.x_points = 21
        self.x = np.linspace(self.x_min, self.x_max, num=self.x_points)

        self.x_dot_min, self.x_dot_max = -2, 2
        self.x_dot_points = 21
        self.x_dot = np.linspace(self.x_dot_max, self.x_dot_min, num=self.x_dot_points)

        self.a_min, self.a_max = -4, 4
        self.a_points = 21
        self.a = np.linspace(self.a_min, self.a_max, num=self.a_points)

        self.gamma = 0.8
        self.converged_threshold = 0.001

        self.support_values = np.zeros((self.x_points * self.x_dot_points, 1))
        self.environment = environment
        self.visualise = visualise if visualise is not None else False

        self.states = np.zeros((self.x_points * self.x_dot_points, 2))
        for i in range(self.x_dot_points):
            for j in range(self.x_points):
                self.states[i * self.x_points + j][0] = self.x[j]
                self.states[i * self.x_points + j][1] = self.x_dot[i]
        
        self.num_fine_points = 250
        self.x_fine = np.linspace(self.x_min, self.x_max, num=self.num_fine_points)
        self.x_dot_fine = np.linspace(self.x_dot_max, self.x_dot_min, num=self.num_fine_points)
        self.states_fine = np.zeros((self.num_fine_points ** 2, 2))

        for i in range(250):
            for j in range(250):
                self.states_fine[i * 250 + j][0] = self.x_fine[j]
                self.states_fine[i * 250 + j][1] = self.x_dot_fine[i]

    def initialise_support_values(self):
        for i in range(self.x_points * self.x_dot_points):
            state = self.states[i]
            reward, _ = self.environment._reward(state, 0)
            self.support_values[i][0] = reward
    
    def learn_value_function(self, states, values):
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 100)) * RBF(length_scale=[0.1, 0.1], length_scale_bounds=(0.05, 10.0))
        gp_val = GaussianProcessRegressor(kernel=kernel, alpha=0.01) 
        gp_val = gp_val.fit(states, values)

        return gp_val

    def visualise_value_function(self, iter_num, maximising_actions=None, show_fig=False):
        if not self.visualise:
            return

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        X, X_DOT = np.meshgrid(self.x_fine, self.x_dot_fine)
        predicted_vals = self.gp_val.predict(self.states_fine).reshape((self.num_fine_points, self.num_fine_points))
        surf = ax.plot_surface(X, X_DOT, predicted_vals, cmap=cm.rainbow, antialiased=True, linewidth=0.001)

        # Customize the z axis.
        ax.set_zlim(np.amin(predicted_vals), np.amax(predicted_vals))
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig("gp%s.png" % iter_num, dpi=300)

        plt.clf()
        contour = plt.contourf(X, X_DOT, predicted_vals)
        plt.colorbar(contour, shrink=0.5)
        plt.savefig("values%s.png" % iter_num, dpi=300)
        
        if maximising_actions is not None:
            plt.clf()
            X, X_DOT = np.meshgrid(self.x, self.x_dot)
            maximising_actions = maximising_actions.reshape((self.x_points, self.x_dot_points))
            contour = plt.contourf(X, X_DOT, maximising_actions)
            plt.colorbar(contour, shrink=0.5)
            plt.savefig("actions%s.png" % iter_num, dpi=300)

        if show_fig:
            plt.show()
    
    def learn_dynamics(self, num_dynamics_examples):
        x = np.random.uniform(low=self.x_min, high=self.x_max, size=num_dynamics_examples)
        x_dot = np.random.uniform(low=self.x_dot_min, high=self.x_dot_max, size=num_dynamics_examples)
        a = np.random.uniform(low=self.a_min, high=self.a_max, size=num_dynamics_examples)

        start_states = list(zip(x, x_dot, a))
        next_xs, next_x_dots = [], []

        for state in start_states:
            x, x_dot, a = state
            self.environment.reset(state=[x,x_dot])
            self.environment.step(a)
            next_x, next_x_dot = self.environment.get_state()
            next_xs.append(next_x)
            next_x_dots.append(next_x_dot)
        
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=[0.25, 0.25, 0.25], length_scale_bounds=(1e-3, 20)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 10.0))
        
        gp_x = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        gp_x.fit(start_states, next_xs)

        gp_x_dot = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        gp_x_dot.fit(start_states, next_x_dots)

        return gp_x, gp_x_dot

    def test_learn_dynamics(self):
        num_training_examples = [25, 50, 75, 100, 125, 150, 200, 250, 300]
        num_validation_examples = 1000

        x = np.random.uniform(low=self.x_min, high=self.x_max, size=num_validation_examples)
        x_dot = np.random.uniform(low=self.x_dot_min, high=self.x_dot_max, size=num_validation_examples)
        a = np.random.uniform(low=self.a_min, high=self.a_max, size=num_validation_examples)

        start_states = list(zip(x, x_dot, a))
        next_xs, next_x_dots = [], []
        x_rmss, x_dot_rmss = [], []

        for state in start_states:
            x, x_dot, a = state
            self.environment.reset(state=[x,x_dot])
            self.environment.step(a)
            next_x, next_x_dot = self.environment.get_state()
            next_xs.append(next_x)
            next_x_dots.append(next_x_dot)

        for num_examples in num_training_examples:
            x_rms, x_dot_rms = 0, 0
            num_reps = 10
            
            for i in range(num_reps):
                gp_x, gp_x_dot = self.learn_dynamics(num_examples)
                estimated_next_xs = gp_x.predict(start_states)
                estimated_next_x_dots = gp_x_dot.predict(start_states)
                x_rms += math.sqrt(mean_squared_error(next_xs, estimated_next_xs))
                x_dot_rms += math.sqrt(mean_squared_error(next_x_dots, estimated_next_x_dots))
            
            x_dot_rmss.append(x_dot_rms/num_reps)
            x_rmss.append(x_rms/num_reps)

            print("Number of training samples: %s, x_rms: %s, x_dot_rms: %s" % (num_examples, x_rms, x_dot_rms))
        
        x_rms_handle, = plt.plot(num_training_examples, x_rmss, label="X")
        x_dot_rms_handle, = plt.plot(num_training_examples, x_dot_rmss, label="X_DOT")
        plt.legend(handles=[x_rms_handle, x_dot_rms_handle])
        plt.ylabel('RMS Error')
        plt.xlabel('Number of Training Examples')
        plt.show()
    
    def learn(self):
        num_training_examples = 50
        
        # Try loading prelearned value function
        try:
            with open ('support_values', 'rb') as fp:
                print("successfully loaded support_values file")
                self.support_values = pickle.load(fp)
                self.gp_x, self.gp_x_dot = self.learn_dynamics(num_training_examples)
                self.gp_val = self.learn_value_function(self.states, self.support_values)
                return
        except FileNotFoundError:
            print("support_values file not found")
            self.initialise_support_values()

        self.gp_val = self.learn_value_function(self.states, self.support_values)
        self.gp_x, self.gp_x_dot = self.learn_dynamics(num_training_examples)

        converged = False
        iter_num = 1

        while not converged:
            k_v_chol = self.gp_val.L_.dot(self.gp_val.L_.T)
            k_v_inv = np.linalg.inv(k_v_chol)

            v_squared, l1, l2 = np.exp(self.gp_val.kernel_.theta)

            print("Learned GP hyperparameters: v_squared: %s, l1: %s, l2: %s" % (v_squared, l1, l2))

            R = np.zeros((self.x_points * self.x_dot_points, 1))
            W = np.zeros((self.x_points * self.x_dot_points, self.x_points * self.x_dot_points))
            maximising_actions = np.zeros((self.x_points * self.x_dot_points, 1))
            
            for state_index in range(len(self.states)):
                x, x_dot = self.states[state_index]

                max_val_index, r, w = self.find_max_action(x, x_dot)

                maximising_actions[state_index][0] = self.a[max_val_index]
                R[state_index][0] = r
                W[state_index] = w
        
            intermediate1 = np.eye(self.x_points * self.x_dot_points) - self.gamma * W.dot(k_v_inv)
            intermediate2 = np.linalg.inv(intermediate1)
            new_v = intermediate2.dot(R)

            change_in_val = mean_squared_error(self.support_values, new_v)
            print("rms change in support point values: %s" % (change_in_val))

            if change_in_val < self.converged_threshold:
                converged = True

            self.support_values = new_v
            self.gp_val = self.learn_value_function(self.states, self.support_values)
            self.visualise_value_function(maximising_actions=maximising_actions, iter_num=iter_num)
            iter_num += 1
        
            with open('support_values', 'wb') as fp:
                pickle.dump(self.support_values, fp)

    def find_max_action(self, x, x_dot):
        target = np.array([self.environment.goal_position, self.environment.goal_velocity]).reshape((-1,1))
        # target = np.array(self.environment.target).reshape((-1,1))
        k_v_inv_v = self.gp_val.alpha_

        v_squared, l1, l2 = np.exp(self.gp_val.kernel_.theta)
        lengths = np.array([l1, l2]).reshape((-1,1))
        lengths_squared = np.square(lengths)

        state_actions = np.array(list(itertools.product([x], [x_dot], self.a)))

        mu_x, std_dev_x = self.gp_x.predict(state_actions, return_std=True)
        mu_x_dot, std_dev_x_dot = self.gp_x_dot.predict(state_actions, return_std=True)

        means = np.array([mu_x, mu_x_dot])
        var = np.square(np.array([std_dev_x, std_dev_x_dot]))
        length_squared_plus_var = np.add(var, lengths_squared)
        state_diffs = np.subtract(self.states[:,:, np.newaxis], means)
        state_diffs_squared = np.square(state_diffs)
        state_diffs_squared_divided_length_plus_var = np.divide(state_diffs_squared, length_squared_plus_var)
        summed = -0.5 * np.sum(state_diffs_squared_divided_length_plus_var, axis=1)
        exponentiated = np.exp(summed)
        product = np.prod(length_squared_plus_var, axis=0)
        square_root = np.sqrt(product)

        w = np.prod(lengths) * v_squared * np.divide(exponentiated, square_root)

        target_minus_mean = np.subtract(target, means)
        squared_target_minus_mean = np.square(target_minus_mean)
        var_plus = var + self.environment.gaussian_reward_length_scale ** 2
        divided = np.divide(squared_target_minus_mean, var_plus)
        summed = -0.5 * np.sum(divided, axis=0)
        exponentiated = np.exp(summed)
        product = np.prod(var_plus, axis=0)
        square_root = np.sqrt(product)
        r = np.divide(exponentiated, square_root)
        r /= (2 * math.pi * 63.66) # Note renormalising

        v = self.gamma * w.T.dot(k_v_inv_v)
        val_i = r + v.T

        max_val_index = np.argmax(val_i)

        return max_val_index, r[max_val_index], w[:,max_val_index]
    
    def act(self, env_state):
        current_x, current_x_dot = env_state
        max_val_index, _, _ = self.find_max_action(current_x, current_x_dot)

        return self.a[max_val_index]
    
    def plot_actions(self, xs, x_dots):
        redline = mlines.Line2D([], [], color='red', label="Actual")
        predicted, = plt.plot(xs, x_dots, label="Predicted")
        start, = plt.plot([-0.5], [0.0], marker='*', markersize=10, color="red", label="Start")
        end, = plt.plot([0.6], [0.0], marker='o', markersize=10, color="green", label="Finish")
        
        plt.legend(handles=[redline, predicted, start, end])
        
        plt.xlabel("x")
        plt.ylabel("dx")

        plt.xlim([-1.5,1.5])
        plt.ylim([-2.5,2.5])
        
        plt.title('Trajectory')
        
        plt.show()
