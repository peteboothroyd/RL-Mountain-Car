import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import utils
import logger
from collections import deque

class MlpPolicy(object):
    def __init__(self, sess, ob_space, action_space, optimizer, exploration=0.2):
        '''Define the tensorflow graph to be used for the policy.'''
        self.sess = sess
        self.ob_space = ob_space
        self.action_space = action_space
        self.optimizer = optimizer

        num_actions = action_space.shape[0]
        ob_shape = (None,) + ob_space.shape

        with tf.name_scope('input'):
            self.states = tf.placeholder(tf.float32, shape=ob_shape, name='obs')

        mean, std_dev = self._mlp(self.states, [64, 64], num_actions, 'policy_network')

        tf.summary.histogram('mean', mean)
        tf.summary.histogram('std_dev', std_dev)
        
        with tf.name_scope('predict_actions'):
            dist = tf.distributions.Normal(loc=mean, scale=std_dev, name='normal')
            self.sample_action = dist.sample(name='sample_action')
            self.sample_action = tf.nn.tanh(self.sample_action)
            self.sample_action = self.sample_action * (action_space.high[0]-action_space.low[0]) * 0.5 + \
                                 (action_space.high[0]+action_space.low[0]) * 0.5

        with tf.name_scope('compute_gradients'):
            self.actions = tf.placeholder(tf.float32, shape=[None,], name='actions')
            self.returns = tf.placeholder(tf.float32, shape=[None,], name='returns')
            log_prob = dist.log_prob(self.actions, name='log_prob')
            # TODO: Deal with case of multiple episodes here
            loss = tf.reduce_mean(-log_prob * self.returns, name='loss')
            # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            # loss += tf.reduce_sum(reg_losses)
            # loss -= tf.reduce_mean(dist.entropy(), name='entropy') * exploration
        
            # Compute gradients
            gradients = optimizer.compute_gradients(loss)

            for grad, var in gradients:
                tf.summary.histogram(var.name, var)
                if grad is not None:
                    tf.summary.histogram(var.name + '/gradient', grad)

        with tf.name_scope('loss'):
            tf.summary.scalar('loss', loss)

        # training update
        with tf.name_scope('train_policy_network'):
            # apply gradients to update policy network
            # self.train_op = self.optimizer.apply_gradients(gradients,
            #                                                global_step=tf.train.get_or_create_global_step())
            self.train_op = self.optimizer.minimize(loss,
                                                    global_step=tf.train.get_or_create_global_step())

        self.summaries = tf.summary.merge_all()

        self.reset()
    
    def pi(self, observation):
        ''' Return the action and negative log probability of that action

        # Params:
            observation: The observation input to the policy

        # Returns:
            a: Action
            nlp: negative log probability of action
        '''
        return self.sess.run(self.sample_action, feed_dict={self.states: observation})
    
    def reset(self):
        ''' Reset the policy.'''
        self.sess.run(tf.global_variables_initializer())
    
    def train(self, returns, actions, states):
        ''' Train the policy'''
        self.sess.run(self.train_op, feed_dict={self.states: states, self.returns: returns, self.actions: actions})
    
    def summarize(self, returns, actions, states):
        '''Summarize key stats for TensorBoard'''
        return self.sess.run(self.summaries, feed_dict={self.states: states, self.returns: returns, self.actions: actions})
    
    def _mlp(self, input_layer, hidden_layers, num_actions, scope, beta=0.1): 
        with tf.variable_scope(scope):
            out = input_layer
            for num_units in hidden_layers:
                out = tf.layers.dense(out,units=num_units,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.glorot_normal_initializer(),
                                          bias_initializer=tf.zeros_initializer(),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(beta))
                # out = tf.layers.dropout(out, rate=0.5)

            mean = tf.layers.dense(out, units=num_actions,
                                        kernel_initializer=tf.glorot_normal_initializer(),
                                        bias_initializer=tf.zeros_initializer(),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(beta),
                                        name='mean')
            std_dev = tf.layers.dense(out, units=num_actions,
                                           activation=tf.nn.softplus,
                                           kernel_initializer=tf.glorot_normal_initializer(),
                                           bias_initializer=tf.zeros_initializer(),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(beta),
                                           name='std_dev')
            std_dev = std_dev + 1e-5
            return mean, std_dev

class PolicyGradientAgent(object):
    def __init__(self, env, visualise, model_dir, max_episode_steps, debug, seed=1, summary_every=50):
        self.debug = debug
        self.sess = tf.Session()

        if self.debug:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

        utils.set_global_seeds(seed)

        self.summary_every = summary_every
        self.visualise = visualise
        self.max_episode_steps = max_episode_steps

        self.env = env
        self.policy = MlpPolicy(self.sess, env.observation_space, env.action_space, tf.train.AdamOptimizer(1e-3))
        self.summary_writer = tf.summary.FileWriter(model_dir, graph=self.sess.graph)

        uninitialized_vars = self.sess.run(tf.report_uninitialized_variables())
        if uninitialized_vars.shape[0] != 0:
            print('Not all variables have been initialized!')

    def act(self, observation):
        return self.policy.pi(observation[np.newaxis,:])

    def learn(self, n_steps=int(1e6)):
        total_timesteps = 0
        episode_history = deque(maxlen=100)
        
        try:
            for episode in range(n_steps):
                visualise = episode % self.summary_every == 0
                rewards, actions, states = self._rollout(visualise)
                returns = self._compute_returns(rewards)
                # print('rewards:', np.array(rewards).shape, rewards[:5])
                np_actions = np.array(actions)
                print('action stats, mean:', np.mean(np_actions), 'std_dev', np.std(np_actions), 'max:', np.amax(np_actions), 'min:', np.amin(np_actions))
                # print('states:', np.array(states).shape, states[:5])
                # print('returns:', np.array(returns).shape, returns[:5], returns[-1])
                # print()

                episode_history.append(returns[-1])
                mean_returns = np.mean(episode_history)
                std_dev_returns = np.std(episode_history)

                self.policy.train(returns, actions, states)
                total_timesteps += len(rewards)

                if episode % self.summary_every == 0:
                    logger.record_tabular('episode', episode)
                    logger.record_tabular('average_returns', mean_returns)
                    logger.record_tabular('std_dev_returns', std_dev_returns)
                    logger.record_tabular('total_timesteps', total_timesteps)
                    logger.dump_tabular()
                    summary = self.policy.summarize(returns, actions, states)
                    self.summary_writer.add_summary(summary, episode)
                    # self.summary_writer.flush()
        finally:
            print('finally...')
            self.summary_writer.close()
            self.env.close()

    # TODO: Change to allow multiple rollouts
    def _compute_returns(self, rewards):
        '''Compute the returns given a rollout of immediate rewards. Used for REINFORCE algorithm.

        # Params:
            rewards ([tf.float32]): List of immediate rewards at different time steps

        # Returns:
            returns ([tf.float32]): List of returns for each trajectory $\sum_{t=0}^{T-1}r_t$
        '''
        ret = np.sum(rewards)
        out = np.ones_like(rewards)*ret
        return out

    # TODO: Change to allow multiple rollouts
    def _compute_future_returns(self, rewards):
        '''Compute the returns for a given rollout of immediate rewards. Used for GMDP algorithm

        # Params:
            rewards ([tf.float32]): List of immediate rewards at different time steps for a rollout

        # Returns:
            returns ([tf.float32]): List of returns from that timestep onwards $\sum_{h=t}^{T-1}r_h$
        '''
        return_sum = 0
        rollout_returns = []

        for reward in list(reversed(rewards)):
            return_sum += reward
            rollout_returns.append(return_sum)

        rollout_returns = list(reversed(rollout_returns))

        return rollout_returns

    # TODO: Change to allow multiple rollouts
    def _rollout(self, render=False):
        '''Generate a trajectory using current policy parameterisation. Return rollout.
        '''
        # TODO: Change back
        state = self.env.reset()
        rewards, actions, states = [], [], []
        for _ in range(self.max_episode_steps):
            if render:
                self.env.render()
            action = self.policy.pi(state[np.newaxis,:])
            state, reward, done, _ = self.env.step(action)
            rewards.append(np.squeeze(reward))
            states.append(state)
            actions.append(np.squeeze(action))

            if done:
                break
        
        return rewards, actions, states
