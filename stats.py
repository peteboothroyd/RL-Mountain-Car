from collections import deque
import numpy as np
import tensorflow as tf

class Stats(object):
  def __init__(self, summary_every):
    self._mean_ep_length_history = []
    self._mean_reward_history = []

    # Store mean rewards between summaries to report progress
    self._reward_buffer = deque(maxlen=summary_every)
    self._ep_length_buffer = deque(maxlen=summary_every)

  def store_episode_stats(self, rollout_rewards, summarise=False):
    ''' Store important stats from the episode for later.
    '''
    num_rollouts = len(rollout_rewards)

    total_steps = 0
    for rollout in rollout_rewards:
      total_steps += len(rollout)

    mean_episode_length = total_steps*1.0/num_rollouts
    self._ep_length_buffer.append(mean_episode_length)

    rewards = np.concatenate([np.array(reward).reshape(
        (-1,)) for reward in rollout_rewards])
    mean_reward = np.mean(rewards)
    self._reward_buffer.append(mean_reward)

    if summarise:
      self._mean_ep_length_history.append(np.mean(self._ep_length_buffer))
      self._mean_reward_history.append(np.mean(self._reward_buffer))
      