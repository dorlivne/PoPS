import random
from collections import namedtuple
from utils.Segment_tree import SumSegmentTree, MinSegmentTree
import numpy as np


class Memory:

    def __init__(self, size):
        self.size = size
        self.currentPosition = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.newStates = []
        self.finals = []

    def getMiniBatch(self, size):
        indices = random.sample(population=range(len(self.states)), k=min(size, len(self.states)))
        miniBatch = []
        for index in indices:
            miniBatch.append({'state': self.states[index], 'action': self.actions[index], 'reward': self.rewards[index],
                              'newState': self.newStates[index], 'isFinal': self.finals[index]})
        return miniBatch

    def getCurrentSize(self):
        return len(self.states)

    def getMemory(self, index):
        return {'state': self.states[index], 'action': self.actions[index], 'reward': self.rewards[index],
                'newState': self.newStates[index], 'isFinal': self.finals[index]}

    def addMemory(self, state, action, reward, newState, isFinal):
        if (self.currentPosition >= self.size - 1):
            self.currentPosition = 0
        if (len(self.states) > self.size):
            self.states[self.currentPosition] = state
            self.actions[self.currentPosition] = action
            self.rewards[self.currentPosition] = reward
            self.newStates[self.currentPosition] = newState
            self.finals[self.currentPosition] = isFinal
        else:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.newStates.append(newState)
            self.finals.append(isFinal)

        self.currentPosition += 1


Supervised_Experience = namedtuple('Supervised_Experience', ['state', 'label', 'action'])
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done', 'next_state'])


class ExperienceReplay:
    """
    This class provides an abstraction to store the [s, a, r, s'] elements of each iteration.
    using Experience object which contains the s-a-r-s-a transition information in an object oriented way
    """

    def __init__(self, size):
            self.size = size
            self.currentPosition = 0
            self.buffer = []

    def add_memory(self, state, action, reward, next_state, is_done):
        exp = Experience(state, action, reward, is_done, next_state)
        if len(self.buffer) < self.size:
            self.buffer.append(exp)
        else:
            self.buffer[self.currentPosition] = exp
        self.currentPosition = (self.currentPosition + 1) % self.size

    def getMiniBatch(self, batch_size):
        indices = random.sample(population=range(len(self.buffer)), k=min(batch_size, len(self.buffer)))
        return [self.buffer[index] for index in indices]


class Supervised_ExperienceReplay(ExperienceReplay):

    """
        This class provides an object to store the [state, Supervised_Q_value] values of each iteration
    """
    def __init__(self, size):
        super(Supervised_ExperienceReplay, self).__init__(size)

    def add_memory(self, state, Supervised_Q_value, action):
        exp = Supervised_Experience(state, Supervised_Q_value, action)
        if len(self.buffer) < self.size:
            self.buffer.append(exp)
        else:
            self.buffer[self.currentPosition] = exp
        self.currentPosition = (self.currentPosition + 1) % self.size


class ExperienceReplayMultistep(ExperienceReplay):
    """
    Multi-step experience replay reviewed in the Rainbow paper, this is basically TD(lamda) that is taught in Silver course.
    we accumulate N steps in episode buffer, and then store a transition based on the first and last state of the N step series
    this is done by the add_to_buffer signal which tells us when to store a transition in buffer
    Note:
        if N is very large it can introduce high variance to the training phase, so be careful with the number of steps.
    """
    def __init__(self, size, gamma):
        super(ExperienceReplayMultistep, self).__init__(size)
        self.gamma = gamma
        self.episode_buffer = []


    def create_first_last_exp(self):
        if self.episode_buffer[-1].done and len(self.episode_buffer) <= 1: # address a special case at restart
            last_state = None
        else:
            last_state = self.episode_buffer[-1].next_state
        total_reward = 0.0
        for exp in reversed(self.episode_buffer):
            total_reward *= self.gamma
            total_reward += exp.reward
        first_exp = self.episode_buffer[0]
        exp = Experience(state=first_exp.state, action=first_exp.action, reward=total_reward, next_state=last_state, done=self.episode_buffer[-1].done)
        self.buffer.append(exp)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)

    def add_memory(self, state, action, reward, next_state, is_done, add_to_buffer):
        exp = Experience(state, action, reward, is_done, next_state)
        self.episode_buffer.append(exp)
        if add_to_buffer or is_done:
            self.create_first_last_exp()
            self.episode_buffer.clear()


class PrioritizedExperienceReplay(ExperienceReplay):
    """
    taken from https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    with adjustments to our code
    """
    def __init__(self, size, alpha):
        super(PrioritizedExperienceReplay, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha
        it_capacity = 1
        while it_capacity < size: # it_capacity is a power of 2
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0


    def add_memory(self, state, action, reward, next_state, is_done):
        idx = self.currentPosition
        super(PrioritizedExperienceReplay, self).add_memory( state, action, reward, next_state, is_done)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha



    def _sample_proportional(self, batch_size):
            res = []
            p_total = self._it_sum.sum(0, len(self.buffer) - 1)  # total priority sum in tree
            every_range_len = p_total / batch_size  # divide into segments, we pick a transition from each segment
            for i in range(batch_size):
                mass = random.random() * every_range_len + i * every_range_len  # mass depicts the rank of the transition
                idx = self._it_sum.find_prefixsum_idx(mass)  # the index of the transition in the tree
                res.append(idx)

            return res

    def getMiniBatch(self, batch_size, beta):
        """Sample a batch of experiences.
               compared to ReplayBuffer.sample
               it also returns importance weights and idxes
               of sampled experiences.
               Parameters
               ----------
               batch_size: int
                   How many transitions to sample.
               beta: float
                   To what degree to use importance weights
                   (0 - no corrections, 1 - full correction)
        Returns:
         1) batch with s-a-r-s-a transitions represented as experience object
         2) weights for each of those transactions
         3) their indexes
        """
        assert beta > 0
        N = len(self.buffer)
        # function to sample via probability of the transactions
        indexes = self._sample_proportional(batch_size)
        weights = []
        batch_transitions = []
        sum = self._it_sum.sum()
        prob_min = self._it_min.min() / sum
        max_weight = (prob_min * N) ** (-beta) # according to PER paper,
                                                        # max weight is used to normalize the weights
        for idx in indexes:
            prob_sample = self._it_sum[idx] / sum
            weight = (prob_sample * N) ** (-beta) # fixes the bias high prob transaction introduce
            weights.append(weight)
            batch_transitions.append(self.buffer[idx])
        weights /= np.ones_like(weights) * max_weight  # normalize
        return batch_transitions, weights, indexes



    def update_priorities(self,indexes, priorities):
        """Update priorities of sampled transitions.
                sets priority of transition at index indexes[i] in buffer
                to priorities[i].
                Parameters
                ----------
                indexes: [int]
                    List of indexes of sampled transitions
                priorities: [float]
                    List of updated priorities corresponding to
                    transitions at the sampled idxes denoted by
                    variable `idxes`.
                """

        assert len(indexes) == len(priorities)
        for index, priority in zip(indexes, priorities):
            assert priority > 0 and 0 <= index < len(self.buffer)
            self._it_sum[index] = priority ** self._alpha
            self._it_min[index] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)



class MultiStepPrioritizedExperienceReplay(PrioritizedExperienceReplay):

    def __init__(self, size, alpha, gamma):
        super(MultiStepPrioritizedExperienceReplay, self).__init__(size=size, alpha=alpha)
        self.gamma = gamma
        self.episode_buffer = []

    def create_first_last_exp(self):
        if self.episode_buffer[-1].done and len(self.episode_buffer) <= 1:  # address a special case at restart
            last_state = None
        else:
            last_state = self.episode_buffer[-1].next_state
        total_reward = 0.0
        for exp in reversed(self.episode_buffer):
            total_reward *= self.gamma
            total_reward += exp.reward
        first_exp = self.episode_buffer[0]
        super(MultiStepPrioritizedExperienceReplay, self).add_memory(state=first_exp.state,
                                                                     action=first_exp.action,
                                                                     reward=total_reward,
                                                                     next_state=last_state,
                                                                     is_done=self.episode_buffer[-1].done)

    def add_memory(self, state, action, reward, next_state, is_done, add_to_buffer):
        exp = Experience(state, action, reward, is_done, next_state)
        self.episode_buffer.append(exp)
        if add_to_buffer or is_done:
            self.create_first_last_exp()
            self.episode_buffer.clear()



class Supervised_Prioritzed_ExperienceReplay(PrioritizedExperienceReplay):
    """
            This class provides an object to store the [state, Supervised_Q_value] values of each iteration
            also, this class incorporates Priority for the experience replay for better training with policy_distillation
    """

    def __init__(self, size, alpha):
            super(Supervised_Prioritzed_ExperienceReplay, self).__init__(size=size, alpha=alpha)

    def add_memory(self, state, Supervised_Q_value, action):
            exp = Supervised_Experience(state, Supervised_Q_value, action)
            idx = self.currentPosition
            if len(self.buffer) < self.size:
                self.buffer.append(exp)
            else:
                self.buffer[self.currentPosition] = exp
            self.currentPosition = (self.currentPosition + 1) % self.size
            self._it_sum[idx] = self._max_priority ** self._alpha
            self._it_min[idx] = self._max_priority ** self._alpha










