from model.consts import Const
from util import DLogger
import tensorflow as tf
import numpy as np


class ModelBeh:

    def __init__(self, a_size, s_size):
        DLogger.logger().debug("number of actions: " + str(a_size))
        DLogger.logger().debug("number of states: " + str(s_size))

        self.s_size = s_size
        self.a_size = a_size

        self.prev_actions = None

        # placeholders

        self.prev_rewards = self.get_pre_reward()
        # DIM: nBatches x (nChoices + 1) x 1

        self.prev_actions = None
        # DIM: nBatches x (nChoices + 1)

        # DIM: nBatches x (nChoices + 1) x nActionTypes
        self.prev_actions_onehot = self.get_prev_actions_onehot()

        # DIM: nBatches x nChoices x nActionTypes :removing the first dummy action
        self.actions_onehot = self.prev_actions_onehot[:, 1:, ]

        self.n_batches = tf.shape(self.prev_rewards)[0]

        if s_size != 0:
            self.prev_states = self.get_pre_state()
            # self.prev_states_onehot = tf.one_hot(self.prev_states, s_size, dtype=Const.FLOAT)
            rnn_in = tf.concat(values=[self.prev_rewards, self.prev_actions_onehot, self.prev_states], axis=2)
            # DIM: nBatches x (nChoices + 1 ) x ( 1 + nActionTypes + nStateTypes)

        else:
            rnn_in = tf.concat(values=[self.prev_rewards, self.prev_actions_onehot], axis=2)
            # DIM: nBatches x (nChoices + 1 ) x ( 1 + nActionTypes)

        self.rnn_in = rnn_in

    def beh_feed(self, actions, rewards, states):
        """
        Created a dict for TensorFlow by adding a dummy action and reward to the beginning of
        actions and rewards and a dummy state to the end of states
        """

        prev_rewards = np.hstack((np.zeros((rewards.shape[0], 1)), rewards))
        prev_actions = np.hstack((-1 * np.ones((actions.shape[0], 1)), actions))
        feed_dict = {self.prev_rewards: prev_rewards[:, :, np.newaxis],
                     self.prev_actions: prev_actions
                     }
        if states is not None:
            prev_states = np.hstack((states, np.zeros(states[:, 0:1].shape)))
            feed_dict[self.prev_states] = prev_states
        return feed_dict

    def get_pre_reward(self):
        return tf.placeholder(shape=[None, None, 1], dtype=Const.FLOAT)

    def get_pre_action(self):
        return tf.placeholder(shape=[None, None], dtype=tf.int32)

    def get_pre_timestep(self):
        return tf.placeholder(shape=[None, None], dtype=tf.int32)

    def get_pre_state(self):
        return tf.placeholder(shape=[None, None, self.s_size], dtype=Const.FLOAT)

    def get_prev_actions_onehot(self):
        self.prev_actions = self.get_pre_action()
        return tf.one_hot(self.prev_actions, self.a_size, dtype=Const.FLOAT, axis=-1)
