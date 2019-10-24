import unittest
from tensorflow.python.ops import math_ops, array_ops, nn_ops
from tensorflow.python.ops.rnn_cell_impl import GRUCell
import tensorflow as tf
import random

from model.consts import Const
from util.logger import DLogger
import numpy as np


def _linear(args, weights, biases):
    # res = math_ops.matmul(args, weights)
    # return nn_ops.bias_add(res, biases)

    res = tf.reduce_sum(tf.multiply(args, tf.transpose(weights, [3, 0, 1, 2])), axis=3)
    return tf.add(tf.transpose(res, [1, 2, 0]), biases)


class GRUCell2(GRUCell):

    def __init__(self,
                 num_units,
                 n_samples,
                 n_T,
                 W1=None, b1=None,
                 W2=None, b2=None,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(GRUCell2, self).__init__(num_units,
                                       activation,
                                       reuse,
                                       kernel_initializer,
                                       bias_initializer)
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.n_samples = n_samples
        self.n_T = n_T

        self.state_in = tf.placeholder(shape=[None, None, self._num_units], dtype=Const.FLOAT)

    @classmethod
    def get_weight_dims(cls, input_dim, state_dim):
        return [input_dim + state_dim, 2 * state_dim], \
               [2 * state_dim], \
               [input_dim + state_dim, state_dim], \
               [state_dim]

    def set_weights(self, W1, b1, W2, b2):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

    def call(self, inputs, state):
        value = math_ops.sigmoid(
            _linear(tf.concat([inputs, state], axis=2), self.W1, self.b1))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=2)
        c = self._activation(
            _linear(tf.concat([inputs, r * state], axis=2), self.W2, self.b2))
        new_h = u * state + (1 - u) * c
        return new_h, new_h

        # value = math_ops.sigmoid(
        #     _linear(tf.concat([inputs, state], axis=1), self.W1, self.b1))
        # r, u = array_ops.split(value=value, num_or_size_splits=2, axis=2)
        # c = self._activation(
        #     _linear(tf.concat([tf.tile(inputs[np.newaxis,], [tf.shape(r)[0], 1, 1]), r * state], axis=2), self.W2,
        #             self.b2))
        # new_h = u * state + (1 - u) * c
        # return new_h, new_h

    def dynamic_rnn(self, rnn_in, step_size):
        rnn_in = tf.tile(rnn_in[np.newaxis], [self.n_samples, 1, 1, 1])

        def learn(curr_trial, state_in, ta):
            input_in = rnn_in[:, :, curr_trial, :]
            _, gru_state = self.call(input_in, state_in)
            ta = ta.write(curr_trial, gru_state)
            return tf.add(curr_trial, 1), gru_state, ta

        def cond(curr_trial, _1, _2):
            return tf.less(curr_trial, step_size[0])

        while_output = tf.while_loop(cond, learn,
                                     loop_vars=[tf.constant(0),
                                                self.state_in,
                                                tf.TensorArray(dtype=Const.FLOAT, size=step_size[0])
                                                ]
                                     )
        _, _, ta = while_output
        ta = tf.transpose(ta.stack(), [1, 2, 0, 3])

        # remove the first dummy variables
        rnn_out = ta[:, :, :]
        last_state = ta[:, :, -1, :]

        return rnn_out, last_state

    def static_rnn(self, rnn_in, step_size):
        rnn_in = tf.tile(rnn_in[np.newaxis], [self.n_samples, 1, 1, 1])

        curr_trial = 0
        state_in = self.state_in
        state_outs = []

        while curr_trial < self.n_T + 1:
            input_in = rnn_in[:, :, curr_trial, :]
            _, state_in = self.call(input_in, state_in)
            state_outs.append(state_in)
            curr_trial += 1

        ta = tf.transpose(tf.stack(state_outs), [1, 2, 0, 3])

        # remove the first dummy variables
        rnn_out = ta[:, :, :]
        last_state = ta[:, :, -1, :]

        return rnn_out, last_state


class TestGRUCell2(unittest.TestCase):

    def test_gru(self):
        DLogger.remove_handlers()
        self.test_output_buf = DLogger.get_string_logger()

        self.run_rnn_grucell2()
        self.test_output_buf.seek(0)
        output1 = self.test_output_buf.read()
        self.test_output_buf.seek(0)

        tf.reset_default_graph()
        self.run_rnn_grucell()
        self.test_output_buf.seek(0)
        output2 = self.test_output_buf.read()
        #
        self.assertEqual(output1, output2)

    def run_rnn_grucell2(self):
        train_input, train_output = self.generate_data()

        DLogger.logger().debug("test and training data loaded")

        data = tf.placeholder(Const.FLOAT, [None, 20, 3])  # Number of examples, number of input, dimension of each input
        target = tf.placeholder(Const.FLOAT, [None, 21])
        num_hidden = 24
        n_samples = 5

        cell = GRUCell2(num_hidden, n_samples)
        n_bacthes = len(train_input)

        dW1, db1, dW2, db2 = cell.get_weight_dims(3, num_hidden)
        W1, b1, W2, b2 = tf.get_variable(name='w1', shape=[n_samples, n_bacthes] + dW1, initializer=tf.constant_initializer(0.1),
                                         dtype=Const.FLOAT), \
                         tf.get_variable(name='b1', shape=[n_samples, n_bacthes] + db1, initializer=tf.constant_initializer(0.1),
                                         dtype=Const.FLOAT), \
                         tf.get_variable(name='w2', shape=[n_samples, n_bacthes] + dW2, initializer=tf.constant_initializer(0.1),
                                         dtype=Const.FLOAT), \
                         tf.get_variable(name='b2', shape=[n_samples, n_bacthes] + db2, initializer=tf.constant_initializer(0.1),
                                         dtype=Const.FLOAT)

        cell.set_weights(W1, b1, W2, b2)

        step_size = tf.ones([tf.shape(data)[0]], dtype=tf.int32) * tf.shape(data)[1]

        state_track, last_state = cell.dynamic_rnn(data, step_size)
        state_track = tf.transpose(state_track[0], [1, 0, 2])

        init_op = tf.global_variables_initializer()
        with tf.Session() as  sess:
            sess.run(init_op)
            entropy = sess.run(state_track, {data: train_input, target: train_output})
            DLogger.logger().debug('Entropy:' + str(entropy))

    def run_rnn_grucell(self):
        train_input, train_output = self.generate_data()

        DLogger.logger().debug("test and training data loaded")

        data = tf.placeholder(Const.FLOAT,
                              [None, 20, 3])  # Number of examples, number of input, dimension of each input
        target = tf.placeholder(Const.FLOAT, [None, 21])
        num_hidden = 24

        cell = tf.nn.rnn_cell.GRUCell(num_hidden,
                                      kernel_initializer=tf.constant_initializer(0.1),
                                      bias_initializer=tf.constant_initializer(0.1))

        val, _ = tf.nn.dynamic_rnn(cell, data, dtype=Const.FLOAT)
        val = tf.transpose(val, [1, 0, 2])

        init_op = tf.global_variables_initializer()
        with tf.Session() as  sess:
            sess.run(init_op)
            entropy = sess.run(val, {data: train_input, target: train_output})
            DLogger.logger().debug('Entropy:' + str(entropy))

        sess.close()

    def generate_data(self):
        NUM_EXAMPLES = 1000
        train_input = ['{0:020b}'.format(i) for i in range(10)]
        np.random.seed(1010)
        random.Random(4).shuffle(train_input, )
        train_input = [map(int, i) for i in train_input]
        ti = []
        for i in train_input:
            temp_list = []
            for j in i:
                temp_list.append([j, j + 1, j + 2])
            ti.append(np.array(temp_list))
        train_input = ti
        train_output = []
        for i in train_input:
            count = 0
            for j in i:
                if j[0] == 1:
                    count += 1
            temp_list = ([0] * 21)
            temp_list[count] = 1
            train_output.append(temp_list)
        train_input = train_input[:NUM_EXAMPLES]
        train_output = train_output[:NUM_EXAMPLES]
        return train_input, train_output


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGRUCell2)
    unittest.TextTestRunner(verbosity=2).run(suite)
