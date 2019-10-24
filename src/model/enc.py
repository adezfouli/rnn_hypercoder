import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import LSTMCell, MultiRNNCell

from model.consts import Const
from model.model_beh import ModelBeh
from util import DLogger


class ENCRNN(ModelBeh):
    def __init__(self, n_cells, a_size, s_size, latent_size, n_samples, n_T, static_loop):

        super().__init__(a_size, s_size)
        DLogger.logger().debug("model created with ncells: " + str(n_cells))

        self.static_loop = static_loop

        self.n_T = n_T

        self.n_samples = n_samples

        self.n_cells = n_cells

        self.latent_size = latent_size

        self.seq_lengths = tf.placeholder(dtype=tf.int32, shape=[None])

        self.rnn_out, _ = self.rnn_cells(n_cells, self.rnn_in)

    def rnn_cells(self, n_cells, rnn_in):
        # RNN component
        # with tf.variable_scope('enc'):
        #     gru = CudnnCompatibleLSTMCell(n_cells)
        #     # step_size = tf.ones([tf.shape(self.prev_actions)[0]], dtype=tf.int32) * tf.shape(self.prev_actions)[1]
        #     output, state = tf.nn.dynamic_rnn(
        #         gru, rnn_in, time_major=False, sequence_length=self.seq_lengths + 1, dtype=Const.FLOAT
        #
        #     )
        #
        # return output, state

        with tf.variable_scope('enc'):
            fw_gru = LSTMCell(n_cells)
            bw_gru = LSTMCell(n_cells)

            if not self.static_loop:
                output, state = tf.nn.bidirectional_dynamic_rnn(
                    fw_gru, bw_gru, rnn_in, time_major=False, sequence_length=self.seq_lengths + 1, dtype=Const.FLOAT)

                return tf.concat(output, axis=2), tf.stack((tf.concat(state[0], axis=1), tf.concat(state[1], axis=1)),
                                                             axis=0)
            else:
                output_static, state_fw, state_bw = tf.nn.static_bidirectional_rnn(
                    fw_gru, bw_gru, tf.unstack(rnn_in, num=self.n_T + 1, axis=1),
                    sequence_length=self.seq_lengths + 1, dtype=Const.FLOAT)

                st1, st2 = tf.concat(tf.stack(output_static, axis=1), axis=2), \
                           tf.stack((tf.concat(state_fw, axis=1), tf.concat(state_bw, axis=1)), axis=0)

                return st1, st2

    def enc_beh_feed(self, actions, rewards, states, seq_lengths):
        dict = super().beh_feed(actions, rewards, states)
        dict[self.seq_lengths] = seq_lengths
        return dict
