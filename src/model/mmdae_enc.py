from tensorflow.python.layers.normalization import batch_normalization
from tensorflow.python.ops.gen_nn_ops import softplus

from model.consts import Const
from model.enc import ENCRNN
import tensorflow as tf
import numpy as np

from util.helper import tf_cov
from util.losses import mmd_loss, maximum_mean_discrepancy
import tensorflow_probability as tfp


class MMDAE(ENCRNN):
    def __init__(self, n_cells, a_size, s_size, latent_size, n_T, static_loops, mmd_loss_coef):
        super().__init__(n_cells, a_size, s_size, latent_size, 1, n_T, static_loops)

        with tf.variable_scope('enc'):
            batch_range = tf.range(tf.shape(self.rnn_in)[0])
            indices = tf.stack([batch_range, self.seq_lengths], axis=1)

            out = tf.gather_nd(self.rnn_out[:, :, :], indices)

            # normalizing the output by seq lengths
            # out = out / tf.cast(self.seq_lengths[:, np.newaxis], Const.FLOAT)
            # out = tf.reshape(self.rnn_out, [tf.shape(self.rnn_out)[0], (n_T + 1) * n_cells * 2])

            dense1 = tf.layers.dense(inputs=out, units=n_cells, activation=tf.nn.relu)
            dense2 = tf.layers.dense(inputs=dense1, units=n_cells, activation=tf.nn.relu)
            dense5 = tf.layers.dense(inputs=dense2, units=10, activation=tf.nn.softplus)

            #DIM: 1 * nBatches * nLatent
            self.z = tf.layers.dense(inputs=dense5, units=latent_size, activation=None)[np.newaxis]

            # added diagonal noise for better numerical stability
            self.z_cov = tf_cov(self.z[0]) + tf.eye(num_rows=tf.shape(self.z)[2], dtype=Const.FLOAT) * 1e-6

            true_samples = tf.random_normal(tf.stack([3000, self.z.shape[2]]), dtype=Const.FLOAT)

            self.loss = mmd_loss_coef * mmd_loss(true_samples, self.z[0], 1) + tfp.distributions.kl_divergence(
                tfp.distributions.MultivariateNormalFullCovariance(loc=tf.reduce_mean(self.z[0], axis=0),
                                                                   covariance_matrix=self.z_cov
                                                                   ),
                tfp.distributions.MultivariateNormalDiag(loc=tf.zeros((self.z.shape[2]), dtype=Const.FLOAT))

            )

            self.dc_loss = tf.constant(0)
            self.z_pred = self.z[0]

    @staticmethod
    def compute_kernel(x, y):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
        tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
        return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, Const.FLOAT))

    @staticmethod
    def compute_mmd(x, y):
        x_kernel = MMDAE.compute_kernel(x, x)
        y_kernel = MMDAE.compute_kernel(y, y)
        xy_kernel = MMDAE.compute_kernel(x, y)
        return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)
