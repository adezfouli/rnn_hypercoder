import unittest

from expr.base_opt import BaseOpt
from expr.data_process import DataProcess
from model.consts import Const
from model.dec import DECRNN
import tensorflow as tf
from model.mmdae_enc import MMDAE
from model.opt import Opt
from util import DLogger
from util.helper import state_to_onehot, fix_init_all_vars
from util.test_env import TestLog, SaveLog
import numpy as np

class HYPMMD:

    def __init__(self, enc_cells, dec_cells, a_size, s_size, latent_size, n_T, static_loops, mmd_coef=2):

        self.enc = MMDAE(enc_cells, a_size, s_size, latent_size, n_T, static_loops, mmd_coef)
        self.dec = DECRNN(dec_cells, a_size, s_size, 1, self.enc.z, n_T, static_loops)

        # self.beta = tf.placeholder(dtype=Const.FLOAT, shape=())
        # self.loss = tf.reduce_mean(self.dec.loss + self.beta * self.enc.loss)


class TestRNN2RNN(unittest.TestCase):

    @classmethod
    def run_mmdae(cls):
        tf.reset_default_graph()
        tf.set_random_seed(12)
        action, reward, state, ids, seq_lengths = cls.get_data()

        rnn2rnn = HYPMMD(enc_cells=10, dec_cells=10, a_size=2, s_size=5, latent_size=3, n_T=action.shape[1],
                         static_loops=True)

        trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        beta = tf.placeholder(Const.FLOAT, shape=())
        lr = tf.placeholder(Const.FLOAT, shape=())

        opt_all = BaseOpt._get_apply_grads(rnn2rnn.dec.loss + beta * rnn2rnn.enc.loss, trainables, lr, tf.train.AdamOptimizer)


        with tf.Session() as sess:
            fix_init_all_vars(sess)


            cur_beta = {beta: 1}
            cur_lr = {lr: 0.1}

            enc_dict_feed = rnn2rnn.enc.enc_beh_feed(action, reward, state, seq_lengths)
            dec_dict_feed = rnn2rnn.dec.dec_beh_feed(action, reward, state, seq_lengths)

            for t in range(20):
                _, enc_loss, dec_loss, discr_loss, rand_loss, z_cov = sess.run([opt_all] + [
                    rnn2rnn.enc.loss,
                    rnn2rnn.dec.loss,
                    rnn2rnn.enc.dc_loss,
                    rnn2rnn.dec.sloss,
                    rnn2rnn.enc.z_cov
                ],
                                                                               feed_dict={**enc_dict_feed,
                                                                                          **dec_dict_feed, **cur_beta,
                                                                                          **cur_lr})

                DLogger.logger().debug("global iter = {:4d} "
                                       "enc loss: {:7.4f} "
                                       "dec loss: {:7.4f} "
                                       "discr loss: {:7.4f} "
                                       "rand loss: {:7.4f} "
                                       "beta: {:7.4f} "
                                       "LR: {:7.4f} "
                                       "z-cov: {}"
                                       .format(t, enc_loss, dec_loss, discr_loss, rand_loss, 1, 0.1,
                                               str(np.array2string(z_cov.flatten(), precision=3).replace('\n', ''))))

    @classmethod
    def get_data(cls):
        import pandas as pd
        data = pd.read_csv("test_data/choices_short.csv", header=0, sep=',', quotechar='"')
        data['action'] = [0 if x == 'R1' else 1 for x in data['action']]
        ids = data['id'].unique().tolist()
        dftr = pd.DataFrame({'id': ids[0:4], 'train': 'train'})
        tdftr = pd.DataFrame({'id': ids[2:4], 'train': 'test'})
        train, test = DataProcess.train_test_between_subject(data, pd.concat((dftr, tdftr)),
                                                             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 2)
        state_to_onehot(train, 5)
        state_to_onehot(test, 5)
        train = DataProcess.merge_data(train)

        action = train['merged'][0]['action']
        reward = train['merged'][0]['reward']
        state = train['merged'][0]['state']
        ids = train['merged'][0]['id']
        seq_lengths = train['merged'][0]['seq_lengths']

        return action, reward, state, ids, seq_lengths


    @classmethod
    def create_MMDAE_logs(cls):
        with SaveLog('test/rnn2rnn_mmdae.tst'):
            TestRNN2RNN.run_mmdae()

    def test_MMDAE_logs(self):
        with TestLog(self, 'test/rnn2rnn_mmdae.tst'):
            TestRNN2RNN.run_mmdae()

    @classmethod
    def create_HYPVAE_logs(cls):
        with SaveLog('test/rnn2rnn_vae.tst'):
            TestRNN2RNN.run_vae()

    def test_HYPVAE_logs(self):
        with TestLog(self, 'test/rnn2rnn_vae.tst'):
            TestRNN2RNN.run_vae()


    @classmethod
    def create_HYPAAE_logs(cls):
        with SaveLog('test/rnn2rnn_aae.tst'):
            TestRNN2RNN.run_aae()

    def test_HYPAAE_logs(self):
        with TestLog(self, 'test/rnn2rnn_aae.tst'):
            TestRNN2RNN.run_aae()

    @classmethod
    def create_true_outputs(cls):
        cls.create_HYPAAE_logs()
        cls.create_HYPVAE_logs()
        cls.create_MMDAE_logs()


if __name__ == '__main__':
    # TestRNN2RNN.create_true_outputs()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRNN2RNN)
    unittest.TextTestRunner(verbosity=2).run(suite)
