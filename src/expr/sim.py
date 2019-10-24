import tensorflow as tf

from actionflow.util.export import Export
from expr.BD import BD
from expr.envs import a2_generic
from model.consts import Const
from model.dec import DECRNN
from model.mmdae_enc import MMDAE
from util.helper import ensure_dir, format_to_training_data
from util.logger import LogFile
import numpy as np
import pandas as pd

class Sim:

    @classmethod
    def create_onpolicy(cls, n_cells,z, n_T, output_path, model_path, mode="ossi"):
        tf.reset_default_graph()
        tf.set_random_seed(1)

        dec = cls.get_enc_dec(n_cells, z)

        ensure_dir(output_path)

        with LogFile(output_path, 'run.log'):
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(model_path)

            with tf.Session() as sess:
                saver.restore(sess, ckpt.model_checkpoint_path)
                freq = 1
                baseline = 4
                if mode == 'A1':
                    a1_period = lambda trial: trial < 15
                elif mode == 'A2':
                    a1_period = lambda trial: False
                else:
                    raise Exception("Unknown mode")
                states, policies, rewards, choices, rnn_states = dec.simulate_env(sess, 10,
                              a2_generic(a1_period,
                                         lambda trial: trial < 1,
                                         lambda trial: trial in [],
                                         init_state=None,
                                         init_action=-1,
                                         init_reward=0
                                         ),
                              greedy=True
                              )

                train = format_to_training_data(rewards, choices, states)

                if output_path is not None:
                    Export.policies({'id1': {'1': policies}}, output_path, 'policies-.csv')
                    Export.export_train(train, output_path, 'train.csv')
                    np.savetxt(output_path + 'z.csv', z[0], delimiter=',')


    @classmethod
    def create_offpolicy(cls, n_cells, z, n_T, output_path, model_path, mode="ossi"):
        tf.reset_default_graph()
        tf.set_random_seed(1)

        dec = cls.get_enc_dec(n_cells, z)

        ensure_dir(output_path)

        with LogFile(output_path, 'run.log'):
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(model_path)

            with tf.Session() as sess:
                saver.restore(sess, ckpt.model_checkpoint_path)
                if mode == 'A1':
                    a1_period = lambda trial: trial < 15
                elif mode == 'A2':
                    a1_period = lambda trial: False
                else:
                    raise Exception("Unknown mode")
                states, policies, rewards, choices, rnn_states = dec.simulate_env(sess, 10,
                              a2_generic(a1_period,
                                         lambda trial: True,
                                         lambda trial: trial in [5],
                                         init_state=None,
                                         init_action=-1,
                                         init_reward=0
                                         ),
                              greedy=True
                              )

                train = format_to_training_data(rewards, choices, states)

                if output_path is not None:
                    Export.policies({'id1': {'1': policies}}, output_path, 'policies-.csv')
                    Export.export_train(train, output_path, 'train.csv')
                    np.savetxt(output_path + 'z.csv', z[0], delimiter=',')


    @classmethod
    def get_enc_dec(cls, n_cells, z):
        enc = MMDAE(n_cells, 2, 0, 2, 0, False, 2)
        dec = DECRNN(3, 2, 0, 1, tf.constant(z, dtype=Const.FLOAT), 0, False)
        return dec

    @classmethod
    def generate_onpolicy(cls, n_cells, model_path, output_path):
        action, reward, state, ids, _ = BD.get_data()
        n_T = action.shape[1]

        for i in range(2):
            j = 0
            for k in np.linspace(-1.2, 1.2, num=15):
                z = np.zeros(shape=(1, 1, 2)) + 0
                z[0, 0, i] = k

                sub_path = output_path + '/dims/A1/' + 'z' + str(i) + '/_' + str(j) + '/'
                cls.create_onpolicy(n_cells, z, n_T, sub_path, model_path, mode="A1")
                pd.DataFrame({'z_dim': [i], 'other_dim': 1 - i}).to_csv(sub_path + 'z_info.csv')

                sub_path = output_path + '/dims/A2/' + 'z' + str(i) + '/_' + str(j) + '/'
                cls.create_onpolicy(n_cells, z, n_T, sub_path, model_path, mode="A2")
                pd.DataFrame({'z_dim': [i], 'other_dim': 1 - i}).to_csv(sub_path + 'z_info.csv')
                j += 1


    @classmethod
    def generate_offpolicy(cls, n_cells, model_path, output_path):
        action, reward, state, ids, _ = BD.get_data()
        n_T = action.shape[1]

        for i in range(2):
            j = 0
            for k in np.linspace(-1.2, 1.2, num=15):
                z = np.zeros(shape=(1, 1, 2)) + 0
                z[0, 0, i] = k

                sub_path = output_path + '/dims/A1/' + 'z' + str(i) + '/_' + str(j) + '/'
                cls.create_offpolicy(n_cells, z, n_T, sub_path, model_path, mode="A1")
                pd.DataFrame({'z_dim': [i], 'other_dim': 1 - i}).to_csv(sub_path + 'z_info.csv')

                sub_path = output_path + '/dims/A2/' + 'z' + str(i) + '/_' + str(j) + '/'
                cls.create_offpolicy(n_cells, z, n_T, sub_path, model_path, mode="A2")
                pd.DataFrame({'z_dim': [i], 'other_dim': 1 - i}).to_csv(sub_path + 'z_info.csv')
                j += 1


if __name__ == '__main__':

    model_path = '../nongit/archive/BD/symms-forpaper/align/aling2-h.1-mm2/model/iter-141/'
    Sim.generate_offpolicy(20, model_path, '../nongit/local/BD/sims/off/')
    Sim.generate_onpolicy(20, model_path, '../nongit/local/BD/sims/on/')

    model_path = '../nongit/archive/synth-random/symms-rep50cells-forpaper/align-h1-50cells/model/iter-224/'
    Sim.generate_offpolicy(50, model_path, '../nongit/local/synth/sims/align/')

    model_path = '../nongit/archive/synth-random/symms-rep50cells-forpaper/align-h1-50cells/model/iter-init/'
    Sim.generate_offpolicy(50, model_path, '../nongit/local/synth/sims/noalign/')