from expr.base_opt import BaseOpt
from expr.data_process import DataProcess
from expr.data_reader import DataReader
from model.consts import Const
from model.rnn2rnn import HYPMDD
from util.helper import ensure_dir, get_total_pionts
from util.logger import LogFile, DLogger
import tensorflow as tf
import pandas as pd


class Synth(BaseOpt):

    @classmethod
    def get_data(cls):
        data = DataReader.read_synth_normal()
        # data = data.loc[data.diag == 'Healthy']
        ids = data['id'].unique().tolist()
        dftr = pd.DataFrame({'id': ids, 'train': 'train'})
        tdftr = pd.DataFrame({'id': ids, 'train': 'test'})
        train, test = DataProcess.train_test_between_subject(data, pd.concat((dftr, tdftr)),
                                                             [1], 2)
        DLogger.logger().debug("total points: " + str(get_total_pionts(train)))
        train = DataProcess.merge_data(train)

        action = train['merged'][0]['action']
        reward = train['merged'][0]['reward']
        state = train['merged'][0]['state']
        ids = train['merged'][0]['id']
        seq_lengths = train['merged'][0]['seq_lengths']

        return action, reward, state, ids, seq_lengths


    @staticmethod
    def train_test(model_path):
        output = '../nongit/local/synth_normal/opt/'
        with LogFile(output, 'run.log'):
            action, reward, state, ids, seq_lengths = Synth.get_data()

            tf.reset_default_graph()

            model = HYPMDD(enc_cells=50, dec_cells=3, a_size=2, s_size=0, latent_size=2, n_T=action.shape[1],
                           static_loops=False, mmd_coef=50)

            ensure_dir(output)

            action, reward, state, seq_lengths, test_action, test_reward, test_state, test_seq_lengths, = \
                Synth.generate_train_test(action, reward, state, seq_lengths, ids, 0.3, output)

            def lr_schedule(t):
                if t < 2000:
                    _lr = 0.001
                elif t < 5000:
                    _lr = 0.001
                else:
                    _lr = 0.0001
                return _lr

            Synth.opt_model_mddae(model,
                                 action, reward, state, seq_lengths,
                                 test_action, test_reward, test_state, test_seq_lengths,
                                 output + '/model/', model_path, hessian_term=False,
                                 lr_schedule=lr_schedule)


    @staticmethod
    def aling_model(model_path):
        output = '../nongit/local/synth_normal/align/'
        with LogFile(output, 'run.log'):
            action, reward, state, ids, seq_lengths = Synth.get_data()

            tf.reset_default_graph()

            model = HYPMDD(enc_cells=50, dec_cells=3, a_size=2, s_size=0, latent_size=2, n_T=action.shape[1],
                           static_loops=True, mmd_coef=50
                           )

            ensure_dir(output)

            action, reward, state, seq_lengths, test_action, test_reward, test_state, test_seq_lengths, = \
                Synth.generate_train_test(action, reward, state, seq_lengths, ids, 0.3, output)

            Synth.opt_model_align(model,
                                 action, reward, state, seq_lengths,
                                 test_action, test_reward, test_state, test_seq_lengths,
                                 output + '/model/', model_path, hessian_term=True,
                                  hessian_lr=0.0001, _beta=1
                                  )

    @staticmethod
    def predict_z(model_path):
        output = '../nongit/local/synth_normal/'
        with LogFile(output, 'run.log'):
            action, reward, state, ids, seq_lengths = Synth.get_data()
            tf.reset_default_graph()
            model = HYPMDD(enc_cells=50, dec_cells=3, a_size=2, s_size=0, latent_size=2 , n_T=action.shape[1],
                           static_loops=False
                           )
            Synth.predict(
                model, action, reward, state, ids, seq_lengths,
                '../nongit/local/synth_normal/',
                model_path,
            )


if __name__ == '__main__':

    # for training the model (stage 1)
    Synth.train_test(None)

    # for training the model (stage 2)
    # replace the below path with the path to the trained model
    # Synth.aling_model('../nongit/archive/synth-random/symms-rep50cells-forpaper/opt-50cells-lr/model/iter-30400/')

    # for predicting z (aligned)
    # replace the below path with the path to the trained model
    # Synth.predict_z('../nongit/archive/synth-random/symms-rep50cells-forpaper/align-h1-50cells/model/iter-224/')

    # for predicting z (unaligned)
    # replace the below path with the path to the trained model
    # Synth.predict_z('../nongit/archive/synth-random/symms-rep50cells-forpaper/align-h1-50cells/model/iter-init/')
