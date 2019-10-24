import tensorflow as tf

from expr.base_opt import BaseOpt
from expr.data_process import DataProcess
from expr.data_reader import DataReader
from model.consts import Const
from model.rnn2rnn import HYPVAE, HYPAAE, HYPMDD
from util.helper import get_total_pionts, ensure_dir, fix_init_all_vars
from util.logger import LogFile, DLogger
import pandas as pd
from util.stratified_train_test_split import stratified_train_test_split


class BD(BaseOpt):

    @classmethod
    def get_data(cls):
        data = DataReader.read_BD()
        # data = data.loc[data.diag == 'Healthy']
        ids = data['id'].unique().tolist()
        dftr = pd.DataFrame({'id': ids, 'train': 'train'})
        tdftr = pd.DataFrame({'id': ids, 'train': 'test'})
        train, test = DataProcess.train_test_between_subject(data, pd.concat((dftr, tdftr)),
                                                             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 2)
        DLogger.logger().debug("total points: " + str(get_total_pionts(train)))
        train = DataProcess.merge_data(train)

        action = train['merged'][0]['action']
        reward = train['merged'][0]['reward']
        state = train['merged'][0]['state']
        ids = train['merged'][0]['id']
        seq_lengths = train['merged'][0]['seq_lengths']

        return action, reward, state, ids, seq_lengths

    @classmethod
    def opt_model_train_test(cls, model_path):
        output = '../nongit/local/BD/opt/'
        with LogFile(output, 'run.log'):
            action, reward, state, ids, seq_lengths = BD.get_data()

            tf.reset_default_graph()

            with tf.device('/device:GPU:0'):
                model = HYPMDD(enc_cells=20, dec_cells=3, a_size=2, s_size=0, latent_size=2, n_T=action.shape[1],
                               static_loops=False, mmd_coef=50)

            ensure_dir(output)

            actions_train, actions_test, rewards_train, rewards_test, seq_train, seq_test, id_train, id_test = \
                stratified_train_test_split(action, reward, state, ids, seq_lengths)

            DLogger.logger().debug("test points: " + str(actions_test.shape[0]))
            DLogger.logger().debug("train points: " + str(actions_train.shape[0]))

            def lr_schedule(t):
                if t < 2000:
                    _lr = 0.001
                elif t < 5000:
                    _lr = 0.0001
                else:
                    _lr = 0.00001
                return _lr

            BD.opt_model_mddae(model,
                               actions_train, rewards_train, None, seq_train,
                               actions_test, rewards_test, None, seq_test,
                               output + '/model/', model_path, hessian_term=False,
                               lr_schedule=lr_schedule
                               )

    @classmethod
    def aling_model(cls, model_path):
        output = '../nongit/local/BD/align/'
        with LogFile(output, 'run.log'):
            action, reward, state, ids, seq_lengths = BD.get_data()

            tf.reset_default_graph()

            with tf.device('/device:GPU:0'):
                model = HYPMDD(enc_cells=20, dec_cells=3, a_size=2, s_size=0, latent_size=2, n_T=action.shape[1],
                               static_loops=True, mmd_coef=2)

            ensure_dir(output)

            actions_train, actions_test, rewards_train, rewards_test, seq_train, seq_test, id_train, id_test = \
                stratified_train_test_split(action, reward, state, ids, seq_lengths)

            DLogger.logger().debug("test points: " + str(actions_test.shape[0]))
            DLogger.logger().debug("train points: " + str(actions_train.shape[0]))

            BD.opt_model_align(model,
                               actions_train, rewards_train, None, seq_train,
                               actions_test, rewards_test, None, seq_test,
                               output + '/model/', model_path, hessian_term=True, _beta=0.5, hessian_lr=0.0001, _h=0.1)

    @classmethod
    def predict_z(cls, model_path):
        output = '../nongit/local/BD/opt/'
        with LogFile(output, 'run.log'):
            action, reward, state, ids, seq_lengths = BD.get_data()
            DLogger.logger().debug("data points: " + str(action.shape[0]))
            tf.reset_default_graph()
            ensure_dir(output)
            model = HYPMDD(enc_cells=20, dec_cells=3, a_size=2, s_size=0, latent_size=2, n_T=action.shape[1],
                           static_loops=False
                           )
            BD.predict(model, action, reward, state, ids, seq_lengths, '../nongit/local/BD/', model_path)


if __name__ == '__main__':

    # see diff.txt in the output folders for differences with committed versions

    # for optimising the model (Stage 1)
    BD.opt_model_train_test('../nongit/archive/BD/symms-forpaper/opt/model/iter-init/')

    # for aligning the model (Stage 2)
    # BD.aling_model('../nongit/archive/BD/symms-forpaper/opt/model/iter-43800/')

    # for getting latent representation
    # BD.predict_z('../nongit/archive/BD/symms-forpaper/align/aling2-h.1-mm2/model/iter-141/')
