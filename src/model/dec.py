import tensorflow as tf
import numpy as np
from expr.data_process import DataProcess
from model.consts import Const
from model.model_beh import ModelBeh
from model.rnn_cell import GRUCell2
from util import DLogger
from util.helper import normalized_columns_initializer, state_to_onehot
from numpy.random import choice


class DECRNN(ModelBeh):
    def __init__(self, n_cells, a_size, s_size, n_samples, z, n_T, static_loop):

        super().__init__(a_size, s_size)

        DLogger.logger().debug("model created with ncells: " + str(n_cells))

        self.z = z
        self.seq_lengths = tf.placeholder(dtype=tf.int32, shape=[None])
        self.n_cells = n_cells
        self.n_samples = n_samples

        # picking k random k
        self.rand_gather = self.rand_withiout_replacement(self.n_batches)

        self.cell = GRUCell2(n_cells, n_samples, n_T)

        W1, W2, Wsoft, b1, b2 = self.z_to_RNNweights(a_size, n_cells, s_size, self.z)

        self.cell.set_weights(W1, b1, W2, b2)

        # for assessing performance on random z
        self.scell = GRUCell2(n_cells, n_samples, n_T)
        self.scell.set_weights(tf.gather(W1, self.rand_gather, axis=1),
                          tf.gather(b1, self.rand_gather, axis=1),
                          tf.gather(W2, self.rand_gather, axis=1),
                          tf.gather(b2, self.rand_gather, axis=1))


        step_size = tf.ones([tf.shape(self.rnn_in)[0]], dtype=tf.int32) * tf.shape(self.rnn_in)[1]

        if static_loop:
            self.state_track, self.last_state = self.cell.static_rnn(self.rnn_in, step_size)
        else:
            self.state_track, self.last_state = self.cell.dynamic_rnn(self.rnn_in, step_size)
        # state_track: nSamples x nBathces x (nChocies + 1) x nCells

        sstate_track, slast_state = self.scell.dynamic_rnn(self.rnn_in, step_size)


        # Output layers for policy and value estimations
        pol_un = tf.matmul(self.state_track, Wsoft) * \
            tf.cast(tf.sequence_mask(self.seq_lengths + 1, maxlen=n_T + 1)[np.newaxis, :, :, np.newaxis], Const.FLOAT)
        self.policy = tf.nn.softmax(pol_un)
        # self.policy = slim.fully_connected(state_track, a_size,
        #                                    activation_fn=tf.nn.softmax,
        #                                    weights_initializer=normalized_columns_initializer(0.01),
        #                                    biases_initializer=None,
        #                                    scope='softmax')

        epsilon = 1e-4

        self.spolicy = tf.nn.softmax(tf.matmul(sstate_track, tf.gather(Wsoft, self.rand_gather, axis=1)))

        # DIM: nSamples x nBatches x (nChoices + 1) x (nActionTypes)

        # this is correction for the missing actions for which the one hot is [0,0, ... 0]
        actions_onehot_corrected = (1 - tf.reduce_sum(self.actions_onehot, axis=2))[:, :, np.newaxis] + \
                                   self.actions_onehot
        # note that we ignore the last element of the policy, since there is no observation after
        # the last reward received.
        action_logs = tf.reduce_sum(tf.log(tf.reduce_sum((self.policy[:, :, :-1, :] + epsilon) *
                                                         actions_onehot_corrected[np.newaxis], axis=3)), axis=[2])
        # DIM: nBatches x nChoices x nActions
        saction_logs = tf.reduce_sum(tf.log(tf.reduce_sum((self.spolicy[:, :, :-1, :] + epsilon) *
                                                         actions_onehot_corrected[np.newaxis], axis=3)), axis=[2])


        # action_logs: dim: nSamples x nBatches

        # uncomment this to calcualte the mean over each datapoint
        # action_logs = tf.div(action_logs, tf.reduce_sum(self.actions_onehot, axis=[1, 2]))

        # sum over all points and average over batches and samples
        self.beh_loss = -tf.reduce_mean(action_logs, axis=0)
        self.loss = tf.reduce_mean(self.beh_loss)

        # loss using random sequence
        self.sloss = tf.reduce_mean(-tf.reduce_mean(saction_logs, axis=0))

        # for the loss in the Hessians matrix

        # this is for normalizing wrt to seq lengths
        # norm_pol = self.policy / tf.cast(self.seq_lengths[np.newaxis, :, np.newaxis, np.newaxis], Const.FLOAT)

        # Cannot calculate second order gradients (use new while loop version in TF if want to use
        # dynamic graphs with second order gradients

        norm_pol = pol_un
        self.unpol = norm_pol[:, :, :, 0] - norm_pol[:, :, :, 1]

        if static_loop:
            a = tf.gradients(norm_pol[:, :, :, 0] - norm_pol[:, :, :, 1], self.z)
            a = tf.gradients(a[0][:, :, 0], self.z)[0][:, :, 1]
            self.hess_loss = tf.reduce_mean(tf.abs(a))
        else:
            DLogger.logger().debug("Hessian loss is deactivated.")
            self.hess_loss = tf.constant(0, Const.FLOAT)

        self.z_grad = tf.reduce_mean(tf.abs(tf.gradients(norm_pol[:, :, :, 0], self.z)[0]), axis=[0, 1])

    def beh_feed(self, actions, rewards, states):
        dict = super().beh_feed(actions, rewards, states)
        dict[self.cell.state_in] = np.zeros((self.n_samples, actions.shape[0], self.n_cells), dtype=np.float64)
        dict[self.scell.state_in] = np.zeros((self.n_samples, actions.shape[0], self.n_cells), dtype=np.float64)
        return dict

    def rand_withiout_replacement(self, n_rands):
        probs = tf.ones(shape=(n_rands,), dtype=Const.FLOAT) / tf.cast(n_rands, dtype=Const.FLOAT)
        ztmp = -tf.log(-tf.log(tf.random_uniform(tf.shape(probs), 0, 1, dtype=Const.FLOAT)))
        _, indices = tf.nn.top_k(probs + ztmp, n_rands)
        return indices

    def z_to_RNNweights(self, a_size, n_cells, s_size, z):
        W1_dim, b1_dim, W2_dim, b2_dim, Wsoft_dim = DECRNN.get_variable_dims(n_cells, a_size, s_size)
        total_weight_dim = W1_dim[0] * W1_dim[1] + b1_dim[0] + W2_dim[0] * W2_dim[1] + b2_dim[0] + Wsoft_dim[0] * \
                           Wsoft_dim[1]

        with tf.variable_scope('dec'):
                dense1 = tf.layers.dense(inputs=z, units=100, activation=tf.nn.tanh)
                dense2 = tf.layers.dense(inputs=dense1, units=100, activation=tf.nn.tanh)
                dense3 = tf.layers.dense(inputs=dense2, units=100, activation=tf.nn.tanh)
                output = tf.layers.dense(inputs=dense3, units=total_weight_dim, activation=None)
                last_ix = 0

        with tf.variable_scope('dec_init'):
            W1 = output[:, :, last_ix:(last_ix + W1_dim[0] * W1_dim[1])]

            last_ix = last_ix + W1_dim[0] * W1_dim[1]
            b1 = output[:, :, last_ix:(last_ix + b1_dim[0])]

            last_ix = last_ix + b1_dim[0]
            W2 = output[:, :, last_ix:(last_ix + W2_dim[0] * W2_dim[1])]

            last_ix = last_ix + W2_dim[0] * W2_dim[1]
            b2 = output[:, :, last_ix:(last_ix + b2_dim[0])]

            last_ix = last_ix + b2_dim[0]
            Wsoft = output[:, :, last_ix:(last_ix + Wsoft_dim[0] * Wsoft_dim[1])]

            W1, b1, W2, b2, Wsoft = tf.reshape(W1, [tf.shape(W1)[0], tf.shape(W1)[1]] + W1_dim), \
                                    tf.reshape(b1, [tf.shape(b1)[0], tf.shape(b1)[1]] + b1_dim), \
                                    tf.reshape(W2, [tf.shape(W2)[0], tf.shape(W2)[1]] + W2_dim), \
                                    tf.reshape(b2, [tf.shape(b2)[0], tf.shape(b2)[1]] + b2_dim), \
                                    tf.reshape(Wsoft, [tf.shape(Wsoft)[0], tf.shape(Wsoft)[1]] + Wsoft_dim)
        return W1, W2, Wsoft, b1, b2

    @classmethod
    def get_variable_dims(cls, n_cells, a_size, s_size):
        input_dim = a_size + s_size + 1
        return GRUCell2.get_weight_dims(input_dim, n_cells) + ([n_cells, a_size],)

    def simulate_env(self, sess, max_time, env_model, greedy=False):
        rnn_state_lists = []
        policy_lists = []
        r_lists = []
        state_lists = []
        actions_list = []

        rnn_state = np.zeros((1, 1, self.n_cells))
        s, r, a = env_model(None, None, -1)
        for t in range(max_time + 1):
            s, r, next_a = env_model(s, a, t)
            feed_dict = {
                self.prev_rewards: [[[r]]],
                self.prev_actions: [[a]],
                self.seq_lengths: [1],
                self.cell.state_in: rnn_state
            }

            if s is not None:
                feed_dict[self.prev_states] = [[s]]

            a_dist, rnn_state_new = sess.run([self.policy, self.last_state], feed_dict=feed_dict)
            rnn_state_new = rnn_state_new
            rnn_state = rnn_state_new
            rnn_state_lists.append(rnn_state_new)
            policy_lists.append(a_dist)
            r_lists.append(r)
            state_lists.append(s)
            actions_list.append(a)

            if next_a is not None:
                a = next_a
            elif greedy:
                a = np.argmax(a_dist, axis=3)[0,0,0]
            else:
                a = choice(np.arange(a_dist.shape[3]), p=a_dist[0][0, 0])[np.newaxis, np.newaxis]

        # note that policy is for the next trial
        return np.hstack(state_lists)[:-1], \
               np.hstack(policy_lists)[0, :-1, 0, :], \
               np.hstack(r_lists)[1:], \
               np.hstack(actions_list)[1:], \
               np.vstack(rnn_state_lists)[0, np.newaxis, 1:, ]

    def dec_beh_feed(self, actions, rewards, states, seq_lengths):
        dict = self.beh_feed(actions, rewards, states)
        dict[self.seq_lengths] = seq_lengths
        return dict



if __name__ == '__main__':
    import pandas as pd

    data = pd.read_csv("test_data/choices_short.csv", header=0, sep=',', quotechar='"')
    data['action'] = [0 if x == 'R1' else 1 for x in data['action']]

    ids = data['id'].unique().tolist()
    dftr = pd.DataFrame({'id': ids[0:2], 'train': 'train'})
    tdftr = pd.DataFrame({'id': ids[2:4], 'train': 'test'})
    train, test = DataProcess.train_test_between_subject(data, pd.concat((dftr, tdftr)), [1, 2], 2)
    state_to_onehot(train, 5)
    state_to_onehot(test, 5)

    train = DataProcess.merge_data(train)

    n_cells = 5
    n_samples = 3
    n_bacthes = train['merged'][0]['action'].shape[0]
    n_states = 5
    n_actions = 2
    dW1, db1, dW2, db2, dSoft = DECRNN.get_variable_dims(n_cells, n_actions, n_states)

    def sub_mat(x):
        return x[:, :]

    np.random.seed(1010)
    W1, b1, W2, b2, Wsoft = tf.get_variable(name='w1', shape=[n_samples,n_bacthes] + dW1,
                                     initializer=tf.constant_initializer(sub_mat(np.random.normal(0, 1, size=[3, 2] + dW1))),
                                     dtype=Const.FLOAT), \
                     tf.get_variable(name='b1', shape=[n_samples, n_bacthes] + db1,
                                     initializer=tf.constant_initializer(sub_mat(np.random.normal(0, 1, size=[3, 2] + db1))),
                                     dtype=Const.FLOAT), \
                     tf.get_variable(name='w2', shape=[n_samples, n_bacthes] + dW2,
                                     initializer=tf.constant_initializer(sub_mat(np.random.normal(0, 1, size=[3, 2] + dW2))),
                                     dtype=Const.FLOAT), \
                     tf.get_variable(name='b2', shape=[n_samples, n_bacthes] + db2,
                                     initializer=tf.constant_initializer(sub_mat(np.random.normal(0, 1, size=[3, 2] + db2))),
                                     dtype=Const.FLOAT), \
                     tf.get_variable(name='Wsoft', shape=[n_samples, n_bacthes] + dSoft,
                                    initializer=tf.constant_initializer(sub_mat(np.random.normal(0, 1, size=[3, 2] + dSoft))),
                                    dtype=Const.FLOAT)

    dec = DECRNN(n_cells, n_actions, n_states, n_samples, W1, b1, W2, b2, Wsoft)

    tf.set_random_seed(1010)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # fix_init_all_vars(sess)
        sess.run(init)
        dict_feed = dec.beh_feed(train['merged'][0]['action'], train['merged'][0]['reward'], train['merged'][0]['state'])
        _loss = sess.run(dec.beh_loss, feed_dict=dict_feed)
        DLogger.logger().debug('loss: ' + str(_loss))
