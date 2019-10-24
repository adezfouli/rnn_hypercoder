from model.consts import Const
from util import DLogger
from util.helper import ensure_dir
import tensorflow as tf
import pandas as pd
import numpy as np


class BaseOpt:

    @classmethod
    def predict(cls, rnn2rnn, action, reward, state, ids, seq_lengths, output_path, model_path):

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_path)

        with tf.Session() as sess:
            saver.restore(sess, ckpt.model_checkpoint_path)
            enc_dict_feed = rnn2rnn.enc.enc_beh_feed(action, reward, state, seq_lengths)
            dec_dict_feed = rnn2rnn.dec.dec_beh_feed(action, reward, state, seq_lengths)

            enc_loss, dec_loss, z_mean = sess.run(
                [rnn2rnn.enc.loss,
                 rnn2rnn.dec.loss,
                 rnn2rnn.enc.z_pred
                 ],
                feed_dict={**enc_dict_feed, **dec_dict_feed})

            cls.report_model(dec_dict_feed, enc_dict_feed, rnn2rnn, sess, -1, {})

            if output_path is not None:
                ensure_dir(output_path)
                df = pd.DataFrame(z_mean)
                df = pd.concat([df, pd.DataFrame({'id': ids})], axis=1)
                df.to_csv(output_path + 'z_mean.csv', index=False)

        return z_mean

    @staticmethod
    def _get_apply_grads(obj, trainables, learning_rate, _optimizer):

        optimizer = _optimizer(learning_rate=learning_rate)
        max_global_norm = 1
        grads = tf.gradients(obj, trainables)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_global_norm)
        grad_var_pairs = zip(grads, trainables)
        apply_grads = optimizer.apply_gradients(grad_var_pairs)
        return apply_grads

    @classmethod
    def random_shuffle_action(cls, action, seq_lengths):
        action = action.copy()
        indx = range(action.shape[0])
        is_ = np.random.choice(indx, size=int(len(indx)/ 2), replace=False)
        for i in is_:
            action[i, 0:seq_lengths[i]] = 1 - action[i, 0:seq_lengths[i]]
        return action

    @classmethod
    def opt_model_mddae(cls, rnn2rnn,
                        action, reward, state, seq_lengths,
                        test_action=None, test_reward=None, test_state=None, test_seq_lengths=None,
                        save_path=None, init_path=None, hessian_term=False,
                        _beta=1, lr_schedule=lambda t: 0.001
                        ):

        cls.write_diff(save_path)
        # Optimizers
        trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        trainables_dec = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dec')
        trainables_enc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='enc')
        beta = tf.placeholder(Const.FLOAT, shape=())
        h = tf.placeholder(Const.FLOAT, shape=())
        lr = tf.placeholder(Const.FLOAT, shape=())

        opt_all = cls._get_apply_grads(rnn2rnn.dec.loss + beta * rnn2rnn.enc.loss + h * rnn2rnn.dec.hess_loss,
                                       trainables, lr, tf.train.AdamOptimizer)

        opt_enc = cls._get_apply_grads(rnn2rnn.dec.loss + beta * rnn2rnn.enc.loss + h * rnn2rnn.dec.hess_loss,
                                       trainables_enc, lr, tf.train.AdamOptimizer)

        opt_dec = cls._get_apply_grads(rnn2rnn.dec.loss + beta * rnn2rnn.enc.loss + h * rnn2rnn.dec.hess_loss,
                                       trainables_dec, lr, tf.train.AdamOptimizer)

        if init_path is not None:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(init_path)
            DLogger.logger().debug('loaded model from: ' + init_path)
        else:
            init = tf.global_variables_initializer()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            if init_path is not None:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(init)

            if save_path:
                iter_path = save_path + '/iter-init/'
                saver = tf.train.Saver()
                ensure_dir(iter_path)
                saver.save(sess, iter_path + "model.ckpt", write_meta_graph=False)
                DLogger.logger().debug("Model saved in path: %s" % iter_path)

            try:

                for t in range(100000):

                    cur_beta = {beta: _beta}

                    # adaptive learning rate
                    _lr = lr_schedule(t)

                    if not hessian_term:
                        _h = 0.0
                    else:
                        _h = min(t / 1000, 0.1)

                    cur_lr = {lr: _lr}
                    cur_h = {h: _h}

                    enc_dict_feed = rnn2rnn.enc.enc_beh_feed(BaseOpt.random_shuffle_action(action, seq_lengths),
                                                             reward, state, seq_lengths)
                    dec_dict_feed = rnn2rnn.dec.dec_beh_feed(BaseOpt.random_shuffle_action(action, seq_lengths),
                                                             reward, state, seq_lengths)
                    _, enc_loss, dec_loss, hess_loss, rand_loss, z_grad, z_cov = sess.run([opt_all] + [
                                    rnn2rnn.enc.loss,
                                    rnn2rnn.dec.loss,
                                    rnn2rnn.dec.hess_loss,
                                    rnn2rnn.dec.sloss,
                                    rnn2rnn.dec.z_grad,
                                    rnn2rnn.enc.z_cov
                                ],
                        feed_dict={**enc_dict_feed, **dec_dict_feed, **cur_beta, **cur_lr, **cur_h})

                    DLogger.logger().debug("global iter = {:4d} "
                                           "enc loss: {:7.4f} "
                                           "dec loss: {:7.4f} "
                                           "hess loss: {:7.4f} "
                                           "rand loss: {:7.4f} "
                                           "beta: {:7.4f} "
                                           "LR: {:7.4f} "
                                           "grad z {} "
                                           "z-cov: {}"
                                           .format(t, enc_loss, dec_loss, hess_loss, rand_loss, _beta, _lr,
                                                   z_grad,
                                                   str(np.array2string(z_cov.flatten(), precision=3).replace('\n', ''))))

                    if t % 200 == 0:
                        if test_action is not None:
                            test_enc_dict_feed = rnn2rnn.enc.enc_beh_feed(
                                BaseOpt.random_shuffle_action(test_action, test_seq_lengths),
                                test_reward, test_state, test_seq_lengths)
                            test_dec_dict_feed = rnn2rnn.dec.dec_beh_feed(
                                BaseOpt.random_shuffle_action(test_action, test_seq_lengths),
                                test_reward, test_state, test_seq_lengths)

                            enc_loss, dec_loss, hess_loss, rand_loss, z_grad, z_cov = sess.run([
                                rnn2rnn.enc.loss,
                                rnn2rnn.dec.loss,
                                rnn2rnn.dec.hess_loss,
                                rnn2rnn.dec.sloss,
                                rnn2rnn.dec.z_grad,
                                rnn2rnn.enc.z_cov
                            ],
                                       feed_dict={**test_enc_dict_feed,
                                                  **test_dec_dict_feed,
                                                  **cur_beta, **cur_lr, **cur_h})

                            DLogger.logger().debug("TEST data: global iter = {:4d} "
                                                   "enc loss: {:7.4f} "
                                                   "dec loss: {:7.4f} "
                                                   "hess loss: {:7.4f} "
                                                   "rand loss: {:7.4f} "
                                                   "beta: {:7.4f} "
                                                   "LR: {:7.4f} "
                                                   "z grad {} "
                                                   "z-cov: {}"
                                                   .format(t, enc_loss, dec_loss, hess_loss, rand_loss, _beta, _lr,
                                                           z_grad,
                                               str(np.array2string(z_cov.flatten(), precision=3).replace('\n',''))))

                        if save_path:
                            iter_path = save_path + '/iter-' + str(t) + '/'
                            saver = tf.train.Saver()
                            ensure_dir(iter_path)
                            saver.save(sess, iter_path + "model.ckpt", write_meta_graph=False)
                            DLogger.logger().debug("Model saved in path: %s" % iter_path)

            finally:
                if save_path:
                    saver = tf.train.Saver()
                    ensure_dir(save_path)
                    save_path = saver.save(sess, save_path + "model.ckpt")
                    DLogger.logger().debug("Model saved in path: %s" % save_path)

    @classmethod
    def opt_model_align(cls, rnn2rnn,
                        action, reward, state, seq_lengths,
                        test_action=None, test_reward=None, test_state=None, test_seq_lengths=None,
                        save_path=None, init_path=None, hessian_term=False,
                        hessian_lr=0.00001, _beta=1, _h=1
                        ):

        cls.write_diff(save_path)
        trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        trainables_dec = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dec')
        trainables_enc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='enc')
        beta = tf.placeholder(Const.FLOAT, shape=())
        h = tf.placeholder(Const.FLOAT, shape=())
        lr = tf.placeholder(Const.FLOAT, shape=())

        opt_all = cls._get_apply_grads(rnn2rnn.dec.loss + beta * rnn2rnn.enc.loss + h * rnn2rnn.dec.hess_loss,
                                       trainables, lr, tf.train.AdamOptimizer)

        opt_enc = cls._get_apply_grads(rnn2rnn.dec.loss + beta * rnn2rnn.enc.loss + h * rnn2rnn.dec.hess_loss,
                                       trainables_enc, lr, tf.train.AdamOptimizer)

        opt_dec = cls._get_apply_grads(rnn2rnn.dec.loss + beta * rnn2rnn.enc.loss + h * rnn2rnn.dec.hess_loss,
                                       trainables_dec, lr, tf.train.AdamOptimizer)

        if init_path is not None:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(init_path)
            DLogger.logger().debug('loaded model from: ' + init_path)
        else:
            init = tf.global_variables_initializer()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            if init_path is not None:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(init)

            if save_path:
                iter_path = save_path + '/iter-init/'
                saver = tf.train.Saver()
                ensure_dir(iter_path)
                saver.save(sess, iter_path + "model.ckpt", write_meta_graph=False)
                DLogger.logger().debug("Model saved in path: %s" % iter_path)


            try:

                for t in range(100000):

                    cur_beta = {beta: _beta}

                    # adaptive learning rate
                    _lr = None

                    if not hessian_term:
                        if t < 2000:
                            _lr = 0.001
                        elif t < 5000:
                            _lr = 0.0001
                        else:
                            _lr = 0.00001
                    else:
                        _lr = hessian_lr

                    if not hessian_term:
                        _h = 0.0
                    else:
                        _h = _h  #redundant -- just for readability

                    cur_lr = {lr: _lr}
                    cur_h = {h: _h}

                    DLogger.logger().debug("DEC opt started")
                    for d in range(50):
                        enc_dict_feed = rnn2rnn.enc.enc_beh_feed(BaseOpt.random_shuffle_action(action, seq_lengths),
                                                                 reward, state, seq_lengths)
                        dec_dict_feed = rnn2rnn.dec.dec_beh_feed(BaseOpt.random_shuffle_action(action, seq_lengths),
                                                                 reward, state, seq_lengths)
                        _, enc_loss, dec_loss, hess_loss, rand_loss, z_grad, z_cov = sess.run([opt_dec] + [
                            rnn2rnn.enc.loss,
                            rnn2rnn.dec.loss,
                            rnn2rnn.dec.hess_loss,
                            rnn2rnn.dec.sloss,
                            rnn2rnn.dec.z_grad,
                            rnn2rnn.enc.z_cov
                        ],
                          feed_dict={
                          **enc_dict_feed,
                          **dec_dict_feed,
                          **cur_beta, **cur_lr,
                          **cur_h})

                        DLogger.logger().debug(
                            "global iter = {:4d} "
                            "dec iter = {:4d} "
                            "enc loss: {:7.4f} "
                            "dec loss: {:7.4f} "
                            "hess loss: {:7.4f} "
                            "rand loss: {:7.4f} "
                            "beta: {:7.4f} "
                            "LR: {:7.4f} "
                            "grad z {} "
                            "z-cov: {}"
                                .format(t, d, enc_loss, dec_loss, hess_loss, rand_loss, _beta, _lr,
                                        z_grad,
                                        str(np.array2string(z_cov.flatten(), precision=3).replace('\n', ''))))

                    DLogger.logger().debug("ENC opt started")
                    for e in range(200):
                        cur_lr = {lr: 0.0001}
                        cur_h = {h: 0}
                        enc_dict_feed = rnn2rnn.enc.enc_beh_feed(BaseOpt.random_shuffle_action(action, seq_lengths),
                                                                 reward, state, seq_lengths)
                        dec_dict_feed = rnn2rnn.dec.dec_beh_feed(BaseOpt.random_shuffle_action(action, seq_lengths),
                                                                 reward, state, seq_lengths)
                        _, enc_loss, dec_loss, hess_loss, rand_loss, z_grad, z_cov = sess.run([opt_enc] + [
                            rnn2rnn.enc.loss,
                            rnn2rnn.dec.loss,
                            rnn2rnn.dec.hess_loss,
                            rnn2rnn.dec.sloss,
                            rnn2rnn.dec.z_grad,
                            rnn2rnn.enc.z_cov
                        ],
                          feed_dict={
                              **enc_dict_feed,
                              **dec_dict_feed,
                              **cur_beta, **cur_lr,
                              **cur_h})

                        DLogger.logger().debug(
                            "global iter = {:4d} "
                            "enc iter = {:4d} "
                            "enc loss: {:7.4f} "
                            "dec loss: {:7.4f} "
                            "hess loss: {:7.4f} "
                            "rand loss: {:7.4f} "
                            "beta: {:7.4f} "
                            "LR: {:7.4f} "
                            "grad z {} "
                            "z-cov: {}"
                                .format(t, e, enc_loss, dec_loss, hess_loss, rand_loss, _beta, _lr,
                                        z_grad,
                                        str(np.array2string(z_cov.flatten(), precision=3).replace('\n', ''))))

                    if test_action is not None:
                        test_enc_dict_feed = rnn2rnn.enc.enc_beh_feed(
                            BaseOpt.random_shuffle_action(test_action, test_seq_lengths),
                            test_reward, test_state, test_seq_lengths)
                        test_dec_dict_feed = rnn2rnn.dec.dec_beh_feed(
                            BaseOpt.random_shuffle_action(test_action, test_seq_lengths),
                            test_reward, test_state, test_seq_lengths)

                        enc_loss, dec_loss, hess_loss, rand_loss, z_grad, z_cov = sess.run([
                            rnn2rnn.enc.loss,
                            rnn2rnn.dec.loss,
                            rnn2rnn.dec.hess_loss,
                            rnn2rnn.dec.sloss,
                            rnn2rnn.dec.z_grad,
                            rnn2rnn.enc.z_cov
                        ],
                            feed_dict={**test_enc_dict_feed,
                                       **test_dec_dict_feed,
                                       **cur_beta, **cur_lr, **cur_h})

                        DLogger.logger().debug("TEST data: global iter = {:4d} "
                                               "enc loss: {:7.4f} "
                                               "dec loss: {:7.4f} "
                                               "hess loss: {:7.4f} "
                                               "rand loss: {:7.4f} "
                                               "beta: {:7.4f} "
                                               "LR: {:7.4f} "
                                               "z grad {} "
                                               "z-cov: {}"
                                               .format(t, enc_loss, dec_loss, hess_loss, rand_loss, _beta, _lr,
                                                       z_grad,
                                                       str(np.array2string(z_cov.flatten(), precision=3).replace(
                                                           '\n',
                                                           ''))))

                    if save_path:
                        iter_path = save_path + '/iter-' + str(t) + '/'
                        saver = tf.train.Saver()
                        ensure_dir(iter_path)
                        saver.save(sess, iter_path + "model.ckpt", write_meta_graph=False)
                        DLogger.logger().debug("Model saved in path: %s" % iter_path)

            finally:
                if save_path:
                    saver = tf.train.Saver()
                    ensure_dir(save_path)
                    save_path = saver.save(sess, save_path + "model.ckpt")
                    DLogger.logger().debug("Model saved in path: %s" % save_path)

    @classmethod
    def write_diff(cls, save_path):
        DLogger.logger().debug("writing git diff to " + save_path + 'diff.txt')
        ensure_dir(save_path)
        try:
            import subprocess
            a = 'no git found'
            a = subprocess.run(["git", "diff"], stdout=subprocess.PIPE)
        finally:
            with open(save_path + "diff.txt", "w") as f:
                f.write(a.stdout.decode('utf-8'))

    @classmethod
    def generate_train_test(cls, action, reward, state , seq_length, ids, test_prop, output_path=None):
        # tf.set_random_seed(1011)

        # generating training/test sets
        DLogger.logger().debug("Test prop: {:7.4f}" .format(test_prop))

        n_batches = action.shape[0]
        np.random.seed(10)
        indx = np.random.choice(range(0, n_batches), n_batches, replace=False)
        tr_indx = indx[:int((1 - test_prop) * n_batches)]
        test_indx = indx[int((1 - test_prop) * n_batches):]

        if output_path is not None:
            pd.DataFrame({'id': [ids[j] for j in tr_indx]}).to_csv(output_path + 'train_IDs.csv', index=False)
            pd.DataFrame({'id': [ids[j] for j in test_indx]}).to_csv(output_path + 'test_IDs.csv', index=False)

        return action[tr_indx], reward[tr_indx], None if state is None else state[tr_indx], seq_length[tr_indx], \
                action[test_indx], reward[test_indx], None if state is None else state[test_indx], seq_length[test_indx]

    @classmethod
    def report_model(cls, dec_dict_feed, enc_dict_feed, rnn2rnn, sess, t, other = {}):
        enc_loss, dec_loss, discr_loss, rand_loss, z_grad, z_cov = sess.run([
            rnn2rnn.enc.loss,
            rnn2rnn.dec.loss,
            rnn2rnn.enc.dc_loss,
            rnn2rnn.dec.sloss,
            rnn2rnn.dec.z_grad,
            rnn2rnn.enc.z_cov
        ],
            feed_dict={**enc_dict_feed, **dec_dict_feed, **other})

        DLogger.logger().debug("global iter = {:4d} "
                               "enc loss: {:7.4f} "
                               "dec loss: {:7.4f} "
                               "discr loss: {:7.4f} "
                               "rand loss: {:7.4f} "
                               "grad z {} "
                                "z-cov: {}"
                               .format(t, enc_loss, dec_loss, discr_loss, rand_loss, z_grad,
                                       str(np.array2string(z_cov.flatten(), precision=3, ))))
        return enc_loss, dec_loss, discr_loss, rand_loss
