from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import gin.tf
from disentanglement_lib.evaluation.metrics.beta_vae import compute_beta_vae_sklearn
from disentanglement_lib.evaluation.metrics.dci import _compute_dci, compute_dci
from disentanglement_lib.evaluation.metrics.factor_vae import compute_factor_vae
from disentanglement_lib.evaluation.metrics.mig import _compute_mig, compute_mig
from disentanglement_lib.evaluation.metrics.modularity_explicitness import modularity, explicitness_per_factor, \
    compute_modularity_explicitness
from disentanglement_lib.evaluation.metrics.sap_score import _compute_sap, compute_sap
import pandas as pd
from disentanglement_lib.evaluation.metrics.utils import _histogram_discretize


class GT:
    def __init__(self, discr=False, limit=True):
        d = pd.read_csv("../data/aling.csv")
        self.mu = np.array(d[['X0', 'X1']]).T
        self.y = np.array(d[['beta_untrans', 'persv']]).T

        dmu = self.mu.copy()
        dy = self.y.copy()

        if discr:
            dmu[0] = pd.qcut(dmu[0], q=15, labels=False)
            dmu[1] = pd.qcut(dmu[1], q=15, labels=False)
            self.mu = dmu

        dy[0] = pd.qcut(dy[0], q=15, labels=False)
        dy[1] = pd.qcut(dy[1], q=15, labels=False)
        self.y = dy

        self.num_factors = 2

        self.d = {}
        for i in range(self.y.shape[1]):
            self.d[str(self.y[:, i])] = i

        np.random.seed(00)
        self.cur_sample = np.arange(0, self.y.shape[1])
        np.random.shuffle(self.cur_sample)
        self.cur_index = 0
        self.limit = limit


    def sample(self, n_points, state):
        out = np.zeros((2, n_points))
        i = 0
        cur_i = []
        while i < n_points:
            out[:, i] = self.y[:, self.cur_sample[self.cur_index]]
            cur_i.append(self.cur_sample[self.cur_index])
            self.cur_index += 1
            if not self.limit:
                if self.cur_index >= 1500:
                    self.cur_index = 0
            i += 1

        return out.T, cur_i

    def sample_observations(self, p, r):
        _, o = self.sample(p, r)
        return np.array(o)

    def sample_factors(self, num, random_state):
        return self.sample(num, random_state)[0]

    def sample_observations_from_factors(self, factors, r):
        out = []
        for f in range(factors.shape[0]):
            out.append(self.d[str(factors[f, :])])
        return out


def reps_fun(gt):
    def reprs(obs):
        reps = np.zeros((2, len(obs)))
        for f in range(len(obs)):
            reps[:, f] = gt.mu[:, obs[f]]
        return reps.T
    return reprs


if __name__ == "__main__":
    gin.bind_parameter("discretizer.discretizer_fn", _histogram_discretize)
    gin.bind_parameter("discretizer.num_bins", 10)


    gt = GT(True)
    print(compute_mig(gt, reps_fun(gt), num_train=1500, random_state=2020))
    #
    gt = GT(True)
    print(compute_dci(gt, reps_fun(gt), num_train=1000, random_state=2020, num_test=500))
    #
    gt = GT(True)
    print(compute_sap(gt, reps_fun(gt), num_train=1000, random_state=np.random.RandomState(0), num_test=500, continuous_factors=False))
    #
    gt = GT(True)
    print(compute_modularity_explicitness(gt, reps_fun(gt), num_train=1000, random_state=np.random.RandomState(0), num_test=500))

    gt = GT(False, False)
    print(compute_beta_vae_sklearn(gt, reps_fun(gt), num_train=1000, batch_size=64, num_eval=500,
                                   random_state = np.random.RandomState(0)))
    gt = GT(False, False)
    print(compute_factor_vae(gt, reps_fun(gt), num_train=1000, batch_size=64, num_eval=500,
                                   random_state = np.random.RandomState(0), num_variance_estimate=10))

