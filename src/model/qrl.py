import numpy as np
import pandas as pd

from util.helper import ensure_dir


class QL:

    @classmethod
    def sim(cls, alpha, beta, persv, p0, p1, trials):
        Q = np.array([0.0, 0.0])
        a_list = []
        r_list  = []
        a = -1
        for t in range(trials):
            pol0 = np.exp(beta * Q[0] + (a == 0) * persv) / \
                   (np.exp(beta * Q[0] + (a == 0) * persv) + np.exp((a == 1) * persv + beta * Q[1]))

            if np.random.uniform(0, 1) < pol0:
                a = 0
                if np.random.uniform(0, 1) < p0:
                    r = 1
                else:
                    r = 0
            else:
                a = 1
                if np.random.uniform(0, 1) < p1:
                    r = 1
                else:
                    r = 0

            a_list.append(a)
            r_list.append(r)
            Q[a] = (1 - alpha) * Q[a] + alpha * r

        return a_list, r_list

    @classmethod
    def sim_OFF(cls, alpha, beta, persv, rewards, actions):
        Q = np.array([0.0, 0.0])
        a_list = []
        r_list  = []
        a = -1
        pol_list = []
        for t in range(len(rewards)):
            pol0 = np.exp(beta * Q[0] + (a == 0) * persv) / \
                   (np.exp(beta * Q[0] + (a == 0) * persv) + np.exp((a == 1) * persv + beta * Q[1]))

            pol_list.append(pol0)
            a = actions[t]
            r = rewards[t]
            a_list.append(a)
            r_list.append(r)
            Q[a] = (1 - alpha) * Q[a] + alpha * r

        return pol_list

    @classmethod
    def generate_off(cls):
        rewards = [0] * 10
        rewards[4] = 1
        # rewards[14] = 1
        actions = [0] * 10

        ind = 0
        for kappa in np.linspace(-1.2, 1.2, num=15):
            beta = 3
            z_dim = 1
            other_dim = 0

            path = "../nongit/local/synth/sims/dims/A1/z0/_" + str(ind) + '/'
            ensure_dir(path)

            pol = np.array(cls.sim_OFF(0.2, beta, kappa, rewards, actions))
            polpd = pd.DataFrame({'0': pol, '1': 1 - pol, 'id': 'id1', 'block': 1})
            polpd.to_csv(path + "policies-.csv")

            train = pd.DataFrame({'reward': rewards, 'action': actions, 'state0':'', 'id': 'id1', 'block': 1})
            train.to_csv(path + "train.csv")

            np.savetxt(path + "z.csv", np.array([[beta, kappa]]), delimiter=',')

            pd.DataFrame({'z_dim': [z_dim], 'other_dim': [other_dim]}).to_csv(path + "z_info.csv")
            ind += 1

        ind = 0
        for beta in np.linspace(0, 9, num=15):
            kappa = 0
            z_dim = 0
            other_dim = 1

            path = "../nongit/local/synth/sims/dims/A1/z1/_" + str(ind) + '/'
            ensure_dir(path)

            pol = np.array(cls.sim_OFF(0.2, beta, kappa, rewards, actions))
            polpd = pd.DataFrame({'0': pol, '1': 1 - pol, 'id': 'id1', 'block': 1})
            polpd.to_csv(path + "policies-.csv")

            train = pd.DataFrame({'reward': rewards, 'action': actions, 'state0':'', 'id': 'id1', 'block': 1})
            train.to_csv(path + "train.csv")

            np.savetxt(path + "z.csv", np.array([[beta, kappa]]), delimiter=',')

            pd.DataFrame({'z_dim': [z_dim], 'other_dim': [other_dim]}).to_csv(path + "z_info.csv")
            ind += 1

    @classmethod
    def sim_subj_fixed(cls):
        dfs = []
        for s in range(1000):
            if np.random.uniform(0, 1) < 0.5:
                a_list, r_list  = cls.sim(0.5, 4.7, 1.2, 0.6, 0.1, 100)
            else:
                a_list, r_list = cls.sim(0.5, 4.7, 1.2, 0.1, 0.6, 100)
            df = pd.DataFrame({'action': a_list, 'reward': r_list, 'id': 'id' + str(s)})
            dfs.append(df)

        pd.concat(dfs).to_csv("../data/synth/fixed.csv", index=False)

    @classmethod
    def sim_subj_Gaussian(cls):

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        dfs = []
        alpha_list = []
        beta_list = []
        unalpha_list = []
        unbeta_list = []
        unpersv_list = []
        persv_list = []
        id_list = []
        for s in range(1500):
            alpha_untrans = np.random.normal() - 1
            alpha = sigmoid(alpha_untrans)
            beta_untrans = np.random.normal()
            beta = abs(beta_untrans * 6)
            persv_untrans =np.random.normal()
            persv = persv_untrans
            if np.random.uniform(0, 1) < 0.5:
                a_list, r_list  = cls.sim(0.2, beta, persv, 0.5, 0.1, 150)
            else:
                a_list, r_list = cls.sim(0.2, beta, persv, 0.1, 0.5, 150)
            df = pd.DataFrame({'action': a_list, 'reward': r_list, 'id': 'id' + str(s)})
            dfs.append(df)

            alpha_list.append(alpha)
            beta_list.append(beta)
            persv_list.append(persv)
            unalpha_list.append(alpha_untrans)
            unbeta_list.append(beta_untrans)
            unpersv_list.append(persv_untrans)
            id_list.append('id' + str(s))

        pd.concat(dfs).to_csv("../data/synth/normal.csv", index=False)
        pd.DataFrame({'alpha': alpha_list,
                      'beta': beta_list,
                      'persv': persv_list,
                      'beta_untrans': unbeta_list,
                      'alpha_untrans': unalpha_list,
                      'persv_untrans': unpersv_list,
                      'id': id_list}).to_csv("../data/synth/params.csv", index=False)


if __name__ == '__main__':
    # QL.sim_subj_fixed()
    # QL.sim_subj_Gaussian()
    QL.generate_off()
