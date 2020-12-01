import numpy as np

def stratified_train_test_split(action, reward, state, ids, seq_lengths, m=8, seed=123):
    # stratified train/test split for actions and rewards
    # allocates a random subset of size m from the total interactions for each
    # user to the training set, and the remainder to the test set

    # for now assume state is none
    assert (state is None)
    np.random.seed(seed)
    N = len(set(ids))
    idxs = []
    for i in range(N):
        idxs.append(np.random.choice(range(12), 12, replace=False))

    seq_len = action.shape[1]
    n = 12 - m    # number of sequences in test set for each sub
    actions_train = np.empty(shape=(m * N, seq_len))
    actions_test = np.empty(shape=(n * N, seq_len))
    rewards_train = np.empty(shape=(m * N, seq_len))
    rewards_test = np.empty(shape=(n * N, seq_len))
    seq_train = np.empty(shape=(m * N), dtype=np.int32)
    seq_test = np.empty(shape=(n * N), dtype=np.int32)
    id_train = np.empty(shape=(m * N), dtype=object)
    id_test = np.empty(shape=(n * N), dtype=object)

    for i in range(0, len(ids)//12):
        # updates the train and test set, one subject for each iteration
        j = n*i; k = m*i; l=(m+n)*i   # keep track of various indexes, make this nicer
        actions_train[k:k+m,:]      = action[l:l+(m+n)][idxs[i][:m], :]
        actions_test[j:j+n,:]       = action[l:l+(m+n)][idxs[i][m:], :]
        rewards_train[k:k+m,:]      = reward[l:l+(m+n)][idxs[i][:m], :]
        rewards_test[j:j+n,:]       = reward[l:l+(m+n)][idxs[i][m:], :]
        seq_train[k:k+m]            = seq_lengths[l:l+(m+n)][idxs[i][:m]]
        seq_test[j:j+n]             = seq_lengths[l:l+(m+n)][idxs[i][m:]]
        id_train[k:k+m]             = np.array(ids)[l:l+(m+n)][idxs[i][:m]]
        id_test[j:j+n]              = np.array(ids)[l:l+(m+n)][idxs[i][m:]]


    return actions_train, actions_test, rewards_train, rewards_test, seq_train, seq_test, id_train, id_test