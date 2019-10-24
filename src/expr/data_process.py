import numpy as np


class DataProcess:

    def __init__(self):
        pass


    @staticmethod
    def train_test_between_subject(data,
                                   indx_data,
                                   train_blocks,
                                   n_actions,
                                   fmri_timeseries=None,
                                   time_series_process=None
                                   ):
        train = {}
        sdata = indx_data.loc[indx_data.train == "train"]

        ids = sdata['id'].unique().tolist()

        for s_id in ids:
            sub_data = data.loc[data.id == s_id]
            train[s_id] = []
            if fmri_timeseries is not None:
                fmri_ts = fmri_timeseries[s_id]
                if time_series_process is not None:
                    fmri_ts = time_series_process(fmri_ts)
            else:
                fmri_ts = None

            for t in train_blocks:
                reward, choices, states, record_time = DataProcess.get_training_data(sub_data, [t])
                train[s_id].append({
                                    'reward': reward,
                                    'action': choices,
                                    'id': s_id,
                                    'block': t,
                                    'record_time': record_time,
                                    'state': states,
                                    'fmri_timeseries': None if fmri_ts is None else np.array(fmri_ts)[np.newaxis, ]
                })

        test = {}
        sdata = indx_data.loc[indx_data.train == "test"]

        idt = sdata['id'].unique().tolist()

        for s_id in idt:

            sub_data = data.loc[data.id == s_id]
            test[s_id] = []
            if fmri_timeseries is not None:
                fmri_ts = fmri_timeseries[s_id]
                if time_series_process is not None:
                    fmri_ts = time_series_process(fmri_ts)
            else:
                fmri_ts = None
            for t in train_blocks:

                reward, choices, states, record_time = DataProcess.get_training_data(sub_data, [t])
                test[s_id].append({
                                    'reward': reward,
                                    'action': choices,
                                    'id': s_id,
                                    'block': t,
                                    'record_time':record_time,
                                    'state': states,
                                    'fmri_timeseries': None if fmri_ts is None else np.array(fmri_ts[np.newaxis, ])
                })

        return train, test

    @staticmethod
    #TODO: remove nactions from arguments
    def get_training_data(data, test_blocks):
        sub_data = data.loc[data.block.isin(test_blocks)]
        reward = sub_data['reward'].values
        choices = sub_data['action'].values
        if 'state' in sub_data:
            state = sub_data['state'].values
        else:
            state = None

        if 'fmri_time' in sub_data:
            record_time = sub_data['fmri_time'].values
        else:
            record_time = None
        return None if reward is None else reward[np.newaxis], \
               None if choices is None else choices[np.newaxis], \
               None if state is None else state[np.newaxis], \
               None if record_time is None else record_time[np.newaxis]

    @staticmethod
    def get_max_seq_len(train):
        max_len = -np.Inf
        max_fmri = -np.Inf
        for s_data in train.values():
            for t_data in s_data:
                if t_data['reward'].shape[1] > max_len:
                    max_len = t_data['reward'].shape[1]

                if not 'fmri_timeseries'in t_data or t_data['fmri_timeseries'] is None:
                    max_fmri = None
                else:
                    if t_data['fmri_timeseries'].shape[1] > max_fmri:
                        max_fmri = t_data['fmri_timeseries'].shape[1]

        return max_len, max_fmri

    @staticmethod
    def merge_blocks(data):
        merged_data = {}

        for k in sorted(data.keys()):
            v = data[k]
            merged_data[k] = DataProcess.merge_data({'merged': v})['merged']

        return merged_data

    @staticmethod
    def merge_data(train, batch_size=-1):

        max_len, _ = DataProcess.get_max_seq_len(train)

        def app_not_None(arr, to_append, max_len):
            if to_append is not None:
                if max_len is not None:
                    pad_shape = [(0, 0), (0, (max_len - to_append.shape[1]))] + [(0,0)] * (len(to_append.shape)-2)
                    arr.append(np.lib.pad(to_append, pad_shape, 'constant', constant_values=(0, -1)))
                else:
                    arr.append(to_append)

        def none_if_empty(arr):
            if len(arr) > 0:
                return np.concatenate(arr)
            return None

        rewards = []
        choices = []
        timesteps = []
        states = []
        fmri_tss = []
        record_time = []
        ids = []
        seq_lengths = []

        batches = []

        cur_size = 0

        for k_data in reversed(sorted(train.keys())):
            s_data = train[k_data]
            for t_data in s_data:
                seq_lengths.append(t_data['reward'].shape[1])
                app_not_None(rewards, t_data['reward'], max_len)
                app_not_None(choices, t_data['action'], max_len)
                app_not_None(states, t_data['state'], max_len)
                if 'fmri_timeseries' in t_data:
                    app_not_None(fmri_tss, t_data['fmri_timeseries'], None)
                if 'record_time' in t_data:
                    app_not_None(record_time, t_data['record_time'], max_len)
                ids.append(t_data['id'])

                cur_size += 1
                if batch_size != -1 and cur_size >= batch_size:
                    batches.append({
                        'reward': none_if_empty(rewards),
                        'action': none_if_empty(choices),
                        'state': none_if_empty(states),
                        'fmri_timeseries': none_if_empty(fmri_tss),
                        'record_time': none_if_empty(record_time),
                        'block': len(batches),
                        'id': str(ids)
                    })

                    rewards = []
                    choices = []
                    timesteps = []
                    states = []
                    fmri_tss = []
                    record_time = []
                    ids = []
                    cur_size = 0

        if cur_size > 0:
            batches.append({
                'reward': none_if_empty(rewards),
                'action': none_if_empty(choices),
                'state': none_if_empty(states),
                'fmri_timeseries': none_if_empty(fmri_tss),
                'record_time': none_if_empty(record_time),
                'block': len(batches),
                'id': ids,
                'seq_lengths': np.array(seq_lengths)
            })

        return {'merged': batches}
