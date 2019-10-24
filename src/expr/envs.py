def a2_generic(a1_period, off_pol_trials, reward_trials, init_state=0, init_action=0, init_reward=0):
    def env(s, a, trial):

        # initial state action
        if trial == -1:
            return init_state, init_reward, init_action

        if off_pol_trials(trial):

            if a1_period(trial):
                next_a = 0
            else:
                next_a = 1

        else:
            next_a = None

        if reward_trials(trial):
            r = 1
        else:
            r = 0

        return s, r, next_a

    return env