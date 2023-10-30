import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import NWB_reader_functions as NWB_read


def build_standard_behavior_table(nwb_list):
    bhv_data = []
    for nwb_file in nwb_list:
        data_frame = NWB_read.get_trial_table(nwb_file)
        mouse_id = NWB_read.get_mouse_id(nwb_file)
        behavior_type, day = NWB_read.get_bhv_type_and_training_day_index(nwb_file)
        session_id = NWB_read.get_session_id(nwb_file)
        data_frame['mouse_id'] = [mouse_id for trial in range(len(data_frame.index))]
        data_frame['session_id'] = [session_id for trial in range(len(data_frame.index))]
        data_frame['behavior'] = [behavior_type for trial in range(len(data_frame.index))]
        data_frame['day'] = [day for trial in range(len(data_frame.index))]
        bhv_data.append(data_frame)

    bhv_data = pd.concat(bhv_data, ignore_index=True)

    # Add performance outcome column for each stimulus.
    bhv_data['outcome_w'] = bhv_data.loc[(bhv_data.trial_type == 'whisker_trial')]['lick_flag']
    bhv_data['outcome_a'] = bhv_data.loc[(bhv_data.trial_type == 'auditory_trial')]['lick_flag']
    bhv_data['outcome_n'] = bhv_data.loc[(bhv_data.trial_type == 'no_stim_trial')]['lick_flag']

    return bhv_data

def build_general_behavior_table(nwb_list):
    bhv_data = []
    for nwb_file in nwb_list:
        data_frame = NWB_read.get_trial_table(nwb_file)
        mouse_id = NWB_read.get_mouse_id(nwb_file)
        behavior_type, day = NWB_read.get_bhv_type_and_training_day_index(nwb_file)
        session_id = NWB_read.get_session_id(nwb_file)
        data_frame['mouse_id'] = [mouse_id for trial in range(len(data_frame.index))]
        data_frame['session_id'] = [session_id for trial in range(len(data_frame.index))]
        data_frame['behavior'] = [behavior_type for trial in range(len(data_frame.index))]
        data_frame['day'] = [day for trial in range(len(data_frame.index))]
        bhv_data.append(data_frame)

    bhv_data = pd.concat(bhv_data, ignore_index=True)

    # Make outcome binary for simplicity
    bhv_data = bhv_data.replace({'trial_outcome': {'Hit': 1, 'Miss': 0}})

    # Add performance outcome column for each stimulus.
    bhv_data['outcome_w'] = bhv_data.loc[(bhv_data.trial_type == 'whisker')]['trial_outcome']
    bhv_data['outcome_a'] = bhv_data.loc[(bhv_data.trial_type == 'auditory')]['trial_outcome']
    bhv_data['outcome_c'] = bhv_data.loc[(bhv_data.trial_type == 'catch')]['trial_outcome']

    return bhv_data


def get_single_session_table(combine_bhv_data, session, block_size=20, verbose=True):
    session_table = combine_bhv_data.loc[(combine_bhv_data['session_id'] == session)]
    session_table = session_table.reset_index(drop=True)
    if verbose:
        print(f" ")
        print(f"Session : {session}, mouse : {session_table['mouse_id'].values[0]}, "
              f"behavior : {session_table['behavior'].values[0]}, "
              f"day : {session_table['day'].values[0]}")

    # Find the block length if context
    if session_table['behavior'].values[0] == "context":
        switches = np.where(np.diff(session_table.wh_reward.values[:]))[0]
        block_length = switches[0] + 1
    else:
        switches = None
        block_length = block_size

    # Add the block info :
    session_table['trial'] = session_table.index
    session_table['block'] = session_table.loc[session_table.early_lick == 0, 'trial'].transform(lambda x: x // block_length)

    # Compute hit rates. Use transform to propagate hit rate to all entries.
    session_table['hr_w'] = session_table.groupby(['block'], as_index=False)['outcome_w'].transform(np.nanmean)
    session_table['hr_a'] = session_table.groupby(['block'], as_index=False)['outcome_a'].transform(np.nanmean)
    session_table['hr_c'] = session_table.groupby(['block'], as_index=False)['outcome_c'].transform(np.nanmean)

    return session_table, switches, block_length


def get_single_mouse_table(combine_bhv_data, mouse):
    mouse_table = combine_bhv_data.loc[(combine_bhv_data['mouse_id'] == mouse)]
    mouse_table = mouse_table.reset_index(drop=True)

    return mouse_table


def get_single_session_time_to_switch(combine_bhv_data, do_single_session_plot=False):
    sessions_list = np.unique(combine_bhv_data['session_id'].values[:])
    n_sessions = len(sessions_list)
    print(f"N sessions : {n_sessions}")
    to_rewarded_transitions_prob = dict()
    to_non_rewarded_transitions_prob = dict()
    for session_id in sessions_list:
        session_table, switches, block_size = get_single_session_table(combine_bhv_data, session=session_id)

        # Keep only the session with context
        if session_table['behavior'].values[0] not in 'context':
            continue

        # Keep only the whisker trials
        whisker_session_table = session_table.loc[session_table.trial_type == 'whisker']
        whisker_session_table = whisker_session_table.reset_index(drop=True)

        # extract licks array
        licks = whisker_session_table.outcome_w.values[:]

        # Extract transitions rwd to non rwd and opposite
        rewarded_transitions = np.where(np.diff(whisker_session_table.wh_reward.values[:]) == 1)[0]
        non_rewarded_transitions = np.where(np.diff(whisker_session_table.wh_reward.values[:]) == -1)[0]

        # Build rewarded transitions matrix from trial -3 to trial +3
        wh_switches = np.where(np.diff(whisker_session_table.wh_reward.values[:]))[0]
        n_trials_around = wh_switches[0] + 1
        trials_above = n_trials_around + 1
        trials_below = n_trials_around - 1
        rewarded_transitions_mat = np.zeros((len(rewarded_transitions), 2 * n_trials_around))
        for index, transition in enumerate(list(rewarded_transitions)):
            if transition + trials_above > len(licks):
                rewarded_transitions_mat = rewarded_transitions_mat[0: len(rewarded_transitions) - 1, :]
                continue
            else:
                rewarded_transitions_mat[index, :] = licks[np.arange(transition - trials_below, transition + trials_above)]
        rewarded_transition_prob = np.mean(rewarded_transitions_mat, axis=0)
        to_rewarded_transitions_prob[session_id] = rewarded_transition_prob

        # Build non_rewarded transitions matrix from trial -3 to trial +3
        non_rewarded_transitions_mat = np.zeros((len(non_rewarded_transitions), 2 * n_trials_around))
        for index, transition in enumerate(list(non_rewarded_transitions)):
            if transition + trials_above > len(licks):
                non_rewarded_transitions_mat = non_rewarded_transitions_mat[0: len(non_rewarded_transitions) - 1, :]
                continue
            else:
                non_rewarded_transitions_mat[index, :] = licks[np.arange(transition - trials_below, transition + trials_above)]
        non_rewarded_transition_prob = np.mean(non_rewarded_transitions_mat, axis=0)
        to_non_rewarded_transitions_prob[session_id] = non_rewarded_transition_prob

        # Do single session plot
        if do_single_session_plot:
            figsize = (6, 4)
            figure, ax = plt.subplots(1, 1, figsize=figsize)
            scale = np.arange(-n_trials_around, n_trials_around + 1)
            scale = np.delete(scale, n_trials_around)
            ax.plot(scale, rewarded_transition_prob, '--go')
            ax.plot(scale, non_rewarded_transition_prob, '--ro')
            ax.set_xlabel('Trial number')
            ax.set_ylabel('Lick probability')
            figure_title = f"{session_table.mouse_id.values[0]}, {session_table.behavior.values[0]} " \
                           f"{session_table.day.values[0]}"
            ax.set_title(figure_title)
            ax.spines[['right', 'top']].set_visible(False)
            ax.set_ylim([-0.1, 1.05])
            plt.xticks(range(-n_trials_around, n_trials_around + 1))
            plt.show()

    return to_rewarded_transitions_prob, to_non_rewarded_transitions_prob

