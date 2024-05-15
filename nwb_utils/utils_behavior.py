import pandas as pd
import numpy as np
import nwb_wrappers.nwb_reader_functions as nwb_read
from nwb_utils.utils_misc import get_continuous_time_periods

def build_standard_behavior_table(nwb_list):
    """
    Build a behavior table from a list of NWB files containing standardized trial tables.
    :param nwb_list:
    :return:
    """
    bhv_data = []
    for nwb_file in nwb_list:
        data_frame = nwb_read.get_trial_table(nwb_file)
        mouse_id = nwb_read.get_mouse_id(nwb_file)
        behavior_type, day = nwb_read.get_bhv_type_and_training_day_index(nwb_file)
        session_id = nwb_read.get_session_id(nwb_file)
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
    bhv_data['correct_choice'] = bhv_data.reward_available == bhv_data.lick_flag

    return bhv_data


def build_standard_behavior_event_table(nwb_list):
    """
        Build a behavior table from a list of NWB files containing standardized trial tables
        as well as behavioural events from the processing module.
        :param nwb_list:
        :return:
        """
    bhv_data = []
    for nwb_file in nwb_list:
        data_frame = nwb_read.get_trial_table(nwb_file)
        mouse_id = nwb_read.get_mouse_id(nwb_file)
        behavior_type, day = nwb_read.get_bhv_type_and_training_day_index(nwb_file)
        session_id = nwb_read.get_session_id(nwb_file)
        data_frame['mouse_id'] = [mouse_id for trial in range(len(data_frame.index))]
        data_frame['session_id'] = [session_id for trial in range(len(data_frame.index))]
        data_frame['behavior'] = [behavior_type for trial in range(len(data_frame.index))]
        data_frame['day'] = [day for trial in range(len(data_frame.index))]

        # Add behavioral events for each trial, relative to start time
        trial_starts = data_frame['response_window_start_time'].values
        trial_stops = trial_starts + 5
        event_dict = nwb_read.get_behavioral_events_time(nwb_file)
        events_to_keep = ['piezo_lick_times']
        for key in events_to_keep: # add each event types
            event_list = []
            for start, stop in zip(trial_starts, trial_stops): # keep events in trial
                try:
                    events_in_trial = [t-start for t in event_dict[key] if t >= start and t <= stop]
                except KeyError as err:
                    print(err, "not found in events dict")
                    events_in_trial = []

                lick_time = data_frame.loc[data_frame['response_window_start_time'] == start, 'lick_time'].values[0]
                reaction_time = lick_time - start
                events_in_trial.insert(0, reaction_time) # insert first lick time in case undetected

                # Assert list of licks not empty if is it a lick trial
                if data_frame.loc[data_frame['response_window_start_time'] == start, 'lick_flag'].values[0]:
                    assert len(events_in_trial) > 0, "No lick detected in lick trial"
                event_list.append(events_in_trial)

            # Add to dataframe
            data_frame[key] = event_list

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
        data_frame = nwb_read.get_trial_table(nwb_file)
        mouse_id = nwb_read.get_mouse_id(nwb_file)
        behavior_type, day = nwb_read.get_bhv_type_and_training_day_index(nwb_file)
        session_id = nwb_read.get_session_id(nwb_file)
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


def get_standard_single_session_table(combine_bhv_data, session, block_size=20, verbose=True):
    """
    Get a single session trial table from the combined behavior table.
    :param combine_bhv_data:
    :param session:
    :param block_size:
    :param verbose:
    :return:
    """
    session_table = combine_bhv_data.loc[(combine_bhv_data['session_id'] == session)]
    session_table = session_table.loc[session_table.early_lick == 0]
    session_table = session_table.reset_index(drop=True)
    if verbose:
        print(f" ")
        print(f"Session : {session}, mouse : {session_table['mouse_id'].values[0]}, "
              f"behavior : {session_table['behavior'].values[0]}, "
              f"day : {session_table['day'].values[0]}")

    # Find the block length if context
    if session_table['behavior'].values[0] == ["context", "whisker_context"]:
        switches = np.where(np.diff(session_table.context.values[:]))[0]
        if len(switches) <= 1:
            block_length = switches[0] + 1
        else:
            block_length = min(np.diff(switches))
    else:
        switches = None
        block_length = block_size

    # Add the block info :
    session_table['trial'] = session_table.index
    session_table['block'] = session_table.loc[session_table.early_lick == 0, 'trial'].transform(lambda x: x // block_length)

    # Compute hit rates. Use transform to propagate hit rate to all entries.
    session_table['hr_w'] = session_table.groupby(['block', 'opto_stim'], as_index=False, dropna=False)['outcome_w'].transform(np.nanmean)
    session_table['hr_a'] = session_table.groupby(['block', 'opto_stim'], as_index=False, dropna=False)['outcome_a'].transform(np.nanmean)
    session_table['hr_n'] = session_table.groupby(['block', 'opto_stim'], as_index=False, dropna=False)['outcome_n'].transform(np.nanmean)
    session_table['correct'] = session_table.groupby(['block', 'opto_stim'], as_index=False, dropna=False)['correct_choice'].transform(np.nanmean)

    # Add whisker contrast:

    return session_table, switches, block_length


def get_single_session_table(combine_bhv_data, session, block_size=20, verbose=True):
    """
    Get a single session trial table from the combined behavior table.
    :param combine_bhv_data:
    :param session:
    :param block_size:
    :param verbose:
    :return:
    """
    session_table = combine_bhv_data.loc[(combine_bhv_data['session_id'] == session)]
    session_table = session_table.reset_index(drop=True)
    if verbose:
        print(f" ")
        print(f"Session : {session}, mouse : {session_table['mouse_id'].values[0]}, "
              f"behavior : {session_table['behavior'].values[0]}, "
              f"day : {session_table['day'].values[0]}")

    # Find the block length if context
    if session_table['behavior'].values[0] in ["context", "whisker_context"]:
        switches = np.where(np.diff(session_table.wh_reward.values[:]))[0]
        if len(switches) <= 1:
            block_length = switches[0] + 1
        else:
            block_length = min(np.diff(switches))
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


def get_standard_multi_session_table(data, block_size=20, verbose=True):
    """
    Get a single session trial table from the combined behavior table.
    :param combine_bhv_data:
    :param session:
    :param block_size:
    :param verbose:
    :return:
    """
    if verbose:
        print(f" ")
        # print(f"Session : {session}, mouse : {session_table['mouse_id'].values[0]}, "
        #       f"behavior : {session_table['behavior'].values[0]}, "
        #       f"day : {session_table['day'].values[0]}")

    # Find the block length if context
    if data['behavior'].values[0] in ["context", "whisker_context"]:
        switches = np.where(np.diff(data.context.values[:]))[0]
        if len(switches) <= 1:
            block_length = switches[0] + 1
        else:
            block_length = min(np.diff(switches))
    else:
        switches = None
        block_length = block_size

    # Add the block info :
    data['trial'] = data.index
    data['block'] = data.loc[data.early_lick == 0, 'trial'].transform(lambda x: x // block_length)

    # Compute hit rates. Use transform to propagate hit rate to all entries.
    data['hr_w'] = data.groupby(['block', 'opto_stim'], as_index=False)['outcome_w'].transform(np.nanmean)
    data['hr_a'] = data.groupby(['block', 'opto_stim'], as_index=False)['outcome_a'].transform(np.nanmean)
    data['hr_n'] = data.groupby(['block', 'opto_stim'], as_index=False)['outcome_n'].transform(np.nanmean)

    return data, switches, block_length


def filter_events_based_on_epochs(events_ts, epochs):
    
    event_in_epoch = []
    for event in events_ts:
        n_epochs = epochs.shape[1]
        keep_event = False
        for sub_epoch in range(n_epochs):
            main_epoch_start = epochs[0, sub_epoch]
            main_epoch_stop = epochs[1, sub_epoch]
            if event > main_epoch_start and event < main_epoch_stop:
                keep_event = True
        if keep_event:
            event_in_epoch.append(True)
        else:
            event_in_epoch.append(False)

    events_filtered = events_ts[event_in_epoch]

    return events_filtered


def compute_single_session_metrics(combine_bhv_data):
    combine_bhv_data['trial'] = combine_bhv_data.groupby(
        'session_id').cumcount()  # Gives an increasing number to each trial in a session, starting from 0
    combine_bhv_data['block'] = combine_bhv_data.groupby('session_id')['context'].transform(lambda x: np.abs(
        np.diff(x, prepend=0)).cumsum())  # Gives an increasing number to each block in a session, starting from 0
    combine_bhv_data['switches'] = combine_bhv_data.groupby('session_id')['context'].transform(
        lambda x: np.abs(np.diff(x, prepend=[0 if x.iloc[0] == 1 else 1])))  # zeros vector with ones at the context switches
    combine_bhv_data['switch_trials'] = np.where(combine_bhv_data.groupby('session_id')['context'].transform(
        lambda x: np.abs(np.diff(x, prepend=[0 if x.iloc[0] == 1 else 1]))) != 0, combine_bhv_data.trial, np.nan)

    combine_bhv_data['hr_w'] = combine_bhv_data.groupby(['session_id', 'block'])['outcome_w'].transform(np.nanmean)
    combine_bhv_data['hr_a'] = combine_bhv_data.groupby(['session_id', 'block'])['outcome_a'].transform(np.nanmean)
    combine_bhv_data['hr_n'] = combine_bhv_data.groupby(['session_id', 'block'])['outcome_n'].transform(np.nanmean)

    return combine_bhv_data


def get_by_block_table(combine_bhv_data):
    by_block = combine_bhv_data.groupby(['session_id', 'block'], sort=False)['mouse_id', 'behavior', 'context', 'context_background','hr_w', 'hr_a', 'hr_n'].agg('max')
    by_block['trial'] = combine_bhv_data.groupby(['session_id', 'block'], sort=False)['trial'].apply(lambda x: int(round(x.mean(), 0)))
    by_block = by_block.reset_index(names=['session_id', 'block'])
    by_block['switches'] = combine_bhv_data['switch_trials'].dropna().reset_index(drop=True)
    by_block['correct_a'] = by_block.hr_a
    by_block['correct_n'] = 1 - by_block.hr_n
    by_block['correct_w'] = [1 - by_block.hr_w.values[i] if by_block.context.values[i] == 0 else by_block.hr_w.values[i] for i in
                      range(len(by_block))]

    by_block['contrast_a'] = by_block.groupby('session_id')['hr_a'].transform(lambda x: compute_contrast(x))
    by_block['contrast_n'] = by_block.groupby('session_id')['hr_n'].transform(lambda x: compute_contrast(x))
    by_block['contrast_w'] = by_block.groupby('session_id')['hr_w'].transform(lambda x: compute_contrast(x))
    by_block['six_contrast_w'] = by_block.contrast_w > 0.375
    by_block['context'] = by_block['context'].map({0:'Non-Rewarded', 1:'Rewarded'})

    return by_block


def compute_contrast(data):

    contrast = [(np.abs(data.values[i] - data.values[i - 1]) + np.abs(data.values[i] - data.values[i + 1])) / 2 for i in np.arange(1, data.size - 1)]
    contrast.insert(0, np.nan)
    contrast.insert(len(contrast), np.nan)

    return contrast


def compute_above_threshold(data, threshold):

    above_threshold = dict()
    n_blocks = data.block.values[-1]
    above_thresh = data.six_contrast_w.values[:]
    continuous_periods = get_continuous_time_periods(above_thresh)
    len_above_thresh = len(np.where(np.array([np.diff(i) for i in continuous_periods]) >= threshold)[0])
    above_threshold['mouse_id'] = data.mouse_id.unique()
    above_threshold['session_id'] = data.session_id.unique()
    above_threshold['n_blocks'] = [n_blocks]
    above_threshold['n_good_blocks'] = [len(np.where(above_thresh)[0])]
    above_threshold['n_4_successive_good_blocks'] = [len_above_thresh]

    return above_threshold