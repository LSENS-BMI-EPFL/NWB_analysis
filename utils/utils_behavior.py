import pandas as pd
import numpy as np
import nwb_wrappers.nwb_reader_functions as nwb_read


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
        for sub_epoch in range(n_epochs):
            main_epoch_start = epochs[0, sub_epoch]
            main_epoch_stop = epochs[1, sub_epoch]
            if event > main_epoch_start and event < main_epoch_stop:
                event_in_epoch.append(True)
            else:
                event_in_epoch.append(False)
    events_filtered = events_ts[event_in_epoch]

    return events_filtered
