import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import NWB_reader_functions as NWB_read


def build_standard_behavior_table(nwb_list):
    """
    Build a behavior table from a list of NWB files containing standardized trial tables.
    :param nwb_list:
    :return:
    """
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
    session_table = session_table.reset_index(drop=True)
    if verbose:
        print(f" ")
        print(f"Session : {session}, mouse : {session_table['mouse_id'].values[0]}, "
              f"behavior : {session_table['behavior'].values[0]}, "
              f"day : {session_table['day'].values[0]}")

    # Find the block length if context
    if session_table['behavior'].values[0] == "context":
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
    session_table['hr_w'] = session_table.groupby(['block'], as_index=False)['outcome_w'].transform(np.nanmean)
    session_table['hr_a'] = session_table.groupby(['block'], as_index=False)['outcome_a'].transform(np.nanmean)
    session_table['hr_n'] = session_table.groupby(['block'], as_index=False)['outcome_n'].transform(np.nanmean)

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
    if session_table['behavior'].values[0] == "context":
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


