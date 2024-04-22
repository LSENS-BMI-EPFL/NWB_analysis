import yaml
import numpy as np
import pandas as pd
import seaborn as sns

import nwb_wrappers.nwb_reader_functions as nwb_read
from nwb_utils import utils_misc


def return_trial_table_with_area_dff_avg(nwb_files, rrs_keys):
    df = []
    for nwb_file in nwb_files:
        session_id = nwb_read.get_session_id(nwb_file)
        mouse_id = session_id[0:5]
        print(f"Mouse : {mouse_id}, session : {session_id}")

        # Get trial table
        trial_table = nwb_read.get_trial_table(nwb_file)
        trial_table['correct_trial'] = trial_table.lick_flag == trial_table.reward_available
        n_trials = len(trial_table)

        # Get activity and brain region ROIs
        rrs_array = nwb_read.get_roi_response_serie_data(nwb_file, keys=rrs_keys)
        rrs_ts = nwb_read.get_roi_response_serie_timestamps(nwb_file, keys=rrs_keys)
        rrs_cell_type_dict = nwb_read.get_cell_indices_by_cell_type(nwb_file, keys=rrs_keys)
        n_cell_types = len(rrs_cell_type_dict.keys())

        # Get all quiet windows
        quiet_window_frames = []
        for trial in range(n_trials):
            quiet_start = trial_table.iloc[trial].abort_window_start_time
            stim_time = trial_table.iloc[trial].start_time
            start_frame = utils_misc.find_nearest(rrs_ts, quiet_start)
            end_frame = utils_misc.find_nearest(rrs_ts, stim_time)
            quiet_window_frames.append(np.arange(start_frame, end_frame))

        # Get average activity in each ROIs for each trial
        trial_by_type = np.zeros((n_trials, n_cell_types))
        for trial in range(n_trials):
            activity = rrs_array[:, quiet_window_frames[trial]]
            mean_activity = np.mean(activity, axis=1)
            trial_by_type[trial, :] = mean_activity

        # Keep ony a few relevant columns
        cols = ['start_time', 'trial_type', 'reward_available', 'lick_flag', 'correct_trial',
                'context', 'context_background']
        sub_trial_table = trial_table[cols]
        session_df = sub_trial_table.copy(deep=True)

        # Add to trial table the average baseline dff for each brain region ROI
        for area, area_index in rrs_cell_type_dict.items():
            session_df[f'{area}_baseline_dff'] = trial_by_type[:, area_index]

        # Add context transition column
        transitions = list(np.diff(session_df['context']))
        transitions.insert(0, 0)
        block_size = np.where(transitions)[0][0]

        # Get context start timestamps and time in context at each trial
        rewarded_context_start_timestamps = nwb_read.get_behavioral_epochs_times(nwb_file, epoch_name='rewarded')[0]
        non_rewarded_context_start_timestamps = nwb_read.get_behavioral_epochs_times(nwb_file, epoch_name='non-rewarded')[0]
        if min(non_rewarded_context_start_timestamps) < min(rewarded_context_start_timestamps):
            context_starts = np.concatenate((non_rewarded_context_start_timestamps, rewarded_context_start_timestamps))
        else:
            context_starts = np.concatenate((rewarded_context_start_timestamps, non_rewarded_context_start_timestamps))
        context_starts = np.sort(context_starts)
        context_starts = np.repeat(context_starts, block_size)
        context_starts = context_starts[0: len(session_df)]
        session_df['context_start'] = context_starts
        session_df['time_in_context'] = session_df['start_time'] - session_df['context_start']

        # Add mouseID and sessionID
        session_df['mouse_id'] = mouse_id
        session_df['session_id'] = session_id

        df.append(session_df)

    return pd.concat(df)


config_file = "//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Robin_Dard/group.yaml"
with open(config_file, 'r', encoding='utf8') as stream:
    config_dict = yaml.safe_load(stream)
# sessions = config_dict['NWB_CI_LSENS']['Context_expert_sessions']
# sessions = config_dict['NWB_CI_LSENS']['Context_good_params']
# sessions = config_dict['NWB_CI_LSENS']['context_expert_widefield']
# sessions = config_dict['NWB_CI_LSENS']['Context_contrast_expert']
sessions = config_dict['NWB_CI_LSENS']['context_contrast_widefield']
files = [session[1] for session in sessions]

all_df = return_trial_table_with_area_dff_avg(nwb_files=files, rrs_keys=['ophys', 'brain_area_fluorescence', 'dff0_traces'])

# --------------------------------------------------------------------------------------------------------------- #
# Here to the good grouping to end up with one point per session or one point per mouse in the plots
print(' ')
print('Average data by session')
# Select columns to drop
# cols_to_drop = ['start_time', 'trial_type', 'reward_available', 'lick_flag',
#                 'correct_trial', 'context_background', 'context_start', 'time_in_context']
# cols_to_drop = ['start_time', 'trial_type', 'reward_available', 'lick_flag',
#                 'context_background', 'context_start', 'time_in_context']
cols_to_drop = ['start_time', 'trial_type', 'reward_available', 'lick_flag', 'context',
                'context_background', 'context_start', 'time_in_context']
selected_df = all_df.drop(labels=cols_to_drop, axis=1)

# Select columns to group
# cols_to_group = ['context', 'correct_trial', 'mouse_id', 'session_id']
cols_to_group = ['correct_trial', 'mouse_id', 'session_id']

# Average data
avg_data = selected_df.groupby(cols_to_group, as_index=False).agg(np.nanmean)

# Choose brain region
brain_region = 'A1'
brain_region += '_baseline_dff'

# Do some plots for baseline
g = sns.catplot(avg_data, y=brain_region, x='context', hue='session_id', kind='point')
g = sns.catplot(avg_data, y=brain_region, x='correct_trial', hue='session_id', kind='point')
g = sns.catplot(avg_data, y=brain_region, x='context', hue='mouse_id', kind='point', col='correct_trial')


g = sns.catplot(selected_df, y=brain_region, x='context', col='mouse_id', hue='session_id', kind='point')
g = sns.catplot(selected_df, y=brain_region, x='context', col='mouse_id', row='correct_trial', hue='session_id', kind='point')
g = sns.lmplot(selected_df, y=brain_region, x='context', hue='session_id', col='correct_trial')

for mouse in selected_df.mouse_id.unique():
    g = sns.catplot(selected_df.loc[selected_df.mouse_id == mouse], y=brain_region, x='context',
                    col='session_id', kind='point')

# Do some plots for baseline over time
g = sns.lmplot(data=all_df, y=brain_region, x='time_in_context', hue='context')
for mouse in all_df.mouse_id.unique():
    g = sns.lmplot(data=all_df.loc[(all_df.mouse_id == mouse)], y=brain_region, x='time_in_context',
                   hue='context', col='session_id')




