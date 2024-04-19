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
        trial_table = nwb_read.get_trial_table(nwb_file)
        trial_table['correct_trial'] = trial_table.lick_flag == trial_table.reward_available
        n_trials = len(trial_table)

        rrs_array = nwb_read.get_roi_response_serie_data(nwb_file, keys=rrs_keys)
        rrs_ts = nwb_read.get_roi_response_serie_timestamps(nwb_file, keys=rrs_keys)
        rrs_cell_type_dict = nwb_read.get_cell_indices_by_cell_type(nwb_file, keys=rrs_keys)
        n_cell_types = len(rrs_cell_type_dict.keys())

        quiet_window_frames = []
        for trial in range(n_trials):
            quiet_start = trial_table.iloc[trial].abort_window_start_time
            stim_time = trial_table.iloc[trial].start_time
            start_frame = utils_misc.find_nearest(rrs_ts, quiet_start)
            end_frame = utils_misc.find_nearest(rrs_ts, stim_time)
            quiet_window_frames.append(np.arange(start_frame, end_frame))

        trial_by_type = np.zeros((n_trials, n_cell_types))
        for trial in range(n_trials):
            activity = rrs_array[:, quiet_window_frames[trial]]
            mean_activity = np.mean(activity, axis=1)
            trial_by_type[trial, :] = mean_activity

        cols = ['trial_type', 'reward_available', 'lick_flag', 'correct_trial', 'context', 'context_background']
        sub_trial_table = trial_table[cols]
        session_df = sub_trial_table.copy(deep=True)

        for area, area_index in rrs_cell_type_dict.items():
            session_df[f'{area}_baseline_dff'] = trial_by_type[:, area_index]

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
merge_trials_df = all_df.drop(labels=['trial_type', 'reward_available'], axis=1)
session_avg_data = all_df.groupby(["mouse_id", "session_id", "context", "context_background",
                                   "trial_type", "lick_flag", "correct_trial"],
                                  as_index=False).agg(np.nanmean)
session_trial_avg_data = merge_trials_df.groupby(["mouse_id", "session_id", "context", "context_background",
                                                  'correct_trial'],
                                                 as_index=False).agg(np.nanmean)
mouse_avg_data = session_trial_avg_data.drop('session_id', axis=1)
mouse_avg_data = mouse_avg_data.groupby(["mouse_id", "context", "context_background"],
                                        as_index=False).agg(np.nanmean)

# g = sns.catplot()




