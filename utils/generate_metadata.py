import os
import time
import numpy as np
import pandas as pd
import nwb_wrappers.nwb_reader_functions as nwb_read
import warnings
warnings.filterwarnings('ignore')

experimenter = 'Pol_Bech'
root_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'NWB')
all_nwb_names = os.listdir(root_path)
nwb_files = [os.path.join(root_path, name) for name in all_nwb_names if 'nwb' in name]

subjects_info = [nwb_read.get_subject_info(nwb) for nwb in nwb_files]
session_id = [nwb_read.get_session_id(nwb) for nwb in nwb_files]
session_date = [session[6:-7] for session in session_id]
session_type = [nwb_read.get_session_type(nwb) for nwb in nwb_files]
session_descriptions = [nwb_read.get_session_metadata(nwb) for nwb in nwb_files]
bhv_day = [nwb_read.get_bhv_type_and_training_day_index(nwb) for nwb in nwb_files]
behavior = [pair[0] for pair in bhv_day]
day = [pair[1] for pair in bhv_day]
video_tstamps = []
top_frames = []
side_frames = []
wf_timestamps = []
wf_frames = []
wf_tbd = []
dlc_tbd = []
for i, nwb in enumerate(nwb_files):
    if nwb_read.get_dlc_timestamps(nwb, ['behavior', 'BehavioralTimeSeries']) is not None:
        video_tstamps += [len(nwb_read.get_dlc_timestamps(nwb, ['behavior', 'BehavioralTimeSeries'])[0])]
        if len(nwb_read.get_dlc_data(nwb, ['behavior', 'BehavioralTimeSeries'], 'whisker_tip_x')) is not None:
            top_frames += [len(nwb_read.get_dlc_data(nwb, ['behavior', 'BehavioralTimeSeries'], 'whisker_tip_x'))]
            side_frames += [len(nwb_read.get_dlc_data(nwb, ['behavior', 'BehavioralTimeSeries'], 'jaw_x'))]
        else:
            top_frames += [0]
            side_frames += [0]
            dlc_tbd += [i]
    else:
        video_tstamps += [0]
        top_frames += [0]
        side_frames += [0]

    if nwb_read.get_widefield_timestamps(nwb, ['ophys', 'dff0']) is not None:
        wf_timestamps += [len(nwb_read.get_widefield_timestamps(nwb, ['ophys', 'dff0']))]
        if nwb_read.get_widefield_dff0_traces(nwb, ['ophys', 'dff0']) is not None:
            wf_frames += [len(nwb_read.get_widefield_dff0_traces(nwb, ['ophys', 'dff0']))]
        else:
            wf_frames += [0]
            wf_tbd += [i]
    else:
        wf_timestamps += [0]
        wf_frames += [0]

top_difference = (np.asarray(video_tstamps) - np.asarray(top_frames)).tolist()
side_difference = (np.asarray(video_tstamps) - np.asarray(side_frames)).tolist()
wf_difference = (np.asarray(wf_timestamps)) - np.asarray(wf_frames).tolist()

# wf_frames = [wf_frames[i] if i not in wf_tbd else 'na' for i in range(len(wf_frames))]
# side_frames[dlc_tbd] = 'na'
# top_frames[dlc_tbd] = 'na'

metadata = pd.DataFrame()
metadata['MouseID'] = [subject_info.subject_id for subject_info in subjects_info]
metadata['MouseSex'] = [subject_info.sex for subject_info in subjects_info]
metadata['MouseAgeDay'] = [subject_info.age[1:-1] for subject_info in subjects_info]
metadata['SessionID'] = session_id
metadata['ExpDate'] = [f'{date[0:4]}_{date[4:6]}_{date[6:8]}' for date in session_date]
metadata['SessionType'] = session_type
metadata['Weight'] = [subject_info.weight for subject_info in subjects_info]
metadata['BehaviorDay'] = day
metadata['BehaviorType'] = behavior
metadata['NWhisker'] = [session_descr.get('wh_stim_weight') for session_descr in session_descriptions]
metadata['NCatch'] = [session_descr.get('no_stim_weight') for session_descr in session_descriptions]
metadata['NAuditory'] = [session_descr.get('aud_stim_weight') for session_descr in session_descriptions]
metadata['BlockSize'] = metadata['NWhisker'] + metadata['NCatch'] + metadata['NAuditory']

metadata['VideoTimestamps'] = video_tstamps
metadata['SideFrames'] = side_frames
metadata['TopFrames'] = top_frames
metadata['SideDiff'] = side_difference
metadata['TopDiff'] = top_difference

metadata['WF_Timestamps'] = wf_timestamps
metadata['WF_Frames'] = wf_frames
metadata['WF_Diff'] = wf_difference

sorted_metadata = metadata.sort_values(by=['MouseID', 'SessionID'], ascending=True)
sorted_metadata = sorted_metadata.reset_index(drop=True)
sorted_metadata.loc[sorted_metadata.BehaviorType == 'context', 'BehaviorType'] = 'whisker_context'
sorted_metadata.to_excel(r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Pol_Bech\context_mice_metadata.xlsx')


metadata = pd.read_excel(r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Pol_Bech\context_mice_metadata.xlsx')
bilateral_sound_date = time.strptime('231102', '%y%m%d')
metadata['Date'] = [time.strptime(metadata.ExpDate[i], '%Y_%m_%d') for i in range(len(metadata))]
metadata.loc[metadata.Date >= bilateral_sound_date, 'AuditoryStim'] = 'bilateral'
metadata.loc[metadata.BehaviorType == 'free_licking', 'AuditoryStim'] = 'na'
metadata.loc[(metadata.Date < time.strptime('231115', '%y%m%d')) & (metadata.AuditoryStim == 'bilateral'), 'AuditoryStimAmp'] = 78
metadata.loc[metadata.BehaviorType == 'free_licking', 'AuditoryStimfreq'] = 'na'
metadata.loc[metadata.Date >= time.strptime('231115', '%y%m%d'), 'WhiskerStimAmp'] = 25
metadata.loc[metadata.BehaviorType.isin(['free_licking', 'auditory']), 'WhiskerStimAmp'] = 'na'
metadata.to_excel(r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Pol_Bech\context_mice_metadata.xlsx')