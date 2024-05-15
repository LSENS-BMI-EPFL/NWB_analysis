import os
import pandas as pd
import time
import nwb_wrappers.nwb_reader_functions as nwb_read

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