from nwb_wrappers import nwb_reader_functions as nwb_read
from utils.lfp_utils import *
from nwb_utils.utils_misc import find_nearest

import yaml
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# MAIN #
# PARAMETERS
task = 'fast-learning'  # 'context' or 'fast-learning'
only_good_units = True  # if False also take MUA

# DATA FOLDER
data_folder = Path('//sv-nas1.rcp.epfl.ch/Petersen-Lab/data')
if task == 'fast-learning':
    save_path = Path(r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Robin_Dard\ripple_results\fastlearning_task\ripple_tables")
else:
    save_path = Path(r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Robin_Dard\ripple_results\context_task\ripple_tables")

# Database to filter
db_df = get_database(task)

if task == 'context':
    group_file = Path(r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Jules_Lebert\group.yaml")
    with open(group_file, 'r', encoding='utf8') as stream:
        group_dict = yaml.safe_load(stream)
    expert = [name.split('.')[0] for name in group_dict['ephys_context']]

if task == 'fast-learning':
    mice_to_do = ['AB147', 'AB150', 'AB156', 'AB157', 'AB158', 'AB159', 'AB164', 'MH009', 'MH031', 'MH036', 'MH039']
    # mice_to_do = ['MH039']
    mice_excluded = ['MH008', 'MH028']

# Mice :
mice_list = db_df.mouse_name.unique()
print(f'{len(mice_list)} mice in data base')

table_list = []
for mouse in mice_list:
    print(' ')
    print(f'Mouse {mouse}')
    if task == 'fast-learning' and mouse not in mice_to_do:
        continue

    experimenter = mouse[0:2]
    if task == 'fast-learning':
        if experimenter == 'AB':
            nwb_folder = os.path.join(r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Axel_Bisi/NWBFull_bis")
        else:
            nwb_folder = os.path.join(r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Myriam_Hamon/NWB")
    else:
        nwb_folder = os.path.join(r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Jules_Lebert/NWB")

    nwb_names = os.listdir(nwb_folder)
    nwb_names = [name for name in nwb_names if mouse in name]
    nwb_files = [os.path.join(nwb_folder, name) for name in nwb_names]

    for nwb_file in nwb_files:
        beh_type, day = nwb_read.get_bhv_type_and_training_day_index(nwb_file)
        if beh_type not in ['whisker', 'context', 'whisker_context']:
            continue
        if (beh_type == 'whisker') and (day != 0):
            continue
        else:
            session_id = nwb_read.get_session_id(nwb_file)
            if task == 'context' and session_id not in expert:
                continue
            print(' ')
            print(f'Session : {session_id}, behavior : {beh_type}, day : {day}')

            # Target
            ripple_target = 'CA1'
            if task == 'fast-learning':
                secondary_target = 'SSp-bfd'
            else:
                secondary_target = 'RSP'
            print(f'Look for {ripple_target} for ripple, {secondary_target} for simultaneous activity')

            # Electrode locations to see if we have both CA1 and secondary target
            electrode_table = nwb_read.get_electrode_table(nwb_file)
            if electrode_table is None:
                print('No electrode table found')
                continue
            is_ca1 = ripple_target in electrode_table.location.unique()
            is_secondary_target_names = len([i for i in electrode_table.location.unique() if secondary_target in i]) > 0
            is_valid_session = is_ca1 and is_secondary_target_names
            if not is_valid_session:
                print(f'Session does not have both {ripple_target} & {secondary_target}')
                continue

            # Get the corresponding probes
            print(f'Location : {electrode_table.location.unique()}')
            ripple_stream = int(electrode_table.loc[electrode_table.location == ripple_target].group_name.values[0][-1])
            secondary_steam = int(electrode_table.loc[electrode_table.location.str.startswith(secondary_target)].group_name.values[0][-1])
            print(f'{ripple_target} on probe {ripple_stream}, {secondary_target} on probe {secondary_steam}')

            # Get reward group
            if task == 'fast-learning':
                rew_group = db_df.loc[db_df.mouse_name == mouse]['reward_group'].values[0]
                print(f'Rewarded group {rew_group}')
            else:
                rew_group = 'NaN'

            # Try loading LFP stream directly if available
            ripple_rec = get_lfp_recordings(data_folder=data_folder, mouse=mouse,
                                            session=session_id, stream=ripple_stream)
            second_rec = get_lfp_recordings(data_folder=data_folder, mouse=mouse,
                                            session=session_id, stream=secondary_steam)
            if ripple_rec is None or second_rec is None:
                print('Raw recordings not found')
                continue

            # Extract sampling rate
            sampling_rate = ripple_rec.get_sampling_frequency()

            # Get ripple sites
            ripples_chs = get_lfp_channels(electrode_table, stream=ripple_stream, rec=ripple_rec,
                                           target=ripple_target, target_type='ripple')
            # Get secondary sites
            second_chs = get_lfp_channels(electrode_table, stream=secondary_steam, rec=second_rec,
                                          target=secondary_target, target_type='secondary')

            # Get trial table
            trial_table = nwb_read.get_trial_table(nwb_file)
            kept_columns = ['trial_id', 'start_time', 'trial_type', 'lick_flag', 'lick_time', 'context']
            bhv_table = trial_table[kept_columns].copy()

            # Get units table
            units_df = nwb_read.get_units_table(nwb_file)

            # Get ripple units
            order_ripple_units = get_units_selection(units_df, target=ripple_target, only_good=only_good_units)
            ripple_spk_times = order_ripple_units.spike_times.values[:]
            n_ripple_units = len(ripple_spk_times)

            # Get secondary region units
            second_units = get_units_selection(units_df, target=secondary_target, only_good=only_good_units)
            second_spk_times = second_units.spike_times.values[:]
            n_second_units = len(second_spk_times)

            print(f'{n_ripple_units} {'units' if only_good_units is True else 'MUA'} in {ripple_target}, '
                  f'{n_second_units} {'units' if only_good_units is True else 'MUA'} in {secondary_target}')

            # Get whisker angle and tongue distance from DLC
            keys = ['behavior', 'BehavioralTimeSeries']
            wh_angle = nwb_read.get_dlc_data(nwb_file, keys, 'whisker_angle')
            tongue_distance = nwb_read.get_dlc_data(nwb_file, keys, 'tongue_distance')
            if wh_angle is not None:
                dlc_ts = nwb_read.get_dlc_timestamps(nwb_file, keys)[0]
            else:
                dlc_ts = []

            # Initialize list for results
            ca1_lfp = []
            ca1_ripple_band_flp = []
            ca1_ripple_power = []
            ca1_ripple_best_ch = []
            ca1_sw_band_lfp = []
            secondary_lfp = []
            secondary_spindle_band_lfp = []
            lfp_ts = []
            ripple_times = []
            is_whisking = []
            ripples_per_trial = []
            ca1_spike_times = []
            secondary_spike_times = []
            whisker_trace = []
            whisker_speed = []
            tongue_trace = []
            dlc_timestamps = []
            trial_duration = []

            # Loop on each trial and extract data
            for trial_id, start_ts in enumerate(bhv_table.start_time.values[:]):
                # Trial info
                print(f'Trial: {trial_id}, type: {bhv_table.loc[trial_id].trial_type}, '
                      f'lick: {bhv_table.loc[trial_id].lick_flag}')

                # Range for data extraction
                if bhv_table.loc[trial_id].trial_type in (['whisker_trial', 'auditory_trial']):
                    start = 0.1
                    start_spike = 0.005
                    stop = 7
                    detection_start = 0.01
                else:
                    start = -1
                    start_spike = -1
                    stop = 7
                    detection_start = 0

                # Timing
                start_frame = int((start_ts + start) * sampling_rate)
                end_frame = int((start_ts + stop) * sampling_rate)
                n_samples = end_frame - start_frame
                time_vec = np.linspace(start_ts + start, start_ts + stop, n_samples)
                lfp_ts.append(time_vec)
                trial_duration.append(stop - start)

                # Extract LFP traces
                ripple_traces = ripple_rec.get_traces(start_frame=start_frame, end_frame=end_frame,
                                                      channel_ids=ripples_chs)
                ca1_lfp.append(ripple_traces)

                secondary_traces = second_rec.get_traces(start_frame=start_frame, end_frame=end_frame,
                                                         channel_ids=second_chs)
                secondary_lfp.append(secondary_traces)

                # LFP band pass filtering
                ripple_traces_filt = lfp_filter(data=ripple_traces, fs=sampling_rate, freq_min=150, freq_max=200)
                ca1_ripple_band_flp.append(ripple_traces_filt)

                sw_traces_filt = lfp_filter(data=ripple_traces, fs=sampling_rate, freq_min=2, freq_max=20)
                ca1_sw_band_lfp.append(sw_traces_filt)

                secondary_traces_filt = lfp_filter(data=secondary_traces, fs=sampling_rate, freq_min=10, freq_max=16)
                secondary_spindle_band_lfp.append(secondary_traces_filt)

                # Get ripples
                ripple_frames, z_scored_power, best_channel = ripple_detect(ca1_sw_lfp=sw_traces_filt,
                                                                            ca1_ripple_lfp=ripple_traces_filt,
                                                                            sampling_rate=sampling_rate, threshold=3,
                                                                            sharp_filter=True, sharp_delay=0.070,
                                                                            detection_delay=detection_start)
                ripple_ts = time_vec[ripple_frames]
                ripples_per_trial.append(len(ripple_frames))
                print(f'{len(ripple_frames)} detected ripples')
                ripple_times.append(ripple_ts)
                ca1_ripple_power.append(z_scored_power)
                ca1_ripple_best_ch.append(best_channel)

                # Get the whisker angle trace and speed
                if wh_angle is not None:
                    wh_trace = wh_angle[find_nearest(dlc_ts, (start_ts + start)): find_nearest(dlc_ts, start_ts + stop)]
                    dlc_trial_ts = dlc_ts[find_nearest(dlc_ts, (start_ts + start)): find_nearest(dlc_ts, start_ts + stop)]
                    wh_speed = np.abs(np.diff(wh_trace))
                    wh_speed = np.insert(wh_speed, 0, 0)
                    if len(ripple_ts) > 0:
                        points = []
                        for i in ripple_ts:
                            if find_nearest(dlc_trial_ts, i) == -1:
                                points.append(int(0))
                            elif find_nearest(dlc_trial_ts, i) == len(dlc_trial_ts):
                                points.append(int(len(dlc_trial_ts) - 1))
                            else:
                                points.append(find_nearest(dlc_trial_ts, i))
                        try:
                            wh_speed_ripple = wh_speed[np.array(points)]
                        except:
                            points = points[points < len(wh_speed)]
                            wh_speed_ripple = wh_speed[np.array(points)]
                        whisking = [speed >= 2 for speed in wh_speed_ripple]
                    else:
                        whisking = []
                else:
                    wh_trace = []
                    dlc_trial_ts = []
                    wh_speed = []
                    whisking = []
                whisker_trace.append(wh_trace)
                whisker_speed.append(wh_speed)
                is_whisking.append(whisking)
                dlc_timestamps.append(dlc_trial_ts)

                # Get tongue trace
                if tongue_distance is not None:
                    tongue_tr = tongue_distance[find_nearest(dlc_ts, (start_ts + start)): find_nearest(dlc_ts, start_ts + stop)]
                else:
                    tongue_tr = []
                tongue_trace.append(tongue_tr)

                # Filter all spike trains
                # CA1 spikes
                ca1_filtered_spikes = [
                    spikes[(spikes >= start_ts + start_spike) & (spikes <= start_ts + stop)]
                    for spikes in ripple_spk_times
                ]
                ca1_spike_times.append(ca1_filtered_spikes)

                # Secondary region spikes
                sspbfd_filtered_spikes = [
                    spikes[(spikes >= start_ts + start_spike) & (spikes <= start_ts + stop)]
                    for spikes in second_spk_times
                ]
                secondary_spike_times.append(sspbfd_filtered_spikes)

            # Add to the table
            bhv_table['trial_duration'] = trial_duration
            bhv_table['ripples_per_trial'] = ripples_per_trial
            bhv_table['ripple_times'] = ripple_times
            bhv_table['ca1_lfp'] = ca1_lfp
            bhv_table['ca1_ripple_band_flp'] = ca1_ripple_band_flp
            bhv_table['ca1_ripple_power'] = ca1_ripple_power
            bhv_table['ca1_ripple_best_ch'] = ca1_ripple_best_ch
            bhv_table['ca1_sw_band_lfp'] = ca1_sw_band_lfp
            bhv_table['secondary_lfp'] = secondary_lfp
            bhv_table['secondary_spindle_band_lfp'] = secondary_spindle_band_lfp
            bhv_table['lfp_ts'] = lfp_ts
            bhv_table['is_whisking'] = is_whisking
            bhv_table['ca1_spike_times'] = ca1_spike_times
            bhv_table['secondary_spike_times'] = secondary_spike_times
            bhv_table['whisker_trace'] = whisker_trace
            bhv_table['whisker_speed'] = whisker_speed
            bhv_table['tongue_trace'] = tongue_trace
            bhv_table['dlc_trial_ts'] = dlc_timestamps
            bhv_table['mouse'] = mouse
            bhv_table['session'] = session_id
            bhv_table['rewarded_group'] = rew_group

            # Save each session df
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            bhv_table.to_pickle(os.path.join(save_path, f'{session_id}_ripple_table.pkl'))

