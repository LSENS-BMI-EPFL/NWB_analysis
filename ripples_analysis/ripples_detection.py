import os
import warnings
import yaml
warnings.filterwarnings('ignore', category=UserWarning)

from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nwb_wrappers import nwb_reader_functions as nwb_read
from utils.lfp_utils import (lfp_filter, plot_lfp_custom, ripple_detect,
                             cluster_ripple_content, build_ripple_population_vectors, get_lfp_recordings)
from nwb_utils.utils_misc import find_nearest


# MAIN #
task = 'fast-learning'  # 'context' or 'fast-learning'

# DATA FOLDER
data_folder = Path('//sv-nas1.rcp.epfl.ch/Petersen-Lab/data')
if task == 'fast-learning':
    save_path = Path(r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Robin_Dard\ripple_results\fastlearning_task\v8")
else:
    save_path = Path(r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Robin_Dard\ripple_results\context_task\v4")

# Database to filter
if task == 'fast-learning':
    db_file_path = Path(r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\z_LSENS\Share\Axel_Bisi_Share\dataset_info")
    db_file = os.path.join(db_file_path, 'joint_probe_insertion_info.xlsx')
else:
    db_file_path = Path(r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Jules_Lebert\mice_info")
    db_file = os.path.join(db_file_path, 'probe_insertion_info.xlsx')
db_df = pd.read_excel(db_file)

if task == 'fast-learning':
    db_df = db_df.loc[
        (db_df.valid == 1) &
        (db_df.reward_group != 'Context') &
        (db_df.nwb_ephys == 1)
    ]
else:
    db_df = db_df.loc[
        (db_df.valid == 1) &
        (db_df.nwb_ephys == 1)
    ]
if task == 'context':
    group_file = Path(r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Jules_Lebert\group.yaml")
    with open(group_file, 'r', encoding='utf8') as stream:
        group_dict = yaml.safe_load(stream)
    expert = [name.split('.')[0] for name in group_dict['ephys_context']]

# Mice :
mice_list = db_df.mouse_name.unique()
print(f'{len(mice_list)} mice in data base')

results_dict = {'mouse_id': [],
                'session_id': [],
                'reward_group': [],
                'n_ripples': [],
                'n_no_stim': [],
                'total_time': [],
                'fz (min-1)': [],
                'wh_perf': []}

for mouse in mice_list:
    print(' ')
    print(f'Mouse {mouse}')
    if task == 'fast-learning':
        if mouse not in ['AB147', 'AB150', 'AB156', 'AB157', 'AB158', 'AB159', 'AB164',
                         'MH009', 'MH031', 'MH036', 'MH039']:
            continue
    if mouse in ['MH008', 'MH028']:
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
            if beh_type == 'whisker':
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

            # Extract information
            sampling_rate = ripple_rec.get_sampling_frequency()

            # Get ripple sites
            ripple_sites = electrode_table.loc[(electrode_table.group_name == f"imec{ripple_stream}") &
                                               (electrode_table.location == ripple_target)]
            if ripple_sites.empty:
                print(f'No {ripple_target} LFP')
                continue
            ripples_ids_list = ripple_sites.index_on_probe.astype(int).to_list()
            if len(ripples_ids_list) >= 15:
                ripples_ids_list = ripples_ids_list[::len(ripples_ids_list)//15]
            ripples_chs = ripple_rec.get_channel_ids()[ripples_ids_list]

            # Get secondary sites
            second_sites = electrode_table.loc[(electrode_table.group_name == f'imec{secondary_steam}') &
                                               (electrode_table.location.str.startswith(secondary_target))]
            if second_sites.empty:
                print(f'No {secondary_target} LFP')
                continue
            second_sites_ids_list = second_sites.index_on_probe.astype(int).to_list()
            if len(second_sites_ids_list) >= 15:
                second_sites_ids_list = second_sites_ids_list[::len(second_sites_ids_list)//15]
            second_chs = second_rec.get_channel_ids()[second_sites_ids_list]

            # Get trial table
            trial_table = nwb_read.get_trial_table(nwb_file)
            if task == 'fast-learning':
                try:
                    wh_perf = trial_table.loc[(trial_table.context == 'active') &
                                              (trial_table.whisker_stim == 1)].lick_flag.mean()
                except:
                    wh_perf = trial_table.loc[(trial_table.whisker_stim == 1)].lick_flag.mean()
            else:
                wh_perf = 'NaN'
            no_stim_table = trial_table.loc[trial_table.trial_type == 'no_stim_trial']

            # Get units table
            units_df = nwb_read.get_units_table(nwb_file)

            # Get ripple units
            try:
                ripple_units = units_df.loc[(units_df.ccf_atlas_acronym == ripple_target) &
                                            (units_df.bc_label == 'good')]
            except:
                ripple_units = units_df.loc[(units_df.ccf_acronym == ripple_target) &
                                            (units_df.bc_label == 'good')]
            order_ripple_units = ripple_units.sort_values('peak_channel', ascending=True)
            ripple_spk_times = order_ripple_units.spike_times.values[:]
            n_ripple_units = len(ripple_spk_times)

            # Get secondary region units
            try:
                second_names = [i for i in units_df.ccf_atlas_acronym.unique() if secondary_target in i]
                second_units = units_df.loc[(units_df.ccf_atlas_acronym.isin(second_names)) &
                                            (units_df.bc_label == 'good')]
            except:
                second_names = [i for i in units_df.ccf_acronym.unique() if secondary_target in i]
                second_units = units_df.loc[(units_df.ccf_acronym.isin(second_names)) &
                                            (units_df.bc_label == 'good')]
            second_spk_times = second_units.spike_times.values[:]
            n_second_units = len(second_spk_times)
            print(f'{n_ripple_units} units in {ripple_target}, {n_second_units} units in {secondary_target}')

            # Get whisker DLC
            keys = ['behavior', 'BehavioralTimeSeries']
            wh_angle = nwb_read.get_dlc_data(nwb_file, keys, 'whisker_angle')
            if wh_angle is not None:
                wh_angle_ts = nwb_read.get_dlc_timestamps(nwb_file, keys)[0]
            else:
                wh_angle_ts = []

            # Get a segment of data for each catch trial
            catch_start_time = no_stim_table.start_time.values[:]
            print(f'{len(catch_start_time)} "no stim" trials ')
            context_list = no_stim_table.context.values[:]
            start = - 1
            stop = 6
            session_ripple = 0
            ca1_ripple_content = []
            secondary_ripple_content = []
            contexts = []
            for catch_id, catch_ts in enumerate(catch_start_time):
                start_frame = int((catch_ts + start) * sampling_rate)
                end_frame = int((catch_ts + stop) * sampling_rate)
                n_samples = end_frame - start_frame
                time_vec = np.linspace(catch_ts + start, catch_ts + stop, n_samples)

                # Extract traces
                ripple_traces = ripple_rec.get_traces(start_frame=start_frame, end_frame=end_frame,
                                                      channel_ids=ripples_chs)
                secondary_traces = second_rec.get_traces(start_frame=start_frame, end_frame=end_frame,
                                                         channel_ids=second_chs)

                # LFP band pass filtering
                ripple_traces_filt = lfp_filter(data=ripple_traces, fs=sampling_rate, freq_min=150, freq_max=200)
                sw_traces_filt = lfp_filter(data=ripple_traces, fs=sampling_rate, freq_min=2, freq_max=20)
                secondary_traces_filt = lfp_filter(data=secondary_traces, fs=sampling_rate, freq_min=10, freq_max=16)

                # Get ripples
                ripple_frames, z_scored_power, best_channel = ripple_detect(ca1_sw_lfp=sw_traces_filt,
                                                                            ca1_ripple_lfp=ripple_traces_filt,
                                                                            sampling_rate=sampling_rate, threshold=3,
                                                                            sharp_filter=True, sharp_delay=0.070)
                if len(ripple_frames) == 0:
                    continue

                # Get ripple times
                ripple_times = time_vec[ripple_frames]

                # Get the whisker angle trace and speed to remove ripples during movement
                if wh_angle is not None:
                    wh_trace = wh_angle[
                               find_nearest(wh_angle_ts, (catch_ts + start)): find_nearest(wh_angle_ts,
                                                                                           catch_ts + stop)]
                    wh_ts = wh_angle_ts[
                            find_nearest(wh_angle_ts, (catch_ts + start)): find_nearest(wh_angle_ts, catch_ts + stop)]
                    wh_speed = np.abs(np.diff(wh_trace))
                    wh_speed_ripple = wh_speed[[find_nearest(wh_ts, i) for i in ripple_times]]
                    quiet = [speed < 2 for speed in wh_speed_ripple]
                    # Filter for no whisker movement
                    ripple_frames = ripple_frames[quiet]
                    if len(ripple_frames) == 0:
                        continue
                    ripple_times = ripple_times[quiet]
                else:
                    wh_trace = []
                    wh_ts = []

                # Count ripple events
                n_ripples = len(ripple_frames)
                session_ripple += n_ripples

                # For plotting
                peaks_1d = np.zeros(ripple_traces.shape[0], dtype=bool)
                peaks_1d[ripple_frames] = True

                # Filter all spike trains
                ca1_filtered_spikes = [
                    spikes[(spikes >= catch_ts + start) & (spikes <= catch_ts + stop)]
                    for spikes in ripple_spk_times
                ]
                sspbfd_filtered_spikes = [
                    spikes[(spikes >= catch_ts + start) & (spikes <= catch_ts + stop)]
                    for spikes in second_spk_times
                ]

                # Main plot for each catch trial
                offset = 50
                print(f'Plot catch : {catch_id}')
                plot_lfp_custom(ca1lfp=ripple_traces, ca_high_filt=ripple_traces_filt, ca1_ripple_power=z_scored_power,
                                sspbfdlfp=secondary_traces, sspbfd_spindle_filt=secondary_traces_filt,
                                time_vec=time_vec, ripple_times=ripple_times, best_channel=best_channel,
                                wh_trace=wh_trace, wh_ts=wh_ts,
                                ca1_spikes=ca1_filtered_spikes, sspbfd_spikes=sspbfd_filtered_spikes,
                                offset=offset, session_id=session_id, catch_id=catch_id, catch_ts=catch_ts,
                                ripple_id=None, fig_size=(16, 20), save_path=save_path)

                # One plot for each detected event
                for ripple_id, ripple_time in enumerate(ripple_times):
                    t_size_s = 0.200
                    if (ripple_time - t_size_s) < time_vec[0] or ripple_time + t_size_s >= time_vec[-1]:
                        continue
                    zoom_time_vec = np.linspace(ripple_time - t_size_s,
                                                ripple_time + t_size_s,
                                                int(sampling_rate * t_size_s * 2))
                    # Filter all spike trains
                    ca1_filtered_spikes = [
                        spikes[(spikes >= ripple_time - t_size_s) & (spikes <= ripple_time + t_size_s)]
                        for spikes in ripple_spk_times
                    ]
                    sspbfd_filtered_spikes = [
                        spikes[(spikes >= ripple_time - t_size_s) & (spikes <= ripple_time + t_size_s)]
                        for spikes in second_spk_times
                    ]

                    # Get the whisker angle trace
                    if wh_angle is not None:
                        wh_trace_zoom = wh_angle[
                                        find_nearest(wh_angle_ts, (ripple_time - t_size_s)):
                                        find_nearest(wh_angle_ts, ripple_time + t_size_s)]
                        wh_ts_zoom = wh_angle_ts[
                                     find_nearest(wh_angle_ts, (ripple_time - t_size_s)):
                                     find_nearest(wh_angle_ts, ripple_time + t_size_s)]
                    else:
                        wh_trace_zoom = []
                        wh_ts_zoom = []

                    ripple_frame = ripple_frames[ripple_id]
                    frame_range = int(sampling_rate * t_size_s)
                    zoom_start = ripple_frame - frame_range
                    zoom_stop = ripple_frame + frame_range

                    # Plot zoomed view on ripple event
                    plot_lfp_custom(ca1lfp=ripple_traces[zoom_start: zoom_stop, :],
                                    ca_high_filt=ripple_traces_filt[zoom_start: zoom_stop, :],
                                    ca1_ripple_power=z_scored_power[zoom_start: zoom_stop, :],
                                    sspbfdlfp=secondary_traces[zoom_start: zoom_stop, :],
                                    sspbfd_spindle_filt=secondary_traces_filt[zoom_start: zoom_stop, :],
                                    time_vec=zoom_time_vec, ripple_times=ripple_time, best_channel=best_channel,
                                    wh_trace=wh_trace_zoom, wh_ts=wh_ts_zoom,
                                    ca1_spikes=ca1_filtered_spikes, sspbfd_spikes=sspbfd_filtered_spikes,
                                    offset=offset, session_id=session_id, catch_id=catch_id, catch_ts=catch_ts,
                                    ripple_id=ripple_id, fig_size=(5, 20),
                                    save_path=os.path.join(save_path, 'single_event'))

                    # Extract CA1 ripple content
                    # Filter all spike trains in a 100ms around ripple
                    ca1_population_vector = build_ripple_population_vectors(all_spikes=ripple_spk_times,
                                                                            ripple_time=ripple_time,
                                                                            delay=0.050)
                    ca1_ripple_content.append(ca1_population_vector)

                    sspbfd_population_vector = build_ripple_population_vectors(all_spikes=second_spk_times,
                                                                               ripple_time=ripple_time,
                                                                               delay=0.050)
                    secondary_ripple_content.append(sspbfd_population_vector)

                    # Get the context block
                    contexts.append(context_list[catch_id])

            # Concatenate basic stats
            results_dict['mouse_id'].append(mouse)
            results_dict['session_id'].append(session_id)
            results_dict['reward_group'].append(rew_group)
            results_dict['n_ripples'].append(session_ripple)
            results_dict['n_no_stim'].append(len(catch_start_time))
            results_dict['total_time'].append((stop - start) * len(catch_start_time))
            results_dict['fz (min-1)'].append(np.round(session_ripple / ((stop - start) * len(catch_start_time)), 3) * 60)
            results_dict['wh_perf'].append(wh_perf)

            print(f'Total : {session_ripple} events, '
                  f'{np.round(session_ripple / ((stop - start) * len(catch_start_time)), 3) * 60} event / min')

            # Session CA1 and SSp-bfd ripple content projection
            if len(ca1_ripple_content) > 0 and len(secondary_ripple_content) > 0:
                ca1_ripple_content_2d = np.array(ca1_ripple_content)
                sspbfd_ripple_content_2d = np.array(secondary_ripple_content)
                cluster_ripple_content(ca1_ripple_array=ca1_ripple_content_2d,
                                       ssp_ripple_array=sspbfd_ripple_content_2d,
                                       session=session_id,
                                       group=rew_group,
                                       context_blocks=contexts,
                                       save_path=os.path.join(save_path, session_id))

# Figure for stat on event frequency
results_df = pd.DataFrame(results_dict)
if len(results_df) > 1:
    results_df.to_csv(os.path.join(save_path, "results.csv"))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
    sns.stripplot(results_df, hue='reward_group', hue_order=['R-', 'R+'], y='fz (min-1)',
                  palette=['darkmagenta', 'green'], dodge=True, legend=False, ax=ax0)
    sns.boxplot(results_df, hue='reward_group', hue_order=['R-', 'R+'], y='fz (min-1)',
                palette=['darkmagenta', 'green'], showfliers=False, ax=ax0)
    sns.lineplot(results_df, x='fz (min-1)', y='wh_perf', hue='reward_group',
                 hue_order=['R-', 'R+'], palette=['darkmagenta', 'green'], ax=ax1)
    sns.despine()
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, "results_figure.png"))
