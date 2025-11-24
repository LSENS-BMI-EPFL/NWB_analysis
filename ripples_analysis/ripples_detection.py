import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from pathlib import Path
import pandas as pd
import spikeinterface.full as si
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nwb_wrappers import nwb_reader_functions as nwb_read
from utils.lfp_utils import lfp_filter, plot_lfp_custom, ripple_detect, \
    cluster_ripple_content, build_ripple_population_vectors
from nwb_utils.utils_misc import find_nearest


# MAIN #
# DATA FOLDER
# data_folder = Path('//sv-nas1.rcp.epfl.ch/Petersen-Lab/data')
# save_path = Path(r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Robin_Dard\ripple_results\v3")
data_folder = Path(r"C:\Users\rdard\Documents\test_data\replay_context_task\data")
save_path = Path(r"C:\Users\rdard\Documents\test_data\replay_context_task\results")

# Database to filter
# db_file_path = Path(r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\z_LSENS\Share\Axel_Bisi_Share\dataset_info")
# db_file = os.path.join(db_file_path, 'joint_probe_insertion_info.xlsx')
# db_df = pd.read_excel(db_file)
# db_df = db_df.loc[
#     (db_df.target_area.isin(['wS1', 'wS2'])) &
#     (db_df.valid == 1) &
#     (db_df.reward_group != 'Context') &
#     (db_df.nwb_ephys == 1)
# ]
#
# db_df = db_df.sort_values('target_area').drop_duplicates(subset='mouse_name', keep='first')

# Mice :
# mice_list = db_df.mouse_name.unique()
mice_list = ['JL002']
print(f'{len(mice_list)} mice in data base')

results_dict = {'mouse_id': [],
                'session_id': [],
                'reward_group': [],
                'n_ripples': [],
                'n_no_stim': [],
                'total_time': [],
                'fz (min-1)': []}

for mouse in mice_list:
    print(f'Mouse {mouse}')
    experimenter = mouse[0:2]
    # if experimenter == 'AB':
    #     nwb_folder = os.path.join(r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Axel_Bisi/NWBFull_bis")
    # else:
    #     nwb_folder = os.path.join(r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Myriam_Hamon/NWB")
    # nwb_names = os.listdir(nwb_folder)
    # nwb_names = [name for name in nwb_names if mouse in name]
    # nwb_files = [os.path.join(nwb_folder, name) for name in nwb_names]
    nwb_files = [os.path.join(data_folder, 'JL002_20250507_135553.nwb')]
    for nwb_file in nwb_files:
        beh_type, day = nwb_read.get_bhv_type_and_training_day_index(nwb_file)
        if (beh_type == 'whisker') and (day != 0):
            continue
        else:
            session_id = nwb_read.get_session_id(nwb_file)
            print(f'Session : {session_id}, behavior : {beh_type}, day : {day}')

            ripple_target = 'CA1'
            if beh_type == 'whisker':
                secondary_target = 'SSp-bfd'
            else:
                secondary_target = 'RSP'

            # stream = db_df.loc[db_df.mouse_name == mouse]['probe_id'].values[0]
            stream = 0
            # rew_group = db_df.loc[db_df.mouse_name == mouse]['reward_group'].values[0]
            rew_group = 'R+'

            # Try loading LFP stream directly if available
            # ephys_path = os.path.join(data_folder, mouse, 'Recording', session_id, 'Ephys')
            # if not os.path.exists(ephys_path):
            #     continue
            # g_index = os.listdir(os.path.join(data_folder, mouse, 'Recording', session_id, 'Ephys'))[0].split('_')[1][1]
            # full_path = os.path.join(ephys_path, f'{mouse}_g{g_index}')
            # if not os.path.exists(full_path):
            #     full_path = os.path.join(ephys_path, f'{mouse}_g0')
            #     if not os.path.exists(full_path):
            #         full_path = os.path.join(ephys_path, f'{mouse}_g1')
            #         if not os.path.exists(full_path):
            #             continue
            # try:
            #     rec = si.read_spikeglx(full_path, stream_name=f"imec{stream}.lf")
            #     print("Using LF stream")
            # except:
            #     print("LF stream not found, using AP stream")
            #     rec = si.read_spikeglx(full_path, stream_name=f"imec{stream}.ap")
            #     rec = si.bandpass_filter(rec, freq_min=0.5, freq_max=500, margin_ms=5000)
            #     rec = si.resample(rec, resample_rate=2500, margin_ms=2000)

            # g_index = os.listdir(os.path.join(data_folder, mouse, 'Recording', session_id, 'Ephys'))[0].split('_')[1][1]
            # try:
            #     rec = si.read_spikeglx(os.path.join(data_folder, mouse, 'Recording', session_id, 'Ephys', f'{mouse}_g{g_index}'),
            #                            stream_name=f"imec{stream}.lf")
            #     print("Using LF stream")
            # except:
            #     print("LF stream not found, using AP stream")
            #     rec = si.read_spikeglx(os.path.join(data_folder, mouse, 'Recording', session_id, 'Ephys', f'{mouse}_g{g_index}'),
            #                            stream_name=f"imec{stream}.ap")
            #     rec = si.bandpass_filter(rec, freq_min=0.5, freq_max=500, margin_ms=5000)
            #     rec = si.resample(rec, resample_rate=2500, margin_ms=2000)
            # rec = si.read_spikeglx(os.path.join(data_folder, f'{mouse}_g0'), stream_name=f"imec{stream}.lf")
            rec = si.read_spikeglx(os.path.join(data_folder, f'{mouse}_20250507_g0'), stream_name=f"imec{stream}.lf")
            # Extract information
            sampling_rate = rec.get_sampling_frequency()
            num_channels = rec.get_num_channels()

            # Electrode locations
            electrode_table = nwb_read.get_electrode_table(nwb_file)
            ca1_sites = electrode_table.loc[(electrode_table.group_name == f"imec{stream}") &
                                            (electrode_table.location == ripple_target)]
            if ca1_sites.empty:
                print('No CA1 LFP')
                continue
            ca1_ids_list = ca1_sites.index_on_probe.astype(int).to_list()
            if len(ca1_ids_list) >= 15:
                ca1_ids_list = ca1_ids_list[::len(ca1_ids_list)//15]
            ca1_ids = rec.get_channel_ids()[ca1_ids_list]

            ssp_bfd_sites = electrode_table.loc[(electrode_table.group_name == f'imec{stream}') &
                                                (electrode_table.location.str.startswith(secondary_target))]
            if ssp_bfd_sites.empty:
                print('No SSp BFD LFP')
                continue
            ssp_bfd_ids_list = ssp_bfd_sites.index_on_probe.astype(int).to_list()
            if len(ssp_bfd_ids_list) >= 15:
                ssp_bfd_ids_list = ssp_bfd_ids_list[::len(ssp_bfd_ids_list)//15]
            ssp_bfd_ids = rec.get_channel_ids()[ssp_bfd_ids_list]

            # Trial table
            trial_table = nwb_read.get_trial_table(nwb_file)
            no_stim_table = trial_table.loc[trial_table.trial_type == 'no_stim_trial']

            # Units table
            units_df = nwb_read.get_units_table(nwb_file)
            try:
                ca1_units = units_df.loc[units_df.ccf_atlas_acronym == ripple_target]
            except:
                ca1_units = units_df.loc[units_df.ccf_acronym == ripple_target]
            order_ca1_units = ca1_units.sort_values('peak_channel', ascending=True)
            ca1_spk_times = order_ca1_units.spike_times.values[:]
            n_ca1_units = len(ca1_spk_times)

            try:
                ssp_bfd_names = [i for i in units_df.ccf_atlas_acronym.unique() if secondary_target in i]
                ssp_bfd_units = units_df.loc[units_df.ccf_atlas_acronym.isin(ssp_bfd_names)]
            except:
                ssp_bfd_names = [i for i in units_df.ccf_acronym.unique() if secondary_target in i]
                ssp_bfd_units = units_df.loc[units_df.ccf_acronym.isin(ssp_bfd_names)]
            ssp_bfd_spk_times = ssp_bfd_units.spike_times.values[:]
            n_ssp_bfd_units = len(ssp_bfd_spk_times)

            # Get whisker DLC
            keys = ['behavior', 'BehavioralTimeSeries']
            wh_angle = nwb_read.get_dlc_data(nwb_file, keys, 'whisker_angle')
            if wh_angle is not None:
                wh_angle_ts = nwb_read.get_dlc_timestamps(nwb_file, keys)[0]

            # Get a segment of data
            catch_start_time = no_stim_table.start_time.values[:]
            print(f'{len(catch_start_time)} "no stim" trials ')
            context_list = no_stim_table.context.values[:]
            start = - 1
            stop = 6
            session_ripple = 0
            ca1_ripple_content = []
            sspbfd_ripple_content = []
            contexts = []
            for catch_id, catch_ts in enumerate(catch_start_time):
                start_frame = int((catch_ts + start) * sampling_rate)
                end_frame = int((catch_ts + stop) * sampling_rate)
                n_samples = end_frame - start_frame
                time_vec = np.linspace(catch_ts + start, catch_ts + stop, n_samples)

                # Extract traces
                ca1_traces = rec.get_traces(start_frame=start_frame, end_frame=end_frame, channel_ids=ca1_ids)
                sspbfd_traces = rec.get_traces(start_frame=start_frame, end_frame=end_frame, channel_ids=ssp_bfd_ids)

                # LFP band pass filtering
                ca1_ripple_traces = lfp_filter(data=ca1_traces, fs=sampling_rate, freq_min=150, freq_max=200)
                ca1_sw_traces = lfp_filter(data=ca1_traces, fs=sampling_rate, freq_min=2, freq_max=20)
                sspbfd_filt_traces = lfp_filter(data=sspbfd_traces, fs=sampling_rate, freq_min=10, freq_max=16)

                # Get ripples
                ripple_frames, z_scored_power, best_channel = ripple_detect(ca1_sw_lfp=ca1_sw_traces,
                                                                            ca1_ripple_lfp=ca1_ripple_traces,
                                                                            sampling_rate=sampling_rate, threshold=3,
                                                                            sharp_filter=True, sharp_delay=0.070)
                if len(ripple_frames) == 0:
                    continue
                ripple_times = time_vec[ripple_frames]

                # Get the whisker angle trace
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
                peaks_1d = np.zeros(ca1_traces.shape[0], dtype=bool)
                peaks_1d[ripple_frames] = True

                # Filter all spike trains
                ca1_filtered_spikes = [
                    spikes[(spikes >= catch_ts + start) & (spikes <= catch_ts + stop)]
                    for spikes in ca1_spk_times
                ]
                sspbfd_filtered_spikes = [
                    spikes[(spikes >= catch_ts + start) & (spikes <= catch_ts + stop)]
                    for spikes in ssp_bfd_spk_times
                ]

                # Main plot for each catch trial
                offset = 50
                print(f'Plot catch : {catch_id}')
                plot_lfp_custom(ca1lfp=ca1_traces, ca_high_filt=ca1_ripple_traces, ca1_ripple_power=z_scored_power,
                                sspbfdlfp=sspbfd_traces, sspbfd_spindle_filt=sspbfd_filt_traces,
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
                        for spikes in ca1_spk_times
                    ]
                    sspbfd_filtered_spikes = [
                        spikes[(spikes >= ripple_time - t_size_s) & (spikes <= ripple_time + t_size_s)]
                        for spikes in ssp_bfd_spk_times
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
                    plot_lfp_custom(ca1lfp=ca1_traces[zoom_start: zoom_stop, :],
                                    ca_high_filt=ca1_ripple_traces[zoom_start: zoom_stop, :],
                                    ca1_ripple_power=z_scored_power[zoom_start: zoom_stop, :],
                                    sspbfdlfp=sspbfd_traces[zoom_start: zoom_stop, :],
                                    sspbfd_spindle_filt=sspbfd_filt_traces[zoom_start: zoom_stop, :],
                                    time_vec=zoom_time_vec, ripple_times=ripple_time, best_channel=best_channel,
                                    wh_trace=wh_trace_zoom, wh_ts=wh_ts_zoom,
                                    ca1_spikes=ca1_filtered_spikes, sspbfd_spikes=sspbfd_filtered_spikes,
                                    offset=offset, session_id=session_id, catch_id=catch_id, catch_ts=catch_ts,
                                    ripple_id=ripple_id, fig_size=(5, 20),
                                    save_path=os.path.join(save_path, 'single_event'))

                    # Extract CA1 ripple content
                    # Filter all spike trains in a 100ms around ripple
                    ca1_population_vector = build_ripple_population_vectors(all_spikes=ca1_spk_times,
                                                                            ripple_time=ripple_time,
                                                                            delay=0.050)
                    ca1_ripple_content.append(ca1_population_vector)

                    sspbfd_population_vector = build_ripple_population_vectors(all_spikes=ssp_bfd_spk_times,
                                                                               ripple_time=ripple_time,
                                                                               delay=0.050)
                    sspbfd_ripple_content.append(sspbfd_population_vector)

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

            print(f'Total : {session_ripple} events, '
                  f'{np.round(session_ripple / ((stop - start) * len(catch_start_time)), 3) * 60} event / min')

            # Session CA1 and SSp-bfd ripple content projection
            ca1_ripple_content_2d = np.array(ca1_ripple_content)
            sspbfd_ripple_content_2d = np.array(sspbfd_ripple_content)
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
    fig, ax = plt.subplots(1, 1)
    sns.boxplot(results_df, x='reward_group', y='fz (min-1)', ax=ax)
    fig.savefig(os.path.join(save_path, "results_figure.png"))
