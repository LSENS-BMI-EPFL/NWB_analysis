import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from pathlib import Path
import pandas as pd
import spikeinterface.full as si
import numpy as np
import scipy as sci
import seaborn as sns
import matplotlib.pyplot as plt
from nwb_wrappers import nwb_reader_functions as nwb_read
from utils.lfp_utils import lfp_filter
from nwb_utils.utils_misc import find_nearest


# MAIN #
# DATA FOLDER
data_folder = Path('//sv-nas1.rcp.epfl.ch/Petersen-Lab/data')
save_path = Path(r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Robin_Dard\ripple_results\v3")

# Database to filter
db_file_path = Path(r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\z_LSENS\Share\Axel_Bisi_Share\dataset_info")
db_file = os.path.join(db_file_path, 'joint_probe_insertion_info.xlsx')
db_df = pd.read_excel(db_file)
db_df = db_df.loc[
    (db_df.target_area.isin(['wS1', 'wS2'])) &
    (db_df.valid == 1) &
    (db_df.reward_group != 'Context') &
    (db_df.nwb_ephys == 1)
]

db_df = db_df.sort_values('target_area').drop_duplicates(subset='mouse_name', keep='first')

# Mice :
mice_list = db_df.mouse_name.unique()
print(f'{len(mice_list)} mice in data base')

results_dict = {'mouse_id': [],
                'session_id': [],
                'reward_group': [],
                'n_ripples': [],
                'n_no_stim': [],
                'total_time': [],
                'fz': []}

for mouse in mice_list:
    print(f'Mouse {mouse}')
    experimenter = mouse[0:2]
    if experimenter == 'AB':
        nwb_folder = os.path.join(r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Axel_Bisi/NWBFull_bis")
    else:
        nwb_folder = os.path.join(r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Myriam_Hamon/NWB")
    nwb_names = os.listdir(nwb_folder)
    nwb_names = [name for name in nwb_names if mouse in name]
    nwb_files = [os.path.join(nwb_folder, name) for name in nwb_names]

    for nwb_file in nwb_files:
        beh_type, day = nwb_read.get_bhv_type_and_training_day_index(nwb_file)
        if day != 0:
            continue
        else:
            session_id = nwb_read.get_session_id(nwb_file)
            print(f'Session : {session_id}, behavior : {beh_type}, day : {day}')
            stream = db_df.loc[db_df.mouse_name == mouse]['probe_id'].values[0]
            rew_group = db_df.loc[db_df.mouse_name == mouse]['reward_group'].values[0]

            # Try loading LFP stream directly if available
            g_index = os.listdir(os.path.join(data_folder, mouse, 'Recording', session_id, 'Ephys'))[0].split('_')[1][1]
            try:
                rec = si.read_spikeglx(os.path.join(data_folder, mouse, 'Recording', session_id, 'Ephys', f'{mouse}_g{g_index}'),
                                       stream_name=f"imec{stream}.lf")
                print("Using LF stream")
            except:
                print("LF stream not found, using AP stream")
                rec = si.read_spikeglx(os.path.join(data_folder, mouse, 'Recording', session_id, 'Ephys', f'{mouse}_g{g_index}'),
                                       stream_name=f"imec{stream}.ap")
                rec = si.bandpass_filter(rec, freq_min=0.5, freq_max=500, margin_ms=5000)
                rec = si.resample(rec, resample_rate=2500, margin_ms=2000)

            # Extract information
            sampling_rate = rec.get_sampling_frequency()
            num_channels = rec.get_num_channels()

            # Electrode locations
            electrode_table = nwb_read.get_electrode_table(nwb_file)
            ca1_sites = electrode_table.loc[(electrode_table.group_name == f"imec{stream}") &
                                            (electrode_table.location == 'CA1')]
            if ca1_sites.empty:
                print('No CA1 LFP')
                continue
            ca1_ids_list = ca1_sites.index_on_probe.astype(int).to_list()
            if len(ca1_ids_list) >= 15:
                ca1_ids_list = ca1_ids_list[::len(ca1_ids_list)//15]
            ca1_ids = rec.get_channel_ids()[ca1_ids_list]

            ssp_bfd_sites = electrode_table.loc[(electrode_table.group_name == f'imec{stream}') &
                                                (electrode_table.location.str.startswith('SSp-bfd'))]
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
            ca1_units = units_df.loc[units_df.ccf_atlas_acronym == 'CA1']
            order_ca1_units = ca1_units.sort_values('peak_channel', ascending=True)
            ca1_spk_times = order_ca1_units.spike_times.values[:]
            n_ca1_units = len(ca1_spk_times)

            ssp_bfd_names = [i for i in units_df.ccf_atlas_acronym.unique() if 'SSp-bfd' in i]
            ssp_bfd_units = units_df.loc[units_df.ccf_atlas_acronym.isin(ssp_bfd_names)]
            ssp_bfd_spk_times = ssp_bfd_units.spike_times.values[:]
            n_ssp_bfd_units = len(ssp_bfd_spk_times)

            # Get whisker DLC
            keys = ['behavior', 'BehavioralTimeSeries']
            wh_angle = nwb_read.get_dlc_data(nwb_file, keys, 'whisker_angle')
            wh_angle_ts = nwb_read.get_dlc_timestamps(nwb_file, keys)[0]

            # Get a segment of data
            catch_start_time = no_stim_table.start_time.values[:]
            print(f'{len(catch_start_time)} "no stim" trials ')
            start = - 1
            stop = 6
            session_ripple = 0
            for catch_id, catch_ts in enumerate(catch_start_time):
                start_frame = int((catch_ts + start) * sampling_rate)
                end_frame = int((catch_ts + stop) * sampling_rate)
                n_samples = end_frame - start_frame
                time_vec = np.linspace(catch_ts + start, catch_ts + stop, n_samples)

                # Extract traces
                ca1_traces = rec.get_traces(start_frame=start_frame, end_frame=end_frame, channel_ids=ca1_ids)
                sspbfd_traces = rec.get_traces(start_frame=start_frame, end_frame=end_frame, channel_ids=ssp_bfd_ids)
                ca1_filt_traces = lfp_filter(data=ca1_traces, fs=sampling_rate, freq_min=150, freq_max=200)
                sspbfd_filt_traces = lfp_filter(data=sspbfd_traces, fs=sampling_rate, freq_min=10, freq_max=16)

                # Power in ripple band
                analytic_signal = sci.signal.hilbert(ca1_filt_traces, axis=0)
                # Extract amplitude envelope for all channels
                amplitude_envelope = np.abs(analytic_signal)
                # Square to get power (time x channels)
                power = amplitude_envelope ** 2
                # Smooth along time axis for all channels
                window_size = int(0.05 * sampling_rate)  # 50 ms
                kernel = np.ones(window_size) / window_size
                smoothed_power = np.apply_along_axis(
                    lambda m: np.convolve(m, kernel, mode='same'),
                    axis=0,
                    arr=power
                )
                z_scored_power = (smoothed_power - np.mean(smoothed_power, axis=0)) / np.std(smoothed_power, axis=0)
                threshold = 3

                # Method 2: Use median across channels for robust detection
                consensus_power = np.median(z_scored_power, axis=1)

                # Detect peaks on consensus signal
                peak_frames, _ = sci.signal.find_peaks(consensus_power, height=threshold,
                                                       distance=int(0.05 * sampling_rate))
                n_events = len(peak_frames)
                session_ripple += n_events

                # For plotting
                peaks_1d = np.zeros(len(consensus_power), dtype=bool)
                peaks_1d[peak_frames] = True

                # Get the whisker angle trace
                wh_trace = wh_angle[
                           find_nearest(wh_angle_ts, (catch_ts + start)): find_nearest(wh_angle_ts, catch_ts + stop)]
                wh_ts = wh_angle_ts[
                        find_nearest(wh_angle_ts, (catch_ts + start)): find_nearest(wh_angle_ts, catch_ts + stop)]

                # Do the first plot
                offset = 50
                fig, axes = plt.subplots(8, 1, figsize=(16, 20))
                for i in range(ca1_traces.shape[1]):
                    axes[0].plot(time_vec, ca1_traces[:, i] + i * offset)
                for i in range(ca1_filt_traces.shape[1]):
                    axes[1].plot(time_vec, ca1_filt_traces[:, i] + i * max(ca1_filt_traces[:, i]))
                for i in range(z_scored_power.shape[1]):
                    axes[2].plot(time_vec, z_scored_power[:, i] + i * 4)
                for i in range(sspbfd_traces.shape[1]):
                    axes[7].plot(time_vec, sspbfd_traces[:, i] + i * offset)
                for i in range(sspbfd_filt_traces.shape[1]):
                    axes[6].plot(time_vec, sspbfd_filt_traces[:, i] + i * max(sspbfd_filt_traces[:, i]))
                # Filter all spike trains
                ca1_filtered_spikes = [
                    spikes[(spikes >= catch_ts + start) & (spikes <= catch_ts + stop)]
                    for spikes in ca1_spk_times
                ]
                sspbfd_filtered_spikes = [
                    spikes[(spikes >= catch_ts + start) & (spikes <= catch_ts + stop)]
                    for spikes in ssp_bfd_spk_times
                ]
                # Plot all at once
                axes[2].scatter(x=time_vec[np.where(peaks_1d)[0]], y=[-5] * len(np.where(peaks_1d)[0]),
                                marker='o', c='k')
                axes[3].eventplot(ca1_filtered_spikes, colors='black', linewidths=0.8)
                axes[4].eventplot(sspbfd_filtered_spikes, colors='black', linewidths=0.8)
                axes[5].plot(wh_ts, wh_trace, c='orange')

                for ax in axes.flatten():
                    ax.spines[['right', 'top']].set_visible(False)

                axes[0].set_title('CA1')
                axes[1].set_title('CA1 - 150-200 Hz')
                axes[2].set_title('Ripple power (z-score)')
                axes[3].set_title('CA1 spike raster')
                axes[4].set_title('SSp-bfd spike raster')
                axes[5].set_title('Whisker angle')
                axes[6].set_title('SSp-bfd - 10-16 Hz')
                axes[7].set_title('SSp-bfd')
                fig.suptitle(f'Catch #{catch_id} at t = {catch_ts} s')
                fig.tight_layout()

                s_path = os.path.join(save_path, session_id)
                if not os.path.exists(s_path):
                    os.makedirs(s_path)
                for f in ['pdf', 'png']:
                    fig.savefig(os.path.join(s_path, f'catch_{catch_id}.{f}'), dpi=400)
                plt.close('all')

                for ripple_id, ripple_frame in enumerate(peak_frames):
                    t_size_s = 0.200
                    offset = 50
                    t_range = int(t_size_s * sampling_rate)
                    t_ripple = time_vec[ripple_frame]
                    if (ripple_frame - t_range) < 0 or ripple_frame + t_range > len(time_vec):
                        continue
                    fig, axes = plt.subplots(8, 1, figsize=(5, 20))
                    for i in range(ca1_traces.shape[1]):
                        axes[0].plot(time_vec[ripple_frame - t_range: ripple_frame + t_range],
                                     ca1_traces[ripple_frame - t_range: ripple_frame + t_range, i] + i * offset)
                    for i in range(ca1_filt_traces.shape[1]):
                        axes[1].plot(time_vec[ripple_frame - t_range: ripple_frame + t_range],
                                     ca1_filt_traces[ripple_frame - t_range: ripple_frame + t_range, i] +
                                     i * max(ca1_filt_traces[ripple_frame - t_range: ripple_frame + t_range, i]))
                    for i in range(z_scored_power.shape[1]):
                        axes[2].plot(time_vec[ripple_frame - t_range: ripple_frame + t_range],
                                     z_scored_power[ripple_frame - t_range: ripple_frame + t_range, i] + i * 4)
                    for i in range(sspbfd_traces.shape[1]):
                        axes[7].plot(time_vec[ripple_frame - t_range: ripple_frame + t_range],
                                     sspbfd_traces[ripple_frame - t_range: ripple_frame + t_range, i] + i * offset)
                    for i in range(sspbfd_filt_traces.shape[1]):
                        axes[6].plot(time_vec[ripple_frame - t_range: ripple_frame + t_range],
                                     sspbfd_filt_traces[ripple_frame - t_range: ripple_frame + t_range, i] +
                                     i * max(sspbfd_filt_traces[ripple_frame - t_range: ripple_frame + t_range, i]))
                    # Filter all spike trains
                    ca1_filtered_spikes = [
                        spikes[(spikes >= t_ripple - t_size_s) & (spikes <= t_ripple + t_size_s)]
                        for spikes in ca1_spk_times
                    ]
                    sspbfd_filtered_spikes = [
                        spikes[(spikes >= t_ripple - t_size_s) & (spikes <= t_ripple + t_size_s)]
                        for spikes in ssp_bfd_spk_times
                    ]
                    # Plot all at once
                    axes[2].scatter(x=t_ripple, y=-5, marker='o', c='k')
                    axes[3].eventplot(ca1_filtered_spikes, colors='black', linewidths=0.8)
                    axes[4].eventplot(sspbfd_filtered_spikes, colors='black', linewidths=0.8)
                    # Get the whisker angle trace
                    wh_trace_zoom = wh_angle[
                               find_nearest(wh_angle_ts, (t_ripple - t_size_s)): find_nearest(wh_angle_ts,
                                                                                           t_ripple + t_size_s)]
                    wh_ts_zoom = wh_angle_ts[
                            find_nearest(wh_angle_ts, (t_ripple - t_size_s)): find_nearest(wh_angle_ts, t_ripple + t_size_s)]
                    axes[5].plot(wh_ts_zoom, wh_trace_zoom, c='orange')

                    for ax in axes.flatten():
                        ax.spines[['right', 'top']].set_visible(False)

                    axes[0].set_title('CA1')
                    axes[1].set_title('CA1 - 150-200 Hz')
                    axes[2].set_title('Ripple power (z-score)')
                    axes[3].set_title('CA1 spike raster')
                    axes[4].set_title('SSp-bfd spike raster')
                    axes[5].set_title('Whisker angle')
                    axes[6].set_title('SSp-bfd - 10-16 Hz')
                    axes[7].set_title('SSp-bfd')
                    fig.suptitle(f'Catch #{catch_id} at t = {catch_ts} s')
                    fig.tight_layout()

                    s_path = os.path.join(save_path, session_id, 'single_events')
                    if not os.path.exists(s_path):
                        os.makedirs(s_path)
                    for f in ['pdf', 'png']:
                        fig.savefig(os.path.join(s_path, f'catch_{catch_id}_ripple_{ripple_id}.{f}'), dpi=400)
                    plt.close('all')

            results_dict['mouse_id'].append(mouse)
            results_dict['session_id'].append(session_id)
            results_dict['reward_group'].append(rew_group)
            results_dict['n_ripples'].append(session_ripple)
            results_dict['n_no_stim'].append(len(catch_start_time))
            results_dict['total_time'].append((stop - start) * len(catch_start_time))
            results_dict['fz'].append(np.round(session_ripple / ((stop - start) * len(catch_start_time)), 3))

            print(f'Total : {session_ripple} events, '
                  f'{np.round(session_ripple / ((stop - start) * len(catch_start_time)), 3) * 60} event / min')

results_df = pd.DataFrame(results_dict)
results_df.to_csv(os.path.join(save_path, "results.csv"))

fig, ax = plt.subplots(1, 1)
sns.boxplot(results_df, x='reward_group', y='fz', ax=ax)
fig.savefig(os.path.join(save_path, "results_figure.png"))
