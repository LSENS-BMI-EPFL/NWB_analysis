import os
import pathlib
import subprocess
from pathlib import Path
import scipy as sci
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface as si
import spikeinterface.preprocessing as sip
from sklearn.manifold import TSNE
from nwb_utils.utils_misc import find_nearest


def get_database(task):
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

    return db_df


def parse_meta(meta_path):
    meta = {}
    with open(meta_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, val = line.strip().split('=', 1)
                meta[key] = val
    return meta


def get_lfp_recordings(data_folder, experimenter, mouse, session, stream):
    experimenter_map = {'AB': 'Axel_Bisi',
                        'PB': 'Pol_Bech',
                        'MH': 'Myriam_Hamon',
                        'JL': 'Jules_Lebert'}
    if mouse[0:2] == 'PB':
        new_data_folder = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\publications\2026\2026_Bech_Dard_eLife\2026_Bech_Dard_eLife_data\raw_data"
        path = os.path.join(new_data_folder, mouse, 'Recording', 'Ephys', session)
        if not os.path.exists(path):
            path = os.path.join(data_folder, mouse, 'Recording', session, 'Ephys')
    else:
        path = os.path.join(data_folder, experimenter_map[experimenter], 'data', mouse, session, 'Ephys')

    if not os.path.exists(path):
        return None

    g_index = os.listdir(path)[0]
    file_id = '_'.join(g_index.split('_')[1:])
    full_path = os.path.join(path, f'{g_index}')

    # Get the specific folder
    subfolder = os.path.join(full_path, f'{file_id}_imec{stream}')
    fs = os.listdir(subfolder)

    # Get lf and ap bin & meta
    lf_files = [f for f in fs if '.lf.bin' in f]
    if len(lf_files) > 0:
        lf_matching_file = lf_files[0]
        lf_bin_path = Path(os.path.join(subfolder, lf_matching_file))
        lf_meta_path = lf_bin_path.with_suffix('.meta')

    ap_files = [f for f in fs if '.ap.bin' in f]
    if len(ap_files) > 0:
        ap_matching_file = ap_files[0]
        ap_bin_path = Path(os.path.join(subfolder, ap_matching_file))
        ap_meta_path = ap_bin_path.with_suffix('.meta')

    try:
        meta = parse_meta(lf_meta_path)
        sampling_rate = float(meta['imSampRate'])
        n_channels = int(meta['nSavedChans'])
        rec = si.core.BinaryRecordingExtractor(
            file_paths=[str(lf_bin_path)],
            sampling_frequency=sampling_rate,
            num_channels=n_channels,
            dtype=np.int16,
            time_axis=0,  # samples are rows in file
            file_offset=0
        )
        print("Using LF stream")

    except:
        print("LF stream not found, using AP stream")
        meta = parse_meta(ap_meta_path)
        sampling_rate = float(meta['imSampRate'])
        n_channels = int(meta['nSavedChans'])
        rec = si.core.BinaryRecordingExtractor(
            file_paths=[str(ap_bin_path)],
            sampling_frequency=sampling_rate,
            num_channels=n_channels,
            dtype=np.int16,
            time_axis=0,  # samples are rows in file
            file_offset=0
        )
        rec = sip.bandpass_filter(rec, freq_min=0.5, freq_max=500, margin_ms=5000)
        rec = sip.resample(rec, resample_rate=2500, margin_ms=2000)

    return rec


def apply_tprime_to_ripple_times(analysis_dir, mouse, session, ref_prob_id, ripple_ts_file_path):
    # Define TPrime executable path at the very beginning
    TPRIME_EXECUTABLE = r"C:\Users\rdard\TPrime-win\TPrime.exe"

    experimenter_map = {'AB': 'Axel_Bisi',
                        'PB': 'Pol_Bech',
                        'MH': 'Myriam_Hamon',
                        'JL': 'Jules_Lebert'}
    experimenter = mouse[0:2]
    if experimenter == 'PB':
        new_data_folder = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\publications\2026\2026_Bech_Dard_eLife\2026_Bech_Dard_eLife_data\raw_data"
        main_dir = os.path.join(new_data_folder, mouse, 'Recording', 'Ephys', session)
        if not os.path.exists(main_dir):
            main_dir = os.path.join(analysis_dir, mouse, 'Recording', session, 'Ephys')
    else:
        main_dir = Path(os.path.join(analysis_dir, experimenter_map[experimenter], 'data', mouse, session, 'Ephys'))

    # Get names & folder structure
    folder_name = os.listdir(main_dir)[0]
    epoch_name = '_'.join(folder_name.split('_')[1:])
    input_dir = os.path.join(main_dir, f'{folder_name}')

    # Get synchronization period
    sglx_metafile_path = os.path.join(input_dir, '{}_tcat.nidq.meta'.format(epoch_name))
    sglx_meta_dict = readSGLX.readMeta(pathlib.Path(sglx_metafile_path))
    syncperiod = float(sglx_meta_dict['syncSourcePeriod'])
    if syncperiod is None:
        syncperiod = 1

    # Write TPrime command line
    nidq_stream_idx = 10
    path_ref_probe = os.path.join(input_dir, '{}_imec{}'.format(epoch_name, ref_prob_id))
    ap_meta_file = os.path.join(input_dir, f'{epoch_name}_imec{ref_prob_id}',
                                '{}_tcat.imec{}.ap.meta'.format(epoch_name, ref_prob_id))
    ap_meta_dict = parse_meta(ap_meta_file)
    ref_probe_edges_file = '{}_tcat.imec{}.ap.xd_{}_6_500.txt'.format(epoch_name,
                                                                      ref_prob_id,
                                                                      int(ap_meta_dict['nSavedChans']) - 1)

    # Convert all paths to strings
    to_stream_path = os.path.join(path_ref_probe, ref_probe_edges_file)
    from_stream_path = os.path.join(input_dir, epoch_name + '_tcat.nidq.xa_0_0.txt')
    output_path = os.path.dirname(ripple_ts_file_path)
    output_file = os.path.join(output_path, f'{session}_ripple_ts_sync.txt')

    # Build command string with full TPrime path
    cmd = (
        f'"{TPRIME_EXECUTABLE}" '
        f'-syncperiod={syncperiod} '
        f'-tostream="{to_stream_path}" '
        f'-fromstream={nidq_stream_idx},"{from_stream_path}" '
        f'-events={nidq_stream_idx},"{ripple_ts_file_path}","{output_file}"'
    )
    
    print('\nRunning TPrime...')
    # Execute the command
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )

    # Check the result
    print(f"Return code: {result.returncode}")
    if result.stdout:
        print(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        print(f"STDERR:\n{result.stderr}")

    # Verify output file was created
    if os.path.exists(output_file):
        print(f"Output file created at: {output_file}")
        return True
    else:
        print(f"ERROR: Output file was not created at: {output_file}")
        if result.returncode != 0:
            print(f"TPrime exited with error code: {result.returncode}")
        return False


def lfp_filter(data, fs, freq_min=150, freq_max=200):
    nyq = 0.5 * fs
    low = freq_min / nyq
    high = freq_max / nyq
    b, a = sci.signal.butter(3, [low, high], btype='band')
    return sci.signal.filtfilt(b, a, data, axis=0)


def ripple_detect(ca1_sw_lfp, ca1_ripple_lfp, sampling_rate, threshold, sharp_filter=False,
                  sharp_delay=0.070, detection_delay=0.010):
    window_size = int(0.05 * sampling_rate)  # 50 ms
    kernel = np.ones(window_size) / window_size

    # Sharp-wave
    sw_envelope = np.abs(sci.signal.hilbert(ca1_sw_lfp, axis=0))
    sw_power = sw_envelope ** 2
    sw_smoothed_power = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode='same'),
        axis=0,
        arr=sw_power
    )
    sw_z = (sw_smoothed_power - np.mean(sw_smoothed_power, axis=0)) / np.std(sw_smoothed_power, axis=0)

    # Ripple
    ripple_envelope = np.abs(sci.signal.hilbert(ca1_ripple_lfp, axis=0))
    ripple_power = ripple_envelope ** 2
    ripple_smoothed_power = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode='same'),
        axis=0,
        arr=ripple_power
    )
    ripple_z = (ripple_smoothed_power - np.mean(ripple_smoothed_power, axis=0)) / np.std(ripple_smoothed_power, axis=0)

    # Use weighted average across channels for robust detection
    weights = np.std(ripple_smoothed_power, axis=0)
    best_channel = np.argmax(weights)

    # Get the delay in frames
    detection_delay_frames = int(detection_delay * sampling_rate)

    # Detect ripples on consensus signal
    ripple_consensus = np.median(ripple_z, axis=1)
    ripple_peak_frames, _ = sci.signal.find_peaks(ripple_consensus, height=threshold,
                                                  distance=int(0.05 * sampling_rate))
    ripple_peak_frames = ripple_peak_frames[ripple_peak_frames >= detection_delay_frames]

    # Detect sharp waves on consensus signal
    sw_consensus = np.median(sw_z, axis=1)
    sw_peak_frames, _ = sci.signal.find_peaks(sw_consensus, height=threshold,
                                              distance=int(0.05 * sampling_rate))

    # Find match
    if sharp_filter:
        if len(sw_peak_frames) == 0:
            ripple_peak_frames = []
        else:
            co_sw_ripple = []
            for ripple_id, ripple_frame in enumerate(ripple_peak_frames):
                nearset_sw = find_nearest(sw_peak_frames, ripple_frame)
                if nearset_sw == len(sw_peak_frames):
                    nearset_sw_frame = sw_peak_frames[-1]
                elif nearset_sw == -1:
                    nearset_sw_frame = sw_peak_frames[0]
                else:
                    nearset_sw_frame = sw_peak_frames[nearset_sw]
                co_sw_ripple.append((np.abs(nearset_sw_frame - ripple_frame) / sampling_rate <= sharp_delay))
            ripple_peak_frames = ripple_peak_frames[co_sw_ripple]

    return ripple_peak_frames, ripple_z, best_channel


def plot_lfp_custom(ca1lfp, ca_high_filt, ca1_ripple_power, sspbfdlfp, sspbfd_spindle_filt,
                    time_vec, ripple_times, best_channel, wh_trace, is_whisking, tongue_trace, wh_ts,
                    ca1_spikes, sspbfd_spikes, offset, session_id, start_id, start_ts, plot_start_ts, ripple_id,
                    ripple_target, secondary_target, trial_selection,
                    fig_size, save_path):

    fig, axes = plt.subplots(8, 1, figsize=fig_size, sharex=True)

    ca1_colors = plt.cm.Blues(np.linspace(0.3, 1, ca1lfp.shape[1]))
    sspbfd_colors = plt.cm.Purples(np.linspace(0.3, 1, sspbfdlfp.shape[1]))

    for i in range(ca1lfp.shape[1]):
        axes[0].plot(time_vec, ca1lfp[:, i] + i * np.max(np.std(ca1lfp, axis=0)) * offset, c=ca1_colors[i])

    for i in range(ca_high_filt.shape[1]):
        axes[1].plot(time_vec, ca_high_filt[:, i] + i * np.max(ca_high_filt), c=ca1_colors[i])

    for i in range(ca1_ripple_power.shape[1]):
        axes[2].plot(time_vec, ca1_ripple_power[:, i] + i * 4, c=ca1_colors[i])

    for i in range(sspbfdlfp.shape[1]):
        axes[7].plot(time_vec, sspbfdlfp[:, i] + i * np.max(np.std(sspbfdlfp, axis=0)) * offset, c=sspbfd_colors[i])

    for i in range(sspbfd_spindle_filt.shape[1]):
        axes[6].plot(time_vec, sspbfd_spindle_filt[:, i] + i * np.max(sspbfd_spindle_filt), c=sspbfd_colors[i])

    if type(ripple_times) != np.float64:
        ripple_marker_c = ['black' if whisk == True else 'red' for whisk in is_whisking]
        axes[2].scatter(x=ripple_times, y=[-10] * len(ripple_times), marker='o', c=ripple_marker_c)
        axes[1].scatter(x=ripple_times, y=[-10] * len(ripple_times), marker='o', c=ripple_marker_c)
    else:
        try:
            ripple_marker_c = 'black' if is_whisking[0] == True else 'red'
        except:
            ripple_marker_c = 'black' if is_whisking == True else 'red'
        axes[2].scatter(x=ripple_times, y=[-10], marker='o', c=ripple_marker_c)
        axes[1].scatter(x=ripple_times, y=[-10], marker='o', c=ripple_marker_c)

    axes[0].scatter(time_vec[0] - (time_vec[1] - time_vec[0]) * 0.8, best_channel * offset, marker='*', c='k')
    axes[3].eventplot(ca1_spikes, colors='black', linewidths=0.8)
    axes[4].eventplot(sspbfd_spikes, colors='black', linewidths=0.8)

    if len(wh_trace) > 0 and len(wh_ts) > 0:
        axes[5].plot(wh_ts, wh_trace, c='darkorange')
        wh_speed = np.abs(np.diff(wh_trace))
        axes[5].plot(wh_ts[1:], wh_speed, c='darkred')
    if len(tongue_trace) > 0 and len(wh_ts) > 0:
        axes[5].plot(wh_ts, 3 * tongue_trace, c='deeppink')

    if trial_selection == 'whisker_trial':
        trial_color = 'darkorange'
    elif trial_selection == 'auditory_trial':
        trial_color = 'royalblue'
    else:
        trial_color = "black"
    for ax in axes.flatten():
        ax.spines[['right', 'top']].set_visible(False)
        if plot_start_ts:
            ax.axvline(x=start_ts, ymin=0, ymax=1, c=trial_color, linestyle='--')

    axes[0].set_title(f'{ripple_target}')
    axes[1].set_title(f'{ripple_target} - 150-200 Hz')
    axes[2].set_title('Ripple power (z-score)')
    axes[3].set_title(f'{ripple_target} spike raster')
    axes[4].set_title(f'{secondary_target} spike raster')
    axes[5].set_title(f'Whisker angle / Tongue distance')
    axes[6].set_title(f'{secondary_target} - 10-16 Hz')
    axes[7].set_title(f'{secondary_target}')
    fig.suptitle(f'{session_id} {trial_selection} #{start_id} at t = {start_ts} s')
    fig.tight_layout()

    for f in ['pdf', 'png']:
        if ripple_id is not None:
            fig.savefig(os.path.join(save_path, f'{trial_selection}_{start_id}_ripple_{ripple_id}.{f}'), dpi=400)
        else:
            fig.savefig(os.path.join(save_path, f'{trial_selection}_{start_id}.{f}'), dpi=400)
    plt.close('all')


def build_ripple_population_vectors(all_spikes, ripple_time, delay, baseline_substraction=False):
    ripple_spikes = [
        spikes[(spikes >= ripple_time - delay) & (spikes <= ripple_time + delay)]
        for spikes in all_spikes
    ]
    if baseline_substraction:
        baseline_spikes = [
            spikes[(spikes >= ripple_time - 2*delay) & (spikes <= ripple_time-delay)]
            for spikes in all_spikes
        ]
        population_vector = [len(ripple) - len(baseline) for ripple, baseline in zip(ripple_spikes, baseline_spikes)]
    else:
        population_vector = [len(spikes) for spikes in ripple_spikes]

    return population_vector


def build_sensory_population_vectors(all_spikes, start_time, delay, baseline_substraction=False):
    ripple_spikes = [
        spikes[(spikes >= start_time) & (spikes <= start_time + delay)]
        for spikes in all_spikes
    ]
    if baseline_substraction:
        baseline_spikes = [
            spikes[(spikes >= start_time - delay) & (spikes <= start_time)]
            for spikes in all_spikes
        ]
        population_vector = [len(ripple) - len(baseline) for ripple, baseline in zip(ripple_spikes, baseline_spikes)]
    else:
        population_vector = [len(spikes) for spikes in ripple_spikes]

    return population_vector

def cluster_ripple_content(ca1_ripple_array, ssp_ripple_array, session, group, context_blocks, save_path):
    # Cluster on CA1 ripple content
    if (ca1_ripple_array.shape[0] > 3) and (ca1_ripple_array.shape[1] > 4):
        ca1_tsne_results = TSNE(n_components=2, learning_rate='auto',
                                init='random', perplexity=3).fit_transform(ca1_ripple_array)
    else:
        ca1_tsne_results = np.zeros((ca1_ripple_array.shape[0], 2))

    # Cluster on second region ripple content
    if (ssp_ripple_array.shape[0] > 3) and (ssp_ripple_array.shape[1] > 4):
        ssp_tsne_results = TSNE(n_components=2, learning_rate='auto',
                                init='random', perplexity=3).fit_transform(ssp_ripple_array)
    else:
        ssp_tsne_results = np.zeros((ssp_ripple_array.shape[0], 2))

    # Figure
    fig, axes = plt.subplots(3, 2, figsize=(8, 15))

    # Population vectors plot
    if ca1_ripple_array.shape[0] > 0 and ca1_ripple_array.shape[1] > 0:
        sns.heatmap(np.transpose(ca1_ripple_array), cmap='viridis', ax=axes[0, 0])
    if ssp_ripple_array.shape[0] > 0 and ssp_ripple_array.shape[1] > 0:
        sns.heatmap(np.transpose(ssp_ripple_array), cmap='viridis', ax=axes[0, 1])
    for ax in axes[0, :].flatten():
        ax.set_xlabel('Ripple events')
        ax.set_ylabel('Units')

    # Correlation matrix plot
    sort_indices = np.argsort(context_blocks)
    n_ripples = len(sort_indices)

    if ca1_ripple_array.shape[0] > 1 and ca1_ripple_array.shape[1] > 1:
        ca1_corr_matrix = np.corrcoef(ca1_ripple_array)
        if (len(np.unique(context_blocks)) > 1) and ('active' not in np.unique(context_blocks)):
            reordered_ca1_corr_matrix = ca1_corr_matrix[sort_indices][:, sort_indices]
            sns.heatmap(reordered_ca1_corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1, 0])
            n_context0 = np.sum(np.array(context_blocks) == 0)
            axes[1, 0].axhline(n_context0 - 0.5, color='black', linewidth=2)
            axes[1, 0].axvline(n_context0 - 0.5, color='black', linewidth=2)
            axes[1, 0].set_xticks(np.arange(n_ripples))
            axes[1, 0].set_yticks(np.arange(n_ripples))
            axes[1, 0].set_xticklabels(sort_indices, rotation=90, fontsize=6)
            axes[1, 0].set_yticklabels(sort_indices, fontsize=8)
        else:
            sns.heatmap(ca1_corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1, 0])
    if ssp_ripple_array.shape[0] > 1 and ssp_ripple_array.shape[1] > 1:
        ssp_corr_matrix = np.corrcoef(ssp_ripple_array)
        if (len(np.unique(context_blocks)) > 1) and ('active' not in np.unique(context_blocks)):
            reordered_ssp_corr_matrix = ssp_corr_matrix[sort_indices][:, sort_indices]
            sns.heatmap(reordered_ssp_corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1, 1])
            n_context0 = np.sum(np.array(context_blocks) == 0)
            axes[1, 1].axhline(n_context0 - 0.5, color='black', linewidth=2)
            axes[1, 1].axvline(n_context0 - 0.5, color='black', linewidth=2)
            axes[1, 1].set_xticks(np.arange(n_ripples))
            axes[1, 1].set_yticks(np.arange(n_ripples))
            axes[1, 1].set_xticklabels(sort_indices, rotation=90, fontsize=6)
            axes[1, 1].set_yticklabels(sort_indices, fontsize=6)
        else:
            sns.heatmap(ssp_corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1, 1])

    # t-SNE results
    if (len(np.unique(context_blocks)) > 1) and ('active' not in np.unique(context_blocks)):
        color = ['darkmagenta' if i == 0 else 'green' for i in context_blocks]
        cmap = None
    else:
        color = range(len(context_blocks))
        cmap = 'Blues'
    axes[2, 0].scatter(ca1_tsne_results[:, 0], ca1_tsne_results[:, 1], c=color,
                       s=100, vmin=0, vmax=len(ca1_tsne_results)-1, cmap=cmap)
    axes[2, 1].scatter(ssp_tsne_results[:, 0], ssp_tsne_results[:, 1], c=color,
                       s=100, vmin=0, vmax=len(ssp_tsne_results)-1, cmap=cmap)
    axes[0, 0].set_title('CA1 ripple content')
    axes[0, 1].set_title('SSp-bfd ripple content')
    for ax in axes[2, :].flatten():
        ax.set_xlabel('t-SNE embedding 1')
        ax.set_ylabel('t-SNE embedding 2')

    fig.tight_layout()
    # Savings
    if len(np.unique(context_blocks)) > 1:
        fig.suptitle(f'{session}')
    else:
        fig.suptitle(f'{session}, {group}')
    for ax in axes.flatten():
        ax.spines[['right', 'top']].set_visible(False)
    for f in ['pdf', 'png']:
        fig.savefig(os.path.join(save_path, f'tsne_ripple_content.{f}'), dpi=400)
    plt.close('all')


def get_units_selection(units_df, target, only_good=False):
    if only_good is True:
        try:
            names = [i for i in units_df.ccf_atlas_acronym.unique() if target in i]
            units = units_df.loc[(units_df.ccf_atlas_acronym.isin(names)) &
                                 (units_df.bc_label == 'good')]
        except:
            names = [i for i in units_df.ccf_acronym.unique() if target in i]
            units = units_df.loc[(units_df.ccf_acronym.isin(names)) &
                                 (units_df.bc_label == 'good')]
    else:
        try:
            names = [i for i in units_df.ccf_atlas_acronym.unique() if target in i]
            units = units_df.loc[(units_df.ccf_atlas_acronym.isin(names))]
        except:
            names = [i for i in units_df.ccf_acronym.unique() if target in i]
            units = units_df.loc[(units_df.ccf_acronym.isin(names))]

    order_units = units.sort_values('peak_channel', ascending=True)

    return order_units


def get_lfp_channels(electrode_table, stream, rec, target, target_type):
    if target_type == 'ripple':
        sites = electrode_table.loc[(electrode_table.group_name == f"imec{stream}") &
                                    (electrode_table.location == target)]
    else:
        sites = electrode_table.loc[(electrode_table.group_name == f"imec{stream}") &
                                    (electrode_table.location.str.startswith(target))]
    if sites.empty:
        return None

    ids_list = sites.index_on_probe.astype(int).to_list()
    n_channels = len(ids_list)
    n_select = min(15, n_channels)
    if n_select == n_channels:
        selected_indices = ids_list
    else:
        selected_indices = [ids_list[int(i)] for i in np.linspace(0, n_channels - 1, n_select)]
    channels = rec.get_channel_ids()[selected_indices]

    return channels


def plot_ripple_frequency_fastlearning(data_folder, trial_types, save_path):

    names = os.listdir(data_folder)
    files = [os.path.join(data_folder, name) for name in names]

    # ======== Load and preprocess all mice data ========
    perf_dict = {'mouse': [], 'wh_perf': []}
    dfs = []

    for file_id, file in enumerate(files):

        mouse_id = names[file_id][:5]
        print(f"Mouse: {mouse_id}")

        df = pd.read_pickle(file)
        perf_dict['mouse'].append(mouse_id)

        # Performance (whisker trials only)
        try:
            wh_perf = df.loc[(df.context == 'active') &
                             (df.trial_type == 'whisker_trial')].lick_flag.mean()
        except:
            wh_perf = df.loc[(df.trial_type == 'whisker_trial')].lick_flag.mean()

        perf_dict['wh_perf'].append(wh_perf)

        cols = [
            'mouse', 'session', 'ripples_per_trial',
            'rewarded_group', 'trial_duration', 'trial_type', 'lick_flag'
        ]
        dfs.append(df.loc[(df.context == 'active'), cols])

    df_all = pd.concat(dfs).copy()
    perf_df = pd.DataFrame(perf_dict)

    # =======================================================
    #        MAIN 3 × 3 FIGURE (strip + box by condition)
    # =======================================================
    fig1, axes1 = plt.subplots(
        nrows=len(trial_types), ncols=3,
        figsize=(11, 3.5 * len(trial_types)),
        sharey=True
    )
    for ax in axes1.flatten():
        ax.spines[['right', 'top']].set_visible(False)

    # =======================================================
    #       CORRELATION 3 × 3 FIGURE (scatter WHR vs ripples)
    # =======================================================
    fig2, axes2 = plt.subplots(
        nrows=len(trial_types), ncols=3,
        figsize=(11, 3.5 * len(trial_types)),
        sharey=True,
        sharex=True
    )
    for ax in axes2.flatten():
        ax.spines[['right', 'top']].set_visible(False)

    # Conditions to loop through
    cond_filters = {
        "all": None,
        "lick": 1,
        "nolick": 0
    }

    for row, tt in enumerate(trial_types):

        for col, (cond_name, cond_val) in enumerate(cond_filters.items()):

            ax1 = axes1[row, col]
            ax2 = axes2[row, col]

            # ======= Filter data per trial type + lick condition ========
            df_tt = df_all.loc[df_all.trial_type == tt].copy()

            if cond_val is not None:
                df_tt = df_tt.loc[df_tt.lick_flag == cond_val]

            if df_tt.empty:
                ax1.set_title(f"{tt} – {cond_name}: no data")
                ax2.set_title(f"{tt} – {cond_name}: no data")
                ax1.axis("off")
                ax2.axis("off")
                continue

            # ======== Aggregate per mouse × session × rewarded group ========
            gdf = df_tt.groupby(
                ['mouse', 'session', 'rewarded_group'], as_index=False
            ).sum()

            gdf['ripple_fz'] = np.round(
                (gdf['ripples_per_trial'] / gdf['trial_duration']) * 60, 3
            )

            gdf = gdf.merge(perf_df, on='mouse', how='left')
            gdf = gdf.rename(columns={'wh_perf': 'whr'})

            # =====================================================
            #                     STRIP + BOX
            # =====================================================
            sns.stripplot(
                data=gdf, hue='rewarded_group', y='ripple_fz',
                hue_order=['R-', 'R+'], palette=['darkmagenta', 'green'],
                dodge=True, legend=False, ax=ax1
            )
            sns.boxplot(
                data=gdf, hue='rewarded_group', y='ripple_fz',
                hue_order=['R-', 'R+'], palette=['darkmagenta', 'green'],
                showfliers=False, ax=ax1
            )
            ax1.set_xlabel('')
            ax1.set_ylabel('Ripple rate (min⁻¹)')
            ax1.set_title(f"{tt} – {cond_name}")
            sns.despine()

            # =====================================================
            #                CORRELATION SCATTER
            # =====================================================
            sns.scatterplot(
                data=gdf,
                x='ripple_fz', y='whr',
                hue='rewarded_group', hue_order=['R-', 'R+'],
                palette=['darkmagenta', 'green'],
                legend=False, ax=ax2
            )
            ax2.set_xlabel('Ripple rate (min⁻¹)')
            ax2.set_ylabel('Session WHR')
            ax2.set_title(f"{tt} – {cond_name}")
            sns.despine()

    fig1.tight_layout()
    fig2.tight_layout()

    # Save
    save_path = os.path.join(save_path, 'average_results')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    fig1.savefig(os.path.join(save_path, "RippleRate_3x3.png"))
    fig1.savefig(os.path.join(save_path, "RippleRate_3x3.pdf"))

    fig2.savefig(os.path.join(save_path, "RippleRate_vs_Performance_3x3.png"))
    fig2.savefig(os.path.join(save_path, "RippleRate_vs_Performance_3x3.pdf"))

    plt.close("all")

    print("Finished generating all figures.")


def plot_all_trials_data(data_folder, task, save_path):
    ripple_target = 'CA1'
    if task == 'fast-learning':
        secondary_target = 'SSp-bfd'
    else:
        secondary_target = 'RSP'

    names = os.listdir(data_folder)
    files = [os.path.join(data_folder, name) for name in names]
    for file_id, file in enumerate(files):
        print(f'Mouse: {names[file_id][0:5]}')
        file_path = os.path.join(data_folder, file)
        df = pd.read_pickle(file_path)

        # Plot each trial:
        for trial_id in range(len(df)):
            if df.loc[trial_id].context == 'passive':
                continue
            trial_type = df.loc[trial_id].trial_type
            lick_tag = 'lick' if df.loc[trial_id].lick_flag == 1 else 'nolick'

            result_folder = os.path.join(save_path, df.loc[trial_id].session, trial_type, lick_tag)
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)

            if len(df.loc[trial_id].ripple_times) > 0:
                print(f'Trial: {trial_id}, type: {df.loc[trial_id].trial_type}, '
                      f'lick: {df.loc[trial_id].lick_flag}')
                plot_lfp_custom(ca1lfp=df.loc[trial_id].ca1_lfp,
                                ca_high_filt=df.loc[trial_id].ca1_ripple_band_flp,
                                ca1_ripple_power=df.loc[trial_id].ca1_ripple_power,
                                sspbfdlfp=df.loc[trial_id].secondary_lfp,
                                sspbfd_spindle_filt=df.loc[trial_id].secondary_spindle_band_lfp,
                                time_vec=df.loc[trial_id].lfp_ts,
                                ripple_times=df.loc[trial_id].ripple_times,
                                best_channel=df.loc[trial_id].ca1_ripple_best_ch,
                                wh_trace=df.loc[trial_id].whisker_trace,
                                is_whisking=df.loc[trial_id].is_whisking,
                                tongue_trace=df.loc[trial_id].tongue_trace,
                                wh_ts=df.loc[trial_id].dlc_trial_ts,
                                ca1_spikes=df.loc[trial_id].ca1_spike_times,
                                sspbfd_spikes=df.loc[trial_id].secondary_spike_times,
                                offset=2, session_id=df.loc[trial_id].session,
                                start_id=trial_id,
                                start_ts=df.loc[trial_id].start_time,
                                plot_start_ts=True,
                                ripple_id=None,
                                ripple_target=ripple_target,
                                secondary_target=secondary_target, trial_selection=trial_type,
                                fig_size=(16, 22), save_path=result_folder)


def plot_single_event_data(data_folder, task, window, only_average, save_path):
    ripple_target = 'CA1'
    if task == 'fast-learning':
        secondary_target = 'SSp-bfd'
    else:
        secondary_target = 'RSP'

    names = os.listdir(data_folder)
    files = [os.path.join(data_folder, name) for name in names]
    for file_id, file in enumerate(files):
        print(f'Mouse: {names[file_id][0:5]}')
        file_path = os.path.join(data_folder, file)
        df = pd.read_pickle(file_path)

        # Plot each trial:
        single_ripple_ca1_lfp = []
        single_ripple_power_ca1_lfp = []
        ca1_aligned_spikes_all = []

        for trial_id in range(len(df)):
            if df.loc[trial_id].context == 'passive':
                continue
            trial_type = df.loc[trial_id].trial_type

            if len(df.loc[trial_id].ripple_times) > 0:
                print(f'Trial: {trial_id}, type: {df.loc[trial_id].trial_type}, '
                      f'lick: {df.loc[trial_id].lick_flag}')
                print(f"{len(df.loc[trial_id].ripple_times)} ripples")

                lick_tag = 'lick' if df.loc[trial_id].lick_flag == 1 else 'nolick'
                result_folder = os.path.join(save_path, df.loc[trial_id].session, trial_type, lick_tag, 'single_event')
                if not os.path.exists(result_folder):
                    os.makedirs(result_folder)

                for ripple_id, ripple_ts in enumerate(df.loc[trial_id].ripple_times):
                    trial_ts = df.loc[trial_id].lfp_ts
                    if (ripple_ts - window) < trial_ts[0] or (ripple_ts + window) > trial_ts[-1]:
                        continue

                    # Get frames in lfp indices
                    ripple_frame = find_nearest(trial_ts, ripple_ts)
                    sampling_rate = 1 / np.median(np.diff(trial_ts))
                    frame_range = int(sampling_rate * window)
                    zoom_start = ripple_frame - frame_range
                    zoom_stop = ripple_frame + frame_range
                    time_vec = np.linspace(ripple_ts - window, ripple_ts + window, len(np.arange(zoom_start, zoom_stop)))

                    # Filter all spike trains
                    ca1_spikes = df.loc[trial_id].ca1_spike_times
                    ca1_filtered_spikes = [
                        spikes[(spikes >= (ripple_ts - window)) & (spikes <= (ripple_ts + window))]
                        for spikes in ca1_spikes
                    ]
                    second_spk_times = df.loc[trial_id].secondary_spike_times
                    second_filtered_spikes = [
                        spikes[(spikes >= (ripple_ts - window)) & (spikes <= (ripple_ts + window))]
                        for spikes in second_spk_times
                    ]

                    # Get the whisker angle trace
                    wh_angle = df.loc[trial_id].whisker_trace
                    tongue_distance = df.loc[trial_id].tongue_trace
                    dlc_ts = df.loc[trial_id].dlc_trial_ts

                    if len(wh_angle) > 0:
                        wh_angle_zoom = wh_angle[find_nearest(dlc_ts, (ripple_ts - window)):
                                                 find_nearest(dlc_ts, (ripple_ts + window))]
                    else:
                        wh_angle_zoom = []

                    if len(tongue_distance) > 0:
                        tongue_distance_zoom = tongue_distance[find_nearest(dlc_ts, (ripple_ts - window)):
                                                               find_nearest(dlc_ts, (ripple_ts + window))]
                    else:
                        tongue_distance_zoom = []

                    if len(dlc_ts) > 0:
                        dlc_ts_zoom = dlc_ts[find_nearest(dlc_ts, (ripple_ts - window)):
                                             find_nearest(dlc_ts, (ripple_ts + window))]
                    else:
                        dlc_ts_zoom = []

                    if not only_average:
                        # Plot zoomed view on ripple event
                        plot_lfp_custom(ca1lfp=df.loc[trial_id].ca1_lfp[zoom_start: zoom_stop, :],
                                        ca_high_filt=df.loc[trial_id].ca1_ripple_band_flp[zoom_start: zoom_stop, :],
                                        ca1_ripple_power=df.loc[trial_id].ca1_ripple_power[zoom_start: zoom_stop, :],
                                        sspbfdlfp=df.loc[trial_id].secondary_lfp[zoom_start: zoom_stop, :],
                                        sspbfd_spindle_filt=df.loc[trial_id].secondary_spindle_band_lfp[zoom_start: zoom_stop, :],
                                        time_vec=time_vec,
                                        ripple_times=ripple_ts,
                                        best_channel=df.loc[trial_id].ca1_ripple_best_ch,
                                        wh_trace=wh_angle_zoom,
                                        is_whisking=df.loc[trial_id].is_whisking[ripple_id],
                                        tongue_trace=tongue_distance_zoom,
                                        wh_ts=dlc_ts_zoom,
                                        ca1_spikes=ca1_filtered_spikes,
                                        sspbfd_spikes=second_filtered_spikes,
                                        offset=2, session_id=df.loc[trial_id].session,
                                        start_id=trial_id,
                                        start_ts=df.loc[trial_id].start_time,
                                        plot_start_ts=False,
                                        ripple_id=ripple_id,
                                        ripple_target=ripple_target,
                                        secondary_target=secondary_target, trial_selection=trial_type,
                                        fig_size=(6, 22), save_path=result_folder)
                    single_ripple_ca1_lfp.append(df.loc[trial_id].ca1_lfp[zoom_start: zoom_stop, :])
                    single_ripple_power_ca1_lfp.append(df.loc[trial_id].ca1_ripple_power[zoom_start: zoom_stop, :])
                    ca1_aligned_spikes = [[t - ripple_ts for t in neuron] for neuron in ca1_filtered_spikes]
                    ca1_aligned_spikes_all.append(ca1_aligned_spikes)  # Collect for all ripples

        n_frames = [data.shape[0] for data in single_ripple_ca1_lfp]

        # Spikes
        bin_size = 0.01
        bins = np.arange(-window, window + bin_size, bin_size)
        n_bins = len(bins) - 1
        n_neurons = len(ca1_aligned_spikes_all[0])

        spike_counts_all = []
        for ca1_aligned_spikes in ca1_aligned_spikes_all:
            spike_counts = np.zeros((n_neurons, n_bins))
            for neuron_idx, spike_times in enumerate(ca1_aligned_spikes):
                if len(spike_times) > 0:
                    counts, _ = np.histogram(spike_times, bins=bins)
                    spike_counts[neuron_idx, :] = counts
            spike_counts_all.append(spike_counts)

        spike_counts_all = np.stack(spike_counts_all, axis=0)  # (n_ripples, n_neurons, n_bins)
        avg_spike_counts = np.mean(spike_counts_all, axis=0)  # (n_neurons, n_bins)

        # CA1 LFP
        single_ripple_ca1_arr = np.zeros((single_ripple_ca1_lfp[0].shape[1], max(n_frames), len(n_frames))) * np.nan
        for i in range(single_ripple_ca1_arr.shape[2]):
            ripple_data = single_ripple_ca1_lfp[i]
            single_ripple_ca1_arr[:, 0:ripple_data.shape[0], i] = ripple_data.T
        avg_ripple_lfp = np.nanmean(single_ripple_ca1_arr, axis=2)

        # CA1 ripple power
        single_ripple_power_ca1_arr = np.zeros((single_ripple_power_ca1_lfp[0].shape[1], max(n_frames), len(n_frames))) * np.nan
        for i in range(single_ripple_power_ca1_arr.shape[2]):
            ripple_data = single_ripple_power_ca1_lfp[i]
            single_ripple_power_ca1_arr[:, 0:ripple_data.shape[0], i] = ripple_data.T
        avg_ripple_power_lfp = np.nanmean(single_ripple_power_ca1_arr, axis=2)

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 4))
        avg_t = np.arange(avg_ripple_lfp.shape[1]) / sampling_rate - window

        # ripple power
        sns.heatmap(avg_ripple_power_lfp, cbar_kws={'label': 'Ripple-power (zscore)'}, ax=ax1)

        # ca1 lfp
        ca1_colors = plt.cm.Blues(np.linspace(0.3, 1, avg_ripple_lfp.shape[0]))
        for ch in range(avg_ripple_lfp.shape[0]):
            ax0.plot(avg_t, avg_ripple_lfp[ch, :] + ch * np.max(np.std(avg_ripple_lfp, axis=1)) * 2, c=ca1_colors[ch])

        # Spike counts
        im = ax2.imshow(avg_spike_counts, aspect='auto', extent=[bins[0], bins[-1], n_neurons, 0],
                        cmap='hot', interpolation='nearest')
        plt.colorbar(im, ax=ax2, label='Spike count (10ms bin)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Neuron #')
        ax2.set_title('CA1 firing')

        for ax in [ax0, ax1]:
            ax.spines[['right', 'top']].set_visible(False)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Channel #')

        for ax in [ax0, ax1]:
            ax.spines[['right', 'top']].set_visible(False)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Channel #')

        n_ticks = 5
        tick_positions = np.linspace(0, len(avg_t) - 1, n_ticks)
        tick_labels = [f'{avg_t[int(pos)]:.2f}' for pos in tick_positions]
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(tick_labels)
        fig.tight_layout()

        avg_saving_folder = os.path.join(save_path, f'{df.session.unique()[0]}', 'ripple_avg')
        if not os.path.exists(avg_saving_folder):
            os.makedirs(avg_saving_folder)

        for f in ['png', 'pdf']:
            fig.savefig(os.path.join(avg_saving_folder, f'{df.session.unique()[0]}_ripple_avg.{f}'))


def build_table_population_vectors(df, window_sensory, window_ripple,substract_baseline=False):
    cols = ['mouse', 'session', 'start_time', 'trial_type', 'lick_flag', 'context', 'ripples_per_trial', 'rewarded_group']
    sub_df = df[cols].copy()

    ca1_vector_list = []
    second_vector_list = []
    ca1_sensory_list = []
    second_sensory_list = []
    for trial_id in range(len(df)):

        ca1_spikes = df.loc[trial_id].ca1_spike_times
        second_spikes = df.loc[trial_id].secondary_spike_times

        # Add sensory response
        ca1_sensory = build_sensory_population_vectors(all_spikes=ca1_spikes,
                                                       start_time=df.loc[trial_id].start_time,
                                                       delay=window_sensory,baseline_substraction=substract_baseline)
        ca1_sensory_list.append(ca1_sensory)
        second_sensory = build_sensory_population_vectors(all_spikes=second_spikes,
                                                          start_time=df.loc[trial_id].start_time,
                                                          delay=window_sensory, baseline_substraction=substract_baseline)
        second_sensory_list.append(second_sensory)

        # Add ripple content to table
        ripple_times = df.loc[trial_id].ripple_times
        trial_ca1_vector = []
        trial_second_vector = []
        if len(ripple_times) > 0:
            for ripple_time in ripple_times:
                ca1_population_vector = build_ripple_population_vectors(all_spikes=ca1_spikes,
                                                                        ripple_time=ripple_time,
                                                                        delay=window_ripple,baseline_substraction=substract_baseline)
                trial_ca1_vector.append(ca1_population_vector)
                sspbfd_population_vector = build_ripple_population_vectors(all_spikes=second_spikes,
                                                                           ripple_time=ripple_time,
                                                                           delay=window_ripple, baseline_substraction=substract_baseline)
                trial_second_vector.append(sspbfd_population_vector)
        if len(ripple_times) > 1:
            ca1_vector_list.append(np.stack(trial_ca1_vector, axis=0))
            second_vector_list.append(np.stack(trial_second_vector, axis=0))
        else:
            ca1_vector_list.append(trial_ca1_vector)
            second_vector_list.append(trial_second_vector)

    sub_df['ca1_ripple_content'] = ca1_vector_list
    sub_df['second_ripple_content'] = second_vector_list
    sub_df['ca1_sensory'] = ca1_sensory_list
    sub_df['second_sensory'] = second_sensory_list

    return sub_df


def plot_wh_hit_trial_ripple_content(data_folder, task, window_sensory, window_ripple, save_path):
    ripple_target = 'CA1'
    if task == 'fast-learning':
        secondary_target = 'SSp-bfd'
    else:
        secondary_target = 'RSP'
    names = os.listdir(data_folder)
    files = [os.path.join(data_folder, name) for name in names]

    results_dict = {'mouse': [],
                    'rewarded_group': [],
                    'whr': [],
                    'ripple_sensory_rho': []}

    for file_id, file in enumerate(files):
        print(' ')
        print(f'Mouse: {names[file_id][0:5]}')
        file_path = os.path.join(data_folder, file)
        df = pd.read_pickle(file_path)
        new_df = build_table_population_vectors(df=df, window_sensory=window_sensory, window_ripple=window_ripple)

        results_dict['mouse'].append(names[file_id][0:5])
        rew_group = new_df.loc[new_df.mouse == names[file_id][0:5]].rewarded_group.unique()[0]
        results_dict['rewarded_group'].append(rew_group)
        print(f'reward group: {rew_group}')

        # Get global whisker perf
        try:
            whr = new_df.loc[(df.context == 'active') & (new_df.trial_type == 'whisker_trial')].lick_flag.mean()
            results_dict['whr'].append(whr)
        except:
            whr = new_df.loc[(new_df.trial_type == 'whisker_trial')].lick_flag.mean()
            results_dict['whr'].append(whr)
        print(f'WHR: {np.round(whr, 3)}')
        wh_hits = new_df.loc[(new_df.trial_type == 'whisker_trial') &
                             (new_df.context == 'active') &
                             (new_df.lick_flag == 1) & (new_df.ripples_per_trial > 0)]

        if wh_hits.empty:
            results_dict['ripple_sensory_rho'].append(np.nan)
            continue
        print(f'{len(wh_hits)} whisker hit trials')
        results_folder = os.path.join(save_path, wh_hits.session.unique()[0], 'whisker_trial', 'lick')
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        # CA1
        ca1_sensory = np.stack(wh_hits.ca1_sensory, axis=0)
        arrays_to_stack = []
        for item in wh_hits.ca1_ripple_content:
            arrays_to_stack.extend(item)
        ca1_ripple = np.stack(arrays_to_stack, axis=0)
        print(f'{ripple_target}: {ca1_ripple.shape[0]} ripples, {ca1_ripple.shape[1]} units')
        if (ca1_ripple.shape[0] > 1) and (ca1_ripple.shape[1] > 1) & (ca1_sensory.shape[0] > 1):
            fig, (ax0, ax1, ax2) = plt.subplots(1, 3, width_ratios=[4, 1, 4], figsize=(6, 3), sharey=True)
            sns.heatmap(np.transpose(ca1_sensory), ax=ax0)
            sns.heatmap(np.transpose(np.mean(ca1_sensory, axis=0, keepdims=True)), ax=ax1)
            sns.heatmap(np.transpose(ca1_ripple), ax=ax2)
            ax0.set_ylabel('Units')
            ax0.set_xlabel('Whisker hits w/ ripples')
            ax0.set_title(f'{ripple_target} - sensory')
            ax1.set_title(f'sensory AVG')
            ax2.set_xlabel('Ripples post whisker hits')
            ax2.set_title(f'{ripple_target} - ripples')
            sns.despine()
            fig.tight_layout()
            for f in ['png', 'pdf']:
                fig.savefig(os.path.join(results_folder,
                                         f'{wh_hits.session.unique()[0]}_whisker_hits_responses_ripples_{ripple_target}.{f}'))
            plt.close('all')

        # SECOND REGION
        second_sensory = np.stack(wh_hits.second_sensory, axis=0)
        arrays_to_stack = []
        for item in wh_hits.second_ripple_content:
            arrays_to_stack.extend(item)
        second_ripple = np.stack(arrays_to_stack, axis=0)
        print(f'{secondary_target}: {second_ripple.shape[0]} ripples, {second_ripple.shape[1]} units')
        if (second_ripple.shape[0] > 1) and (second_ripple.shape[1] > 1) and (second_sensory.shape[0] > 1):
            fig, (ax0, ax1, ax2) = plt.subplots(1, 3, width_ratios=[4, 1, 4], figsize=(6, 3))
            sns.heatmap(np.transpose(second_sensory), ax=ax0)
            sns.heatmap(np.transpose(np.mean(second_sensory, axis=0, keepdims=True)), ax=ax1)
            sns.heatmap(np.transpose(second_ripple), ax=ax2)
            ax0.set_ylabel('Units')
            ax0.set_xlabel('Whisker hits w/ ripples')
            ax0.set_title(f'{secondary_target} - sensory')
            ax1.set_title(f'sensory AVG')
            ax2.set_xlabel('Ripples post whisker hits')
            ax2.set_title(f'{secondary_target} - ripples')
            sns.despine()
            fig.tight_layout()
            for f in ['png', 'pdf']:
                fig.savefig(os.path.join(results_folder,
                                         f'{wh_hits.session.unique()[0]}_whisker_hits_responses_ripples_{secondary_target}.{f}'))
            plt.close('all')

            avg_sensory_ripple = np.mean(second_sensory, axis=0)
            rho_list = []
            for ripple in range(second_ripple.shape[0]):
                rho_list.append(np.corrcoef(avg_sensory_ripple, second_ripple[ripple, :]))
            results_dict['ripple_sensory_rho'].append(np.nanmean(rho_list))
            print(f'Ripples - Sensory correlation: {np.round(np.nanmean(rho_list), 3)}')
        else:
            results_dict['ripple_sensory_rho'].append(np.nan)
            print(f'Ripples - Sensory correlation: NaN')

    results_df = pd.DataFrame(results_dict)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
    sns.stripplot(results_df, y='ripple_sensory_rho', hue='rewarded_group', hue_order=['R-', 'R+'],
                  palette=['darkmagenta', 'green'], legend=False, dodge=True, ax=ax0)
    sns.boxplot(results_df, y='ripple_sensory_rho', hue='rewarded_group', hue_order=['R-', 'R+'],
                palette=['darkmagenta', 'green'], legend=False, dodge=True, showfliers=False, ax=ax0)
    sns.scatterplot(results_df, x='ripple_sensory_rho', y='whr', hue='rewarded_group', hue_order=['R-', 'R+'],
                    palette=['darkmagenta', 'green'], ax=ax1)
    sns.despine()
    fig.tight_layout()
    results_folder = os.path.join(save_path, 'average_results', 'whisker_trial', 'lick')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    for f in ['png', 'pdf']:
        fig.savefig(os.path.join(results_folder, f'wh_hits_reactivation_to_perf.{f}'))
    plt.close('all')


def plot_hist_ripples_time(data_folder, trial_types, save_path, bin_width=0.5):
    """
    Plot ripple timing distributions for multiple trial types.

    Parameters:
    -----------
    data_folder : str
        Path to folder containing pickle files
    trial_types : list
        List of trial types to analyze (e.g., ['CS+', 'CS-'])
    save_path : str
        Path to save figures
    bin_width : float
        Width of histogram bins in seconds (default: 0.5)
    """
    names = os.listdir(data_folder)
    files = [os.path.join(data_folder, name) for name in names]

    # Load all data
    dfs = []
    for file_id, file in enumerate(files):
        print(f'Mouse: {names[file_id][0:5]}')
        file_path = os.path.join(data_folder, file)
        df = pd.read_pickle(file_path)

        selected_df = df.loc[df.context == 'active']

        cols = ['mouse', 'session', 'context', 'rewarded_group', 'trial_type',
                'start_time', 'lick_time', 'ripple_times', 'trial_duration', 'lick_flag']

        dfs.append(selected_df[cols])

    df_all = pd.concat(dfs).copy()

    # Loop through trial types
    for trial_type in trial_types:
        print(f'\nProcessing trial type: {trial_type}')

        # Filter for current trial type
        df_trial = df_all[df_all.trial_type == trial_type].copy()

        # Separate lick and no-lick trials
        df_lick = df_trial[df_trial.lick_flag == 1].copy()
        df_nolick = df_trial[df_trial.lick_flag == 0].copy()

        # Calculate delays for all trials (stimulus-aligned) - FIXED
        df_trial['ripple_stim_delay'] = df_trial.apply(
            lambda row: np.array(row['ripple_times']) - row['start_time']
            if isinstance(row['ripple_times'], (list, np.ndarray)) else [],
            axis=1
        )

        # Calculate delays for lick trials (lick-aligned) - FIXED
        df_lick['ripple_lick_delay'] = df_lick.apply(
            lambda row: np.array(row['ripple_times']) - row['lick_time']
            if isinstance(row['ripple_times'], (list, np.ndarray)) else [],
            axis=1
        )

        # Calculate delays for no-lick trials (stimulus-aligned) - FIXED
        df_nolick['ripple_stim_delay'] = df_nolick.apply(
            lambda row: np.array(row['ripple_times']) - row['start_time']
            if isinstance(row['ripple_times'], (list, np.ndarray)) else [],
            axis=1
        )

        # Create expanded dataframes with one row per ripple (keeping mouse_id)
        ripple_stim_all_data = []
        ripple_lick_aligned_data = []
        ripple_stim_nolick_data = []

        # Process all trials (stimulus-aligned)
        for idx, row in df_trial.iterrows():
            if isinstance(row['ripple_stim_delay'], (list, np.ndarray)):
                for delay in row['ripple_stim_delay']:
                    ripple_stim_all_data.append({
                        'delay': delay,
                        'rewarded_group': row['rewarded_group'],
                        'mouse': row['mouse']
                    })

        # Process lick trials (lick-aligned)
        for idx, row in df_lick.iterrows():
            if isinstance(row['ripple_lick_delay'], (list, np.ndarray)):
                for delay in row['ripple_lick_delay']:
                    ripple_lick_aligned_data.append({
                        'delay': delay,
                        'rewarded_group': row['rewarded_group'],
                        'mouse': row['mouse']
                    })

        # Process no-lick trials (stimulus-aligned)
        for idx, row in df_nolick.iterrows():
            if isinstance(row['ripple_stim_delay'], (list, np.ndarray)):
                for delay in row['ripple_stim_delay']:
                    ripple_stim_nolick_data.append({
                        'delay': delay,
                        'rewarded_group': row['rewarded_group'],
                        'mouse': row['mouse']
                    })

        df_stim_all = pd.DataFrame(ripple_stim_all_data)
        df_lick_aligned = pd.DataFrame(ripple_lick_aligned_data)
        df_stim_nolick = pd.DataFrame(ripple_stim_nolick_data)

        # Create figure with 6 subplots (2 rows x 3 columns)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=True)

        rewarded_groups = ['R-', 'R+']
        colors = ['darkmagenta', 'green']

        for row_idx, rewarded_group in enumerate(rewarded_groups):
            # Column 0: Stimulus-aligned (ALL trials)
            print('Processing all trials')
            ax_stim_all = axes[row_idx, 0]
            df_group = df_stim_all[df_stim_all.rewarded_group == rewarded_group]
            vals = df_group['delay'].dropna()

            if len(vals) > 1 and vals.max() > vals.min():
                sns.histplot(data=df_group, x='delay', binwidth=bin_width,
                             stat='probability', color=colors[row_idx], ax=ax_stim_all)
                ax_stim_all.axvline(x=0, ymin=0, ymax=1, ls='--', c='darkorange', linewidth=2)
                ax_stim_all.set_xlabel('Ripple-Stimulus Delay (s)', fontsize=11)
                ax_stim_all.set_ylabel('Ripple Probability', fontsize=11)
                ax_stim_all.set_title(f'{rewarded_group} - Stim-aligned (All trials)\n'
                                      f'n_ripples = {len(df_group)}', fontsize=12)
                sns.despine(ax=ax_stim_all)
            else:
                ax_stim_all.text(0.5, 0.5, 'No valid ripple delays',
                                 ha='center', va='center', transform=ax_stim_all.transAxes)

            # Column 1: Lick-aligned (lick trials only)
            print('Processing lick trials lick aligned')
            ax_lick = axes[row_idx, 1]
            df_group = df_lick_aligned[df_lick_aligned.rewarded_group == rewarded_group]
            vals = df_group['delay'].dropna()

            if len(vals) > 1 and vals.max() > vals.min():
                sns.histplot(data=df_group, x='delay', binwidth=bin_width,
                             stat='probability', color=colors[row_idx], ax=ax_lick)
                ax_lick.axvline(x=0, ymin=0, ymax=1, ls='--', c='deeppink', linewidth=2)
                ax_lick.set_xlabel('Ripple-Lick Delay (s)', fontsize=11)
                ax_lick.set_ylabel('Ripple Probability', fontsize=11)
                ax_lick.set_title(f'{rewarded_group} - Lick-aligned (Lick trials)\n'
                                  f'n_ripples = {len(df_group)}', fontsize=12)
                sns.despine(ax=ax_lick)
            else:
                ax_lick.text(0.5, 0.5, 'No valid ripple delays',
                             ha='center', va='center', transform=ax_lick.transAxes)

            # Column 2: Stimulus-aligned (no-lick trials only)
            print('Processing no-lick trials stim aligned')
            ax_stim_nolick = axes[row_idx, 2]
            df_group = df_stim_nolick[df_stim_nolick.rewarded_group == rewarded_group]

            vals = df_group['delay'].dropna()
            if len(vals) > 1 and vals.max() > vals.min():
                sns.histplot(data=df_group, x='delay', binwidth=bin_width,
                             stat='probability', color=colors[row_idx], ax=ax_stim_nolick)
                ax_stim_nolick.axvline(x=0, ymin=0, ymax=1, ls='--', c='darkorange', linewidth=2)
                ax_stim_nolick.set_xlabel('Ripple-Stimulus Delay (s)', fontsize=11)
                ax_stim_nolick.set_ylabel('Ripple Probability', fontsize=11)
                ax_stim_nolick.set_title(f'{rewarded_group} - Stim-aligned (No-lick trials)\n'
                                         f'n_ripples = {len(df_group)}', fontsize=12)
                sns.despine(ax=ax_stim_nolick)
            else:
                ax_stim_nolick.text(0.5, 0.5, 'No valid ripple delays',
                                    ha='center', va='center', transform=ax_stim_nolick.transAxes)

        n_mice = df_trial.mouse.nunique()
        n_lick_trials = len(df_lick)
        n_nolick_trials = len(df_nolick)
        fig.suptitle(f'{trial_type} - All trials\n'
                     f'{n_mice} mice, {n_lick_trials} lick trials, {n_nolick_trials} no-lick trials',
                     fontsize=14, fontweight='bold')
        fig.tight_layout()

        # Save figure
        fig_path = os.path.join(save_path, 'average_results', trial_type)
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        for f in ['png', 'pdf']:
            fig.savefig(os.path.join(fig_path,
                                     f"{trial_type}_ripple_timing_by_group.{f}"),
                        dpi=300, bbox_inches='tight')

        plt.close('all')


def plot_ripple_similarity(data_folder, task, window, save_path):
    ripple_target = 'CA1'
    if task == 'fast-learning':
        secondary_target = 'SSp-bfd'
    else:
        secondary_target = 'RSP'
    names = os.listdir(data_folder)
    files = [os.path.join(data_folder, name) for name in names]

    for file_id, file in enumerate(files):
        print(' ')
        print(f'Mouse: {names[file_id][0:5]}')
        file_path = os.path.join(data_folder, file)
        df = pd.read_pickle(file_path)
        new_df = build_table_population_vectors(df=df, window_sensory=0.050, window_ripple=window)
        new_df = new_df.loc[new_df.context == 'active']
        session = new_df.session.unique()[0]

        # Keep original order for reference
        original_df = new_df.copy()
        original_df['original_idx'] = range(len(original_df))

        # Sort for ordered version
        sorted_new_df = new_df.sort_values(['trial_type', 'lick_flag'])
        sorted_new_df['original_idx'] = new_df.index

        for target in [ripple_target, secondary_target]:
            col_name = 'ca1_ripple_content' if target == ripple_target else 'second_ripple_content'

            # Build ripple data array - each row is one ripple event
            arrays_to_stack = []
            original_indices = []
            for trial_idx, row in original_df.iterrows():
                ripple_list = row[col_name]
                if isinstance(ripple_list, list):
                    for ripple in ripple_list:
                        # Check if ripple is a valid numpy array with data
                        if hasattr(ripple, '__len__') and len(ripple) > 0:
                            arrays_to_stack.append(ripple)
                            original_indices.append(row['original_idx'])

            # Skip if no ripples found
            if len(arrays_to_stack) == 0:
                print(f'No ripples found for {target}, skipping plot')
                continue

            ripple_data = np.stack(arrays_to_stack, axis=0)  # n_ripples x n_units
            ripple_cor = np.corrcoef(ripple_data)  # n_ripples x n_ripples

            # Build ordered ripple data
            ordered_arrays_to_stack = []
            ordered_indices = []
            for trial_idx, row in sorted_new_df.iterrows():
                ripple_list = row[col_name]
                if isinstance(ripple_list, list) and len(ripple_list) > 0:
                    for ripple in ripple_list:
                        ordered_arrays_to_stack.append(ripple)
                        ordered_indices.append(row['original_idx'])

            ordered_ripple_data = np.stack(ordered_arrays_to_stack, axis=0)
            ordered_ripple_cor = np.corrcoef(ordered_ripple_data)

            # Collect block information for labels
            block_info = []
            prev_trial_type = None
            prev_lick_flag = None
            block_start = 0
            cumulative_ripples = 0

            for trial_idx, row in sorted_new_df.iterrows():
                current_trial_type = row['trial_type']
                current_lick_flag = row['lick_flag']

                if (prev_trial_type is not None and
                        (current_trial_type != prev_trial_type or current_lick_flag != prev_lick_flag)):
                    # Save previous block info
                    block_info.append({
                        'start': block_start,
                        'end': cumulative_ripples,
                        'trial_type': prev_trial_type,
                        'lick_flag': prev_lick_flag
                    })
                    block_start = cumulative_ripples

                ripple_list = row[col_name]
                if isinstance(ripple_list, list) and len(ripple_list) > 0:
                    cumulative_ripples += len(ripple_list)

                prev_trial_type = current_trial_type
                prev_lick_flag = current_lick_flag

            # Add the last block
            block_info.append({
                'start': block_start,
                'end': cumulative_ripples,
                'trial_type': prev_trial_type,
                'lick_flag': prev_lick_flag
            })

            # NOW create the figure after we know we have data
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))

            # Original data and correlation
            sns.heatmap(ripple_data.T, ax=axes[0, 0], cmap='viridis', cbar_kws={'label': 'Activity'})
            axes[0, 0].set_title('Original Ripple Data')
            axes[0, 0].set_xlabel('Ripple Events')
            axes[0, 0].set_ylabel('Units')

            sns.heatmap(ripple_cor, ax=axes[1, 0], cmap='coolwarm', center=0,
                        square=True, cbar_kws={'label': 'Correlation'})
            axes[1, 0].set_title('Original Ripple Correlation')
            axes[1, 0].set_xlabel('Ripple Events')
            axes[1, 0].set_ylabel('Ripple Events')

            # Ordered data and correlation with block lines
            sns.heatmap(ordered_ripple_data.T, ax=axes[0, 1], cmap='viridis', cbar_kws={'label': 'Activity'})
            axes[0, 1].set_title('Ordered Ripple Data (by trial type & lick)')
            axes[0, 1].set_xlabel('Ripple Events')
            axes[0, 1].set_ylabel('Units')

            # Add vertical lines for blocks
            for block in block_info[:-1]:  # Don't draw line after last block
                axes[0, 1].axvline(x=block['end'], color='red', linestyle='-', linewidth=1.5, alpha=0.5)

            # Add labels at the top
            for block in block_info:
                center = (block['start'] + block['end']) / 2
                label = f"{block['trial_type']}\n{'Lick' if block['lick_flag'] else 'No Lick'}"
                axes[0, 1].text(center, -0.5, label,
                                ha='center', va='bottom',
                                fontsize=8, rotation=0,
                                transform=axes[0, 1].get_xaxis_transform())

            sns.heatmap(ordered_ripple_cor, ax=axes[1, 1], cmap='coolwarm', center=0,
                        square=True, cbar_kws={'label': 'Correlation'})
            axes[1, 1].set_title('Ordered Ripple Correlation')
            axes[1, 1].set_xlabel('Ripple Events')
            axes[1, 1].set_ylabel('Ripple Events')

            # Add lines for blocks in correlation matrix (yellow with reduced transparency)
            for block in block_info[:-1]:
                axes[1, 1].axhline(y=block['end'], color='yellow', linestyle='-', linewidth=1.5, alpha=0.5)
                axes[1, 1].axvline(x=block['end'], color='yellow', linestyle='-', linewidth=1.5, alpha=0.5)

            # Add labels at the top of correlation plot
            for block in block_info:
                center = (block['start'] + block['end']) / 2
                label = f"{block['trial_type']}\n{'Lick' if block['lick_flag'] else 'No Lick'}"
                axes[1, 1].text(center, -0.5, label,
                                ha='center', va='bottom',
                                fontsize=8, rotation=0,
                                transform=axes[1, 1].get_xaxis_transform())

            plt.tight_layout()
            # Save figure
            results_folder = os.path.join(save_path, 'average_results', 'ripple_similarity', f'{session}')
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
            for f in ['png', 'pdf']:
                save_file = os.path.join(results_folder, f'{session}_{target}_ripple_similarity.{f}')
                plt.savefig(save_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f'Saved {target} plot / Total ripples: {len(arrays_to_stack)}')


def plot_ripple_frequency_over_session(data_folder, block_size, save_path):
    names = os.listdir(data_folder)
    files = [os.path.join(data_folder, name) for name in names]

    # ======== Load and preprocess all mice data ========
    dfs = []

    for file_id, file in enumerate(files):

        mouse_id = names[file_id][:5]
        print(f"Mouse: {mouse_id}")

        df = pd.read_pickle(file)

        cols = [
            'mouse', 'session', 'ripples_per_trial',
            'rewarded_group', 'trial_duration', 'trial_type', 'lick_flag'
        ]
        dfs.append(df.loc[(df.context == 'active'), cols])

    df_all = pd.concat(dfs).copy()

    m_dfs = []
    for mouse in df_all.mouse.unique():
        m_df = df_all.loc[df_all.mouse == mouse].copy()
        m_df['outcome_w'] = m_df.loc[(m_df.trial_type == 'whisker_trial')]['lick_flag']
        m_df['outcome_a'] = m_df.loc[(m_df.trial_type == 'auditory_trial')]['lick_flag']
        m_df['outcome_n'] = m_df.loc[(m_df.trial_type == 'no_stim_trial')]['lick_flag']
        m_df['block_index'] = m_df.groupby('session').cumcount() // block_size

        for outcome, new_col in zip(['outcome_w', 'outcome_a', 'outcome_n'], ['hr_w', 'hr_a', 'hr_n']):
            m_df[new_col] = m_df.groupby(['mouse', 'session', 'block_index'],
                                         as_index=False)[outcome].transform('mean')

        m_df['block_trial_duration'] = m_df.groupby(['mouse', 'session', 'block_index', 'trial_type', 'lick_flag'],
                                                    as_index=False)['trial_duration'].transform('sum')

        m_df['block_n_ripples'] = m_df.groupby(['mouse', 'session', 'block_index', 'trial_type', 'lick_flag'],
                                               as_index=False)['ripples_per_trial'].transform('sum')

        m_df['block_ripples_fz'] = np.round((m_df['block_n_ripples'] / m_df['block_trial_duration']) * 60, 3)

        m_df = m_df.reset_index(drop=True)

        block_m_df = m_df[int(block_size / 2)::block_size]

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        for hr, c in zip(['hr_w', 'hr_n', 'hr_a'], ['darkorange', 'black', 'royalblue']):
            sns.lineplot(block_m_df, x='block_index', y=hr, color=c, label=f'{hr}', ax=ax0)
        ax0.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax0.set_ylim(0, 1.05)
        ax0.set_ylabel('Hit rate')

        # Plot ripple frequency for each trial type / lick flag combination
        trial_types = ['whisker_trial', 'no_stim_trial', 'auditory_trial']
        colors = ['darkorange', 'black', 'royalblue']
        valid_blocks = block_m_df['block_index'].unique()
        for tt, c in zip(trial_types, colors):
            # Hit trials - filter by valid blocks
            hit_data = m_df[(m_df['trial_type'] == tt) &
                            (m_df['lick_flag'] == 1) &
                            (m_df['block_index'].isin(valid_blocks))]
            sns.lineplot(hit_data, x='block_index', y='block_ripples_fz',
                         color=c, linestyle='-', ax=ax1, label=f'{tt}_hit')
            # Miss trials - filter by valid blocks
            miss_data = m_df[(m_df['trial_type'] == tt) &
                             (m_df['lick_flag'] == 0) &
                             (m_df['block_index'].isin(valid_blocks))]
            sns.lineplot(miss_data, x='block_index', y='block_ripples_fz',
                         color=c, linestyle='--', ax=ax1, label=f'{tt}_miss')
        ax1.set_ylabel('Ripple (min-1)')
        ax1.set_xlabel('Block index')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        sns.despine()
        fig.suptitle(f'{m_df.session.unique()[0]}')
        fig.tight_layout()

        result_folder = os.path.join(save_path, m_df.session.unique()[0], 'behavior')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        for f in ['png', 'pdf']:
            fig.savefig(os.path.join(result_folder, f'{m_df.session.unique()[0]}_bhv_ripple_plot.{f}'))
        plt.close()

        m_dfs.append(m_df)

    m_dfs = pd.concat(m_dfs)

    for r_group in m_dfs.rewarded_group.unique():
        group_df = m_dfs.loc[m_dfs.rewarded_group == r_group].copy()

        # Average hit rates across mice
        hr_avg = group_df.groupby(['mouse', 'block_index'], as_index=False)[['hr_w', 'hr_a', 'hr_n']].mean()

        # Average ripple frequencies across mice for each trial_type and lick_flag
        ripple_avg = group_df.groupby(['mouse', 'block_index', 'trial_type', 'lick_flag'],
                                      as_index=False)['block_ripples_fz'].mean()

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # Plot average hit rates
        hr_avg = hr_avg.loc[hr_avg.block_index <= 10]
        for hr, c, label in zip(['hr_w', 'hr_n', 'hr_a'],
                                ['darkorange', 'black', 'royalblue'],
                                ['whisker', 'no_stim', 'auditory']):
            sns.lineplot(hr_avg, x='block_index', y=hr, errorbar=('ci', 95), color=c, label=f'{label}_hit', ax=ax0)
        ax0.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax0.set_ylim(0, 1.05)
        ax0.set_ylabel('Hit rate (avg)')
        ax0.set_title(f'Rewarded Group: {r_group}')

        # Plot average ripple frequencies
        trial_types = ['whisker_trial', 'no_stim_trial', 'auditory_trial']
        colors = ['darkorange', 'black', 'royalblue']
        ripple_avg = ripple_avg.loc[ripple_avg.block_index <= 10]
        for tt, c in zip(trial_types, colors):
            hit_data = ripple_avg[(ripple_avg['trial_type'] == tt) &
                                  (ripple_avg['lick_flag'] == 1)]
            sns.lineplot(hit_data, x='block_index', y='block_ripples_fz',
                         color=c, linestyle='-', errorbar=('ci', 95), ax=ax1, label=f'{tt}_hit')
            miss_data = ripple_avg[(ripple_avg['trial_type'] == tt) &
                                   (ripple_avg['lick_flag'] == 0)]
            sns.lineplot(miss_data, x='block_index', y='block_ripples_fz',
                         color=c, linestyle='--', errorbar=('ci', 95), ax=ax1, label=f'{tt}_miss')
        ax1.set_ylabel('Ripple (min-1) (avg)')
        ax1.set_xlabel('Block index')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        sns.despine()
        fig.tight_layout()

        result_folder = os.path.join(save_path, 'average_results', 'behavior')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        for f in ['png', 'pdf']:
            plt.savefig(f"{result_folder}/{r_group}_avg_ripple_frequency.{f}", dpi=300, bbox_inches='tight')
        plt.close()

