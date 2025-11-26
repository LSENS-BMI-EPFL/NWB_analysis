import os
from pathlib import Path
import scipy as sci
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface.full as si
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


def get_lfp_recordings(data_folder, mouse, session, stream):
    if mouse[0:2] == 'PB':
        new_data_folder = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\publications\2026\2026_Bech_Dard_eLife\2026_Bech_Dard_eLife_data\raw_data"
        path = os.path.join(new_data_folder, mouse, 'Recording', 'Ephys', session)
        if not os.path.exists(path):
            path = os.path.join(data_folder, mouse, 'Recording', session, 'Ephys')
    else:
        path = os.path.join(data_folder, mouse, 'Recording', session, 'Ephys')

    if not os.path.exists(path):
        return None

    g_index = os.listdir(path)[0]
    full_path = os.path.join(path, f'{g_index}')

    if not os.path.exists(full_path):
        full_path = os.path.join(path, f'{session}')
        if not os.path.exists(full_path):
            full_path = os.path.join(path, f'{session}_g0')
            if not os.path.exists(full_path):
                full_path = os.path.join(path, f'{mouse}_g0')
                if not os.path.exists(full_path):
                    full_path = os.path.join(path, f'{mouse}_g1')
                    if not os.path.exists(full_path):
                        return None
    try:
        rec = si.read_spikeglx(full_path, stream_name=f"imec{stream}.lf")
        print("Using LF stream")

    except:
        print("LF stream not found, using AP stream")
        rec = si.read_spikeglx(full_path, stream_name=f"imec{stream}.ap")
        rec = si.bandpass_filter(rec, freq_min=0.5, freq_max=500, margin_ms=5000)
        rec = si.resample(rec, resample_rate=2500, margin_ms=2000)

    return rec


def lfp_filter(data, fs, freq_min=150, freq_max=200):
    nyq = 0.5 * fs
    low = freq_min / nyq
    high = freq_max / nyq
    b, a = sci.signal.butter(3, [low, high], btype='band')
    return sci.signal.filtfilt(b, a, data, axis=0)


def ripple_detect(ca1_sw_lfp, ca1_ripple_lfp, sampling_rate, threshold, sharp_filter=False, sharp_delay=0.070):
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

    # Detect ripples on consensus signal
    ripple_peak_frames, _ = sci.signal.find_peaks(np.median(ripple_z, axis=1), height=threshold,
                                                  distance=int(0.05 * sampling_rate))

    # Detect sharp waves
    sw_peak_frames, _ = sci.signal.find_peaks(np.median(sw_z, axis=1), height=threshold,
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
                    time_vec, ripple_times, best_channel, wh_trace, tongue_trace, wh_ts,
                    ca1_spikes, sspbfd_spikes, offset, session_id, start_id, start_ts, ripple_id,
                    ripple_target, secondary_target, trial_selection,
                    fig_size, save_path):

    fig, axes = plt.subplots(8, 1, figsize=fig_size, sharex=True)

    for i in range(ca1lfp.shape[1]):
        axes[0].plot(time_vec, ca1lfp[:, i] + i * offset)

    for i in range(ca_high_filt.shape[1]):
        axes[1].plot(time_vec, ca_high_filt[:, i] + i * max(ca_high_filt[:, i]))

    for i in range(ca1_ripple_power.shape[1]):
        axes[2].plot(time_vec, ca1_ripple_power[:, i] + i * 4)

    for i in range(sspbfdlfp.shape[1]):
        axes[7].plot(time_vec, sspbfdlfp[:, i] + i * offset)

    for i in range(sspbfd_spindle_filt.shape[1]):
        axes[6].plot(time_vec, sspbfd_spindle_filt[:, i] + i * max(sspbfd_spindle_filt[:, i]))

    if type(ripple_times) != np.float64:
        axes[2].scatter(x=ripple_times, y=[-5] * len(ripple_times), marker='o', c='k')
    else:
        axes[2].scatter(x=ripple_times, y=[-5], marker='o', c='k')

    axes[0].scatter(time_vec[0] - (time_vec[1] - time_vec[0]) * 0.8, best_channel * offset, marker='*', c='k')
    axes[3].eventplot(ca1_spikes, colors='black', linewidths=0.8)
    axes[4].eventplot(sspbfd_spikes, colors='black', linewidths=0.8)

    if len(wh_trace) > 0 and len(wh_ts) > 0:
        axes[5].plot(wh_ts, wh_trace, c='darkorange')
        wh_speed = np.abs(np.diff(wh_trace))
        axes[5].plot(wh_ts[1:], wh_speed, c='darkred')
    if len(tongue_trace) > 0 and len(wh_ts) > 0:
        axes[5].plot(wh_ts, 3 * tongue_trace, c='deeppink')

    for ax in axes.flatten():
        ax.spines[['right', 'top']].set_visible(False)

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


def build_ripple_population_vectors(all_spikes, ripple_time, delay):
    ripple_spikes = [
        spikes[(spikes >= ripple_time - delay) & (spikes <= ripple_time + delay)]
        for spikes in all_spikes
    ]
    population_vector = [len(spikes) for spikes in ripple_spikes]

    return population_vector


def build_sensory_population_vectors(all_spikes, start_time, delay):
    ripple_spikes = [
        spikes[(spikes >= start_time) & (spikes <= start_time + delay)]
        for spikes in all_spikes
    ]
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
    if len(ids_list) >= 15:
        ids_list = ids_list[::len(ids_list) // 15]
    channels = rec.get_channel_ids()[ids_list]

    return channels


def plot_ripple_frequency_fastlearning(data_folder, trial_type, lick_flag, save_path):
    names = os.listdir(data_folder)
    files = [os.path.join(data_folder, name) for name in names]
    dfs = []
    wh_perf = []
    for file_id, file in enumerate(files):
        print(f'Mouse: {names[file_id][0:5]}')
        file_path = os.path.join(data_folder, file)
        df = pd.read_pickle(file_path)

        # Get global whisker perf
        try:
            wh_perf.append(df.loc[(df.context == 'active') & (df.trial_type == 'whisker_trial')].lick_flag.mean())
        except:
            wh_perf.append(df.loc[(df.trial_type == 'whisker_trial')].lick_flag.mean())

        # Global ripple count figure
        if lick_flag is not None:
            selected_df = df.loc[(df.context == 'active') &
                                 (df.trial_type == trial_type) &
                                 (df.lick_flag == lick_flag)]
        else:
            selected_df = df.loc[(df.context == 'active') & (df.trial_type == trial_type)]

        cols = ['mouse', 'session', 'ripples_per_trial', 'rewarded_group', 'trial_duration']

        dfs.append(selected_df[cols])

    df_to_plot = pd.concat(dfs).copy()

    # Plot
    grouped_df = df_to_plot.groupby(['mouse', 'session', 'rewarded_group'], as_index=False).sum()
    grouped_df['ripple_fz'] = np.round((grouped_df['ripples_per_trial'] / grouped_df['trial_duration']) * 60, 3)
    grouped_df['whr'] = wh_perf
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 3))
    sns.stripplot(grouped_df, hue='rewarded_group', hue_order=['R-', 'R+'], y='ripple_fz',
                  palette=['darkmagenta', 'green'], dodge=True, legend=False, ax=ax0)
    sns.boxplot(grouped_df, hue='rewarded_group', hue_order=['R-', 'R+'], y='ripple_fz',
                palette=['darkmagenta', 'green'], showfliers=False, ax=ax0)
    sns.scatterplot(grouped_df, x='ripple_fz', y='whr', hue='rewarded_group',
                    hue_order=['R-', 'R+'], palette=['darkmagenta', 'green'], legend=False, ax=ax1)
    sns.despine()
    fig.suptitle(f'Ripple occurrence in {trial_type} trial')
    fig.tight_layout()
    if lick_flag is None:
        fig_path = os.path.join(save_path, 'average_results', f'{trial_type}')
    else:
        if lick_flag == 0:
            fig_path = os.path.join(save_path, f'{trial_type}', 'nolick')
        else:
            fig_path = os.path.join(save_path, f'{trial_type}', 'lick')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    for f in ['png', 'pdf']:
        fig.savefig(os.path.join(fig_path, f"{trial_type}_ripple_occurrence.{f}"))
    plt.close('all')


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
                                tongue_trace=df.loc[trial_id].tongue_trace,
                                wh_ts=df.loc[trial_id].dlc_trial_ts,
                                ca1_spikes=df.loc[trial_id].ca1_spike_times,
                                sspbfd_spikes=df.loc[trial_id].secondary_spike_times,
                                offset=50, session_id=df.loc[trial_id].session,
                                start_id=trial_id,
                                start_ts=df.loc[trial_id].start_time,
                                ripple_id=None,
                                ripple_target=ripple_target,
                                secondary_target=secondary_target, trial_selection=trial_type,
                                fig_size=(16, 22), save_path=result_folder)


def plot_single_event_data(data_folder, task, window, save_path):
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

                    # Plot zoomed view on ripple event
                    plot_lfp_custom(ca1lfp=df.loc[trial_id].ca1_lfp[zoom_start: zoom_stop, :],
                                    ca_high_filt=df.loc[trial_id].ca1_ripple_band_flp[zoom_start: zoom_stop, :],
                                    ca1_ripple_power=df.loc[trial_id].ca1_ripple_power[zoom_start: zoom_stop, :],
                                    sspbfdlfp=df.loc[trial_id].secondary_lfp[zoom_start: zoom_stop, :],
                                    sspbfd_spindle_filt=df.loc[trial_id].secondary_spindle_band_lfp[zoom_start: zoom_stop, :],
                                    time_vec=time_vec,
                                    ripple_times=df.loc[trial_id].ripple_times,
                                    best_channel=df.loc[trial_id].ca1_ripple_best_ch,
                                    wh_trace=wh_angle_zoom,
                                    tongue_trace=tongue_distance_zoom,
                                    wh_ts=dlc_ts_zoom,
                                    ca1_spikes=ca1_filtered_spikes,
                                    sspbfd_spikes=second_filtered_spikes,
                                    offset=50, session_id=df.loc[trial_id].session,
                                    start_id=trial_id,
                                    start_ts=df.loc[trial_id].start_time,
                                    ripple_id=ripple_id,
                                    ripple_target=ripple_target,
                                    secondary_target=secondary_target, trial_selection=trial_type,
                                    fig_size=(6, 22), save_path=result_folder)


def build_table_population_vectors(df, window):
    cols = ['mouse', 'session', 'start_time', 'trial_type', 'lick_flag', 'context', 'ripples_per_trial', 'rewarded_group']
    sub_df = df[cols].copy()

    ca1_vector_list = []
    second_vector_list = []
    ca1_sensory_list = []
    second_sensory_list = []
    for trial_id in range(len(df)):
        print(f'Trial: {trial_id}, type: {df.loc[trial_id].trial_type}, '
              f'lick: {df.loc[trial_id].lick_flag}')

        ca1_spikes = df.loc[trial_id].ca1_spike_times
        second_spikes = df.loc[trial_id].secondary_spike_times

        # Add sensory response
        ca1_sensory = build_sensory_population_vectors(all_spikes=ca1_spikes,
                                                       start_time=df.loc[trial_id].start_time,
                                                       delay=window)
        ca1_sensory_list.append(ca1_sensory)
        second_sensory = build_sensory_population_vectors(all_spikes=second_spikes,
                                                          start_time=df.loc[trial_id].start_time,
                                                          delay=window)
        second_sensory_list.append(second_sensory)

        # Add ripple content to table
        ripple_times = df.loc[trial_id].ripple_times
        trial_ca1_vector = []
        trial_second_vector = []
        if len(ripple_times) > 0:
            for ripple_time in ripple_times:
                ca1_population_vector = build_ripple_population_vectors(all_spikes=ca1_spikes,
                                                                        ripple_time=ripple_time,
                                                                        delay=window)
                trial_ca1_vector.append(ca1_population_vector)
                sspbfd_population_vector = build_ripple_population_vectors(all_spikes=second_spikes,
                                                                           ripple_time=ripple_time,
                                                                           delay=window)
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


def plot_trial_ripple_content(data_folder, task, window, save_path):
    names = os.listdir(data_folder)
    files = [os.path.join(data_folder, name) for name in names]
    for file_id, file in enumerate(files):
        print(f'Mouse: {names[file_id][0:5]}')
        file_path = os.path.join(data_folder, file)
        df = pd.read_pickle(file_path)
        new_df = build_table_population_vectors(df=df, window=window)

        wh_hits = new_df.loc[(new_df.trial_type == 'whisker_trial') &
                             (new_df.context == 'active') &
                             (new_df.lick_flag == 1) & (new_df.ripples_per_trial > 0)]

        ca1_sensory = np.stack(wh_hits.ca1_sensory, axis=0)
        arrays_to_stack = []
        for item in wh_hits.ca1_ripple_content:
            if isinstance(item, list):
                for i in item:
                    arrays_to_stack.append(item)  # Add all arrays from the list
            else:
                arrays_to_stack.append(item)  # Add single array
        ca1_ripple = np.concatenate(arrays_to_stack, axis=0)
        fig, (ax0, ax1) = plt.subplots(1, 2)
        sns.heatmap(np.transpose(ca1_sensory), ax=ax0)
        sns.heatmap(np.transpose(ca1_ripple), ax=ax1)

        second_sensory = np.stack(wh_hits.second_sensory, axis=0)
        arrays_to_stack = []
        for item in wh_hits.second_ripple_content:
            if isinstance(item, list):
                for i in item:
                    arrays_to_stack.append(item)  # Add all arrays from the list
            else:
                arrays_to_stack.append(item)  # Add single array
        second_ripple = np.concatenate(arrays_to_stack, axis=0)
        fig, (ax0, ax1) = plt.subplots(1, 2)
        sns.heatmap(np.transpose(second_sensory), ax=ax0)
        sns.heatmap(np.transpose(second_ripple), ax=ax1)


