import os
import scipy as sci
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import spikeinterface.full as si
from sklearn.manifold import TSNE
from nwb_utils.utils_misc import find_nearest


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
                    time_vec, ripple_times, best_channel, wh_trace, wh_ts,
                    ca1_spikes, sspbfd_spikes, offset, session_id, catch_id, catch_ts, ripple_id,
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

    axes[0].scatter(time_vec[0] - 0.050, best_channel * offset, marker='*', c='k')
    axes[3].eventplot(ca1_spikes, colors='black', linewidths=0.8)
    axes[4].eventplot(sspbfd_spikes, colors='black', linewidths=0.8)

    if len(wh_trace) > 0 and len(wh_ts) > 0:
        axes[5].plot(wh_ts, wh_trace, c='orange')
        wh_speed = np.abs(np.diff(wh_trace))
        axes[5].plot(wh_ts[1:], wh_speed, c='red')

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
    fig.suptitle(f'{session_id} Catch #{catch_id} at t = {catch_ts} s')
    fig.tight_layout()

    s_path = os.path.join(save_path, session_id)
    if not os.path.exists(s_path):
        os.makedirs(s_path)
    for f in ['pdf', 'png']:
        if ripple_id is not None:
            fig.savefig(os.path.join(s_path, f'catch_{catch_id}_{ripple_id}.{f}'), dpi=400)
        else:
            fig.savefig(os.path.join(s_path, f'catch_{catch_id}.{f}'), dpi=400)
    plt.close('all')


def build_ripple_population_vectors(all_spikes, ripple_time, delay):
    ripple_spikes = [
        spikes[(spikes >= ripple_time - delay) & (spikes <= ripple_time + delay)]
        for spikes in all_spikes
    ]
    population_vector = [len(spikes) for spikes in ripple_spikes]

    return population_vector


def cluster_ripple_content(ca1_ripple_array, ssp_ripple_array, session, group, context_blocks, save_path):
    # Cluster on CA1 ripple content
    if ca1_ripple_array.shape[1] > 4:
        ca1_tsne_results = TSNE(n_components=2, learning_rate='auto',
                                init='random', perplexity=3).fit_transform(ca1_ripple_array)
    else:
        ca1_tsne_results = np.zeros((ca1_ripple_array.shape[0], 2))

    # Cluster on second region ripple content
    if ssp_ripple_array.shape[1] > 4:
        ssp_tsne_results = TSNE(n_components=2, learning_rate='auto',
                                init='random', perplexity=3).fit_transform(ssp_ripple_array)
    else:
        ssp_tsne_results = np.zeros((ssp_ripple_array.shape[0], 2))

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # Population vectors plot
    if ca1_ripple_array.shape[1] > 0:
        sns.heatmap(np.transpose(ca1_ripple_array), cmap='viridis', ax=axes[0, 0])
    if ssp_ripple_array.shape[1] > 0:
        sns.heatmap(np.transpose(ssp_ripple_array), cmap='viridis', ax=axes[0, 1])
    for ax in axes[0, :].flatten():
        ax.set_xlabel('Ripple events')
        ax.set_ylabel('Units')

    # t-SNE results
    if len(np.unique(context_blocks)) > 1:
        color = ['darkmagenta' if i == 0 else 'green' for i in context_blocks]
        cmap = None
    else:
        color = range(len(context_blocks))
        cmap = 'Blues'
    axes[1, 0].scatter(ca1_tsne_results[:, 0], ca1_tsne_results[:, 1], c=color,
                       s=100, vmin=0, vmax=len(ca1_tsne_results)-1, cmap=cmap)
    axes[1, 1].scatter(ssp_tsne_results[:, 0], ssp_tsne_results[:, 1], c=color,
                       s=100, vmin=0, vmax=len(ssp_tsne_results)-1, cmap=cmap)
    axes[0, 0].set_title('CA1 ripple content')
    axes[0, 1].set_title('SSp-bfd ripple content')
    for ax in axes[1, :].flatten():
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



