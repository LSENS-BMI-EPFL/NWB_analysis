import yaml
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from nwb_utils.utils_misc import find_nearest
from nwb_wrappers import nwb_reader_functions as nwb_read


config_file = r"Z:\z_LSENS\Share\Pol_Bech\Session_list\context_sessions_gcamp_expert.yaml"
# config_file = "//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Robin_Dard/group.yaml"
with open(config_file, 'r', encoding='utf8') as stream:
    config_dict = yaml.safe_load(stream)
# sessions = config_dict['NWB_CI_LSENS']['Context_expert_sessions']
# sessions = config_dict['NWB_CI_LSENS']['Context_contrast_expert']
# sessions = config_dict['NWB_CI_LSENS']['context_contrast_widefield']
# files = [session[1] for session in sessions]
nwb_files = config_dict['Session path']

root_folder = r'Z:\analysis\Robin_Dard\Pop_results\Context_behaviour\test_PCA_analysis'
output_folder = os.path.join(root_folder, '20241018', 'full_session_pca_and_proj_baseline_kept')
save_fig = True
rrs_keys = ['ophys', 'brain_area_fluorescence', 'dff0_traces']
components_to_plot = [0, 4]
remove_baseline = False
n_frames_before_stim = 100
n_frames_after_stim = 25
session_data_dict = dict()
color_dict = {'RWD-Wh-hit': ['darkgreen', 'limegreen'], 'RWD-Wh-miss': ['black', 'dimgrey'],
              'NN-RWD-Wh-hit': ['darkred', 'red'], 'NN-RWD-Wh-miss': ['darkblue', 'blue']}

for file in nwb_files:
    session = nwb_read.get_session_id(file)
    mouse = session[0:5]
    print(' ')
    print(f"Mouse: {mouse}, Session: {session}")
    saving_folder = os.path.join(output_folder, f'{mouse}', f'{session}')
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    # Extract neural data
    traces = nwb_read.get_roi_response_serie_data(nwb_file=file, keys=rrs_keys)
    rrs_ts = nwb_read.get_roi_response_serie_timestamps(nwb_file=file, keys=rrs_keys)
    if len(rrs_ts) > traces.shape[1]:
        rrs_ts = rrs_ts[0: traces.shape[1]]

    # Extract area names
    area_dict = nwb_read.get_cell_indices_by_cell_type(nwb_file=file, keys=rrs_keys)
    sorted_areas = sorted(area_dict, key=area_dict.get)

    # Build pd dataframe
    df = pd.DataFrame(np.transpose(traces), index=rrs_ts, columns=sorted_areas)

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Run PCA
    pca = PCA(n_components=8)  # Choose the number of components you want
    results = pca.fit(scaled_data)
    principal_components = pca.transform(scaled_data)

    # Create a DataFrame for the principal components
    pca_df = pd.DataFrame(data=principal_components, index=df.index, columns=[f'PC{i}' for i in range(8)])
    n_tot_f = len(pca_df)

    # Get the time of rewarded context
    epochs = nwb_read.get_behavioral_epochs_names(nwb_file=file)
    rewarded_times = nwb_read.get_behavioral_epochs_times(nwb_file=file, epoch_name='rewarded')
    non_rewarded_times = nwb_read.get_behavioral_epochs_times(nwb_file=file, epoch_name='non-rewarded')

    # Assign context to each imaging frame
    rewarded_img_frame = [0 for i in range(len(rrs_ts))]
    for i, ts in enumerate(rrs_ts):
        for epoch in range(rewarded_times.shape[1]):
            if (ts >= rewarded_times[0][epoch]) and (ts <= rewarded_times[1][epoch]):
                rewarded_img_frame[i] = 1
                break
            else:
                continue

    # Get the jaw opening
    dlc_ts = nwb_read.get_dlc_timestamps(nwb_file=file, keys=['behavior', 'BehavioralTimeSeries'])
    if dlc_ts is not None:
        jaw_angle = nwb_read.get_dlc_data(nwb_file=file, keys=['behavior', 'BehavioralTimeSeries'], part='jaw_angle')
        dlc_frames = []
        for img_ts in rrs_ts:
            dlc_frames.append(find_nearest(array=dlc_ts[0], value=img_ts, is_sorted=True))
        aligned_jaw_angle = jaw_angle[dlc_frames]
        jaw_filt = gaussian_filter1d(input=aligned_jaw_angle, sigma=20, axis=-1, order=0)
    else:
        jaw_filt = np.empty(len(rewarded_img_frame))
        jaw_filt[:] = np.nan

    # ----------------- FIGURES ------------------- #
    # Figure 1 : loadings and explained variance
    color = iter(cm.rainbow(np.linspace(0, 1, 8)))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    for i in range(components_to_plot[0], components_to_plot[1]):
        c = next(color)
        ax0.plot(np.abs(results.components_[i]), label=f'Principal Component {i}', color=c, marker='o')
        ax0.set_xticklabels(['-10'] + sorted_areas)
        ax0.set_ylabel('PC Loadings (absolute value)')
        ax0.set_xlabel('Area')
    ax0.legend(loc='upper left')
    ax0.set_ylim([-0.05, 1])
    ax1.plot(np.cumsum(results.explained_variance_ratio_), marker='o')
    ax1.set_ylabel('Explained variance')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylim([0.50, 1.1])
    ax1.axhline(y=0.95, linestyle='--', color='black')
    ax0.spines[['right', 'top']].set_visible(False)
    ax1.spines[['right', 'top']].set_visible(False)
    fig.suptitle(f'{session}')
    fig.tight_layout()
    if save_fig:
        fig.savefig(os.path.join(saving_folder, f'{session}_PC_loadings_and_variance.png'))
    else:
        plt.show()

    # Figure 2 : PC time courses with jaw opening and context
    color = iter(cm.rainbow(np.linspace(0, 1, 8)))
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(15, 9))
    for i in range(components_to_plot[0], components_to_plot[1]):
        c = next(color)
        axes[i].plot(range(n_tot_f), pca_df[f'PC{i}'], label=f'Principal Component {i}', color=c)
        axes[i].set_ylabel(f'PC{i}')
        axes[i].fill_between(range(n_tot_f), 0, 1, where=np.array(rewarded_img_frame).astype(bool), alpha=0.4,
                             color='green', transform=axes[i].get_xaxis_transform(), label='RWD context')
        axes[i].plot(range(n_tot_f), jaw_filt, color='red', label='Jaw angle filtered')
        axes[i].legend(loc='upper right')
        axes[i].spines[['right', 'top']].set_visible(False)
    axes[3].set_xlabel('Time (in frames)')
    axes[0].set_title('Temporal PCA - Principal Components Over Time')
    fig.suptitle(f'{session}')
    fig.tight_layout()
    if save_fig:
        fig.savefig(os.path.join(saving_folder, f'{session}_PC_time_courses.png'))
    else:
        plt.show()

    # ---------------------------- Try to link all this to behavior or context -------------------------- #
    # BEHAVIOR
    print(' ')
    print('Link PC to behavior')
    if len(jaw_filt[np.where(~np.isnan(jaw_filt))[0]]) == len(pca_df[f'PC0']):
        # With filtered lick trace
        model = LinearRegression()
        r_squared_values = []
        corr_values = []
        for i in range(components_to_plot[0], components_to_plot[1]):
            model.fit(jaw_filt.reshape(-1, 1), pca_df[f'PC{i}'])
            r_squared = model.score(jaw_filt.reshape(-1, 1), pca_df[f'PC{i}'])
            print(f"LinearRegression: R-squared for PC{i} vs aligned_jaw_angle_filt: {r_squared:.3f}")
            r_squared_values.append(r_squared)
            correlation, p_value = pearsonr(jaw_filt, pca_df[f'PC{i}'])
            print(f"Pearson: Correlation between PC{i} and aligned_jaw_angle_filt: {correlation:.3f}, p-value: {p_value}")
            corr_values.append(correlation)
    else:
        print('DLC frames missing')
        r_squared_values = None
        corr_values = None

    # CONTEXT
    print(' ')
    print('Link PC to context')
    # Know if each imaging frames is 'rewarded' or not

    # Create a pd DataFrame
    df = pd.DataFrame({'TPC0': pca_df[f'PC0'], 'TPC1': pca_df[f'PC1'],
                       'TPC2': pca_df[f'PC2'], 'TPC3': pca_df[f'PC3'],
                       'Context': rewarded_img_frame})

    # Figure : Boxplot for TPCs and context
    fig, axes = plt.subplots(1, 4, figsize=(12, 6), sharey=True)
    for i in range(components_to_plot[0], components_to_plot[1]):
        sns.boxplot(x='Context', y=f'TPC{i}', data=df, ax=axes[i], palette=['red', 'green'], showfliers=False)
        axes[i].set_ylabel(f'TPC{i}')
        axes[i].spines[['right', 'top']].set_visible(False)
    fig.suptitle(f'{session}')
    fig.tight_layout()
    if save_fig:
        fig.savefig(os.path.join(saving_folder, f'{session}_PC_time_course_vs_context.png'))
    else:
        plt.show()

    # Logistic regression
    accuracies = []
    for i in range(components_to_plot[0], components_to_plot[1]):
        model = LogisticRegression()
        scores = cross_val_score(model, pca_df[f'PC{i}'].values[:].reshape(-1, 1), rewarded_img_frame, cv=5)
        print(f"Classification accuracy: {np.mean(scores):.3f}")
        accuracies.append(np.mean(scores))

    # Rapid estimate of 'lick' and 'context' PC
    context_pc = np.argmax(accuracies)
    if r_squared_values is not None and corr_values is not None:
        lick_pc = np.argmax(r_squared_values)
    else:
        if context_pc == 0:
            lick_pc = 1
        else:
            lick_pc = 0
    if context_pc == lick_pc:
        print('Best PC is the same for lick and context, take second best for context')
        context_pc = np.argsort(accuracies)[1]
    print(f'Estimate of best PC: PC#{lick_pc} for lick, PC#{context_pc} for context')

    # ------------------------------- TRIAL PROJECTION IN PC SPACE
    # Get trial table
    trial_table = nwb_read.get_trial_table(nwb_file=file)
    n_trials = len(trial_table)

    # Get full traces around stim
    selected_frames = []
    outcome = []
    pre_stim = n_frames_before_stim
    post_stim = n_frames_after_stim
    nf = pre_stim + post_stim
    for trial in range(n_trials):
        if trial_table.iloc[trial].trial_type != 'whisker_trial':
            continue
        stim_time = trial_table.iloc[trial].start_time
        stim_frame = find_nearest(rrs_ts, stim_time)
        start_frame = stim_frame - pre_stim
        end_frame = stim_frame + post_stim
        selected_frames.append(np.arange(start_frame, end_frame))
        epoch = trial_table.iloc[trial].context
        lick = trial_table.iloc[trial].lick_flag
        if epoch == 1:
            if lick == 1:
                outcome.append(np.zeros(len(np.arange(start_frame, end_frame))))
            else:
                outcome.append(np.ones(len(np.arange(start_frame, end_frame))))
        else:
            if lick == 1:
                outcome.append(np.zeros(len(np.arange(start_frame, end_frame))) + 3)
            else:
                outcome.append(np.zeros(len(np.arange(start_frame, end_frame))) + 2)

    # Full data
    selected_frames = np.concatenate(selected_frames, axis=0)
    selected_frames_traces = traces[:, selected_frames]
    outcome = np.concatenate(outcome, axis=0)

    # Keep only baseline subtracted average across trials for each trial type:
    # Get data by whisker trial type and put them into dict
    rwd_hits = selected_frames_traces[:, np.where(outcome == 0)[0]]
    rwd_hits = np.reshape(rwd_hits, (rwd_hits.shape[0], int(rwd_hits.shape[1] / nf), nf))
    session_data_dict['RWD-Wh-hit'] = rwd_hits

    rwd_miss = selected_frames_traces[:, np.where(outcome == 1)[0]]
    rwd_miss = np.reshape(rwd_miss, (rwd_miss.shape[0], int(rwd_miss.shape[1] / nf), nf))
    session_data_dict['RWD-Wh-miss'] = rwd_miss

    nn_rwd_hits = selected_frames_traces[:, np.where(outcome == 3)[0]]
    nn_rwd_hits = np.reshape(nn_rwd_hits, (nn_rwd_hits.shape[0], int(nn_rwd_hits.shape[1] / nf), nf))
    session_data_dict['NN-RWD-Wh-hit'] = nn_rwd_hits

    nn_rwd_miss = selected_frames_traces[:, np.where(outcome == 2)[0]]
    nn_rwd_miss = np.reshape(nn_rwd_miss, (nn_rwd_miss.shape[0], int(nn_rwd_miss.shape[1] / nf), nf))
    session_data_dict['NN-RWD-Wh-miss'] = nn_rwd_miss

    # Figure of trial averaged activity
    fig, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)
    ax_to_use = 0
    for key, data in session_data_dict.items():
        avg_data = np.mean(data, axis=1)
        if remove_baseline:
            avg_data -= np.nanmean(avg_data[:, 0:n_frames_before_stim], axis=1, keepdims=True)
        for i in range(avg_data.shape[0]):
            axes.flatten()[ax_to_use].plot(avg_data[i, :], label=f'{sorted_areas[i]}')
            axes.flatten()[ax_to_use].axvline(x=n_frames_before_stim, linestyle='--', color='grey')
            axes.flatten()[ax_to_use].spines[['right', 'top']].set_visible(False)
            axes.flatten()[ax_to_use].legend(ncol=2, loc="upper left")
            axes.flatten()[ax_to_use].set_xlabel('Time (in frames)')
            axes.flatten()[ax_to_use].set_ylabel('df/f')
        axes.flatten()[ax_to_use].set_title(f'{key} ({data.shape[1]} trials)')
        ax_to_use += 1
    fig.suptitle(f'{session} Average trial response')
    fig.tight_layout()
    if save_fig:
        save_formats = ['png', 'svg']
        for save_format in save_formats:
            fig.savefig(os.path.join(saving_folder, f'{session}_average_trial_response.{save_format}'))

    # Transform each trial type average with PC loadings and plot projection of trial average
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    for key, data in session_data_dict.items():
        avg_data = np.mean(data, axis=1)
        if remove_baseline:
            avg_data -= np.nanmean(avg_data[:, 0:n_frames_before_stim], axis=1, keepdims=True)
        reduced_data = np.transpose(pca.transform(np.transpose(avg_data)))
        ax0.plot(reduced_data[lick_pc, :], reduced_data[context_pc, :], c=color_dict[key][0], label=f'{key}')
        ax0.scatter(reduced_data[lick_pc, pre_stim], reduced_data[context_pc, pre_stim], c='gold', marker='*', s=40,
                    zorder=5)
        ax1.scatter(reduced_data[lick_pc, :], reduced_data[context_pc, :], c=color_dict[key][0], label=f'{key}',
                    alpha=np.linspace(0.1, 1, reduced_data.shape[1]))
        ax1.scatter(reduced_data[lick_pc, pre_stim], reduced_data[context_pc, pre_stim], c='gold', marker='*', s=40,
                    zorder=5)
    ax0.spines[['right', 'top']].set_visible(False)
    ax0.set_xlabel(f'PC #{lick_pc} (Lick)')
    ax0.set_ylabel(f'PC #{context_pc} (Context)')
    ax0.legend(ncol=2, loc='upper left')
    ax1.spines[['right', 'top']].set_visible(False)
    ax1.set_xlabel(f'PC #{lick_pc} (Lick)')
    ax1.set_ylabel(f'PC #{context_pc} (Context)')
    ax1.legend(ncol=2, loc='upper left')
    fig.suptitle(f'{session} Projection of trial averaged activity')
    fig.tight_layout()
    if save_fig:
        fig.savefig(os.path.join(saving_folder, f'{session}_trial_averaged_activity_projected.png'))
    else:
        plt.show()

    # Transform each (individual) trial with PC loadings and plot projections
    fig, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)
    ax_to_use = 0
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
    for key, data in session_data_dict.items():
        reduced_activity = np.zeros(data.shape)
        for trial in range(data.shape[1]):
            trial_data = data[:, trial, :]
            if remove_baseline:
                trial_data -= np.nanmean(trial_data[:, 0:n_frames_before_stim], axis=1, keepdims=True)
            reduced_data = np.transpose(pca.transform(np.transpose(trial_data)))
            axes.flatten()[ax_to_use].plot(reduced_data[lick_pc, :], reduced_data[context_pc, :], c=color_dict[key][1])
            axes.flatten()[ax_to_use].spines[['right', 'top']].set_visible(False)
            axes.flatten()[ax_to_use].set_xlabel(f'PC #{lick_pc} (Lick)')
            axes.flatten()[ax_to_use].set_ylabel(f'PC #{context_pc} (Context)')
            reduced_activity[:, trial, :] = reduced_data
        avg_trajectory = np.mean(reduced_activity, axis=1)
        axes.flatten()[ax_to_use].plot(avg_trajectory[lick_pc, :], avg_trajectory[context_pc, :], c=color_dict[key][0],
                                       label='Average trajectory')
        axes.flatten()[ax_to_use].legend(loc='upper left')
        axes.flatten()[ax_to_use].set_title(f'{key} ({data.shape[1]} trials)')
        ax2.plot(avg_trajectory[lick_pc, :], avg_trajectory[context_pc, :], c=color_dict[key][0], label=f'{key}')
        ax_to_use += 1
    fig.suptitle(f'{session} Projection of individual trial and average trajectory')
    ax2.spines[['right', 'top']].set_visible(False)
    ax2.set_xlabel(f'PC #{lick_pc} (Lick)')
    ax2.set_ylabel(f'PC #{context_pc} (Context)')
    ax2.legend(loc='upper left', ncol=2)
    fig2.suptitle(f'{session} Average of individual trial trajectories')
    fig.tight_layout()
    fig2.tight_layout()
    if save_fig:
        fig.savefig(os.path.join(saving_folder, f'{session}_single_trial_activity_projection.png'))
        fig2.savefig(os.path.join(saving_folder, f'{session}_trials_projections_average.png'))
    else:
        plt.show()

    plt.close('all')
