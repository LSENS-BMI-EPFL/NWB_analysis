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
output_folder = os.path.join(root_folder, '20241009')
save_fig = False
rrs_keys = ['ophys', 'brain_area_fluorescence', 'dff0_traces']
components_to_plot = [0, 4]

for file in nwb_files:
    session = nwb_read.get_session_id(file)
    if session in ['RD039_20240222_145509', 'RD057_20240823_114642']:
        continue
    mouse = session[0:5]
    print(' ')
    print(f"Mouse: {mouse}, Session: {session}")
    saving_folder = os.path.join(output_folder, f'{mouse}')
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    # Extract neural data
    traces = nwb_read.get_roi_response_serie_data(nwb_file=file, keys=rrs_keys)
    rrs_ts = nwb_read.get_roi_response_serie_timestamps(nwb_file=file, keys=rrs_keys)

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
    principal_components = pca.fit_transform(scaled_data)
    results = pca.fit(scaled_data)

    # Create a DataFrame for the principal components
    pca_df = pd.DataFrame(data=principal_components, index=df.index, columns=[f'PC{i}' for i in range(8)])

    # Get the time of rewarded context
    epochs = nwb_read.get_behavioral_epochs_names(nwb_file=file)
    rewarded_times = nwb_read.get_behavioral_epochs_times(nwb_file=file, epoch_name='rewarded')
    non_rewarded_times = nwb_read.get_behavioral_epochs_times(nwb_file=file, epoch_name='non-rewarded')

    # ----------------- FIGURES ------------------- #
    # figure1
    color = iter(cm.rainbow(np.linspace(0, 1, 8)))
    fig, ax = plt.subplots(1, 1, figsize=(15, 9))
    for i in range(components_to_plot[0], components_to_plot[1]):
        c = next(color)
        ax.plot(pca_df.index, pca_df[f'PC{i}'], label=f'Principal Component {i}', color=c)
    for i in range(len(rewarded_times[0])):
        ax.axvspan(rewarded_times[0][i], rewarded_times[1][i], facecolor='g', alpha=0.7)
    ax.set_title('Temporal PCA - Principal Components Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Principal Components')
    ax.legend()
    fig.suptitle(f'{session}')
    if save_fig:
        fig.savefig(os.path.join(saving_folder, f'{session}_PC_timecourse.png'))
    else:
        plt.show()

    # figure2
    color = iter(cm.rainbow(np.linspace(0, 1, 8)))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 9))
    for i in range(components_to_plot[0], components_to_plot[1]):
        c = next(color)
        ax0.plot(results.components_[i], label=f'Principal Component {i}', color=c, marker='o')
        ax0.set_xticklabels(['-10'] + sorted_areas)
        ax0.set_ylabel('PC Loadind')
        ax0.set_xlabel('Area')
    ax0.legend()
    ax1.plot(np.cumsum(results.explained_variance_ratio_), marker='o')
    ax1.set_ylabel('Explained variance')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylim([0.50, 1.1])
    ax1.axhline(y=0.95, linestyle='--', color='black')
    fig.suptitle(f'{session}')
    if save_fig:
        fig.savefig(os.path.join(saving_folder, f'{session}_PC_loadings_and_variance.png'))
    else:
        plt.show()

    # figure3
    color = iter(cm.rainbow(np.linspace(0, 1, 8)))
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(15, 9))
    for i in range(components_to_plot[0], components_to_plot[1]):
        c = next(color)
        axes[i].plot(pca_df.index, pca_df[f'PC{i}'], label=f'Principal Component {i}', color=c)
        axes[i].set_ylabel(f'PC{i}')
        for j in range(len(rewarded_times[0])):
            axes[i].axvspan(rewarded_times[0][j], rewarded_times[1][j], facecolor='g', alpha=0.7)
    axes[3].set_xlabel('Time')
    axes[0].set_title('Temporal PCA - Principal Components Over Time')
    fig.suptitle(f'{session}')
    if save_fig:
        fig.savefig(os.path.join(saving_folder, f'{session}_PC_timecourse_separated.png'))
    else:
        plt.show()

    # Try to link all this to behavior or context :
    jaw_angle = nwb_read.get_dlc_data(nwb_file=file, keys=['behavior', 'BehavioralTimeSeries'], part='jaw_angle')
    dlc_ts = nwb_read.get_dlc_timestamps(nwb_file=file, keys=['behavior', 'BehavioralTimeSeries'])[0]

    dlc_frames = []
    for img_ts in rrs_ts:
        dlc_frames.append(find_nearest(array=dlc_ts, value=img_ts, is_sorted=True))
    aligned_jaw_angle = jaw_angle[dlc_frames]
    aligned_jaw_angle_filt = gaussian_filter1d(input=aligned_jaw_angle, sigma=20, axis=-1, order=0)

    # Figure with PC and jaw
    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(15, 9))
    ax0.plot(range(len(pca_df[f'PC0'])), aligned_jaw_angle_filt)
    ax0.plot(range(len(pca_df[f'PC0'])), aligned_jaw_angle)
    ax0.plot(range(len(pca_df[f'PC0'])), pca_df[f'PC0'].values[:])
    ax1.plot(range(len(pca_df[f'PC1'])), aligned_jaw_angle_filt)
    ax1.plot(range(len(pca_df[f'PC1'])), aligned_jaw_angle)
    ax1.plot(range(len(pca_df[f'PC1'])), pca_df[f'PC1'].values[:])
    if save_fig:
        fig.savefig(os.path.join(saving_folder, f'{session}_PC_timecourse_vs_jaw.png'))
    else:
        plt.show()

    # Link PC and jaw trace
    # With non-filtered lick trace
    # model = LinearRegression()
    # for i in range(components_to_plot[0], components_to_plot[1]):
    #     model.fit(aligned_jaw_angle.reshape(-1, 1), pca_df[f'PC{i}'])
    #     r_squared = model.score(aligned_jaw_angle.reshape(-1, 1), pca_df[f'PC{i}'])
    #     print(f"R-squared for PC{i} vs aligned_jaw_angle: {r_squared:.3f}")
    #
    #     correlation, p_value = pearsonr(aligned_jaw_angle, pca_df[f'PC{i}'])
    #     print(f"Correlation between PC{i} and aligned_jaw_angle: {correlation:.3f}, p-value: {p_value}")

    # With filtered lick trace
    model = LinearRegression()
    for i in range(components_to_plot[0], components_to_plot[1]):
        model.fit(aligned_jaw_angle_filt.reshape(-1, 1), pca_df[f'PC{i}'])
        r_squared = model.score(aligned_jaw_angle_filt.reshape(-1, 1), pca_df[f'PC{i}'])
        print(f"R-squared for PC{i} vs aligned_jaw_angle_filt: {r_squared:.3f}")
        correlation, p_value = pearsonr(aligned_jaw_angle_filt, pca_df[f'PC{i}'])
        print(f"Correlation between PC{i} and aligned_jaw_angle_filt: {correlation:.3f}, p-value: {p_value}")

    # Know if each imaging frames is 'rewarded' or not
    rewarded_img_frame = [0 for i in range(len(rrs_ts))]
    for i, ts in enumerate(rrs_ts):
        for epoch in range(rewarded_times.shape[1]):
            if (ts >= rewarded_times[0][epoch]) and (ts <= rewarded_times[1][epoch]):
                rewarded_img_frame[i] = 1
                break
            else:
                continue

    # Create a pd DataFrame
    df = pd.DataFrame({'TPC0': pca_df[f'PC0'], 'TPC1': pca_df[f'PC1'],
                       'TPC2': pca_df[f'PC2'], 'TPC3': pca_df[f'PC3'],
                       'Category': rewarded_img_frame})

    # Boxplot for TPCs and context
    fig, axes = plt.subplots(1, 4, figsize=(15, 9))
    for i in range(components_to_plot[0], components_to_plot[1]):
        sns.boxplot(x='Category', y=f'TPC{i}', data=df, ax=axes[i])
        axes[i].set_ylabel(f'TPC{i}')
    if save_fig:
        fig.savefig(os.path.join(saving_folder, f'{session}_PC_timecourse_vs_context.png'))
    else:
        plt.show()

    # Logisitic regression
    for i in range(components_to_plot[0], components_to_plot[1]):
        model = LogisticRegression()
        scores = cross_val_score(model, pca_df[f'PC{i}'].values[:].reshape(-1, 1), rewarded_img_frame, cv=5)
        print(f"Classification accuracy: {np.mean(scores):.3f}")
