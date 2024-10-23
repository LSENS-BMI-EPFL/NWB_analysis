import yaml
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.pyplot import cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from nwb_utils.utils_misc import find_nearest
from nwb_wrappers import nwb_reader_functions as nwb_read


config_file = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\\z_LSENS\Share\Pol_Bech\Session_list\context_sessions_gcamp_expert.yaml"
# config_file = "//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Robin_Dard/group.yaml"
with open(config_file, 'r', encoding='utf8') as stream:
    config_dict = yaml.safe_load(stream)
# sessions = config_dict['NWB_CI_LSENS']['Context_expert_sessions']
# sessions = config_dict['NWB_CI_LSENS']['Context_contrast_expert']
# sessions = config_dict['NWB_CI_LSENS']['context_contrast_widefield']
# files = [session[1] for session in sessions]
nwb_files = config_dict['Session path']

root_folder = r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\\\analysis\Robin_Dard\Pop_results\Context_behaviour\PCA_pixels'
output_folder = os.path.join(root_folder, '20241022')
save_fig = True
dff0_keys = ['ophys', 'dff0']
time_to_take = 80  # in minutes

# quick sort of nwb files to keep only one mouse:
# nwb_files = [file for file in nwb_files if 'RD043' in file]

for file in nwb_files:
    session = nwb_read.get_session_id(file)
    mouse = session[0:5]
    print(' ')
    print(f"Mouse: {mouse}, Session: {session}")
    saving_folder = os.path.join(output_folder, f'{mouse}', f'{session}')
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    # Extract neural data
    print('Extract neuronal data')
    dff0 = nwb_read.get_widefield_dff0(nwb_file=file, keys=dff0_keys, start=0, stop=int(time_to_take * 60 * 100))
    dff0_ts = nwb_read.get_widefield_timestamps(nwb_file=file, keys=dff0_keys)
    dff0_ts = dff0_ts[0: dff0.shape[0]]
    n_frames, height, width = dff0.shape
    dff0 = np.reshape(dff0, (n_frames, height * width))
    print('Neuronal data extracted')

    # Filter for NaN pixels
    bad_pixels = np.isnan(dff0).any(axis=0)
    dff0 = dff0[:, np.where(bad_pixels==False)[0]]

    # Scale the data
    print('Scale the data for PCA')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dff0)

    # Run PCA
    print('Run PCA')
    pca = PCA(n_components=50)  # Choose the number of components you want
    results = pca.fit(scaled_data)
    pc_time_course = pca.transform(dff0)
    print('Decomposition done')

    # Visualize explained variance
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(np.cumsum(results.explained_variance_ratio_), marker='o')
    ax.set_ylabel('Explained variance')
    ax.set_xlabel('Principal Component')
    ax.spines[['right', 'top']].set_visible(False)
    fig.tight_layout()
    if save_fig:
        fig.savefig(os.path.join(saving_folder, f'{session}_PC_explained_variance.png'))
    else:
        plt.show()

    # Get the time of rewarded context
    epochs = nwb_read.get_behavioral_epochs_names(nwb_file=file)
    rewarded_times = nwb_read.get_behavioral_epochs_times(nwb_file=file, epoch_name='rewarded')
    non_rewarded_times = nwb_read.get_behavioral_epochs_times(nwb_file=file, epoch_name='non-rewarded')

    # Assign context to each imaging frame
    rewarded_img_frame = [0 for i in range(len(dff0_ts))]
    for i, ts in enumerate(dff0_ts):
        for epoch in range(rewarded_times.shape[1]):
            if (ts >= rewarded_times[0][epoch]) and (ts <= rewarded_times[1][epoch]):
                rewarded_img_frame[i] = 1
                break
            else:
                continue

    print('Link PC to context')
    # Logistic regression
    accuracies = []
    for i in range(50):
        model = LogisticRegression()
        scores = cross_val_score(model, pc_time_course[:, i].reshape(-1, 1), rewarded_img_frame, cv=5)
        accuracies.append(np.mean(scores))
    context_pcs = np.argsort(accuracies)[::-1]
    print(f'Top five context PCs : {context_pcs[0:5]}')

    # Get the jaw opening
    dlc_ts = nwb_read.get_dlc_timestamps(nwb_file=file, keys=['behavior', 'BehavioralTimeSeries'])
    if dlc_ts is not None:
        jaw_angle = nwb_read.get_dlc_data(nwb_file=file, keys=['behavior', 'BehavioralTimeSeries'], part='jaw_angle')
        dlc_frames = []
        for img_ts in dff0_ts:
            dlc_frames.append(find_nearest(array=dlc_ts[0], value=img_ts, is_sorted=True))
        aligned_jaw_angle = jaw_angle[dlc_frames]
        jaw_filt = gaussian_filter1d(input=aligned_jaw_angle, sigma=20, axis=-1, order=0)
    else:
        jaw_filt = np.empty(len(rewarded_img_frame))
        jaw_filt[:] = np.nan

    print('Link PC to behavior')
    if len(jaw_filt[np.where(~np.isnan(jaw_filt))[0]]) == len(pc_time_course[:, 0]):
        # With filtered lick trace
        model = LinearRegression()
        r_squared_values = []
        corr_values = []
        for pc in range(50):
            model.fit(jaw_filt.reshape(-1, 1), pc_time_course[:, pc])
            r_squared = model.score(jaw_filt.reshape(-1, 1), pc_time_course[:, pc])
            r_squared_values.append(r_squared)
            correlation, p_value = pearsonr(jaw_filt, pc_time_course[:, pc])
            corr_values.append(correlation)
    else:
        print('DLC frames missing')
        r_squared_values = None
        corr_values = None
    lick_pcs = np.argsort(corr_values)[::-1]
    print(f'Top five lick PCs : {lick_pcs[0:5]}')

    # Visualize loadings
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for pc in range(25):
        pc_loadings = np.full(len(bad_pixels), np.nan)
        pc_loadings[np.where(bad_pixels == False)[0]] = pca.components_[pc]
        pc_loadings_reshaped = pc_loadings.reshape(height, width)
        im = axes.flatten()[pc].imshow(pc_loadings_reshaped, cmap='hot', vmin=-0.03, vmax=0.03)
        if (pc in context_pcs[0:5]) and (pc in lick_pcs[0:5]):
            axes.flatten()[pc].set_title(f'PC{pc} loadings l/c*')
        elif pc in context_pcs[0:5]:
            axes.flatten()[pc].set_title(f'PC{pc} loadings c*')
        elif pc in lick_pcs[0:5]:
            axes.flatten()[pc].set_title(f'PC{pc} loadings l*')
        else:
            axes.flatten()[pc].set_title(f'PC{pc} loadings')
        if pc == 24:
            divider = make_axes_locatable(axes.flatten()[pc])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
    fig.suptitle(f'Session: {session}')
    fig.tight_layout()
    if save_fig:
        fig.savefig(os.path.join(saving_folder, f'{session}_PC_pixel_loadings.png'))
    else:
        plt.show()
    print('end of debugging')

