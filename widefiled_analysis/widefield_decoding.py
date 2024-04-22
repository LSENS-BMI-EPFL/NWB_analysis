import os
import matplotlib.colors
import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt
import nwb_wrappers.nwb_reader_functions as nwb_read

import warnings
warnings.filterwarnings("ignore")

from PIL import Image
from tqdm import tqdm
from matplotlib.cm import get_cmap
from skimage.transform import rescale
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from nwb_utils import server_path, utils_misc, utils_behavior
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap


def get_wf_scalebar(scale = 1, plot=False, savepath=None):
    file = r"M:\analysis\Pol_Bech\Parameters\Widefield\wf_scalebars\reference_grid_20240314.tif"
    grid = Image.open(file)
    im = np.array(grid)
    im = im.reshape(int(im.shape[0] / 2), 2, int(im.shape[1] / 2), 2).mean(axis=1).mean(axis=2) # like in wf preprocessing
    x = [62*scale, 167*scale]
    y = [162*scale, 152*scale]
    fig, ax = plt.subplots()
    ax.imshow(rescale(im, scale, anti_aliasing=False))
    ax.plot(x, y, c='r')
    ax.plot(x, [y[0], y[0]], c='k')
    ax.plot([x[1], x[1]], y, c='k')
    ax.text(x[0] + int((x[1] - x[0]) / 2), 175*scale, f"{x[1] - x[0]} px")
    ax.text(170*scale, 168*scale, f"{np.abs(y[1] - y[0])} px")
    c = np.sqrt((x[1] - x[0]) ** 2 + (y[0] - y[1]) ** 2)
    ax.text(100*scale, 145*scale, f"{round(c)} px")
    ax.text(200*scale, 25*scale, f"{round(c / 6)} px/mm", color="r")
    if plot:
        fig.show()
    if savepath:
        fig.savefig(savepath+rf'\wf_scalebar_scale{scale}.png')
    return round(c / 6)


def get_allen_ccf(bregma = (528, 315), root=r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Robin_Dard\Parameters\Widefield\allen_brain"):
    """Find in utils the AllenSDK file to generate the npy files"""

     ## all images aligned to 240,175 at widefield video alignment, after expanding image, goes to this. Set manually.
    iso_mask = np.load(root + r"\allen_isocortex_tilted_500x640.npy")
    atlas_mask = np.load(root + r"\allen_brain_tilted_500x640.npy")
    bregma_coords = np.load(root + r"\allen_bregma_tilted_500x640.npy")

    displacement_x = int(bregma[0] - np.round(bregma_coords[0] + 20))
    displacement_y = int(bregma[1] - np.round(bregma_coords[1]))

    margin_y = atlas_mask.shape[0]-np.abs(displacement_y)
    margin_x = atlas_mask.shape[1]-np.abs(displacement_x)

    if displacement_y >= 0 and displacement_x >= 0:
        atlas_mask[displacement_y:, displacement_x:] = atlas_mask[:margin_y, :margin_x]
        atlas_mask[:displacement_y, :] *= 0
        atlas_mask[:, :displacement_x] *= 0

        iso_mask[displacement_y:, displacement_x:] = iso_mask[:margin_y, :margin_x]
        iso_mask[:displacement_y, :] *= 0
        iso_mask[:, :displacement_x] *= 0

    elif displacement_y < 0 and displacement_x>=0:
        atlas_mask[:displacement_y, displacement_x:] = atlas_mask[-margin_y:, :margin_x]
        atlas_mask[displacement_y:, :] *= 0
        atlas_mask[:, :displacement_x] *= 0

        iso_mask[:displacement_y, displacement_x:] = iso_mask[-margin_y:, :margin_x]
        iso_mask[displacement_y:, :] *= 0
        iso_mask[:, :displacement_x] *= 0

    elif displacement_y >= 0 and displacement_x<0:
        atlas_mask[displacement_y:, :displacement_x] = atlas_mask[:margin_y, -margin_x:]
        atlas_mask[:displacement_y, :] *= 0
        atlas_mask[:, displacement_x:] *= 0

        iso_mask[displacement_y:, :displacement_x] = iso_mask[:margin_y, -margin_x:]
        iso_mask[:displacement_y, :] *= 0
        iso_mask[:, displacement_x:] *= 0

    else:
        atlas_mask[:displacement_y, :displacement_x] = atlas_mask[-margin_y:, -margin_x:]
        atlas_mask[displacement_y:, :] *= 0
        atlas_mask[:, displacement_x:] *= 0

        iso_mask[:displacement_y, :displacement_x] = iso_mask[-margin_y:, -margin_x:]
        iso_mask[displacement_y:, :] *= 0
        iso_mask[:, displacement_x:] *= 0

    return iso_mask, atlas_mask, bregma_coords


def get_colormap(cmap='hotcold'):
    hotcold = ['#aefdff', '#60fdfa', '#2adef6', '#2593ff', '#2d47f9', '#3810dc', '#3d019d',
               '#313131',
               '#97023d', '#d90d39', '#f8432d', '#ff8e25', '#f7da29', '#fafd5b', '#fffda9']

    cyanmagenta = ['#00FFFF', '#FFFFFF', '#FF00FF']

    if cmap == 'cyanmagenta':
        cmap = LinearSegmentedColormap.from_list("Custom", cyanmagenta)

    elif cmap == 'whitemagenta':
        cmap = LinearSegmentedColormap.from_list("Custom", ['#FFFFFF', '#FF00FF'])

    elif cmap == 'hotcold':
        cmap = LinearSegmentedColormap.from_list("Custom", hotcold)

    elif cmap == 'grays':
        cmap = get_cmap('Greys')

    elif cmap == 'viridis':
        cmap = get_cmap('viridis')

    elif cmap == 'blues':
        cmap = get_cmap('Blues')

    elif cmap == 'magma':
        cmap = get_cmap('magma')

    elif cmap == 'seismic':
        cmap = get_cmap('seismic')

    else:
        cmap = get_cmap(cmap)

    cmap.set_bad(color='k', alpha=0.1)

    return cmap


def plot_image_stats(image, y_binary, classify_by, save_path):
    cat_a = image[np.where(y_binary == 1)[0]]
    cat_b = image[np.where(y_binary == 0)[0]]

    fig, ax = plt.subplots(1, 2, figsize=(10,7))
    ax[0].scatter(np.nanmean(cat_a, axis=0).flatten(), np.nanmean(cat_b, axis=0).flatten(), c='k', alpha=0.5, s=2)
    ax[0].set_xlabel('Lick' if classify_by == 'lick' else 'Rewarded')
    ax[0].set_ylabel('No Lick' if classify_by == 'lick' else 'Non-Rewarded')
    ax[0].plot(ax[0].get_xlim(), ax[0].get_ylim(), ls='--', c='r')
    ax[0].set_box_aspect(1)

    ax[1].hist(np.nanmean(cat_a, axis=0).flatten(), 100, alpha=0.5, label='Lick' if classify_by == 'lick' else 'Rewarded')
    ax[1].hist(np.nanmean(cat_b, axis=0).flatten(), 100, alpha=0.5, label='No Lick' if classify_by == 'lick' else 'Non-Rewarded')
    ax[1].set_xlabel("MinMax Scores")
    ax[1].set_ylabel("Counts")
    ax[1].set_box_aspect(1)
    fig.legend()
    fig.tight_layout()

    for ext in ['png', 'svg']:
        fig.savefig(save_path + f".{ext}")

def plot_single_frame(data, title, norm=True, colormap='seismic', save_path=None, vmin=-0.5, vmax=0.5, show=False):
    bregma = (488, 290)
    scale = 4
    scalebar = get_wf_scalebar(scale=scale)
    iso_mask, atlas_mask, allen_bregma = get_allen_ccf(bregma)

    fig, ax = plt.subplots(1, figsize=(7, 7))
    fig.suptitle(title)
    cmap = get_colormap(colormap)
    if norm:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm= matplotlib.colors.NoNorm()

    single_frame = np.rot90(rescale(data, scale, anti_aliasing=False))
    single_frame = np.pad(single_frame, [(0, 650 - single_frame.shape[0]), (0, 510 - single_frame.shape[1])],
                          mode='constant', constant_values=np.nan)

    mask = np.pad(iso_mask, [(0, 650 - iso_mask.shape[0]), (0, 510 - iso_mask.shape[1])], mode='constant',
                  constant_values=np.nan)
    single_frame = np.where(mask > 0, single_frame, np.nan)

    im = ax.imshow(single_frame, norm=norm, cmap=cmap)
    ax.contour(atlas_mask, levels=np.unique(atlas_mask), colors='gray',
                       linewidths=1)
    ax.contour(iso_mask, levels=np.unique(np.round(iso_mask)), colors='black',
                       linewidths=2, zorder=2)
    ax.scatter(bregma[0], bregma[1], marker='+', c='k', s=100, linewidths=2,
                       zorder=3)
    ax.hlines(25, 25, 25 + scalebar * 3, linewidth=2, colors='k')
    ax.text(50, 100, "3 mm", size=10)
    ax.set_title(f"{title}")
    fig.colorbar(im, ax=ax)
    fig.axes[1].set(ylabel="Coefficients")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path + ".png")
        fig.savefig(save_path + ".svg")
    if show:
        fig.show()


def get_frames_by_epoch(nwb_file, trials, wf_timestamps):
    frames = []
    for tstamp in trials:
        frame = utils_misc.find_nearest(wf_timestamps, tstamp)
        data = nwb_read.get_widefield_dff0(nwb_file, ['ophys', 'dff0'], frame - 200, frame + 200)
        if data.shape != (400, 125, 160):
            continue
        frames.append(data)

    data_frames = np.array(frames)
    data_frames = np.stack(data_frames, axis=0)
    return data_frames


def ols_statistics(coefficients, confidence=0.95):
    """
    Calculate OLS statistics.
    """
    lower_percentile = (1 - confidence) / 2
    upper_percentile = 1 - lower_percentile

    # Calculate mean and standard error of coefficients
    coef_mean = np.mean(coefficients, axis=0)

    coef_std_error = np.std(coefficients, axis=0, ddof=1)

    # Calculate confidence intervals
    if coefficients.shape[0]==1:
        lower_bound = np.percentile(coefficients, lower_percentile * 100, axis=1)
        upper_bound = np.percentile(coefficients, upper_percentile * 100, axis=1)
    else:
        lower_bound = np.percentile(coefficients, lower_percentile * 100, axis=0)
        upper_bound = np.percentile(coefficients, upper_percentile * 100, axis=0)

    return coef_mean, coef_std_error, lower_bound, upper_bound


def logregress_model(image, y_binary):

    X = np.nan_to_num(image.reshape(image.shape[0], -1), 0)

    # Step 1: Split data into training and temporary set (validation + test)
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=0)

    # Step 2: Split the temporary set into validation and test sets
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=0)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    scores = cross_val_score(model, X, y_binary, cv=50)
    print("Model CV finished with %0.2f accuracy and a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    y_pred = model.predict(X_val)

    results = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "cross_val_scores": scores,
        "coefficients": model.coef_,
    }

    print("Model trained successfully")
    print(f"Accuracy: {round(accuracy_score(y_val, y_pred), 2)}; ", f"Precision: {precision_score(y_val, y_pred)}; ", f"CV score: {round(scores.mean(), 2)}")

    return results


def logregress_shuffle(image, y_binary, n_shuffles=1000):

    coefficients = np.zeros([n_shuffles, image.shape[1]])
    for i in tqdm(range(n_shuffles)):
        shuffle = np.random.shuffle(y_binary)

        X = np.nan_to_num(image.reshape(image.shape[0], -1), 0)

        # Step 1: Split data into training and temporary set (validation + test)
        X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=0)

        model = LogisticRegression()
        model.fit(X_train, y_train)
        coefficients[i] = model.coef_

    return coefficients


def compute_logreg_and_shuffle(image, y_binary, classify_by, save_path=None):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    image_scaled = scaler.fit_transform(image.reshape(image.shape[0], -1))

    if save_path is not None:
        plot_image_stats(image_scaled, y_binary, classify_by, save_path)

    results = logregress_model(image_scaled, y_binary)
    coefficients_shuffle = logregress_shuffle(image_scaled, y_binary)

    alpha = 0.95
    coef_mean, coef_std_error, lower_bound, upper_bound = ols_statistics(coefficients_shuffle, confidence=alpha)

    results['coefficients_shuffle'] = coefficients_shuffle
    results['shuffle_mean'] = coef_mean
    results['shuffle_std'] = coef_std_error
    results['alpha'] = alpha
    results['lower_bound'] = lower_bound
    results['upper_bound'] = upper_bound

    return results


def logregress_classification(nwb_files, classify_by, n_chunks, output_path, show=False):
    print(f"Widefield image classification")

    results_total = pd.DataFrame()
    for nwb_file in nwb_files:
        mouse_id = nwb_read.get_mouse_id(nwb_file)
        session_id = nwb_read.get_session_id(nwb_file)
        print(" ")
        print(f"Analyzing session {session_id}")
        session_type = nwb_read.get_session_type(nwb_file)
        if 'wf' not in session_type:
            print(f"{session_id} is not a widefield session")
            continue

        if n_chunks == 1:
            save_path = os.path.join(output_path, f"{classify_by}_decoding", f"{mouse_id}", f"{session_id}")
        else:
            save_path = os.path.join(output_path, f"{classify_by}_decoding", f"{mouse_id}", f"{session_id}", "by_timebins")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        trial_table = nwb_read.get_trial_table(nwb_file)
        epochs = nwb_read.get_behavioral_epochs_names(nwb_file)
        wf_timestamps = nwb_read.get_widefield_timestamps(nwb_file, ['ophys', 'dff0'])

        data_frames = get_frames_by_epoch(nwb_file, trial_table.start_time, wf_timestamps)

        if n_chunks == 1:
            split = np.array([0])
            step = int(data_frames.shape[1]/2)
        else:
            split = np.linspace(0, int(data_frames.shape[1]/2), n_chunks, endpoint=False)
            step = np.unique(np.diff(split))[0]

        if classify_by == 'context':
            y_binary = trial_table.context
        elif classify_by == 'lick':
            y_binary = trial_table.lick_flag

        for i, start in enumerate(split):
            image = np.nanmean(data_frames[:, int(start):int(start + step), :, :], axis=(1))
            results = compute_logreg_and_shuffle(image, y_binary, classify_by, os.path.join(save_path, f"{classify_by}_image_stats_chunk{i}"))

            results['mouse_id'] = mouse_id
            results['session_id'] = session_id
            results['start_frame'] = start
            results['stop_frame'] = start + step
            np.save(os.path.join(save_path, f"{classify_by}_model_scores_chunk{i}.npy"), results)

            results_total = results_total.append(results, ignore_index=True)

            coef_image = results['coefficients'].reshape(125, -1)
            title = f"{int((start - (data_frames.shape[1] / 2)) * 1000 / 100)} - {int((start - (data_frames.shape[1] / 2) + step) * 1000 / 100)} ms"
            plot_single_frame(coef_image,
                              title,
                              colormap='seismic',
                              vmin=-0.5,
                              vmax=0.5,
                              save_path=os.path.join(save_path, f"{session_id}_{classify_by}_coef_chunk{i}"),
                              show=show)

            CI_out = (results['coefficients'] > results['upper_bound']) | (results['coefficients'] < results['lower_bound'])
            title = f"{int((start - (data_frames.shape[1] / 2)) * 1000 / 100)} - {int((start - (data_frames.shape[1] / 2) + step) * 1000 / 100)} ms statistics"
            plot_single_frame(CI_out.reshape(125, -1),
                              title,
                              norm=False,
                              colormap='Greys_r',
                              vmin=0,
                              vmax=0.5,
                              save_path=os.path.join(save_path, f"{session_id}_{classify_by}_stats_chunk{i}"),
                              show=show)

        save_path = os.path.join(output_path, f"{classify_by}_decoding", f"{mouse_id}", f"{session_id}")
        start= 0
        step = int(data_frames.shape[1]/2)

        image = np.nanmean(data_frames[:, int(start):int(start + step), :, :], axis=(1))
        results = compute_logreg_and_shuffle(image, y_binary, classify_by, os.path.join(save_path, f"{classify_by}_image_stats_full"))

        results['mouse_id'] = mouse_id
        results['session_id'] = session_id
        results['start_frame'] = start
        results['stop_frame'] = start + step
        np.save(os.path.join(save_path, f"{classify_by}_model_scores_full.npy"), results)

        results_total = results_total.append(results, ignore_index=True)

        coef_image = results['coefficients'].reshape(125, -1)
        title = f"{int((start - (data_frames.shape[1] / 2)) * 1000 / 100)} - {int((start - (data_frames.shape[1] / 2) + step) * 1000 / 100)} ms"
        plot_single_frame(coef_image,
                          title,
                          colormap='seismic',
                          vmin=-0.5,
                          vmax=0.5,
                          save_path=os.path.join(save_path, f"{session_id}_{classify_by}_coef_full"),
                          show=show)

        CI_out = (results['coefficients'] > results['upper_bound']) | (results['coefficients'] < results['lower_bound'])
        title = f"{int((start - (data_frames.shape[1] / 2)) * 1000 / 100)} - {int((start - (data_frames.shape[1] / 2) + step) * 1000 / 100)} ms statistics"
        plot_single_frame(CI_out.reshape(125, -1),
                          title,
                          norm=False,
                          colormap='Greys_r',
                          vmin=0,
                          vmax=0.5,
                          save_path=os.path.join(save_path, f"{session_id}_{classify_by}_stats_full"),
                          show=show)

    results_total.to_csv(os.path.join(output_path, "results.csv"))
    return 0


if __name__ == "__main__":

    config_file = "//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Robin_Dard/group.yaml"
    with open(config_file, 'r', encoding='utf8') as stream:
        config_dict = yaml.safe_load(stream)

    sessions = config_dict['NWB_CI_LSENS']['context_contrast_widefield']
    session_to_do = [session[0] for session in sessions]

    subject_ids = list(np.unique([session[0:5] for session in session_to_do]))

    experimenter_initials = subject_ids[0][0:2]

    root_path = server_path.get_experimenter_nwb_folder(experimenter_initials)

    output_path = os.path.join(f'{server_path.get_experimenter_saving_folder_root("PB")}',
                               'Pop_results', 'Context_behaviour', 'Test_decoding')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    all_nwb_names = os.listdir(root_path)

    session_dit = {'Sessions': session_to_do}
    with open(os.path.join(output_path, "session_to_do.yaml"), 'w') as stream:
        yaml.dump(session_dit, stream, default_flow_style=False, explicit_start=True)

    for subject_id in subject_ids:
        nwb_names = [name for name in all_nwb_names if subject_id in name]
        nwb_files = []
        for session in session_to_do:
            nwb_files += [os.path.join(root_path, name) for name in nwb_names if session in name]
        print(" ")
        print(f"nwb_files : {nwb_files}")
        # logregress_classification(nwb_files, classify_by='context', n_chunks=1, output_path=output_path)
        logregress_classification(nwb_files, classify_by='context', n_chunks=10, output_path=output_path)
        # logregress_classification(nwb_files, classify_by='lick', n_chunks=1, output_path=output_path)
        logregress_classification(nwb_files, classify_by='lick', n_chunks=10, output_path=output_path)