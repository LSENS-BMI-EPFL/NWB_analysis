import os
import matplotlib.colors
import pandas as pd
import yaml
import numpy as np
import seaborn as sns
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
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve
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


def get_traces_by_epoch(nwb_file, trials, wf_timestamps, start=-200, stop=0):
    wf_data = pd.DataFrame(columns=['A1', 'ALM', 'tjM1', 'tjS1', 'wM1', 'wM2', 'wS1', 'wS2'])
    indices = nwb_read.get_cell_indices_by_cell_type(nwb_file, ['ophys', 'brain_area_fluorescence', 'dff0_traces'])
    for key in indices.keys():
        wf_data[key] = nwb_read.get_widefield_dff0_traces(nwb_file, ['ophys', 'brain_area_fluorescence', 'dff0_traces'])[indices[key][0]]

    data = []
    for tstamp in trials:
        frame = utils_misc.find_nearest(wf_timestamps, tstamp)
        wf = wf_data.loc[frame+start:frame+stop-1].to_numpy()
        if wf.shape != (len(np.arange(start, stop)), wf_data.shape[1]):
            continue
        data += [wf]

    data = np.array(data)
    data = np.stack(data, axis=0)
    return data

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


# def logregress_model(X, y_binary):
#
#     # Step 1: Split data into training and temporary set (validation + test)
#     X_trainval, X_test, y_trainval, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=0)
#
#     # Step 2: Split the temporary set into validation and test sets
#     X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=0)
#
#     model = LogisticRegression()
#
#     model.fit(X_train, y_train)
#
#     scores = cross_val_score(model, X, y_binary, cv=50)
#     print("Model CV finished with %0.2f accuracy and a standard deviation of %0.2f" % (scores.mean(), scores.std()))
#
#     y_pred = model.predict(X_val)
#
#     results = {
#         "accuracy": accuracy_score(y_val, y_pred),
#         "precision": precision_score(y_val, y_pred),
#         "cross_val_scores": scores,
#         "coefficients": model.coef_,
#     }
#
#     print("Model trained successfully")
#     print(f"Accuracy: {round(accuracy_score(y_val, y_pred), 2)}; ", f"Precision: {precision_score(y_val, y_pred)}; ", f"CV score: {round(scores.mean(), 2)}")
#
#     return results

def logregress_model(df, y_binary, result_path, strat=True):
    # Step 1: Split data into training and temporary set (validation + test)
    if strat:
        stratify = df.regressor
    else:
        stratify = None

    X = np.stack(df.wf_scaled.to_numpy()).squeeze()
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=0, stratify=stratify)

    model = LogisticRegression(solver='liblinear', penalty='l1', C=3)

    skf = StratifiedKFold(n_splits=10)
    scoring = ['accuracy', 'precision', 'f1', 'recall', 'r2', 'explained_variance', 'neg_mean_squared_error', 'roc_auc']
    scores = pd.DataFrame.from_dict(cross_validate(model, X, y_binary, cv=skf, scoring=scoring, return_train_score=True))
    scores.to_csv(os.path.join(result_path, 'cross_validated_scores.csv'))
    print("Model CV finished with %0.2f accuracy and a standard deviation of %0.2f" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std()))

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "coefficients": model.coef_,
        "coefficients_idx": ['A1', 'ALM', 'tjM1', 'tjS1', 'wM1', 'wM2', 'wS1', 'wS2'],
        "fpr": fpr,
        "tpr": tpr,
        "conf_mat_tp": tn,
        "conf_mat_fp": fp,
        "conf_mat_fn": fn,
        "conf_mat_tn": fp,
    }

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC: {roc_auc_score(y_test, y_pred_prob):.2f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='best')
    fig.savefig(os.path.join(result_path, 'roc_curve.png'))
    fig.savefig(os.path.join(result_path, 'roc_curve.svg'))

    # Plot Precision-Recall Curve
    fig, ax = plt.subplots()
    ax.plot(recall[:-1], precision[:-1], label='Scores')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()
    ax.set_title('Precision-Recall Curve')
    fig.savefig(os.path.join(result_path, 'precision_recall_curve.png'))
    fig.savefig(os.path.join(result_path, 'precision_recall_curve.svg'))

    fig, ax =plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    fig.savefig(os.path.join(result_path, 'confusion_matrix.png'))
    fig.savefig(os.path.join(result_path, 'confusion_matrix.svg'))

    coef = pd.Series(model.coef_[0], index=['A1', 'ALM', 'tjM1', 'tjS1', 'wM1', 'wM2', 'wS1', 'wS2'])
    coef = coef.sort_values()

    fig, ax = plt.subplots()
    ax.barh(coef.index, coef.values)
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Feature Importance (Logistic Regression Coefficients)')
    fig.savefig(os.path.join(result_path, 'feature_importance.png'))
    fig.savefig(os.path.join(result_path, 'feature_importance.svg'))

    return results


def logregress_shuffle(df, y_binary, classify_by, n_shuffles=1000):

    X = np.stack(df.wf_scaled.to_numpy()).squeeze()
    coefficients = np.zeros([n_shuffles, X.shape[1]])
    accuracy = np.zeros(coefficients.shape[0])
    precision = np.zeros(coefficients.shape[0])
    trials = np.arange(len(y_binary))

    for i in tqdm(range(n_shuffles)):
        if classify_by == 'context':
            block_id = np.abs(np.diff(y_binary, prepend=0)).cumsum()
            shuffle_idx = np.hstack([np.where(block_id == i) for i in np.random.permutation(np.unique(block_id))])[0]
            shuffle = y_binary[shuffle_idx]
        else:
            shuffle = np.random.permutation(y_binary)

        # Step 1: Split data into training and temporary set (validation + test)
        X_train, X_test, y_train, y_test = train_test_split(X, shuffle, test_size=0.2, random_state=0, stratify=df.regressor)

        model = LogisticRegression(solver='liblinear', penalty='l1', C=3)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        coefficients[i] = model.coef_
        accuracy[i] = accuracy_score(y_test, y_pred)
        precision[i] = precision_score(y_test, y_pred)

    return accuracy, precision, coefficients

def logregress_leaveoneout(X, y_binary, label):

    # Step 1: Split data into training and temporary set (validation + test)
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=0)

    model = LogisticRegression(solver='liblinear', penalty='l1', C=3)

    skf = StratifiedKFold(n_splits=10)
    scoring = ['accuracy', 'precision', 'f1', 'recall', 'r2', 'explained_variance', 'neg_mean_squared_error', 'roc_auc']
    scores = pd.DataFrame.from_dict(cross_validate(model, X, y_binary, cv=skf, scoring=scoring, return_train_score=True))
    scores['del'] = label

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # print(f"Model CV for del {label} finished with %0.2f accuracy and a standard deviation of %0.2f" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std()))

    return scores


def compute_logreg_and_shuffle(df, y_binary, classify_by, result_path):

    image = np.stack(df.wf_mean.to_numpy())
    image = image - np.nanmean(image, axis=0)
    image_scaled = (image.T - np.nanmean(image, axis=1)).T

    df['wf_scaled'] = [[np.vstack(image_scaled)[i, :]] for i in range(np.vstack(image_scaled).shape[0])]
    df['regressor'] = y_binary

    results = logregress_model(df, y_binary, result_path=result_path)

    accuracy_shuffle, precision_shuffle, coefficients_shuffle = logregress_shuffle(df, y_binary, classify_by)

    alpha = 0.95
    coef_mean, coef_std_error, lower_bound, upper_bound = ols_statistics(coefficients_shuffle, confidence=alpha)

    results['accuracy_shuffle'] = accuracy_shuffle
    results['precision_shuffle'] = precision_shuffle
    results['coefficients_shuffle'] = coefficients_shuffle
    results['shuffle_mean'] = coef_mean
    results['shuffle_std'] = coef_std_error
    results['alpha'] = alpha
    results['lower_bound'] = lower_bound
    results['upper_bound'] = upper_bound

    labels = ['A1', 'ALM', 'tjM1', 'tjS1', 'wM1', 'wM2', 'wS1', 'wS2', 'motor', 'sensory']
    matrix = np.ones([len(labels) - 2, len(labels) - 2])
    np.fill_diagonal(matrix, 0)
    sensory = np.zeros_like(matrix[0])
    sensory[[0, 6, 7]] = 1
    motor = np.where(sensory != 1, np.ones_like(sensory), 0)
    matrix = np.vstack([matrix, sensory, motor])

    results_leaveoneout = []
    for i in range(matrix.shape[0]):
        image = np.stack(df.wf_mean.to_numpy())
        image = image[:, np.where(matrix[i] == 1)].squeeze()
        image = image - np.nanmean(image, axis=0)
        image_scaled = (image.T - np.nanmean(image, axis=1)).T

        leaveoneout = logregress_leaveoneout(image_scaled, y_binary, labels[i])
        results_leaveoneout += [leaveoneout]
    results_leaveoneout = pd.concat(results_leaveoneout, ignore_index=True)
    results_leaveoneout.to_csv(os.path.join(result_path, 'leave_one_out_scores.csv'))

    return results


def logregress_classification(trial_table, classify_by, decode, n_chunks, output_path):
    print(f"Widefield image classification")

    if decode == 'baseline':
        split = np.linspace(0, 200, n_chunks, endpoint=False)
        step = np.unique(np.diff(split))[0]
        start = -200
        stop = 0
    elif decode == 'stim':
        split = np.linspace(0, 20, n_chunks, endpoint=False)
        step = np.unique(np.diff(split))[0]
        start = 0
        stop = 20

    results_total = pd.DataFrame()
    mouse_id = trial_table.mouse_id.unique()[0]
    session_id = trial_table.session_id.unique()[0]

    save_path = os.path.join(output_path, decode, f"{classify_by}_decoding")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    trial_table['correct_choice'] = trial_table.reward_available == trial_table.lick_flag

    if classify_by == 'context':
        y_binary = trial_table.context

    elif classify_by == 'lick':
        y_binary = trial_table.lick_flag

    elif classify_by == 'tone':
        y_binary = trial_table.context_background.map({'pink': 1, 'brown': 0})

    elif classify_by == 'correct':
        y_binary = trial_table.correct_choice

    data_frames = np.stack(trial_table.wf)

    for i, start in enumerate(split):
        trial_table['wf_mean'] = trial_table.wf.apply(lambda x: np.nanmean(x[int(start):int(start + step), :], axis=0))
        result_path = os.path.join(save_path, f'{start}-{start+step}')
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        results = compute_logreg_and_shuffle(trial_table[['mouse_id', 'session_id', 'wf_mean']], y_binary, classify_by, result_path=result_path)

        results['mouse_id'] = mouse_id
        results['session_id'] = session_id
        results['start_frame'] = start
        results['stop_frame'] = start + step

        results_total = results_total.append(results, ignore_index=True)
        print(f"Analysis ran successfully for chunk {i}, continuing")

    start = 0
    stop = -1

    # image = np.nanmean(data_frames[:, int(start):int(stop), :], axis=1)
    trial_table['wf_mean'] = trial_table.wf.apply(lambda x: np.nanmean(x[int(start):int(stop), :], axis=0))
    results = compute_logreg_and_shuffle(trial_table[['mouse_id', 'session_id', 'wf_mean']], y_binary, classify_by, result_path=save_path)

    results['mouse_id'] = mouse_id
    results['session_id'] = session_id
    results['start_frame'] = start
    results['stop_frame'] = stop

    results_total = results_total.append(results, ignore_index=True)

    results_total.to_json(os.path.join(save_path, "results.json"))

    return 0


def get_data(nwb_file, decode):

    print(nwb_file.split('\\')[-1])

    if decode == 'baseline':
        start = -200
        stop = 0
    elif decode == 'stim':
        start = 0
        stop = 20

    trial_table = nwb_read.get_trial_table(nwb_file)
    trial_table['mouse_id'] = nwb_read.get_mouse_id(nwb_file)
    trial_table['session_id'] = nwb_read.get_session_id(nwb_file)
    wf_timestamps = nwb_read.get_widefield_timestamps(nwb_file, ['ophys', 'dff0'])
    data_frames = get_traces_by_epoch(nwb_file, trial_table.start_time, wf_timestamps, start=start, stop=stop)
    if len(trial_table) != data_frames.shape[0]:
        os.system('echo "Different number of trials and wf frames"')
        difference = len(trial_table) - data_frames.shape[0]
        if difference == 1:
            os.system('echo "One more trial than wf frames, removing"')
            trial_table = trial_table.drop(-1)

    trial_table['wf'] = [data_frames[i] for i in range(data_frames.shape[0])]

    return trial_table


if __name__ == "__main__":

    for state in ['naive', 'expert']:
        config_file = f"//sv-nas1.rcp.epfl.ch/Petersen-Lab/z_LSENS/Share/Pol_Bech/Session_list/context_sessions_gcamp_{state}.yaml"
        with open(config_file, 'r', encoding='utf8') as stream:
            config_dict = yaml.safe_load(stream)

        nwb_files = config_dict['Session path']

        if os.path.exists(nwb_files[0]):
            subject_ids = list(np.unique([nwb_read.get_mouse_id(file) for file in nwb_files]))
        else:
            subject_ids = list(np.unique([session[0:5] for session in nwb_files]))

        experimenter_initials_1 = "RD"
        experimenter_initials_2 = "PB"
        root_path_1 = server_path.get_experimenter_nwb_folder(experimenter_initials_1)
        root_path_2 = server_path.get_experimenter_nwb_folder(experimenter_initials_2)

        output_path = os.path.join(f'{server_path.get_experimenter_saving_folder_root("PB")}',
                                   'Pop_results', 'Context_behaviour', f'widefield_decoding_area_gcamp_{state}_saga_elasticnet_001')

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # nwb_files = [file for file in nwb_files if 'PB' in file]
        for decode in ['baseline', 'stim']:
            for nwb_file in nwb_files:
                results = get_data(nwb_file, decode=decode)
                for classify_by in ['context', 'lick', 'tone']:
                    print(f"Decoding {state} {nwb_file.split('/')[-1]} {decode} {classify_by}")
                    logregress_classification(results, classify_by=classify_by, decode=decode, n_chunks=5,
                                              output_path=output_path)
