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


def get_traces_by_epoch(nwb_file, trials, wf_timestamps, start=-200, stop=0):
    wf_data = pd.DataFrame(columns=['A1', 'ALM', 'tjM1', 'tjS1', 'wM1', 'wM2', 'wS1', 'wS2'])
    indices = nwb_read.get_cell_indices_by_cell_type(nwb_file, ['ophys', 'brain_area_fluorescence', 'dff0_traces'])
    for key in indices.keys():
        wf_data[key] = \
            nwb_read.get_widefield_dff0_traces(nwb_file, ['ophys', 'brain_area_fluorescence', 'dff0_traces'])[
                indices[key][0]]

    data = []
    for tstamp in trials:
        frame = utils_misc.find_nearest(wf_timestamps, tstamp)
        wf = wf_data.loc[frame + start:frame + stop - 1].to_numpy()
        if wf.shape != (len(np.arange(start, stop)), 8):
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
    if coefficients.shape[0] == 1:
        lower_bound = np.percentile(coefficients, lower_percentile * 100, axis=1)
        upper_bound = np.percentile(coefficients, upper_percentile * 100, axis=1)
    else:
        lower_bound = np.percentile(coefficients, lower_percentile * 100, axis=0)
        upper_bound = np.percentile(coefficients, upper_percentile * 100, axis=0)

    return coef_mean, coef_std_error, lower_bound, upper_bound


def logregress_model(X, y_binary):
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
    print(f"Accuracy: {round(accuracy_score(y_val, y_pred), 2)}; ", f"Precision: {precision_score(y_val, y_pred)}; ",
          f"CV score: {round(scores.mean(), 2)}")

    return results


def logregress_shuffle(X, y_binary, n_shuffles=1000):
    coefficients = np.zeros([n_shuffles, X.shape[1]])
    accuracy = np.zeros(coefficients.shape[0])
    precision = np.zeros(coefficients.shape[0])

    for i in tqdm(range(n_shuffles)):
        shuffle = np.random.permutation(y_binary)

        # Step 1: Split data into training and temporary set (validation + test)
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, shuffle, test_size=0.2, random_state=0)

        # Step 2: Split the temporary set into validation and test sets
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=0)

        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        coefficients[i] = model.coef_
        accuracy[i] = accuracy_score(y_val, y_pred)
        precision[i] = precision_score(y_val, y_pred)

    return accuracy, precision, coefficients


def logregress_leaveoneout(results, X, y_binary, label):
    # Step 1: Split data into training and temporary set (validation + test)
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=0)

    # Step 2: Split the temporary set into validation and test sets
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=0)

    model = LogisticRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    results[f"{label}_accuracy"] = accuracy_score(y_val, y_pred)
    results[f"{label}_precision"] = precision_score(y_val, y_pred)
    results[f"{label}_coefficients"] = model.coef_

    print(f"Leave {label} out model trained successfully")
    print(f"Accuracy: {round(accuracy_score(y_val, y_pred), 2)}; ",
          f"Precision: {precision_score(y_val, y_pred)}")

    return results


def compute_logreg_and_shuffle(image, y_binary):
    image_scaled = image - np.nanmean(image, axis=0)
    image_scaled = image_scaled - np.nanmean(image_scaled, axis=1)

    results = logregress_model(image_scaled, y_binary)

    accuracy_shuffle, precision_shuffle, coefficients_shuffle = logregress_shuffle(image_scaled, y_binary)

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

    for i in range(matrix.shape[0]):
        scaler = MinMaxScaler(feature_range=(0, 1))
        image_scaled = scaler.fit_transform(image[:, np.where(matrix[i] == 1)].squeeze().T).T
        results = logregress_leaveoneout(results, image_scaled, y_binary, labels[i])

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

    print(" ")
    print(f"Analyzing session {session_id}")

    save_path = os.path.join(output_path, mouse_id, session_id, f"{classify_by}_decoding")

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
        image = np.nanmean(data_frames[:, int(start):int(start + step), :], axis=1)
        results = compute_logreg_and_shuffle(image, y_binary)

        results['mouse_id'] = mouse_id
        results['session_id'] = session_id
        results['start_frame'] = start
        results['stop_frame'] = start + step

        results_total = results_total.append(results, ignore_index=True)
        print(f"Analysis ran successfully for chunk {i}, continuing")

    start = 0
    stop = -1

    image = np.nanmean(data_frames[:, int(start):int(stop), :], axis=1)
    results = compute_logreg_and_shuffle(image, y_binary)

    results['mouse_id'] = mouse_id
    results['session_id'] = session_id
    results['start_frame'] = start
    results['stop_frame'] = stop

    results_total = results_total.append(results, ignore_index=True)

    results_total.to_json(os.path.join(save_path, "results.json"))

    return 0


def get_data(nwb_files, decode):
    wf_data = []
    for nwb_file in nwb_files:

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
        wf_data += [trial_table]

    return pd.concat(wf_data, ignore_index=True)


if __name__ == "__main__":

    # config_file = r"M:\analysis\Pol_Bech\Sessions_list\context_contrast_expert_widefield_sessions_path.yaml"
    config_file = r"M:\analysis\Robin_Dard\Sessions_list\context_naïve_mice_widefield_sessions_path.yaml"
    with open(config_file, 'r', encoding='utf8') as stream:
        config_dict = yaml.safe_load(stream)

    nwb_files = config_dict['Sessions path']

    if os.path.exists(nwb_files[0]):
        subject_ids = list(np.unique([nwb_read.get_mouse_id(file) for file in nwb_files]))
    else:
        subject_ids = list(np.unique([session[0:5] for session in nwb_files]))

    experimenter_initials_1 = "RD"
    experimenter_initials_2 = "PB"
    root_path_1 = server_path.get_experimenter_nwb_folder(experimenter_initials_1)
    root_path_2 = server_path.get_experimenter_nwb_folder(experimenter_initials_2)

    output_path = os.path.join(f'{server_path.get_experimenter_saving_folder_root("PB")}',
                               'Pop_results', 'Context_behaviour', 'widefield_decoding_area_naive_stim')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    nwb_files = [file for file in nwb_files if 'RD' in file]

    for decode in ['stim', 'baseline']:
        results = get_data(nwb_files, decode=decode)
        for classify_by in ['context', 'lick', 'tone']:
            logregress_classification(results, classify_by=classify_by, decode=decode, n_chunks=10,
                                      output_path=output_path)
