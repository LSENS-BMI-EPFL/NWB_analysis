import os
import sys
sys.path.append("/home/bechvila/NWB_analysis")
import numpy as np
import pandas as pd
import nwb_wrappers.nwb_reader_functions as nwb_read

import warnings
warnings.filterwarnings("ignore")

from multiprocessing import Pool
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from nwb_utils import server_path, utils_misc, utils_behavior


def get_frames_by_epoch(nwb_file, trials, wf_timestamps, start=-200, stop=200):
    frames = []
    for tstamp in trials:
        frame = utils_misc.find_nearest(wf_timestamps, tstamp)
        data = nwb_read.get_widefield_dff0(nwb_file, ['ophys', 'dff0'], int(frame + start), int(frame + stop))
        if data.shape != (len(np.arange(start, stop)), 125, 160):
            continue
        frames.append(np.nanmean(data, axis=0))

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


def logregress_model(X, y_binary):

    # Step 1: Split data into training and temporary set (validation + test)
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=0)

    # Step 2: Split the temporary set into validation and test sets
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=0)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    scores = cross_val_score(model, X, y_binary, cv=50)
    output = "Model CV finished with %0.2f accuracy and a standard deviation of %0.2f" % (scores.mean(), scores.std())
    os.system('echo ' + output)

    y_pred = model.predict(X_val)

    results = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "cross_val_scores": scores,
        "coefficients": model.coef_,
    }

    os.system('echo "Model trained successfully"')
    output = f"Accuracy: {round(accuracy_score(y_val, y_pred), 2)}"
    os.system('echo ' + output)
    output = f"Precision: {round(precision_score(y_val, y_pred), 2)}"
    os.system('echo ' + output)

    return results


def logregress_shuffle(X, y_binary, n_shuffles=1000):

    coefficients = np.zeros([n_shuffles, X.shape[1]])
    accuracy = np.zeros(coefficients.shape[0])
    precision = np.zeros(coefficients.shape[0])
    os.system("echo 'Starting logress_shuffle'")

    for i in range(n_shuffles):
        if i % 100 ==0:
            output = f"Executed {i} iterations"
            os.system("echo " + output)
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


def compute_logreg_and_shuffle(image, y_binary):

    image = gaussian_filter(np.nan_to_num(image, 0), sigma=(0, 2, 2))
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # image_scaled = scaler.fit_transform(image.reshape(image.shape[0], -1).T).T
    image = image.reshape(image.shape[0], -1)
    image_scaled = image - np.nanmean(image, axis=0)
    image_scaled = image_scaled.T - np.nanmean(image_scaled, axis=1)

    results = logregress_model(image_scaled.T, y_binary)
    accuracy_shuffle, precision_shuffle, coefficients_shuffle = logregress_shuffle(image_scaled.T, y_binary)

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

    return results


def logregress_classification(nwb_file, classify_by, decode, n_chunks, output_path):
    os.system("echo 'Widefield image classification'")

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
    else:
        os.system('echo "Wrong period to decode, valid options are stim or baseline"')
        return 0

    results_total = pd.DataFrame()
    mouse_id = nwb_read.get_mouse_id(nwb_file)
    session_id = nwb_read.get_session_id(nwb_file)
    output = f"Analyzing session {session_id}"
    os.system("echo " + output)
    session_type = nwb_read.get_session_type(nwb_file)

    if 'wf' not in session_type:
        os.system(f'echo "{session_id} is not a widefield session"')
        return 1

    save_path = os.path.join(output_path, f"{classify_by}_decoding")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    trial_table = nwb_read.get_trial_table(nwb_file)
    trial_table['correct_choice'] = trial_table.reward_available == trial_table.lick_flag
    wf_timestamps = nwb_read.get_widefield_timestamps(nwb_file, ['ophys', 'dff0'])

    if classify_by == 'context':
        y_binary = trial_table.loc[trial_table.correct_choice, 'context'].values
        trials = trial_table.loc[trial_table.correct_choice, 'start_time']

    elif classify_by == 'lick':
        y_binary = trial_table.lick_flag.values
        trials = trial_table.start_time

    elif classify_by == 'tone':
        y_binary = trial_table.context_background.map({'pink': 1, 'brown': 0})
        trials = trial_table.start_time
    else:
        os.system('echo"Wrong thing to decode"')
        return 1

    # split = np.linspace(0, 200, n_chunks, endpoint=False)
    # step = np.unique(np.diff(split))[0]

    for i, st in enumerate(split):
        image = get_frames_by_epoch(nwb_file, trials, wf_timestamps, start=int(start + st), stop=int(start + st + step))
        if len(y_binary) != image.shape[0]:
            os.system('echo "Different number of trials and wf frames"')
            difference = len(y_binary) - image.shape[0]
            if difference == 1:
                os.system('echo "One more trial than wf frames, removing"')
                y_binary = y_binary[:-1]
            elif difference == -1:
                image = image[:-1]

        results = compute_logreg_and_shuffle(image, y_binary)

        results['mouse_id'] = mouse_id
        results['session_id'] = session_id
        results['start_frame'] = st
        results['stop_frame'] = st + step

        results_total = results_total.append(results, ignore_index=True)
        print(f"Analysis ran successfully for chunk {i}, continuing")

    # start= -200
    # stop = 0

    image = get_frames_by_epoch(nwb_file, trials, wf_timestamps, start=start, stop=stop)
    if len(y_binary) != image.shape[0]:
        os.system('echo "Different number of trials and wf frames"')
        difference = len(y_binary) - image.shape[0]
        if difference == 1:
            os.system('echo "One more trial than wf frames, removing"')
            y_binary = y_binary[:-1]
        elif difference == -1:
            image = image[:-1]

    results = compute_logreg_and_shuffle(image, y_binary)

    results['mouse_id'] = mouse_id
    results['session_id'] = session_id
    results['start_frame'] = start
    results['stop_frame'] = stop

    results_total = results_total.append(results, ignore_index=True)

    results_total.to_json(os.path.join(save_path, "results.json"))

    return 0


if __name__ == "__main__":
    nwb_file = sys.argv[1]
    output_path = sys.argv[2]
    classify_by = sys.argv[3]
    decode = sys.argv[4]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(" ")
    print(f"nwb_files : {nwb_file}")
    logregress_classification(nwb_file, classify_by=classify_by, decode=decode, n_chunks=5, output_path=output_path)
