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


def get_frames_by_epoch(nwb_file, trials, wf_timestamps):
    os.system("echo 'Getting wf frames'")
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
    output = f"Accuracy: {round(accuracy_score(y_val, y_pred), 2)}; Precision: {precision_score(y_val, y_pred)}; CV score: {round(scores.mean(), 2)}"
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
    scaler = MinMaxScaler(feature_range=(-1, 1))
    image_scaled = scaler.fit_transform(image.reshape(image.shape[0], -1).T).T

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

    return results


def logregress_classification(nwb_file, classify_by, n_chunks, output_path):
    os.system("echo 'Widefield image classification'")

    results_total = pd.DataFrame()
    mouse_id = nwb_read.get_mouse_id(nwb_file)
    session_id = nwb_read.get_session_id(nwb_file)
    output = f"Analyzing session {session_id}"
    os.system("echo " + output)
    session_type = nwb_read.get_session_type(nwb_file)
    if 'wf' not in session_type:
        print(f"{session_id} is not a widefield session")
        return 0

    save_path = os.path.join(output_path, f"{classify_by}_decoding")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    trial_table = nwb_read.get_trial_table(nwb_file)
    trial_table['correct_choice'] = trial_table.reward_available == trial_table.lick_flag
    wf_timestamps = nwb_read.get_widefield_timestamps(nwb_file, ['ophys', 'dff0'])

    if classify_by == 'context':
        y_binary = trial_table.loc[trial_table.correct_choice, 'context'].values
        data_frames = get_frames_by_epoch(nwb_file, trial_table.loc[trial_table.correct_choice, 'start_time'], wf_timestamps)

    elif classify_by == 'lick':
        y_binary = trial_table.lick_flag.values
        data_frames = get_frames_by_epoch(nwb_file, trial_table.start_time, wf_timestamps)

    elif classify_by == 'tone':
        y_binary = trial_table.context_background.map({'pink': 1, 'brown': 0})
        data_frames = get_frames_by_epoch(nwb_file, trial_table.start_time, wf_timestamps)


    if n_chunks == 1:
        split = np.array([0])
        step = int(data_frames.shape[1]/2)
    else:
        split = np.linspace(0, int(data_frames.shape[1]/2), n_chunks, endpoint=False)
        step = np.unique(np.diff(split))[0]

    if len(y_binary) != data_frames.shape[0]:
        os.system('echo "Different number of trials and wf frames"')
        difference = len(y_binary) - data_frames.shape[0]
        if difference == 1:
            os.system('echo "One more trial than wf frames, removing"')
            y_binary = y_binary[:-1]

    for i, start in enumerate(split):
        print(f"Analyzing split {i}/{n_chunks}: {(start - 200) * 1000 / 100} to {(start - 200 + step)*1000/100}")
        image = np.nanmean(data_frames[:, int(start):int(start + step), :, :], axis=1)
        results = compute_logreg_and_shuffle(image, y_binary)

        results['mouse_id'] = mouse_id
        results['session_id'] = session_id
        results['start_frame'] = start
        results['stop_frame'] = start + step

        results_total = results_total.append(results, ignore_index=True)
        print(f"Analysis ran successfully for chunk {i}, continuing")

    start= 0
    stop = 200

    image = np.nanmean(data_frames[:, int(start):int(stop), :, :], axis=(1))
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

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(" ")
    print(f"nwb_files : {nwb_file}")
    logregress_classification(nwb_file, classify_by='context', n_chunks=10, output_path=output_path)
    logregress_classification(nwb_file, classify_by='lick', n_chunks=10, output_path=output_path)