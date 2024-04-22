import os
import sys
sys.path.append("/home/bechvila/NWB_analysis")
import numpy as np
import pandas as pd
import nwb_wrappers.nwb_reader_functions as nwb_read

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from nwb_utils import server_path, utils_misc, utils_behavior


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
    for i in range(n_shuffles):
        if i % 100 ==0:
            print(f"Executed {i} iterations")
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

    results = logregress_model(image_scaled, y_binary)

    print("Starting shuffle")
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


def logregress_classification(nwb_file, classify_by, n_chunks, output_path):
    print(f"Widefield image classification")

    results_total = pd.DataFrame()
    mouse_id = nwb_read.get_mouse_id(nwb_file)
    session_id = nwb_read.get_session_id(nwb_file)
    print(" ")
    print(f"Analyzing session {session_id}")
    session_type = nwb_read.get_session_type(nwb_file)
    if 'wf' not in session_type:
        print(f"{session_id} is not a widefield session")
        return 0

    if n_chunks == 1:
        save_path = os.path.join(output_path, f"{classify_by}_decoding", f"{mouse_id}", f"{session_id}")
    else:
        save_path = os.path.join(output_path, f"{classify_by}_decoding", f"{mouse_id}", f"{session_id}", "by_timebins")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    trial_table = nwb_read.get_trial_table(nwb_file)
    epochs = nwb_read.get_behavioral_epochs_names(nwb_file)
    wf_timestamps = nwb_read.get_widefield_timestamps(nwb_file, ['ophys', 'dff0'])

    print("Loading WF frames")
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
        print(f"Analyzing split {i}/{n_chunks}: {(start - 200) * 1000 / 100} to {(start - 200 + step)*1000/100}")
        image = np.nanmean(data_frames[:, int(start):int(start + step), :, :], axis=(1))
        results = compute_logreg_and_shuffle(image, y_binary, classify_by, os.path.join(save_path, f"{classify_by}_image_stats_chunk{i}"))

        results['mouse_id'] = mouse_id
        results['session_id'] = session_id
        results['start_frame'] = start
        results['stop_frame'] = start + step
        np.save(os.path.join(save_path, f"{classify_by}_model_scores_chunk{i}.npy"), results)

        results_total = results_total.append(results, ignore_index=True)
        print(f"Analysis ran successfully for chunk {i}, continuing")

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

    results_total.to_csv(os.path.join(output_path, "results.csv"))

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