import os
import sys
sys.path.append("/home/bechvila/NWB_analysis")
import numpy as np
import pandas as pd
import nwb_wrappers.nwb_reader_functions as nwb_read

import warnings
warnings.filterwarnings("ignore")

from scipy.ndimage import gaussian_filter
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve
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


def logregress_model(X, y_binary, result_path, strat=True):
    if strat:
        stratify = y_binary
    else:
        stratify = None

    # Step 1: Split data into training and test set (train + test)
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=0, stratify=stratify)

    model = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.01)

    skf = StratifiedKFold(n_splits=10)
    scoring = ['accuracy', 'precision', 'f1', 'recall', 'r2', 'explained_variance', 'neg_mean_squared_error', 'roc_auc']
    scores = pd.DataFrame.from_dict(cross_validate(model, X, y_binary, cv=skf, scoring=scoring, return_train_score=True))
    scores.to_csv(os.path.join(result_path, 'cross_validated_scores.csv'))

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "coefficients": model.coef_,
        "fpr": fpr,
        "tpr": tpr,
        "conf_mat_tp": tn,
        "conf_mat_fp": fp,
        "conf_mat_fn": fn,
        "conf_mat_tn": fp,
    }

    os.system('echo "Model trained successfully"')
    output = f"Accuracy: {round(accuracy_score(y_test, y_pred), 2)}"
    os.system('echo ' + output)
    output = f"Precision: {round(precision_score(y_test, y_pred), 2)}"
    os.system('echo ' + output)

    return results


def logregress_shuffle(X, y_binary, classify_by, n_shuffles=1000, strat=True):

    coefficients = np.zeros([n_shuffles, X.shape[1]])
    accuracy = np.zeros(coefficients.shape[0])
    precision = np.zeros(coefficients.shape[0])
    os.system("echo 'Starting logress_shuffle'")
    trials = np.arange(len(y_binary))

    for i in range(n_shuffles):
        if i % 100 ==0:
            output = f"Executed {i} iterations"
            os.system("echo " + output)

        if classify_by == 'context':
            block_id = np.abs(np.diff(y_binary, prepend=0)).cumsum()
            shuffle_idx = np.hstack([np.where(block_id == i) for i in np.random.permutation(np.unique(block_id))])[0]
            shuffle = y_binary[shuffle_idx]
        else:
            shuffle = np.random.permutation(y_binary)

        if strat:
            stratify = shuffle
        else:
            stratify = None

        # Step 1: Split data into training and temporary set (validation + test)
        X_train, X_test, y_train, y_test = train_test_split(X, shuffle, test_size=0.2, random_state=0, stratify=stratify)
        
        model = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.01)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        coefficients[i] = model.coef_
        accuracy[i] = accuracy_score(y_test, y_pred)
        precision[i] = precision_score(y_test, y_pred)

    return accuracy, precision, coefficients


def compute_logreg_and_shuffle(image, y_binary, classify_by, result_path):

    image = gaussian_filter(np.nan_to_num(image, 0), sigma=(0, 2, 2))
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # image_scaled = scaler.fit_transform(image.reshape(image.shape[0], -1).T).T
    image = image.reshape(image.shape[0], -1)
    image_scaled = image - np.nanmean(image, axis=0)
    image_scaled = image_scaled.T - np.nanmean(image_scaled, axis=1)

    results = logregress_model(image_scaled.T, y_binary, result_path=result_path, strat=True)
    accuracy_shuffle, precision_shuffle, coefficients_shuffle = logregress_shuffle(image_scaled.T, y_binary, classify_by)

    alpha = 0.95
    coef_mean, coef_std_error, lower_bound, upper_bound = ols_statistics(coefficients_shuffle, confidence=alpha)
    results['accuracy_shuffle'] = accuracy_shuffle
    results['precision_shuffle'] = precision_shuffle
    #results['coefficients_shuffle'] = coefficients_shuffle
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
    #trial_table['correct_choice'] = trial_table.reward_available == trial_table.lick_flag
    wf_timestamps = nwb_read.get_widefield_timestamps(nwb_file, ['ophys', 'dff0'])

    if classify_by == 'context':
        correct = []
        for i, x in trial_table.iterrows():
            if x['trial_type'] == 'auditory_trial' and x['lick_flag'] == 1:
                correct += [1]
            elif x['trial_type'] == 'whisker_trial' and x['context'] == 1 and x['lick_flag'] == 1:
                correct += [1]
            elif x['trial_type'] == 'whisker_trial' and x['context'] == 0 and x['lick_flag'] == 0:
                correct += [1]
            elif x['trial_type'] == 'no_stim' and x['lick_flag'] == 0:
                correct += [1]
            else:
                correct += [0]

        trial_table['correct_trial'] = correct
        #y_binary = trial_table.loc[trial_table.correct_trial==1, 'context'].values
        #trials = trial_table.loc[trial_table.correct_trial==1, 'start_time']
        y_binary = trial_table.context.values
        trials = trial_table.start_time

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

        results = compute_logreg_and_shuffle(image, y_binary, classify_by, result_path = save_path)

        results['mouse_id'] = mouse_id
        results['session_id'] = session_id
        results['start_frame'] = int(start + st)
        results['stop_frame'] = int(start + st + step)

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

    os.system(f'echo "Classify by {classify_by}"')
    results = compute_logreg_and_shuffle(image, y_binary, classify_by, result_path = save_path)

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
