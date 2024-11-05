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
from sklearn.model_selection import cross_val_score, StratifiedKFold
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
    if len(trials) == 0:
        data = []
    else:
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


def logregress_model(df, y_binary, result_path, strat=True):
    # Step 1: Split data into training and temporary set (validation + test)
    if strat:
        stratify = df.regressor
    else:
        stratify = None

    X = np.stack(df.wf_scaled.to_numpy()).squeeze()
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=0, stratify=stratify)

    # Step 2: Split the temporary set into validation and test sets
    # X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=0)

    # model = LogisticRegression()
    model = LogisticRegression(penalty='l2', C=0.2)

    model.fit(X_train, y_train)

    skf = StratifiedKFold(n_splits=10)
    scoring = ['accuracy', 'precision', 'f1', 'recall', 'r2', 'explained_variance', 'neg_mean_squared_error', 'roc_auc']
    from sklearn.model_selection import cross_validate
    scoring = ['accuracy', 'precision', 'f1', 'recall', 'r2', 'explained_variance', 'neg_mean_squared_error', 'roc_auc']
    scores = pd.DataFrame.from_dict(cross_validate(model, X, y_binary, cv=skf, scoring=scoring, return_train_score=True))
    scores.to_csv(os.path.join(result_path, 'cross_validated_scores.csv'))
    print("Model CV finished with %0.2f accuracy and a standard deviation of %0.2f" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std()))

    y_pred_prob = model.predict_proba(X_test)[:, 1]

    import seaborn as sns
    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC: {roc_auc_score(y_test, y_pred_prob):.2f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='best')
    fig.savefig(os.path.join(result_path, 'roc_curve.png'))
    fig.savefig(os.path.join(result_path, 'roc_curve.svg'))

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

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
    y_pred = model.predict(X_test)
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

    results = {
        "accuracy": accuracy_score(y_test, model.predict(X_test)),
        "precision": precision_score(y_test, model.predict(X_test)),
        "cross_val_scores": scores,
        "coefficients": model.coef_,
    }

    print("Model trained successfully")
    print(f"Accuracy: {round(accuracy_score(y_test,  model.predict(X_test)), 2)}; ", f"Precision: {precision_score(y_test,  model.predict(X_test))}; ",
          f"CV score: {round(scores.mean(), 2)}")

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

        # Step 2: Split the temporary set into validation and test sets
        # X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=0)

        model = LogisticRegression(penalty='l2', C=0.2)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        coefficients[i] = model.coef_
        accuracy[i] = accuracy_score(y_test, y_pred)
        precision[i] = precision_score(y_test, y_pred)

    return accuracy, precision, coefficients


def logregress_leaveoneout(results, X, y_binary, label):
    # Step 1: Split data into training and temporary set (validation + test)

    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=0, stratify=y_binary)

    # Step 2: Split the temporary set into validation and test sets
    # X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=0, stratify=df.regressor)

    model = LogisticRegression(penalty='l2', C=0.2)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results[f"{label}_accuracy"] = accuracy_score(y_test, y_pred)
    results[f"{label}_precision"] = precision_score(y_test, y_pred)
    results[f"{label}_coefficients"] = model.coef_

    print(f"Leave {label} out model trained successfully")
    print(f"Accuracy: {round(accuracy_score(y_test, y_pred), 2)}; ",
          f"Precision: {precision_score(y_test, y_pred)}")

    return results


def compute_logreg_and_shuffle(df, y_binary, classify_by, result_path):

    # scaler = MinMaxScaler(feature_range=(0, 1))
    # image_scaled = image.groupby('session_id')['wf_mean'].transform(lambda x: [i for i in scaler.fit_transform(np.stack(x).T).T])
    # image_scaled = np.stack(image_scaled)
    # image_scaled = scaler.fit_transform(image.T)
    image_scaled = []
    for session, group in df.groupby('session_id'):
        image = np.stack(group.wf_mean.to_numpy())
        image = image - np.nanmean(image, axis=0)
        # image_scaled += [(image.T - np.nanmean(image, axis=1)).T]
        image_scaled += [image]

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

    for i in range(matrix.shape[0]):
        image_scaled = []
        for session, group in df.groupby('session_id'):
            image = np.stack(group.wf_mean.to_numpy())
            image = image[:, np.where(matrix[i] == 1)].squeeze()
            image = image - np.nanmean(image, axis=0)
            # image_scaled += [(image.T - np.nanmean(image, axis=1)).T]
            image_scaled += [image]
        results = logregress_leaveoneout(results, np.vstack(image_scaled), y_binary, labels[i])

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


def get_data(nwb_files, decode):
    wf_data = []
    for nwb_file in nwb_files:
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
        wf_data += [trial_table]

    return pd.concat(wf_data, ignore_index=True)


if __name__ == "__main__":

    config_file = f"//sv-nas1.rcp.epfl.ch/Petersen-Lab/z_LSENS/Share/Pol_Bech/Session_list/context_sessions_gcamp_naive.yaml"
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
                               'Pop_results', 'Context_behaviour', 'widefield_decoding_area_gcamp_naive_sept_2024_allmice_l2reg')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # nwb_files = [file for file in nwb_files if 'PB' in file]
    for decode in ['baseline', 'stim']:
        results = get_data(nwb_files, decode=decode)
        for classify_by in ['context', 'lick', 'tone']:
            logregress_classification(results, classify_by=classify_by, decode=decode, n_chunks=10,
                                      output_path=output_path)
