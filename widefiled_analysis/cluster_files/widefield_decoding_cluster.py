import os
import sys
sys.path.append("/home/bechvila/NWB_analysis")
import random
import numpy as np
import pandas as pd
import nwb_wrappers.nwb_reader_functions as nwb_read

import warnings
warnings.filterwarnings("ignore")

from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
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


def correct_vs_incorrect_logress_model(X, y_binary, correct_choice, result_path):
    trials = np.arange(y_binary.shape[0])
    block_id = np.abs(np.diff(y_binary, prepend=0)).cumsum()

    if len(np.where(block_id == block_id[-1])[0])<20:
        block_id = block_id[np.where(block_id != block_id[-1])[0]]
        trials = trials[:len(block_id)]
        correct_choice = correct_choice[:len(block_id)]

    if len(np.unique(block_id)) % 2 != 0:
        block_id = block_id[np.where(block_id != block_id[-1])]
        trials = trials[:len(block_id)]
        correct_choice = correct_choice[:len(block_id)]

    even = np.unique(block_id)[::2]
    odd = np.unique(block_id)[1::2]
    n_test_blocks = np.ceil(odd.shape[0] * 0.2).astype(int)

    avg_results = []
    trial_based_accuracy = []
    accuracy = []
    coefficients = []
    fpr_total = []
    tpr_total = []
    roc_total = []
    thres_total = []
    for i in range(200):

        test_blocks = random.sample(even.tolist(), n_test_blocks)
        test_blocks.extend(random.sample(odd.tolist(), n_test_blocks))

        test = trials[np.isin(block_id, test_blocks)]
        # test = sorted(random.sample(trials.tolist(), round(trials.shape[0] * 0.2)))
        train = [trial for trial in trials if trial not in test]

        x_train, y_train = X[train], y_binary[train]
        x_test, y_test = X[test], y_binary[test]

        train_mean, train_std = np.nanmean(x_train, axis=0), np.nanstd(x_train, axis=0)
        z_train, z_test = (x_train-train_mean)/train_std, (x_test-train_mean)/train_std

        model = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.01)
        model.fit(z_train, y_train)


    return

def trialbased_logregress_model(X, y_binary, result_path):

    trials = np.arange(y_binary.shape[0])
    block_id = np.abs(np.diff(y_binary, prepend=0)).cumsum()

    if len(np.where(block_id == block_id[-1])[0])<20: # Remove incomplete blocks at the end of session
        block_id = block_id[np.where(block_id != block_id[-1])[0]]
        trials = trials[:len(block_id)]

    if len(np.unique(block_id)) % 2 != 0: # Take same number of blocks for each context
        block_id = block_id[np.where(block_id != block_id[-1])]
        trials = trials[:len(block_id)]

    even = np.unique(block_id)[::2]
    odd = np.unique(block_id)[1::2]
    n_test_blocks = np.ceil(odd.shape[0] * 0.2).astype(int) # select number of blocks for test (20%)

    avg_results = []
    trial_based_accuracy = []
    accuracy = []
    coefficients = []
    fpr_total = []
    tpr_total = []
    roc_total = []
    thres_total = []
    for i in range(200): # 200 folds

        test_blocks = random.sample(even.tolist(), n_test_blocks)
        test_blocks.extend(random.sample(odd.tolist(), n_test_blocks))

        test = trials[np.isin(block_id, test_blocks)]
        # test = sorted(random.sample(trials.tolist(), round(trials.shape[0] * 0.2)))
        train = [trial for trial in trials if trial not in test]

        x_train, y_train = X[train], y_binary[train]
        x_test, y_test = X[test], y_binary[test]

        train_mean, train_std = np.nanmean(x_train, axis=0), np.nanstd(x_train, axis=0) # z-score data with the same transformation as done in the train set
        z_train, z_test = np.nan_to_num((x_train - train_mean) / train_std, nan=0), np.nan_to_num((x_test - train_mean) / train_std, nan=0)

        #model = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.01)
        model = LogisticRegression(solver='lbfgs', penalty='l2')
        model.fit(z_train, y_train)

        y_pred = model.predict(z_test)
        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(z_test)[:, 1])
        accuracy += [accuracy_score(y_test, y_pred)]
        coefficients += [model.coef_]
        fpr_total += [fpr]
        tpr_total += [tpr]
        thres_total += [thresholds]
        roc_total += [roc_auc_score(y_test, model.predict_proba(z_test)[:, 1])]
        try:
            trial_based_results = {
                'data': ['full_model' for j in range(len(test))],
                'iter': [i for j in range(len(test))],
                'block_id': block_id[test],
                'trials': test,
                'true': y_test,
                'prediction': y_pred,
                'correct': [1 if y_test[trial] == y_pred[trial] else 0 for trial in range(len(test))]}
            trial_based_accuracy += [trial_based_results]
        except:
            output = f"data = 'full_model', y_pred len = {len(y_pred)}, y_test len = {len(y_test)}"
            os.system("echo " + output)

    avg_results += [{
        "data": 'full_model',
        "accuracy": accuracy,
        "fpr": fpr_total,
        "tpr": tpr_total,
        "thresholds": thres_total,
        "roc": roc_total
    }]
    np.save(os.path.join(result_path, 'coefficients.npy'), coefficients)

    for sh_index in range(1000):
        if sh_index % 100 ==0:
            output = f"Executed {sh_index} shuffles"
            os.system("echo " + output)

        accuracy = []
        coefficients = []
        fpr_total = []
        tpr_total = []
        roc_total = []
        thres_total = []

        ## Block shuffle
        shuffle_idx = np.hstack([np.where(block_id == i) for i in np.random.permutation(np.unique(block_id))])[0]
        shuffle = y_binary[shuffle_idx]

        for i in range(200):
            even = np.unique(block_id)[::2]
            odd = np.unique(block_id)[1::2]
            n_test_blocks = np.ceil(odd.shape[0] * 0.2).astype(int)

            test_blocks = random.sample(even.tolist(), n_test_blocks)
            test_blocks.extend(random.sample(odd.tolist(), n_test_blocks))

            test = trials[np.isin(block_id[shuffle_idx], test_blocks)]
            train = [trial for trial in trials if trial not in test]

            x_train, y_train = X[train], shuffle[train]
            x_test, y_test = X[test], shuffle[test]

            train_mean, train_std = np.nanmean(x_train, axis=0), np.nanstd(x_train, axis=0)
            z_train, z_test = np.nan_to_num((x_train - train_mean) / train_std, nan=0), np.nan_to_num((x_test - train_mean) / train_std, nan=0)

            #model = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.01)
            model = LogisticRegression(solver='lbfgs', penalty='l2')
            model.fit(z_train, y_train)

            y_pred = model.predict(z_test)
            fpr, tpr, thresholds = roc_curve(y_test, y_pred)
            accuracy += [accuracy_score(y_test, y_pred)]
            coefficients += [model.coef_]
            fpr_total += [fpr]
            tpr_total += [tpr]
            thres_total += [thresholds]
            roc_total += [roc_auc_score(y_test, model.predict_proba(z_test)[:, 1])]
            try:
                trial_based_results = {
                    'data': ['shuffle' for j in range(len(test))],
                    'iter': [i for j in range(len(test))],
                    'block_id': block_id[shuffle_idx][test],
                    'trials': test,
                    'true': y_test,
                    'prediction': y_pred,
                    'correct': [1 if y_test[trial] == y_pred[trial] else 0 for trial in range(len(test))]}
                trial_based_accuracy += [trial_based_results]
            except:
                output = f"data = 'shuffle', iter = {i}, y_pred len = {len(y_pred)}, y_test len = {len(y_test)}"
                os.system("echo " + output)

        avg_results += [{
            "data": 'shuffle',
            "accuracy": accuracy,
            "fpr": fpr_total,
            "tpr": tpr_total,
            "thresholds": thres_total,
            "roc": roc_total
        }]

    np.save(os.path.join(result_path, 'coefficients_shuffle.npy'), coefficients)

    trial_based_accuracy = pd.concat([pd.DataFrame(res) for res in trial_based_accuracy])
    trial_based_accuracy.to_csv(os.path.join(result_path, 'trial_based_scores.csv'))

    avg_results = pd.concat([pd.DataFrame(res) for res in avg_results])
    avg_results.to_json(os.path.join(result_path, 'results.json'))

    return 0

def compute_logreg_and_shuffle(image, y_binary, correct_choice, result_path):

    image = gaussian_filter(np.nan_to_num(image, 0), sigma=(0, 2, 2))
    image = image.reshape(image.shape[0], -1)
    trialbased_logregress_model(image, y_binary, result_path=result_path)
    # correct_vs_incorrect_logress_model(image, y_binary, correct_choice, result_path=result_path)

    return 0


def logregress_classification(nwb_file, classify_by, decode, n_chunks, output_path):
    os.system("echo 'Widefield image classification'")

    if decode == 'baseline':
        split = np.linspace(0, 50, n_chunks, endpoint=False)
        step = np.unique(np.diff(split))[0]
        start = -50
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
    wf_timestamps = nwb_read.get_widefield_timestamps(nwb_file, ['ophys', 'dff0'])

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

    if classify_by == 'context':
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

    if decode == 'stim':
        for i, st in enumerate(split):
            baseline = get_frames_by_epoch(nwb_file, trials, wf_timestamps, start=-50, stop=0)
            image = get_frames_by_epoch(nwb_file, trials, wf_timestamps, start=int(start + st), stop=int(start + st + step))
            image = image-baseline # remove baseline effects from stim classification
            if len(y_binary) != image.shape[0]:
                os.system('echo "Different number of trials and wf frames"')
                difference = len(y_binary) - image.shape[0]
                if difference == 1:
                    os.system('echo "One more trial than wf frames, removing"')
                    y_binary = y_binary[:-1]
                elif difference == -1:
                    image = image[:-1]

            save_path = os.path.join(save_path, f'split_{int(start + st)}_{int(start + st + step)}')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            results = compute_logreg_and_shuffle(image, y_binary, trial_table['correct_trial'], result_path=save_path)

            # results['mouse_id'] = mouse_id
            # results['session_id'] = session_id
            # results['start_frame'] = int(start + st)
            # results['stop_frame'] = int(start + st + step)

            # results_total = results_total.append(results, ignore_index=True)
            print(f"Analysis ran successfully for chunk {i}, continuing")

    else:
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
        results = compute_logreg_and_shuffle(image, y_binary, trial_table['correct_trial'], result_path = save_path)

    #     results['mouse_id'] = mouse_id
    #     results['session_id'] = session_id
    #     results['start_frame'] = start
    #     results['stop_frame'] = stop
    #
    # results_total = results_total.append(results, ignore_index=True)
    #
    # results_total.to_json(os.path.join(save_path, "results.json"))

    return 0


if __name__ == "__main__":

    if 'COMPUTERNAME' in os.environ.keys():
        import yaml
        for state in ['naive', 'expert']:
            config_file = f"//sv-nas1.rcp.epfl.ch/Petersen-Lab/z_LSENS/Share/Pol_Bech/Session_list/context_sessions_gcamp_{state}.yaml"
            with open(config_file, 'r', encoding='utf8') as stream:
                config_dict = yaml.safe_load(stream)

            nwb_files = config_dict['Session path']

            if os.path.exists(nwb_files[0]):
                subject_ids = list(np.unique([nwb_read.get_mouse_id(file) for file in nwb_files]))
            else:
                subject_ids = list(np.unique([session[0:5] for session in nwb_files]))

            output_path = os.path.join(f'{server_path.get_experimenter_saving_folder_root("PB")}',
                                       'Pop_results', 'Context_behaviour', f'widefield_decoding_area_gcamp_{state}_saga_elasticnet_001')

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # nwb_files = [file for file in nwb_files if 'PB' in file]
            for decode in ['baseline', 'stim']:
                for classify_by in ['context', 'lick', 'tone']:
                    for nwb_file in nwb_files:
                        logregress_classification(nwb_file, classify_by=classify_by, decode=decode, n_chunks=5,
                                                output_path=output_path)

    else:
        nwb_file = sys.argv[1]
        output_path = sys.argv[2]
        classify_by = sys.argv[3]
        decode = sys.argv[4]

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        print(" ")
        print(f"nwb_files : {nwb_file}")
        logregress_classification(nwb_file, classify_by=classify_by, decode=decode, n_chunks=5, output_path=output_path)
