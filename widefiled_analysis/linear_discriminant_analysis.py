import os
import sys
sys.path.append(".")
import random
import numpy as np
import pandas as pd
import nwb_wrappers.nwb_reader_functions as nwb_read

import warnings
warnings.filterwarnings("ignore")

from utils.wf_plotting_utils import reduce_im_dimensions
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from nwb_utils import server_path, utils_misc, utils_behavior

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns


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


def lda_analysis(image, y_binary, correct, result_path):
    session = result_path.split("\\")[-1]

    im_mean, im_std = np.nanmean(image, axis=0), np.nanstd(image, axis=0)  # z-score data with the same transformation as done in the train set
    z_image = np.nan_to_num((image - im_mean) / im_std, nan=0)
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(z_image, y_binary)

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.suptitle(session)
    ax.hist(X_lda[y_binary == 0], color='blue', label="Non-rewarded", alpha=0.7, density=True)
    ax.hist(X_lda[y_binary == 1], color='red', label="Rewarded", alpha=0.7, density=True)
    ax.set_xlabel("LDA Component Score")
    ax.set_ylabel("Density")
    # fig.show()
    for ext in ['.png', '.svg']:
        fig.savefig(os.path.join(result_path, f"hist_lda{ext}"))

    df_lda = pd.DataFrame({"LDA Score": X_lda.ravel(), "Context": y_binary})
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.violinplot(x="Context", y="LDA Score", data=df_lda, palette=["blue", "red"])
    fig.suptitle(f"{session} Violin Plot of LDA Scores")
    # fig.show()
    for ext in ['.png', '.svg']:
        fig.savefig(os.path.join(result_path, f"violin_lda{ext}"))

    ## Correct choice only
    corr_im = image[np.asarray(correct, dtype=bool), :]
    im_mean, im_std = np.nanmean(corr_im, axis=0), np.nanstd(corr_im, axis=0)  # z-score data with the same transformation as done in the train set
    z_image = np.nan_to_num((corr_im - im_mean) / im_std, nan=0)
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(z_image, y_binary[np.asarray(correct, dtype=bool)])
    
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.suptitle(session)
    ax.hist(X_lda[y_binary[np.asarray(correct, dtype=bool)] == 0], color='blue', label="Non-rewarded", alpha=0.7, density=True)
    ax.hist(X_lda[y_binary[np.asarray(correct, dtype=bool)] == 1], color='red', label="Rewarded", alpha=0.7, density=True)
    ax.set_xlabel("LDA Component Score")
    ax.set_ylabel("Density")
    # fig.show()
    for ext in ['.png', '.svg']:
        fig.savefig(os.path.join(result_path, f"hist_lda_correct{ext}"))


def compute_logreg_and_shuffle(image, y_binary, correct_choice, result_path):

    image = gaussian_filter(np.nan_to_num(image, 0), sigma=(0, 2, 2))

    image, coords = reduce_im_dimensions(image)
    # image = image.reshape(image.shape[0], -1)

    # ## simulate image
    # image = np.zeros([y_binary.shape[0], 42])
    # image[y_binary == 0, :] = np.random.normal(np.zeros(image.shape[1]), 0.5)
    # image[y_binary == 1, :] = np.random.normal(np.zeros(image.shape[1]), 0.5) + 1
    # np.save(os.path.join(result_path, 'dim_red_coords.npy'), np.asarray(coords))
    lda = lda_analysis(image, y_binary, correct_choice, result_path)


def logregress_classification(nwb_file, classify_by, decode, n_chunks, output_path):
    os.system("echo 'Widefield image classification'")

    if decode == 'baseline':
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
        results = compute_logreg_and_shuffle(image, y_binary, correct, result_path = save_path)


if __name__ == "__main__":

    import yaml
    for state in ['expert', 'naive']:
        config_file = f"//sv-nas1.rcp.epfl.ch/Petersen-Lab/z_LSENS/Share/Pol_Bech/Session_list/context_sessions_gcamp_{state}.yaml"
        with open(config_file, 'r', encoding='utf8') as stream:
            config_dict = yaml.safe_load(stream)

        nwb_files = config_dict['Session path']

        if os.path.exists(nwb_files[0]):
            subject_ids = list(np.unique([nwb_read.get_mouse_id(file) for file in nwb_files]))
        else:
            subject_ids = list(np.unique([session[0:5] for session in nwb_files]))

        output_path = os.path.join(f'{server_path.get_experimenter_saving_folder_root("PB")}',
                                   'Pop_results', 'Context_behaviour', f'linear_discriminant_analysis')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # nwb_files = [file for file in nwb_files if 'PB' in file]
        for decode in ['baseline']:
            for classify_by in ['context']:
                for nwb_file in nwb_files:
                    mouse_id = nwb_read.get_mouse_id(nwb_file)
                    session_id = nwb_read.get_session_id(nwb_file)
                    result_path = os.path.join(output_path, state, decode, classify_by, mouse_id, session_id)
                    if not os.path.exists(result_path):
                        os.makedirs(result_path)

                    logregress_classification(nwb_file, classify_by=classify_by, decode=decode, n_chunks=1,
                                            output_path=result_path)