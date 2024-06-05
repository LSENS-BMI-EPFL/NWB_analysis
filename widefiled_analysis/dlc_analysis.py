import os
import re

import matplotlib.pyplot as plt
import yaml
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import nwb_wrappers.nwb_reader_functions as nwb_read

import warnings
warnings.filterwarnings("ignore")

from nwb_utils import utils_misc
from nwb_utils import server_path, utils_misc, utils_behavior


def filter_part_by_camview(view):
    if view == 'side':
        return ['jaw_angle', 'jaw_distance', 'jaw_velocity',
                'nose_angle', 'nose_distance',
                'particle_x', 'particle_y',
                'pupil_area', 'spout_y',
                'tongue_angle', 'tongue_distance', 'tongue_velocity']

    elif view == 'top':
        return ['top_nose_angle', 'top_nose_distance', 'top_nose_velocity',
                'top_particle_x', 'top_particle_y',
                'whisker_angle', 'whisker_velocity']

    else:
        print('Wrong view name')
        return 0


def get_likelihood_filtered_bodypart(nwb_file, keys, part, threshold=0.8):

    kinematic = part.split("_")[-1]
    root = re.sub(kinematic, '', part)
    suffix = 'tip_likelihood' if 'whisker' in part or 'top_nose' in part else 'likelihood'
    data = nwb_read.get_dlc_data(nwb_file, keys, part)
    likelihood = nwb_read.get_dlc_data(nwb_file, keys, root+suffix)

    return np.where(likelihood > threshold, data, 0 if 'tongue' in part else np.nan)


def get_traces_by_epoch(nwb_file, trials, timestamps, view, parts='all', start=-200, stop=200):

    keys = ['behavior', 'BehavioralTimeSeries']

    if parts == 'all':
        dlc_parts = filter_part_by_camview(view)
    else:
        dlc_parts = [part for part in filter_part_by_camview(view) if part in parts]

    dlc_data = pd.DataFrame(columns=dlc_parts)

    for part in dlc_parts:
        print(f"Getting data for {part}")
        dlc_data[part] = get_likelihood_filtered_bodypart(nwb_file, keys, part, threshold=0.6)

    view_timestamps = timestamps[0 if view == 'side' else 1][:len(dlc_data)]

    trial_data = []
    for tstamp in trials:
        frame = utils_misc.find_nearest(view_timestamps, tstamp)

        trace = dlc_data.loc[frame+(start+1):frame+stop]

        if trace.shape == (len(np.arange(start, stop)), len(dlc_parts)):
            trace['time'] = np.arange(start/100, stop/100, 0.01)
            trial_data += [trace]
        else:
            print(f'{view} has less data for this trial than requested: {tstamp}')

    return pd.concat(trial_data)


def plot_dlc_traces(data, x, y, hue, ax):
    sns.lineplot(data=data, x=x, y=y, hue=hue, ax=ax, estimator='mean', errorbar=('ci', 95))
    ax.set_xlabel('Time (s)')
    if 'angle' in y:
        ylabel = 'Angle (deg)'
    elif 'area' in y:
        ylabel = 'Surface (mm^2)'
    elif 'velocity' in y:
        ylabel = 'Velocity (mm/10ms)'
    else:
        ylabel = 'Distance (mm)'

    ax.set_ylabel(ylabel)
    return


def compute_combined_data(nwb_files, parts):
    combined_side_data, combined_top_data = [], []
    for nwb_index, nwb_file in enumerate(nwb_files):
        mouse_id = nwb_read.get_mouse_id(nwb_file)
        session_id = nwb_read.get_session_id(nwb_file)

        print(" ")
        print(f"Analyzing session {session_id}")
        session_type = nwb_read.get_session_type(nwb_file)

        save_path = os.path.join(output_path, mouse_id, session_id, f"dlc_results")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        trial_table = nwb_read.get_trial_table(nwb_file)
        trial_table['correct_choice'] = trial_table.reward_available == trial_table.lick_flag
        trial_table['context'] = trial_table['context'].map({0: 'non-rewarded', 1: 'rewarded'})

        epochs = nwb_read.get_behavioral_epochs_names(nwb_file)
        epochs = [epoch for epoch in epochs if epoch in ['rewarded', 'non-rewarded']]

        mouse_trial_avg_data = dict.fromkeys(epochs)
        mouse_trial_avg_data['mouse_id'] = mouse_id
        mouse_trial_avg_data['session_id'] = session_id

        trial_types = nwb_read.get_behavioral_events_names(nwb_file)
        trial_types = [trial_type for trial_type in trial_types if trial_type.split("_")[0] not in ['jaw', 'tongue']]

        timestamps = nwb_read.get_dlc_timestamps(nwb_file, keys=['behavior', 'BehavioralTimeSeries'])

        if len(epochs) > 0:
            epoch_trial_permutations = list(itertools.product(epochs, trial_types))

            for epoch_trial in epoch_trial_permutations:
                print(f"Epoch : {epoch_trial[0]}, Trials : {epoch_trial[1]}")
                if nwb_index == 0:
                    mouse_trial_avg_data[f'{epoch_trial[0]}_{epoch_trial[1]}'] = []

                epoch_times = nwb_read.get_behavioral_epochs_times(nwb_file, epoch_trial[0])
                trials = nwb_read.get_behavioral_events_times(nwb_file, epoch_trial[1])[0]
                trials_kept = utils_behavior.filter_events_based_on_epochs(events_ts=trials, epochs=epoch_times)
                print(f"Total of {len(trials_kept)} trials in {epoch_trial[0]} epoch")
                if len(trials_kept) == 0:
                    print("No trials in this condition, skipping")
                    continue

                side_data = get_traces_by_epoch(nwb_file, trials, timestamps, 'side', parts=parts, start=-200, stop=200)
                side_data['mouse_id'] = mouse_id
                side_data['session_id'] = session_id
                side_data['context'] = epoch_trial[0]
                side_data['trial_type'] = epoch_trial[1]
                side_data['context_background'] = trial_table.groupby('context').get_group(epoch_trial[0])['context_background'].unique()[0]
                combined_side_data += [side_data]

                top_data = get_traces_by_epoch(nwb_file, trials, timestamps, 'top', parts=parts, start=-200, stop=200)
                top_data['mouse_id'] = mouse_id
                top_data['session_id'] = session_id
                top_data['context'] = epoch_trial[0]
                top_data['trial_type'] = epoch_trial[1]
                side_data['context_background'] = trial_table.groupby('context').get_group(epoch_trial[0])['context_background'].unique()[0]
                combined_top_data += [top_data]

    return pd.concat(combined_side_data), pd.concat(combined_top_data)

def main(nwb_files, output_path, recompute_traces=True):

    parts = ['jaw_angle', 'jaw_distance', 'nose_angle', 'nose_distance', 'particle_x', 'particle_y', 'pupil_area', 'spout_y', 'tongue_angle', 'tongue_distance',
             'top_nose_angle', 'top_nose_distance', 'whisker_angle', 'whisker_velocity']

    if recompute_traces or not os.path.exists(os.path.join(output_path, 'side_dlc_results.csv')):
        combined_side_data, combined_top_data = compute_combined_data(nwb_files, parts)
        combined_side_data.to_csv(os.path.join(output_path, 'side_dlc_results.csv'))
        combined_top_data.to_csv(os.path.join(output_path, 'top_dlc_results.csv'))
    else:
        combined_side_data = pd.read_csv(os.path.join(output_path, 'side_dlc_results.csv'))
        combined_top_data = pd.read_csv(os.path.join(output_path, 'top_dlc_results.csv'))

    combined_side_data['correct_choice'] = combined_side_data['trial_type'].map(
        {'auditory_hit_trial': 1, 'auditory_miss_trial': 0,
         'correct_rejection_trial': 1, 'false_alarm_trial': 0})
    combined_side_data.loc[(combined_side_data['context'] == "rewarded") & (
                combined_side_data['trial_type'] == 'whisker_hit_trial'), 'correct_choice'] = 1
    combined_side_data.loc[(combined_side_data['context'] == "non-rewarded") & (
                combined_side_data['trial_type'] == 'whisker_miss_trial'), 'correct_choice'] = 1

    combined_top_data['correct_choice'] = combined_top_data['trial_type'].map(
        {'auditory_hit_trial': 1, 'auditory_miss_trial': 0,
         'correct_rejection_trial': 1, 'false_alarm_trial': 0})
    combined_top_data.loc[(combined_top_data['context'] == "rewarded") & (
                combined_top_data['trial_type'] == 'whisker_hit_trial'), 'correct_choice'] = 1
    combined_top_data.loc[(combined_top_data['context'] == "non-rewarded") & (
                combined_top_data['trial_type'] == 'whisker_miss_trial'), 'correct_choice'] = 1

    agg_side_data = combined_side_data.groupby(['session_id', 'context', 'trial_type', 'time']).agg('mean').reset_index()
    agg_top_data = combined_top_data.groupby(['session_id', 'context', 'trial_type', 'time']).agg('mean').reset_index()

    for epoch in ['rewarded', 'non-rewarded']:
        save_path = os.path.join(output_path, epoch)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i, part in enumerate(['jaw_angle', 'jaw_distance', 'tongue_angle', "tongue_distance", 'pupil_area', 'nose_angle', 'nose_distance', 'particle_x']):
            fig, ax = plt.subplots(figsize=(7,7))
            fig.suptitle(f"{part} {epoch} trials")
            plot_dlc_traces(data=agg_side_data.loc[agg_side_data['context']==epoch],
                            x='time',
                            y=part,
                            hue='trial_type',
                            ax=ax)
            fig.savefig(os.path.join(save_path, f'{part}_all_trial_psth.png'))

        for i, part in enumerate(['whisker_angle', 'whisker_velocity', 'top_nose_angle', 'top_nose_distance']):
            fig, ax = plt.subplots(figsize=(7,7))
            fig.suptitle(f"{part} {epoch} trials")
            plot_dlc_traces(data=agg_top_data.loc[agg_side_data['context']==epoch],
                            x='time',
                            y=part,
                            hue='trial_type',
                            ax=ax)
            fig.savefig(os.path.join(save_path, f'{part}_all_trial_psth.png'))

    subset_side = [agg_side_data.loc[agg_side_data['trial_type'] == trial, :] for trial in
              ['whisker_hit_trial', 'whisker_miss_trial']]
    subset_side = pd.concat(subset_side)

    subset_top = [agg_top_data.loc[agg_top_data['trial_type'] == trial, :] for trial in
              ['whisker_hit_trial', 'whisker_miss_trial']]
    subset_top = pd.concat(subset_top)

    for i, part in enumerate(
            ['jaw_angle', 'jaw_distance', 'tongue_angle', "tongue_distance", 'pupil_area', 'nose_angle',
             'nose_distance', 'particle_x']):

        fig, ax = plt.subplots(figsize=(7, 7))
        fig.suptitle(f"{part} {epoch} trials")
        plot_dlc_traces(data=subset_side.loc[subset_top['correct_choice']==1],
                        x='time',
                        y=part,
                        hue='context',
                        ax=ax)
        fig.savefig(os.path.join(output_path, f'{part}_all_trial_psth.png'))

    for i, part in enumerate(['whisker_angle', 'whisker_velocity', 'top_nose_angle', 'top_nose_distance']):
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.suptitle(f"{part} {epoch} trials")
        plot_dlc_traces(data=subset_top.loc[subset_top['correct_choice']==1],
                        x='time',
                        y=part,
                        hue='context',
                        ax=ax)
        fig.savefig(os.path.join(output_path, f'{part}_all_trial_psth.png'))

if __name__ == '__main__':

    config_file = r"M:\analysis\Pol_Bech\Sessions_list\context_contrast_expert_widefield_sessions_path.yaml"
    # config_file = r"M:\analysis\Robin_Dard\Sessions_list\context_naïve_mice_widefield_sessions_path.yaml"
    with open(config_file, 'r', encoding='utf8') as stream:
        config_dict = yaml.safe_load(stream)

    output_path = os.path.join(f'{server_path.get_experimenter_saving_folder_root("PB")}',
                               'Pop_results', 'Context_behaviour', 'dlc_results')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    nwb_files = config_dict['Sessions path']
    main(nwb_files, output_path=output_path)