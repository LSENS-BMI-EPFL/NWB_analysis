import os
import re
import sys
sys.path.append("/home/bechvila/NWB_analysis")
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
from utils.haas_utils import *
from nwb_utils import utils_io, utils_misc, utils_behavior


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

    return np.where(likelihood >= threshold, data, 0 if 'tongue' in part else np.nan)


def get_traces_by_epoch(nwb_file, trials, timestamps, view, parts='all', start=-200, stop=200):

    keys = ['behavior', 'BehavioralTimeSeries']

    if parts == 'all':
        dlc_parts = filter_part_by_camview(view)
    else:
        dlc_parts = [part for part in filter_part_by_camview(view) if part in parts]

    dlc_data = pd.DataFrame(columns=dlc_parts)

    for part in dlc_parts:
        # print(f"Getting data for {part}")
        dlc_data[part] = get_likelihood_filtered_bodypart(nwb_file, keys, part, threshold=0.5)

    view_timestamps = timestamps[0 if view == 'side' else 1][:len(dlc_data)]
    if len(view_timestamps) == 0:
        trial_data = np.ones([len(trials), len(dlc_parts)])*np.nan
        return pd.DataFrame(trial_data, columns=dlc_parts)
    
    trial_data = []
    for tstamp in trials:
        frame = utils_misc.find_nearest(view_timestamps, tstamp)

        trace = dlc_data.loc[frame+(start+1):frame+stop]
        # print(tstamp, frame, trace.shape)
        if trace.shape == (len(np.arange(start, stop)), len(dlc_parts)):
            trace = trace.apply(lambda x: x - np.nanmean(x.iloc[175:200]))
        elif trace.shape == (len(np.arange(start, stop))-1, len(dlc_parts)):
            print(f"{view} has one frame less than requested")
            trace = dlc_data.loc[frame+(start+1):frame+stop+1]
            print(f"New shape {trace.shape[0]}")
        elif trace.shape == (len(np.arange(start, stop))+1, len(dlc_parts)):
            print(f"{view} has one frame more than requested")
            trace = trace[:-1, :]
            print(f"New shape {trace.shape[0]}")

        else:
            print(f'{view} has less data for this trial than requested: {trace.__len__()} frames')
            trace = pd.DataFrame(np.ones([len(np.arange(start, stop)), len(dlc_parts)])*np.nan, columns=trace.keys())

        trace['time'] = np.arange(start/100, stop/100, 0.01)
        trial_data += [trace]
    return pd.concat(trial_data)


def plot_dlc_traces(data, x, y, hue, hue_order, style, ax):
    sns.lineplot(data=data, x=x, y=y, hue=hue, hue_order=hue_order, style=style, palette=['#348A18', '#6E188A'], ax=ax, estimator='mean', errorbar=('ci', 95))
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

                side_data = get_traces_by_epoch(nwb_file, trials_kept, timestamps, 'side', parts=parts, start=-200, stop=200)
                side_data['mouse_id'] = mouse_id
                side_data['session_id'] = session_id
                side_data['context'] = epoch_trial[0]
                side_data['trial_type'] = epoch_trial[1]
                side_data['context_background'] = trial_table.groupby('context').get_group(epoch_trial[0])['context_background'].unique()[0]
                combined_side_data += [side_data]

                top_data = get_traces_by_epoch(nwb_file, trials_kept, timestamps, 'top', parts=parts, start=-200, stop=200)
                top_data['mouse_id'] = mouse_id
                top_data['session_id'] = session_id
                top_data['context'] = epoch_trial[0]
                top_data['trial_type'] = epoch_trial[1]
                top_data['context_background'] = trial_table.groupby('context').get_group(epoch_trial[0])['context_background'].unique()[0]
                combined_top_data += [top_data]

            for context in ['rewarded', 'non-rewarded']:
                epoch_times = nwb_read.get_behavioral_epochs_times(nwb_file, context)
                side_data = get_traces_by_epoch(nwb_file, epoch_times[0], timestamps, 'side', parts=parts, start=-200, stop=200)
                side_data['mouse_id'] = mouse_id
                side_data['session_id'] = session_id
                side_data['context'] = context
                side_data['trial_type'] = "to_rewarded" if context == 'rewarded' else 'to_non_rewarded'
                side_data['context_background'] = \
                trial_table.groupby('context').get_group(epoch_trial[0])['context_background'].unique()[0]
                combined_side_data += [side_data]

                top_data = get_traces_by_epoch(nwb_file,  epoch_times[0], timestamps, 'top', parts=parts, start=-200, stop=200)
                top_data['mouse_id'] = mouse_id
                top_data['session_id'] = session_id
                top_data['context'] = context
                top_data['trial_type'] = "to_rewarded" if context == 'rewarded' else 'to_non_rewarded'
                top_data['context_background'] = \
                trial_table.groupby('context').get_group(epoch_trial[0])['context_background'].unique()[0]
                combined_top_data += [top_data]

    return pd.concat(combined_side_data), pd.concat(combined_top_data)

def main(nwb_files, output_path, recompute_traces=False):

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
    combined_side_data.loc[(combined_side_data['context'] == "rewarded") & (
                combined_side_data['trial_type'] == 'whisker_miss_trial'), 'correct_choice'] = 0
    combined_side_data.loc[(combined_side_data['context'] == "non-rewarded") & (
                combined_side_data['trial_type'] == 'whisker_miss_trial'), 'correct_choice'] = 1
    combined_side_data.loc[(combined_side_data['context'] == "non-rewarded") & (
                combined_side_data['trial_type'] == 'whisker_hit_trial'), 'correct_choice'] = 0

    combined_top_data['correct_choice'] = combined_top_data['trial_type'].map(
        {'auditory_hit_trial': 1, 'auditory_miss_trial': 0,
         'correct_rejection_trial': 1, 'false_alarm_trial': 0})
    combined_top_data.loc[(combined_top_data['context'] == "rewarded") & (
                combined_top_data['trial_type'] == 'whisker_hit_trial'), 'correct_choice'] = 1
    combined_top_data.loc[(combined_top_data['context'] == "rewarded") & (
                combined_top_data['trial_type'] == 'whisker_miss_trial'), 'correct_choice'] = 0
    combined_top_data.loc[(combined_top_data['context'] == "non-rewarded") & (
                combined_top_data['trial_type'] == 'whisker_miss_trial'), 'correct_choice'] = 1
    combined_top_data.loc[(combined_top_data['context'] == "non-rewarded") & (
                combined_top_data['trial_type'] == 'whisker_hit_trial'), 'correct_choice'] = 0

    agg_side_data = combined_side_data.groupby(['session_id', 'context', 'trial_type', 'time']).agg('mean').reset_index()
    agg_side_data['mouse_id'] = agg_side_data.apply(lambda x: x.session_id.split("_")[0], axis=1)
    agg_top_data = combined_top_data.groupby(['session_id', 'context', 'trial_type', 'time']).agg('mean').reset_index()
    agg_top_data['mouse_id'] = agg_top_data.apply(lambda x: x.session_id.split("_")[0], axis=1)

    total_avg_side = agg_side_data.groupby(['mouse_id', 'context', 'trial_type', 'time']).agg('mean').reset_index()
    total_avg_top = agg_top_data.groupby(['mouse_id', 'context', 'trial_type', 'time']).agg('mean').reset_index()

    for stim in ['whisker', 'auditory']:
        save_path = os.path.join(output_path, f'{stim}_trials')
        if not os.path.exists(save_path):
            os.makedirs(os.path.join(save_path, '200ms'))

        for i, part in enumerate(['jaw_angle', 'jaw_distance', 'tongue_angle', "tongue_distance", 'pupil_area', 'nose_angle', 'nose_distance', 'particle_x']):
            fig, ax = plt.subplots(figsize=(7,7))
            fig.suptitle(f"{part} whisker trials")
            plot_dlc_traces(data=total_avg_side.loc[(total_avg_side['trial_type'].isin([f'{stim}_hit_trial', f'{stim}_miss_trial']))],
                            x='time',
                            y=part,
                            hue='context',
                            hue_order=['rewarded', 'non-rewarded'],
                            style='trial_type',
                            ax=ax)
            ax.set_xlim(-2, 2)

            fig.savefig(os.path.join(save_path, f'{part}_{stim}_trial_psth.png'))
            fig.savefig(os.path.join(save_path, f'{part}_{stim}_trial_psth.svg'))

            ax.set_xlim(-0.05, 0.2)

            fig.savefig(os.path.join(save_path, '200ms', f'{part}_{stim}_trial_psth_200ms.png'))
            fig.savefig(os.path.join(save_path, '200ms', f'{part}_{stim}_trial_psth_200ms.svg'))

        for i, part in enumerate(['whisker_angle', 'whisker_velocity', 'top_nose_angle', 'top_nose_distance']):
            fig, ax = plt.subplots(figsize=(7,7))
            fig.suptitle(f"{part} whisker trials")
            plot_dlc_traces(data=total_avg_top.loc[(total_avg_top['trial_type'].isin(['whisker_hit_trial', 'whisker_miss_trial']))],
                            x='time',
                            y=part,
                            hue='context',
                            hue_order=['rewarded', 'non-rewarded'],
                            style='trial_type',
                            ax=ax)
            ax.set_xlim(-2, 2)

            fig.savefig(os.path.join(save_path, f'{part}_{stim}_trial_psth.png'))
            fig.savefig(os.path.join(save_path, f'{part}_{stim}_trial_psth.svg'))

            ax.set_xlim(-0.05, 0.2)

            fig.savefig(os.path.join(save_path, '200ms', f'{part}_{stim}_trial_psth_200ms.png'))
            fig.savefig(os.path.join(save_path, '200ms', f'{part}_{stim}_trial_psth_200ms.svg'))

    save_path = os.path.join(output_path, 'transition')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, part in enumerate(['jaw_angle', 'jaw_distance', 'tongue_angle', "tongue_distance", 'pupil_area', 'nose_angle', 'nose_distance', 'particle_x']):
        fig, ax = plt.subplots(figsize=(7,7))
        fig.suptitle(f"{part} whisker trials")
        plot_dlc_traces(data=total_avg_side.loc[(total_avg_side['trial_type'].isin(['to_rewarded', 'to_non_rewarded']))],
                        x='time',
                        y=part,
                        hue='trial_type',
                        hue_order=['to_rewarded', 'to_non_rewarded'],
                        style=None,
                        ax=ax)
        ax.set_xlim(-2, 2)

        fig.savefig(os.path.join(save_path, f'{part}_transition_psth.png'))
        fig.savefig(os.path.join(save_path, f'{part}_transition_psth.svg'))

    for i, part in enumerate(['whisker_angle', 'whisker_velocity', 'top_nose_angle', 'top_nose_distance']):
        fig, ax = plt.subplots(figsize=(7,7))
        fig.suptitle(f"{part} whisker trials")
        plot_dlc_traces(data=total_avg_top.loc[(total_avg_top['trial_type'].isin(['to_rewarded', 'to_non_rewarded']))],
                        x='time',
                        y=part,
                        hue='trial_type',
                        hue_order=['to_rewarded', 'to_non_rewarded'],
                        style=None,
                        ax=ax)
        ax.set_xlim(-2, 2)

        fig.savefig(os.path.join(save_path, f'{part}_transition_psth.png'))
        fig.savefig(os.path.join(save_path, f'{part}_transition_psth.svg'))


if __name__ == '__main__':

    for dtype in ['controls_gfp']: #'jrgeco', 'gcamp', 'controls_gfp', 'controls_tdtomato'
        config_file = f"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Session_list/context_sessions_{dtype}_expert.yaml"
        config_file = haas_pathfun(config_file)
        # config_file = r"M:\analysis\Robin_Dard\Sessions_list\context_naïve_mice_widefield_sessions_path.yaml"
        with open(config_file, 'r', encoding='utf8') as stream:
            config_dict = yaml.safe_load(stream)

        output_path = os.path.join(f'{utils_io.get_experimenter_saving_folder_root("PB")}',
                                'Pop_results', 'Context_behaviour', 'dlc_results', dtype)
        output_path = haas_pathfun(output_path.replace("\\", "/"))
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        nwb_files = config_dict['Session path']
        nwb_files = [haas_pathfun(nwb_file.replace("\\", "/")) for nwb_file in nwb_files]
        # nwb_files = [f for f in nwb_files if 'RD049' not in f]
        main(nwb_files, output_path=output_path, recompute_traces=True)