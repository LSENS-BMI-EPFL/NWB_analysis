import os
import re
import sys
sys.path.append("/home/bechvila/NWB_analysis")
import glob
import matplotlib.pyplot as plt
import yaml
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp, ttest_rel
import nwb_wrappers.nwb_reader_functions as nwb_read

import warnings
warnings.filterwarnings("ignore")

from nwb_utils import utils_misc
from utils.haas_utils import *
from nwb_utils import utils_io, utils_misc, utils_behavior


def filter_part_by_camview(view):
    if view == 'side':
        return ['jaw_x', 'jaw_y', 'jaw_angle', 'jaw_distance', 'jaw_velocity',
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


def get_traces_by_epoch(nwb_file, trials, timestamps, view, center=True, parts='all', start=-200, stop=200):

    keys = ['behavior', 'BehavioralTimeSeries']
    nframes = abs(start-stop)

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
        # if trace.shape == (len(np.arange(start, stop)), len(dlc_parts)):
            
        if trace.shape[0] == (nframes-1, len(dlc_parts)):
            print(f"{view} has one frame less than requested")
            trace = dlc_data.loc[frame+(start+1):frame+stop+1]
            print(f"New shape {trace.shape[0]}")
        elif trace.shape[0] > nframes:
            print(f"{view} has one frame more than requested")
            trace = trace[:nframes, :]
            print(f"New shape {trace.shape[0]}")
        elif trace.shape[0] < nframes-1:
            print(f'{view} has less data for this trial than requested: {trace.__len__()} frames')
            trace = pd.DataFrame(np.ones([len(np.arange(start, stop)), len(dlc_parts)])*np.nan, columns=trace.keys())

        if center:
            trace = trace.apply(lambda x: x - np.nanmean(x.iloc[175:200]))

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


def compute_combined_data(nwb_files, parts, center=True):
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

                side_data = get_traces_by_epoch(nwb_file, trials_kept, timestamps, 'side', center=center, parts=parts, start=-200, stop=200)
                side_data['mouse_id'] = mouse_id
                side_data['session_id'] = session_id
                side_data['context'] = epoch_trial[0]
                side_data['trial_type'] = epoch_trial[1]
                side_data['context_background'] = trial_table.groupby('context').get_group(epoch_trial[0])['context_background'].unique()[0]
                side_data['correct_choice'] = side_data['trial_type'].map({'auditory_hit_trial': 1, 'auditory_miss_trial': 0,'correct_rejection_trial': 1, 'false_alarm_trial': 0})
                side_data.loc[(side_data['context'] == "rewarded") & (
                            side_data['trial_type'] == 'whisker_hit_trial'), 'correct_choice'] = 1
                side_data.loc[(side_data['context'] == "rewarded") & (
                            side_data['trial_type'] == 'whisker_miss_trial'), 'correct_choice'] = 0
                side_data.loc[(side_data['context'] == "non-rewarded") & (
                            side_data['trial_type'] == 'whisker_miss_trial'), 'correct_choice'] = 1
                side_data.loc[(side_data['context'] == "non-rewarded") & (
                            side_data['trial_type'] == 'whisker_hit_trial'), 'correct_choice'] = 0
                combined_side_data += [side_data]

                top_data = get_traces_by_epoch(nwb_file, trials_kept, timestamps, 'top', center=center, parts=parts, start=-200, stop=200)
                top_data['mouse_id'] = mouse_id
                top_data['session_id'] = session_id
                top_data['context'] = epoch_trial[0]
                top_data['trial_type'] = epoch_trial[1]
                top_data['context_background'] = trial_table.groupby('context').get_group(epoch_trial[0])['context_background'].unique()[0]
                top_data['correct_choice'] = top_data['trial_type'].map({'auditory_hit_trial': 1, 'auditory_miss_trial': 0,'correct_rejection_trial': 1, 'false_alarm_trial': 0})
                top_data.loc[(top_data['context'] == "rewarded") & (
                            top_data['trial_type'] == 'whisker_hit_trial'), 'correct_choice'] = 1
                top_data.loc[(top_data['context'] == "rewarded") & (
                            top_data['trial_type'] == 'whisker_miss_trial'), 'correct_choice'] = 0
                top_data.loc[(top_data['context'] == "non-rewarded") & (
                            top_data['trial_type'] == 'whisker_miss_trial'), 'correct_choice'] = 1
                top_data.loc[(top_data['context'] == "non-rewarded") & (
                            top_data['trial_type'] == 'whisker_hit_trial'), 'correct_choice'] = 0
                combined_top_data += [top_data]

            for context in ['rewarded', 'non-rewarded']:
                epoch_times = nwb_read.get_behavioral_epochs_times(nwb_file, context)
                side_data = get_traces_by_epoch(nwb_file, epoch_times[0], timestamps, 'side', center=center, parts=parts, start=-200, stop=200)
                side_data['mouse_id'] = mouse_id
                side_data['session_id'] = session_id
                side_data['context'] = context
                side_data['trial_type'] = "to_rewarded" if context == 'rewarded' else 'to_non_rewarded'
                side_data['context_background'] = trial_table.groupby('context').get_group(epoch_trial[0])['context_background'].unique()[0]
                side_data['correct_choice'] = np.nan
                combined_side_data += [side_data]

                top_data = get_traces_by_epoch(nwb_file,  epoch_times[0], timestamps, 'top', center=center, parts=parts, start=-200, stop=200)
                top_data['mouse_id'] = mouse_id
                top_data['session_id'] = session_id
                top_data['context'] = context
                top_data['trial_type'] = "to_rewarded" if context == 'rewarded' else 'to_non_rewarded'
                top_data['context_background'] = trial_table.groupby('context').get_group(epoch_trial[0])['context_background'].unique()[0]
                top_data['correct_choice'] = np.nan
                combined_top_data += [top_data]

    return pd.concat(combined_side_data), pd.concat(combined_top_data)


def plot_movement_between_contexts(df, output_path, height=4, aspect=0.5, dodge=0.3):

    n_comparisons = 72 # 6 bodyparts in side + 6 bodyparts in top * 2 contexts *3 trial types
    stats = []

    for name, group in df.groupby(by=['bodypart', 'context', 'stim_type']):
        correct= group.loc[group.correct_choice==1]
        incorrect = group.loc[group.correct_choice==0]
        if correct.shape[0] != incorrect.shape[0]:
            correct = correct[correct.mouse_id.isin(incorrect.mouse_id)]

        t, p = ttest_rel(correct['value'].values, incorrect['value'].values)
        results = {
         'bodypart': name[0],
         'context': name[1],
         'trial_type': name[2],
         'dof': correct.mouse_id.unique().shape[0]-1,
         'mean_correct': correct['value'].mean(),
         'std_correct': correct['value'].std(),
         'mean_incorrect': incorrect['value'].mean(),
         'std_incorrect': incorrect['value'].std(),
         't': t,
         'p': np.round(p,8),
         'p_corr': p*n_comparisons,
         'alpha': 0.05,
         'alpha_corr': 0.05/n_comparisons,
         'significant': p*n_comparisons<0.05,
         'd_prime': abs(t/np.sqrt(correct.mouse_id.unique().shape[0]))
         }
        stats+=[results]
    stats= pd.DataFrame(stats)
    stats.to_csv(os.path.join(f"{output_path}_stats.csv"))

    fig, ax = plt.subplots(1, df.bodypart.unique().shape[0], figsize=(height * aspect, height))
    for i, part in enumerate(df.bodypart.unique()):
        group = df[df.bodypart==part]
        g = sns.pointplot(group,
                        x='legend', 
                        y='value', 
                        estimator='mean',
                        errorbar=('ci', 95),
                        order=['non-rewarded - incorrect', 'non-rewarded - correct', 'rewarded - incorrect', 'rewarded - correct'], 
                        hue='legend', 
                        hue_order=['non-rewarded - incorrect', 'non-rewarded - correct', 'rewarded - incorrect', 'rewarded - correct'], 
                        palette=['#C5A2D0', '#6E188A', '#ADD0A2', '#348A18'],
                        ax=ax.flat[i],
                        dodge=False
                      )
        ['#348A18', '#ADD0A2', '#6E188A', '#C5A2D0']
        sns.stripplot(group,
                        x='legend', 
                        y='value', 
                        order=['non-rewarded - incorrect', 'non-rewarded - correct', 'rewarded - incorrect', 'rewarded - correct'], 
                        hue='legend', 
                        hue_order=['non-rewarded - incorrect', 'non-rewarded - correct', 'rewarded - incorrect', 'rewarded - correct'], 
                        palette=['#C5A2D0', '#6E188A', '#ADD0A2', '#348A18'],
                        ax= ax.flat[i],
                        dodge=0.3,
                        alpha=0.5
        )

        ax.flat[i].get_legend().remove()
        ax.flat[i].set_title(part)
        ax.flat[i].spines[['top', 'right']].set_visible(False)
        ax.flat[i].set_xticklabels([])

        if stats.loc[stats.bodypart==part, 'significant'].any():
            star_loc = max(ax.flat[i].get_ylim())
            ax.flat[i].scatter([0.5, 1.5], stats.loc[(stats.bodypart==part), 'significant'].map({True: 1}).to_numpy()*star_loc*0.9, marker='*', s=100, c='k')

    for ext in ['png', 'svg']:    
        fig.tight_layout()
        fig.savefig(f"{output_path}.{ext}")


def compute_dlc_data(nwb_files, output_path):

    # if load and os.path.exists(os.path.join(output_path, 'side_dlc_results.csv')):
       
    #     combined_side_data = pd.read_csv(os.path.join(output_path, 'side_dlc_results.csv'))
    #     combined_top_data = pd.read_csv(os.path.join(output_path, 'top_dlc_results.csv'))

    #     uncentered_combined_side_data = pd.read_csv(os.path.join(output_path, 'uncentered_side_dlc_results.csv'))
    #     uncentered_combined_top_data = pd.read_csv(os.path.join(output_path, 'uncentered_top_dlc_results.csv'))

    # else: 
    parts = ['jaw_angle', 'jaw_distance', 'jaw_x', 'jaw_y', 'nose_angle', 'nose_distance', 'particle_x', 'particle_y', 'pupil_area', 'spout_y', 'tongue_angle', 'tongue_distance',
    'top_nose_angle', 'top_nose_distance', 'whisker_angle', 'whisker_velocity', 'top_particle_x', 'top_particle_y']
    combined_side_data, combined_top_data = compute_combined_data(nwb_files, parts)
    combined_side_data.to_csv(os.path.join(output_path, 'side_dlc_results.csv'))
    combined_top_data.to_csv(os.path.join(output_path, 'top_dlc_results.csv'))

    uncentered_combined_side_data, uncentered_combined_top_data = compute_combined_data(nwb_files, parts, center=False)
    uncentered_combined_side_data.to_csv(os.path.join(output_path, 'uncentered_side_dlc_results.csv'))
    uncentered_combined_top_data.to_csv(os.path.join(output_path, 'uncentered_top_dlc_results.csv'))

    return combined_side_data, combined_top_data, uncentered_combined_side_data, uncentered_combined_top_data


def plot_stim_aligned_movement(file_list, output_path):

    combined_side_data = []
    combined_top_data = []
    for file in file_list:
        if 'side' in file:
            df = pd.read_csv(file)
            combined_side_data += [df]
        else:
            df = pd.read_csv(file)
            combined_top_data += [df]

    combined_side_data = pd.concat(combined_side_data)
    combined_top_data = pd.concat(combined_top_data)

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


def plot_baseline_differences(file_list, output_path):
    from matplotlib.colors import LogNorm
    save_path = os.path.join(output_path, 'baseline')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    uncentered_combined_side_data = []
    uncentered_combined_top_data = []
    for file in file_list:
        if 'side' in file:
            df = pd.read_csv(file)
            uncentered_combined_side_data += [df]
        else:
            df = pd.read_csv(file)
            uncentered_combined_top_data += [df]

    uncentered_combined_side_data = pd.concat(uncentered_combined_side_data)
    uncentered_combined_top_data = pd.concat(uncentered_combined_top_data)

    uncentered_combined_side_data = uncentered_combined_side_data[uncentered_combined_side_data.time<0]
    uncentered_combined_top_data = uncentered_combined_top_data[uncentered_combined_top_data.time<0]

    uncentered_agg_side_data = uncentered_combined_side_data.groupby(['session_id', 'context', 'trial_type', 'correct_choice', 'time']).agg('mean').reset_index()
    uncentered_agg_side_data['mouse_id'] = uncentered_agg_side_data.apply(lambda x: x.session_id.split("_")[0], axis=1)
    uncentered_agg_top_data = uncentered_combined_top_data.groupby(['session_id', 'context', 'trial_type', 'correct_choice', 'time']).agg('mean').reset_index()
    uncentered_agg_top_data['mouse_id'] = uncentered_agg_top_data.apply(lambda x: x.session_id.split("_")[0], axis=1)

    uncentered_total_avg_side = uncentered_agg_side_data.groupby(['mouse_id', 'context', 'trial_type', 'correct_choice', 'time']).agg('mean').reset_index()
    uncentered_total_avg_top = uncentered_agg_top_data.groupby(['mouse_id', 'context', 'trial_type', 'correct_choice', 'time']).agg('mean').reset_index()

    norm = LogNorm(vmin=0.1, vmax=100000)
    subset = uncentered_combined_top_data[uncentered_combined_top_data.trial_type.str.contains('trial')].reset_index(drop=True)
    subset['top_particle_x'] = subset.groupby('session_id').apply(lambda x: x.top_particle_x - np.nanmean(x.loc[x.context=='rewarded', 'top_particle_x'].values)).reset_index()['top_particle_x']
    subset['top_particle_y'] = subset.groupby('session_id').apply(lambda x: x.top_particle_y - np.nanmean(x.loc[x.context=='rewarded', 'top_particle_y'].values)).reset_index()['top_particle_y']
    g=sns.displot(subset, 
                  x='top_particle_x', 
                  y='top_particle_y', 
                  kind='hist',
                  hue='context', 
                  hue_order=['rewarded', 'non-rewarded'],
                  palette=['#348A18', '#6E188A'],
                  alpha=0.7,
                  binwidth=1, 
                  cbar=True, 
                  norm=norm,
                  vmin=None,
                  vmax=None,
                  col='correct_choice',
                  height=4, aspect=1)      
    for ax in g.axes.flat:
        ax.set_aspect('equal')
        ax.set_ylim([-10, 10])
        ax.set_xlim([-10, 10])
    g.figure.tight_layout()
    g.figure.savefig(os.path.join(save_path,'particle_set_point_heatmap.png'))
    g.figure.savefig(os.path.join(save_path,'particle_set_point_heatmap.svg'))

    subset = uncentered_combined_side_data[uncentered_combined_side_data.trial_type.str.contains('trial')].reset_index(drop=True)
    subset['particle_x'] = subset.groupby('session_id').apply(lambda x: x.particle_x - np.nanmean(x.loc[x.context=='rewarded', 'particle_x'].values)).reset_index()['particle_x']
    subset['particle_y'] = subset.groupby('session_id').apply(lambda x: x.particle_y - np.nanmean(x.loc[x.context=='rewarded', 'particle_y'].values)).reset_index()['particle_y']
    g=sns.displot(subset, 
                  x='particle_x', 
                  y='particle_y', 
                  kind='hist', 
                  hue='context', 
                  hue_order=['rewarded', 'non-rewarded'],
                  palette=['#348A18', '#6E188A'],
                  alpha=0.7,
                  binwidth=1, 
                  cbar=True, 
                  norm=norm,
                  vmin=None,
                  vmax=None,
                  col='correct_choice',
                  height=4, aspect=1)      
    for ax in g.axes.flat:
        ax.set_aspect('equal')
        ax.set_ylim([-10, 10])
        ax.set_xlim([-10, 10])
    g.figure.tight_layout()
    g.figure.savefig(os.path.join(save_path,'particle_set_point_heatmap_side.png'))
    g.figure.savefig(os.path.join(save_path,'particle_set_point_heatmap_side.svg'))

    subset = uncentered_combined_side_data[uncentered_combined_side_data.trial_type.str.contains('trial')].reset_index(drop=True)
    subset['jaw_x'] = subset.groupby('session_id').apply(lambda x: x.jaw_x - np.nanmean(x.loc[x.context=='rewarded', 'jaw_x'].values)).reset_index()['jaw_x']
    subset['jaw_y'] = subset.groupby('session_id').apply(lambda x: x.jaw_y - np.nanmean(x.loc[x.context=='rewarded', 'jaw_y'].values)).reset_index()['jaw_y']
    g=sns.displot(subset, 
                  x='jaw_x', 
                  y='jaw_y', 
                  kind='hist', 
                  hue='context', 
                  hue_order=['rewarded', 'non-rewarded'],
                  palette=['#348A18', '#6E188A'],
                  alpha=0.7,
                  binwidth=1, 
                  cbar=True, 
                  norm=norm,
                  vmin=None,
                  vmax=None,
                  col='correct_choice',
                  height=4, aspect=1)      
    for ax in g.axes.flat:
        ax.set_aspect('equal')
        ax.set_ylim([-20, 20])
        ax.set_xlim([-20, 20])
    
    g.figure.tight_layout()
    g.figure.savefig(os.path.join(save_path,'jaw_set_point_heatmap_side.png'))
    g.figure.savefig(os.path.join(save_path,'jaw_set_point_heatmap_side.svg'))

    uncentered_total_avg_side = uncentered_total_avg_side.loc[uncentered_total_avg_side.trial_type.str.contains('trial')].melt(id_vars=['mouse_id', 'context', 'trial_type', 'correct_choice', 'time'], value_vars=['jaw_angle', 'jaw_distance', 'nose_distance', 'particle_x','particle_y', 'pupil_area'], var_name='bodypart')
    uncentered_total_avg_top = uncentered_total_avg_top.loc[uncentered_total_avg_top.trial_type.str.contains('trial')].melt(id_vars=['mouse_id', 'context', 'trial_type', 'correct_choice', 'time'], value_vars=['top_nose_angle', 'top_nose_distance', 'top_particle_x', 'top_particle_y', 'whisker_angle', 'whisker_velocity'], var_name='bodypart')

    side_mean = uncentered_total_avg_side.groupby(by=['mouse_id', 'context', 'trial_type', 'correct_choice', 'bodypart']).agg('mean').reset_index()
    side_mean['legend'] = side_mean.apply(lambda x: f"{x.context} - {'correct' if x.correct_choice==1 else 'incorrect'}", axis=1)
    side_mean['stim_type'] = side_mean.apply(lambda x: x.trial_type.split("_")[0], axis=1)
    side_mean.loc[(~side_mean.trial_type.str.contains('whisker')) & (~side_mean.trial_type.str.contains('auditory')), 'stim_type'] ='catch'

    top_mean = uncentered_total_avg_top.groupby(by=['mouse_id', 'context', 'trial_type', 'correct_choice', 'bodypart']).agg('mean').reset_index()
    top_mean['legend'] = top_mean.apply(lambda x: f"{x.context} - {'correct' if x.correct_choice==1 else 'incorrect'}", axis=1)
    top_mean['stim_type'] = top_mean.apply(lambda x: x.trial_type.split("_")[0], axis=1)
    top_mean.loc[(~top_mean.trial_type.str.contains('whisker')) & (~top_mean.trial_type.str.contains('auditory')), 'stim_type'] ='catch'

    side_std = uncentered_total_avg_side.groupby(by=['mouse_id', 'context', 'trial_type', 'correct_choice', 'bodypart']).agg('std').reset_index()
    side_std['legend'] = side_std.apply(lambda x: f"{x.context} - {'correct' if x.correct_choice==1 else 'incorrect'}", axis=1)
    side_std['stim_type'] = side_std.apply(lambda x: x.trial_type.split("_")[0], axis=1)
    side_std.loc[(~side_std.trial_type.str.contains('whisker')) & (~side_std.trial_type.str.contains('auditory')), 'stim_type'] ='catch'
    
    top_std = uncentered_total_avg_top.groupby(by=['mouse_id', 'context', 'trial_type', 'correct_choice', 'bodypart']).agg('std').reset_index()
    top_std['legend'] = top_std.apply(lambda x: f"{x.context} - {'correct' if x.correct_choice==1 else 'incorrect'}", axis=1)
    top_std['stim_type'] = top_std.apply(lambda x: x.trial_type.split("_")[0], axis=1)
    top_std.loc[(~top_std.trial_type.str.contains('whisker')) & (~top_std.trial_type.str.contains('auditory')), 'stim_type'] ='catch'

    model_side_mean = smf.ols('value  ~ context + correct_choice + stim_type', data=side_mean).fit()
    with open(os.path.join(save_path, 'side_set_point_OLS.csv'),'w') as f:
        f.write(model_side_mean.summary().as_csv())

    model_top_mean = smf.ols('value  ~ context + correct_choice + stim_type', data=top_mean).fit()
    with open(os.path.join(save_path, 'top_set_point_OLS.csv'),'w') as f:
        f.write(model_top_mean.summary().as_csv())

    model_side_std = smf.ols('value  ~ context + correct_choice + stim_type', data=side_std).fit()
    with open(os.path.join(save_path, 'side_movement_OLS.csv'),'w') as f:
        f.write(model_side_std.summary().as_csv())

    model_top_std = smf.ols('value  ~ context + correct_choice + stim_type', data=top_std).fit()
    with open(os.path.join(save_path, 'top_movement_OLS.csv'),'w') as f:
        f.write(model_top_std.summary().as_csv())


    for trial in ['auditory', 'whisker', 'catch']:
        plot_movement_between_contexts(side_mean[side_mean.stim_type==trial], os.path.join(save_path, f'mean_position_sideview_{trial}'), height=4, aspect=3, dodge=0.75)
        plot_movement_between_contexts(top_mean[top_mean.stim_type==trial], os.path.join(save_path, f'mean_position_topview_{trial}'), height=4, aspect=3, dodge=0.75)

        plot_movement_between_contexts(side_std[side_std.stim_type==trial], os.path.join(save_path, f'std_position_sideview_{trial}'), height=4, aspect=3, dodge=0.75) 
        plot_movement_between_contexts(top_std[top_std.stim_type==trial], os.path.join(save_path, f'std_position_topview_{trial}'), height=4, aspect=3, dodge=0.75)

def main(data_path,  output_path):

    uncentered_files = [file for file in data_path if 'uncentered' in file]
    centered_files = [file for file in data_path if 'uncentered' not in file]

    print("Analyzing dlc data")
    plot_stim_aligned_movement(centered_files, output_path=output_path)

    plot_baseline_differences(uncentered_files, output_path=output_path)


if __name__ == '__main__':

    recompute_data = True
    all_nwb_files =[]
    for dtype in ['jrgeco', 'gcamp', 'controls_gfp', 'controls_tdtomato']: #'jrgeco', 'gcamp', 'controls_gfp', 'controls_tdtomato'
        config_file = f"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Session_list/context_sessions_{dtype}_expert.yaml"
        config_file = haas_pathfun(config_file)
        # config_file = r"M:\analysis\Robin_Dard\Sessions_list\context_naïve_mice_widefield_sessions_path.yaml"
        with open(config_file, 'r', encoding='utf8') as stream:
            config_dict = yaml.safe_load(stream)

        nwb_files = config_dict['Session path']
        nwb_files = [haas_pathfun(nwb_file.replace("\\", "/")) for nwb_file in nwb_files]
        # nwb_files = [f for f in nwb_files if 'RD049' not in f]
        output_path = os.path.join(f'{utils_io.get_experimenter_saving_folder_root("PB")}', 'Pop_results', 'Context_behaviour', 'combined_dlc_results', dtype)
        output_path = haas_pathfun(output_path.replace("\\", "/"))
        if not os.path.exists(output_path):
            os.makedirs(os.path.join(output_path, 'results'))
        
        if recompute_data:
            print("ATTENTION: Preprocessing dlc data")
            combined_side_data, combined_top_data, uncentered_combined_side_data, uncentered_combined_top_data = compute_dlc_data(nwb_files, output_path)

        data_path = glob.glob(os.path.join(output_path, '*.csv'))
        main(data_path, output_path=os.path.join(output_path, 'results'))
        
    output_path = os.path.join(f'{utils_io.get_experimenter_saving_folder_root("PB")}', 'Pop_results', 'Context_behaviour', 'combined_dlc_results')
    output_path = haas_pathfun(output_path.replace("\\", "/"))

    data_path = glob.glob(os.path.join(output_path, '**', '*.csv'))
    main(data_path, output_path=os.path.join(output_path, 'results'))