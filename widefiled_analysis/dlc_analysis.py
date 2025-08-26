import os
import re
import sys
sys.path.append("/home/bechvila/NWB_analysis")
import glob
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
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
    suffix = 'base_likelihood' if 'whisker' in part or 'top_nose' in part else 'likelihood'
    data = nwb_read.get_dlc_data(nwb_file, keys, part)
    likelihood = nwb_read.get_dlc_data(nwb_file, keys, root+suffix)

    if ((likelihood >=threshold).sum()/ likelihood.shape[0])*100 < 70 and 'tongue' not in part and 'pupil' not in part:
        data = np.zeros_like(data)*np.nan
        print(f"{nwb_read.get_session_id(nwb_file)} {part} has more than 30% of NaN values, discard")

    return np.where(likelihood >= threshold, data, 0 if 'tongue' in part else np.nan)


def get_traces_by_epoch(nwb_file, trials, timestamps, view, center=True, parts='all', start=-2, stop=2):

    parts = ['jaw_angle', 'jaw_distance', 'jaw_x', 'jaw_y', 'pupil_area', 'tongue_angle', 'tongue_distance', 'whisker_angle', 'whisker_velocity', 'top_particle_x', 'top_particle_y']
        
    thresholds = {
        'jaw_angle': 0.6, 
        'jaw_distance': 0.6, 
        'jaw_x': 0.6, 
        'jaw_y': 0.6,
        'pupil_area': 0.6, 
        'tongue_angle': 0.5, 
        'tongue_distance': 0.5,
        'whisker_angle':0.8, 
        'whisker_velocity':0.8, 
        'top_particle_x':0.8, 
        'top_particle_y':0.8
        }
    fr = 1/np.round(np.median(np.diff(timestamps[0 if view == 'side' else 1])),3)
    keys = ['behavior', 'BehavioralTimeSeries']

    nframes = int(abs(start*fr-stop*fr))

    if parts == 'all':
        dlc_parts = filter_part_by_camview(view)
    else:
        dlc_parts = [part for part in filter_part_by_camview(view) if part in parts]

    dlc_data = pd.DataFrame(columns=dlc_parts)

    for part in dlc_parts:

        dlc_data[part] = get_likelihood_filtered_bodypart(nwb_file, keys, part, threshold=thresholds[part])
        if part in ['jaw_x', 'jaw_y'] and len(dlc_data[part].dropna()) != 0:
            ref = np.percentile(dlc_data[part].dropna(), 5)
            dlc_data[part] = dlc_data[part] - ref

    view_timestamps = timestamps[0 if view == 'side' else 1][:len(dlc_data)]
    if len(view_timestamps) == 0:
        trial_data = np.ones([len(trials), len(dlc_parts)])*np.nan
        return pd.DataFrame(trial_data, columns=dlc_parts)
    
    trial_data = []
    for tstamp in trials:
        frame = utils_misc.find_nearest(view_timestamps, tstamp)

        trace = dlc_data.loc[frame+(start*fr)+1:frame+(stop*fr)]

            
        if trace.shape[0] == (nframes-1, len(dlc_parts)):
            trace = dlc_data.loc[frame+(start*fr)+1:frame+stop*fr+1]
        elif trace.shape[0] > nframes:
            trace = trace.iloc[:nframes, :]
        elif trace.shape[0] < nframes-1:
            trace = pd.DataFrame(np.ones([nframes, len(dlc_parts)])*np.nan, columns=trace.keys())

        if center:
            trace = trace.apply(lambda x: x - np.nanmean(x.iloc[175:200]))

        trace['time'] = np.round(np.arange(start, stop, 1/fr),2)
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
                # print(f"Epoch : {epoch_trial[0]}, Trials : {epoch_trial[1]}")
                if nwb_index == 0:
                    mouse_trial_avg_data[f'{epoch_trial[0]}_{epoch_trial[1]}'] = []

                epoch_times = nwb_read.get_behavioral_epochs_times(nwb_file, epoch_trial[0])
                trials = nwb_read.get_behavioral_events_times(nwb_file, epoch_trial[1])[0]
                trials_kept = utils_behavior.filter_events_based_on_epochs(events_ts=trials, epochs=epoch_times)
                # print(f"Total of {len(trials_kept)} trials in {epoch_trial[0]} epoch")
                if len(trials_kept) == 0:
                    # print("No trials in this condition, skipping")
                    continue

                side_data = get_traces_by_epoch(nwb_file, trials_kept, timestamps, 'side', center=center, parts=parts, start=-2, stop=2)
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

                top_data = get_traces_by_epoch(nwb_file, trials_kept, timestamps, 'top', center=center, parts=parts, start=-2, stop=2)
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
                side_data = get_traces_by_epoch(nwb_file, epoch_times[0], timestamps, 'side', center=center, parts=parts, start=-2, stop=2)
                side_data['mouse_id'] = mouse_id
                side_data['session_id'] = session_id
                side_data['context'] = context
                side_data['trial_type'] = "to_rewarded" if context == 'rewarded' else 'to_non_rewarded'
                side_data['context_background'] = trial_table.groupby('context').get_group(epoch_trial[0])['context_background'].unique()[0]
                side_data['correct_choice'] = np.nan
                combined_side_data += [side_data]

                top_data = get_traces_by_epoch(nwb_file,  epoch_times[0], timestamps, 'top', center=center, parts=parts, start=-2, stop=2)
                top_data['mouse_id'] = mouse_id
                top_data['session_id'] = session_id
                top_data['context'] = context
                top_data['trial_type'] = "to_rewarded" if context == 'rewarded' else 'to_non_rewarded'
                top_data['context_background'] = trial_table.groupby('context').get_group(epoch_trial[0])['context_background'].unique()[0]
                top_data['correct_choice'] = np.nan
                combined_top_data += [top_data]

    return pd.concat(combined_side_data), pd.concat(combined_top_data)


def plot_movement_between_contexts(df, output_path, height=4, aspect=0.5, dodge=0.3):

    n_comparisons = 20 # 2 bodyparts in side + 2 bodyparts in top * 2 contexts *3 trial types
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
            ax.flat[i].scatter([0.5, 2.5], stats.loc[(stats.bodypart==part), 'significant'].map({True: 1}).to_numpy()*star_loc*0.9, marker='*', s=100, c='k')

    for ext in ['png', 'svg']:    
        fig.tight_layout()
        fig.savefig(f"{output_path}.{ext}")


def compute_dlc_data(nwb_files, output_path):

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
    combined_top_data['whisker_speed'] = combined_top_data['whisker_velocity'].abs()

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

        for i, part in enumerate(['jaw_angle', 'jaw_y', 'tongue_angle', "tongue_distance", 'pupil_area']):
            fig, ax = plt.subplots(figsize=(7,7))
            fig.suptitle(f"{part} {stim} trials")
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

        for i, part in enumerate(['whisker_angle', 'whisker_velocity', 'whisker_speed']):
            fig, ax = plt.subplots(figsize=(7,7))
            fig.suptitle(f"{part} {stim} trials")
            plot_dlc_traces(data=total_avg_top.loc[(total_avg_top['trial_type'].isin([f'{stim}_hit_trial', f'{stim}_miss_trial']))],
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

    for i, part in enumerate(['jaw_angle', 'jaw_y', 'tongue_angle', "tongue_distance", 'pupil_area']):
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

    for i, part in enumerate(['whisker_angle', 'whisker_velocity']):
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


    for i, part in enumerate(['jaw_y']):
        rt_side = total_avg_side[~total_avg_side.trial_type.isin(['correct_rejection_trial', 'false_alarm_trial'])].groupby(by=['mouse_id', 'trial_type', 'context', 'correct_choice']).apply(
            lambda x: np.where(abs(x[f'{part}']/np.nanstd(x[part]))>=1, x.time, np.nan)).explode(0).reset_index().dropna()
        rt_side = rt_side.loc[(rt_side[0]>0) & (rt_side.trial_type.isin(['auditory_hit_trial', 'whisker_hit_trial', 'false_alarm_trial']))]
        rt_side = rt_side.groupby(by=['mouse_id', 'trial_type', 'context', 'correct_choice']).apply(lambda x: np.round(np.min(x)[0], 2)).reset_index()

        fig, ax= plt.subplots(figsize=(2,4))
        g = sns.pointplot(rt_side.dropna(),
                        x='trial_type',
                        y=0,
                        estimator='mean',
                        errorbar=('ci', 95),
                        order=['auditory_hit_trial', 'whisker_hit_trial'],
                        hue='context',
                        hue_order=['non-rewarded', 'rewarded'],
                        palette=['#6E188A', '#348A18'],
                        ax=ax,
                        dodge=True,
                        join=False
                            )
        sns.stripplot(rt_side.dropna(),
                    x='trial_type',
                    y=0,
                    order=['auditory_hit_trial', 'whisker_hit_trial'],
                    hue='context',
                    hue_order=['non-rewarded', 'rewarded'],
                    palette=['#6E188A', '#348A18'],
                    ax= ax,
                    dodge=0.3,
                    alpha=0.5)

        ax.get_legend().remove()
        ax.set_title(part)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xticklabels(['Auditory', 'Whisker'])
        ax.set_ylim([0,0.3])
        ax.set_ylabel('Reaction Time (s)')

        fig.tight_layout()
        fig.savefig(os.path.join(output_path, f'rt_{part}_all_trials.png'))
        fig.savefig(os.path.join(output_path, f'rt_{part}_all_trials.svg'))
    
    for i, part in enumerate(['whisker_angle']):

        rt_top = total_avg_top[~total_avg_top.trial_type.isin(['correct_rejection_trial', 'false_alarm_trial'])].groupby(by=['mouse_id', 'trial_type', 'context', 'correct_choice']).apply(lambda x: np.where(abs(x[f'{part}']/np.nanstd(x[part]))>=1, x.time, np.nan)).explode(0).reset_index().dropna()
        rt_top = rt_top.loc[(rt_top[0]>0) & (rt_top.trial_type.isin(['auditory_hit_trial', 'whisker_hit_trial', 'false_alarm_trial']))]
        rt_top = rt_top.groupby(by=['mouse_id', 'trial_type', 'context', 'correct_choice']).apply(lambda x: np.round(np.min(x)[0], 2)).reset_index()

        fig, ax= plt.subplots(figsize=(2,4))
        g = sns.pointplot(rt_top.dropna(),
                        x='trial_type',
                        y=0,
                        estimator='mean',
                        errorbar=('ci', 95),
                        order=['auditory_hit_trial', 'whisker_hit_trial'],
                        hue='context',
                        hue_order=['non-rewarded', 'rewarded'],
                        palette=['#6E188A', '#348A18'],
                        ax=ax,
                        dodge=True,
                        join=False
                            )
        sns.stripplot(rt_top.dropna(),
                    x='trial_type',
                    y=0,
                    order=['auditory_hit_trial', 'whisker_hit_trial'],
                    hue='context',
                    hue_order=['non-rewarded', 'rewarded'],
                    palette=['#6E188A', '#348A18'],
                    ax= ax,
                    dodge=0.3,
                    alpha=0.5)

        ax.get_legend().remove()
        ax.set_title(part)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xticklabels(['Auditory', 'Whisker'])
        ax.set_ylim([0,0.3])
        ax.set_ylabel('Reaction Time (s)')

        fig.tight_layout()
        fig.savefig(os.path.join(output_path, f'rt_{part}_all_trials.png'))
        fig.savefig(os.path.join(output_path, f'rt_{part}_all_trials.svg'))


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
    uncentered_combined_side_data['jaw_angle'] = 90 - uncentered_combined_side_data['jaw_angle']
    uncentered_combined_side_data['trial_count'] = (uncentered_combined_side_data['time'].diff().abs()>1).cumsum()
    uncentered_combined_side_data['jaw_speed'] = uncentered_combined_side_data.groupby(by=['trial_count']).apply(lambda x: np.pad(abs(np.diff(x.jaw_y.to_numpy())), (1,0), 'constant', constant_values=np.nan)).explode().reset_index()[0].values
    uncentered_combined_side_data['jaw_speed'] = uncentered_combined_side_data['jaw_speed']*200

    uncentered_combined_top_data = pd.concat(uncentered_combined_top_data)
    uncentered_combined_top_data['whisker_speed'] = uncentered_combined_top_data['whisker_velocity'].abs()*200

    uncentered_combined_top_data['trial_count'] = (uncentered_combined_top_data['time'].diff().abs()>1).cumsum()

    uncentered_combined_side_data = uncentered_combined_side_data[(uncentered_combined_side_data.time<0)]# & (uncentered_combined_side_data.time>-0.5)]
    uncentered_combined_top_data = uncentered_combined_top_data[(uncentered_combined_top_data.time<0)]# & (uncentered_combined_top_data.time>-0.5)]

    uncentered_combined_top_data['correct_choice'] = uncentered_combined_top_data['correct_choice'].astype(bool)
    uncentered_combined_side_data['correct_choice'] = uncentered_combined_side_data['correct_choice'].astype(bool)

    uncentered_agg_side_data = uncentered_combined_side_data.groupby(['session_id', 'context', 'trial_type', 'correct_choice', 'time']).agg('mean').reset_index()
    uncentered_agg_side_data['mouse_id'] = uncentered_agg_side_data.apply(lambda x: x.session_id.split("_")[0], axis=1)
    uncentered_agg_top_data = uncentered_combined_top_data.groupby(['session_id', 'context', 'trial_type', 'correct_choice', 'time']).agg('mean').reset_index()
    uncentered_agg_top_data['mouse_id'] = uncentered_agg_top_data.apply(lambda x: x.session_id.split("_")[0], axis=1)

    # norm = LogNorm(vmin=0.1, vmax=100000)
    # for i, choice in enumerate(uncentered_combined_top_data['correct_choice'].unique()):
    #     if np.isnan(choice):
    #         continue
    #     subset = uncentered_combined_top_data.loc[(uncentered_combined_top_data.trial_type.str.contains('whisker')) & (uncentered_combined_top_data.correct_choice==choice)].reset_index(drop=True)
    #     subset['top_particle_x'] = subset.groupby('session_id').apply(lambda x: x.top_particle_x - np.nanmean(x.loc[x.context=='rewarded', 'top_particle_x'].values)).reset_index()['top_particle_x']
    #     subset['top_particle_y'] = subset.groupby('session_id').apply(lambda x: x.top_particle_y - np.nanmean(x.loc[x.context=='rewarded', 'top_particle_y'].values)).reset_index()['top_particle_y']
    #     g=sns.jointplot(subset, 
    #                 x='top_particle_x', 
    #                 y='top_particle_y', 
    #                 kind='hist',
    #                 hue='context', 
    #                 hue_order=['rewarded', 'non-rewarded'],
    #                 palette=['#348A18', '#6E188A'],
    #                 alpha=0.5,
    #                 xlim=[-10, 10],
    #                 ylim=[-10, 10],
    #                 binwidth=1, 
    #                 cbar=False, 
    #                 norm=norm,
    #                 vmin=None,
    #                 vmax=None,
    #                 height=4,
    #                 ratio=3,
    #                 )      
    #     g.ax_joint.invert_yaxis()
    #     g.ax_joint.set_aspect('equal')

    #     g.ax_marg_x.clear()
    #     g.ax_marg_y.clear()
    #     for c, color in zip(['rewarded', 'non-rewarded'], ['#348A18', '#6E188A']):
    #         sns.histplot(data=subset.loc[subset.context==c], x='top_particle_x', ax=g.ax_marg_x, color=color, alpha=0.5, binwidth =1)
    #         sns.histplot(data=subset.loc[subset.context==c], y='top_particle_y', ax=g.ax_marg_y, color=color, alpha=0.5, binwidth =1)

    #     g.ax_marg_x.set_yscale('log')
    #     g.ax_marg_x.set_ylim([0.1, 1000000])
    #     g.ax_marg_x.set_axis_off()

    #     g.ax_marg_y.set_xscale('log')
    #     g.ax_marg_y.set_xlim([0.1, 1000000])
    #     g.ax_marg_y.invert_yaxis()
    #     g.ax_marg_y.set_axis_off()

    #     g.figure.tight_layout()
    #     g.figure.savefig(os.path.join(save_path,f'particle_set_point_heatmap_wh_trials_{"correct" if choice else "incorrect"}.png'))
    #     g.figure.savefig(os.path.join(save_path,f'particle_set_point_heatmap_wh_trials_{"correct" if choice else "incorrect"}.svg'))

    # for i, choice in enumerate(uncentered_combined_top_data['correct_choice'].unique()):
    #     if np.isnan(choice):
    #         continue
    #     subset = uncentered_combined_side_data.loc[(uncentered_combined_side_data.trial_type.str.contains('whisker')) & (uncentered_combined_side_data.correct_choice==choice)].reset_index(drop=True)
    #     subset['jaw_x'] = subset.groupby('session_id').apply(lambda x: x.jaw_x - np.nanmean(x.loc[x.context=='rewarded', 'jaw_x'].values)).reset_index()['jaw_x']
    #     subset['jaw_y'] = subset.groupby('session_id').apply(lambda x: x.jaw_y - np.nanmean(x.loc[x.context=='rewarded', 'jaw_y'].values)).reset_index()['jaw_y']
    #     g=sns.jointplot(subset, 
    #                 x='jaw_x', 
    #                 y='jaw_y', 
    #                 kind='hist',
    #                 hue='context', 
    #                 hue_order=['rewarded', 'non-rewarded'],
    #                 palette=['#348A18', '#6E188A'],
    #                 alpha=0.5,
    #                 xlim=[-20, 20],
    #                 ylim=[-20, 20],
    #                 binwidth=1, 
    #                 cbar=False, 
    #                 norm=norm,
    #                 vmin=None,
    #                 vmax=None,
    #                 height=4,
    #                 ratio=3,
    #                 )      
    #     g.ax_joint.invert_yaxis()
    #     g.ax_joint.set_aspect('equal')

    #     g.ax_marg_x.clear()
    #     g.ax_marg_y.clear()
    #     for c, color in zip(['rewarded', 'non-rewarded'], ['#348A18', '#6E188A']):
    #         sns.histplot(data=subset.loc[subset.context==c], x='jaw_x', ax=g.ax_marg_x, color=color, alpha=0.5, binwidth =1)
    #         sns.histplot(data=subset.loc[subset.context==c], y='jaw_y', ax=g.ax_marg_y, color=color, alpha=0.5, binwidth =1)

    #     g.ax_marg_x.set_yscale('log')
    #     g.ax_marg_x.set_ylim([0.1, 1000000])
    #     g.ax_marg_x.set_axis_off()

    #     g.ax_marg_y.set_xscale('log')
    #     g.ax_marg_y.set_xlim([0.1, 1000000])
    #     g.ax_marg_y.invert_yaxis()
    #     g.ax_marg_y.set_axis_off()

    #     g.figure.tight_layout()
    #     g.figure.savefig(os.path.join(save_path,f'jaw_set_point_heatmap_side_wh_trials_{"correct" if choice else "incorrect"}.png'))
    #     g.figure.savefig(os.path.join(save_path,f'jaw_set_point_heatmap_side_wh_trials_{"correct" if choice else "incorrect"}.svg'))

    uncentered_combined_side_data['legend'] = uncentered_combined_side_data.apply(lambda x: f"{x.context} - {'correct' if x.correct_choice==1 else 'incorrect'}", axis=1)
    uncentered_combined_side_data['stim_type'] = uncentered_combined_side_data.apply(lambda x: x.trial_type.split("_")[0], axis=1)
    uncentered_combined_side_data = uncentered_combined_side_data.loc[uncentered_combined_side_data.trial_type.str.contains('trial')]

    uncentered_combined_top_data['legend'] = uncentered_combined_top_data.apply(lambda x: f"{x.context} - {'correct' if x.correct_choice==1 else 'incorrect'}", axis=1)
    uncentered_combined_top_data['stim_type'] = uncentered_combined_top_data.apply(lambda x: x.trial_type.split("_")[0], axis=1)
    uncentered_combined_top_data = uncentered_combined_top_data.loc[uncentered_combined_top_data.trial_type.str.contains('trial')]

    data = uncentered_combined_side_data.groupby(by=['mouse_id', 'session_id', 'context', 'context_background', 'trial_type', 'correct_choice', 'legend', 'stim_type', 'trial_count']).agg({'jaw_y':np.nanmean, 'jaw_speed':np.nanmean, 'pupil_area':np.nanmean}).reset_index()
    data = data.merge(uncentered_combined_top_data.groupby(by=['mouse_id', 'session_id', 'context', 'context_background', 'trial_type', 'correct_choice', 'legend', 'stim_type', 'trial_count']).agg({'whisker_angle':np.nanmean, 'whisker_speed':np.nanmean}).reset_index()[['trial_count', 'whisker_angle', 'whisker_speed']], on='trial_count')
    # data = data.melt(id_vars=['mouse_id', 'session_id', 'context', 'trial_type', 'correct_choice', 'legend', 'stim_type', 'trial_count'], value_vars=['jaw_angle', 'jaw_speed', 'pupil_area', 'whisker_angle', 'whisker_speed'], var_name='bodypart')
    data = data.melt(id_vars=['mouse_id', 'session_id', 'context', 'trial_type', 'correct_choice', 'legend', 'stim_type', 'trial_count'], value_vars=['jaw_y', 'jaw_speed', 'pupil_area', 'whisker_angle', 'whisker_speed'], var_name='bodypart')
    data['correct_choice'] = data.correct_choice.astype(bool)
    data['lick'] = data['legend'].map({'non-rewarded - incorrect': 1, 'non-rewarded - correct': 0, 'rewarded - correct': 1, 'rewarded - incorrect': 0})
    data = data[data.stim_type=='whisker']
    n_comparisons = 20

    context_data = data.groupby(by=['mouse_id', 'session_id', 'context', 'bodypart']).agg('mean').reset_index()
    context_data = context_data.groupby(by=['mouse_id', 'context', 'bodypart']).agg('mean').reset_index()

    stats = []
    for name, group in context_data.groupby(by='bodypart'):
        correct= group.loc[group.context=='rewarded'].dropna()
        incorrect = group.loc[group.context=='non-rewarded'].dropna()
        if correct.shape[0] != incorrect.shape[0]:
            correct = correct[correct.mouse_id.isin(incorrect.mouse_id)]

        t, p = ttest_rel(correct['value'].values, incorrect['value'].values)
        results = {
         'bodypart': name,
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
         'd_prime': abs((correct['value'].mean()-incorrect['value'].mean()))/np.std(correct['value'].to_numpy()-incorrect['value'].to_numpy())
         }
        stats+=[results]
    stats= pd.DataFrame(stats)
    stats.to_csv(os.path.join(save_path, 'stats_context_gral_effect.csv'))

    fig, axes = plt.subplots(1, len(context_data.bodypart.unique()), figsize=(12,3))
    for ax, part in zip(axes.flat, context_data.bodypart.unique()):
        subset = context_data[context_data.bodypart==part]
        ax.set_title(f"{part} \nn = {stats.loc[(stats.bodypart==part), 'dof'].to_numpy()[0]+1}")
        ax.spines[['top', 'right']].set_visible(False)

        g = sns.pointplot(subset,
                        x='context', 
                        y='value', 
                        order=['non-rewarded', 'rewarded'],
                        palette=['#6E188A', '#348A18'],
                        estimator='mean',
                        errorbar=('ci', 95),
                        markers='o',
                        scale=1.3,
                        join=False,
                        dodge=True,
                        ax = ax
                        )
    
        pivoted = subset.pivot(index='mouse_id', columns='context', values='value')
        pivoted = pivoted.dropna()
        for _, row in pivoted.iterrows():
            ax.plot([0.1, 0.9], row.values, color='gray', alpha=0.4, linewidth=3)

        if stats.loc[stats.bodypart==part, 'significant'].any():
            star_loc = max(ax.get_ylim())
            ax.scatter(.5, stats.loc[(stats.bodypart==part), 'significant'].map({True: 1}).to_numpy()*star_loc*0.9, marker='*', s=100, c='k')

        ax.margins(x=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path,'context_gral_effect.png'))
    fig.savefig(os.path.join(save_path,'context_gral_effect.svg'))

    choice_data = data.groupby(by=['mouse_id', 'session_id', 'correct_choice', 'bodypart']).agg('mean').reset_index()
    choice_data = choice_data.groupby(by=['mouse_id', 'correct_choice', 'bodypart']).agg('mean').reset_index()

    stats = []
    for name, group in choice_data.groupby(by='bodypart'):
        correct= group.loc[group.correct_choice==True].dropna()
        incorrect = group.loc[group.correct_choice==False].dropna()
        if correct.shape[0] != incorrect.shape[0]:
            correct = correct[correct.mouse_id.isin(incorrect.mouse_id)]

        t, p = ttest_rel(correct['value'].values, incorrect['value'].values)
        results = {
         'bodypart': name,
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
         'd_prime': abs((correct['value'].mean()-incorrect['value'].mean()))/np.std(correct['value'].to_numpy()-incorrect['value'].to_numpy())
         }
        stats+=[results]
    stats= pd.DataFrame(stats)
    stats.to_csv(os.path.join(save_path, 'stats_choice_gral_effect.csv'))

    fig, axes = plt.subplots(1, len(choice_data.bodypart.unique()), figsize=(12,3))
    for ax, part in zip(axes.flat, choice_data.bodypart.unique()):
        subset = choice_data[choice_data.bodypart==part]
        ax.set_title(f"{part} \nn = {stats.loc[(stats.bodypart==part), 'dof'].to_numpy()[0]+1}")
        ax.spines[['top', 'right']].set_visible(False)

        g = sns.pointplot(subset,
                        x='correct_choice', 
                        y='value', 
                        order=[False, True],
                        palette=['#a0a0a0', '#000000'],
                        estimator='mean',
                        errorbar=('ci', 95),
                        markers='o',
                        scale=1.3,
                        join=False,
                        dodge=True,
                        ax = ax
                        )
    
        pivoted = subset.pivot(index='mouse_id', columns='correct_choice', values='value')
        pivoted = pivoted.dropna()
        for _, row in pivoted.iterrows():
            ax.plot([0.1, 0.9], row.values, color='gray', alpha=0.4, linewidth=3)
            
        if stats.loc[stats.bodypart==part, 'significant'].any():
            star_loc = max(ax.get_ylim())
            ax.scatter(.5, stats.loc[(stats.bodypart==part), 'significant'].map({True: 1}).to_numpy()*star_loc*0.9, marker='*', s=100, c='k')

        ax.margins(x=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path,'choice_gral_effect.png'))
    fig.savefig(os.path.join(save_path,'choice_gral_effect.svg'))

    lick_data = data.groupby(by=['mouse_id', 'session_id', 'lick', 'bodypart']).agg('mean').reset_index()
    lick_data = lick_data.groupby(by=['mouse_id', 'lick', 'bodypart']).agg('mean').reset_index()

    stats = []
    for name, group in lick_data.groupby(by='bodypart'):
        correct= group.loc[group.lick==True].dropna()
        incorrect = group.loc[group.lick==False].dropna()
        if correct.shape[0] != incorrect.shape[0]:
            correct = correct[correct.mouse_id.isin(incorrect.mouse_id)]

        t, p = ttest_rel(correct['value'].values, incorrect['value'].values)
        results = {
         'bodypart': name,
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
         'd_prime': abs((correct['value'].mean()-incorrect['value'].mean()))/np.std(correct['value'].to_numpy()-incorrect['value'].to_numpy())
         }
        stats+=[results]
    stats= pd.DataFrame(stats)
    stats.to_csv(os.path.join(save_path, 'stats_lick_gral_effect.csv'))

    fig, axes = plt.subplots(1, len(lick_data.bodypart.unique()), figsize=(12,3))
    for ax, part in zip(axes.flat, lick_data.bodypart.unique()):
        subset = lick_data[lick_data.bodypart==part]
        ax.set_title(f"{part} \nn = {stats.loc[(stats.bodypart==part), 'dof'].to_numpy()[0]+1}")
        ax.spines[['top', 'right']].set_visible(False)

        g = sns.pointplot(subset,
                        x='lick', 
                        y='value', 
                        order=[False, True],
                        palette=['#a0a0a0', '#000000'],
                        estimator='mean',
                        errorbar=('ci', 95),
                        markers='o',
                        scale=1.3,
                        join=False,
                        dodge=True,
                        ax = ax
                        )
    
        pivoted = subset.pivot(index='mouse_id', columns='lick', values='value')
        pivoted = pivoted.dropna()
        for _, row in pivoted.iterrows():
            ax.plot([0.1, 0.9], row.values, color='gray', alpha=0.4, linewidth=3)
            
        if stats.loc[stats.bodypart==part, 'significant'].any():
            star_loc = max(ax.get_ylim())
            ax.scatter(.5, stats.loc[(stats.bodypart==part), 'significant'].map({True: 1}).to_numpy()*star_loc*0.9, marker='*', s=100, c='k')

        ax.margins(x=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path,'lick_gral_effect.png'))
    fig.savefig(os.path.join(save_path,'lick_gral_effect.svg'))

    choice_vs_context_data = data.groupby(by=['mouse_id', 'session_id', 'context', 'correct_choice', 'bodypart']).agg('mean').reset_index()
    choice_vs_context_data = choice_vs_context_data.groupby(by=['mouse_id', 'context', 'correct_choice', 'bodypart']).agg('mean').reset_index()

    stats = []
    for name, group in choice_vs_context_data.groupby(by=['bodypart', 'context']):
        correct= group.loc[group.correct_choice==True].dropna()
        incorrect = group.loc[group.correct_choice==False].dropna()
        if correct.shape[0] != incorrect.shape[0]:
            correct = correct[correct.mouse_id.isin(incorrect.mouse_id)]

        t, p = ttest_rel(correct['value'].values, incorrect['value'].values)
        results = {
         'bodypart': name[0],
         'context': name[1],
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
         'd_prime': abs((correct['value'].mean()-incorrect['value'].mean()))/np.std(correct['value'].to_numpy()-incorrect['value'].to_numpy())
         }
        stats+=[results]
    stats= pd.DataFrame(stats)
    stats.to_csv(os.path.join(save_path, 'stats_context_vs_choice_mixed_effects.csv'))

    palette = {'non-rewarded - incorrect': '#C5A2D0',
               'non-rewarded - correct': '#6E188A',
                'rewarded - incorrect': '#ADD0A2',
                'rewarded - correct': '#348A18'}
    fig, axes = plt.subplots(1, len(choice_vs_context_data.bodypart.unique()), figsize=(12,3))
    for ax, part in zip(axes.flat, choice_vs_context_data.bodypart.unique()):
        ax.set_title(f"{part} \nn = {stats.loc[(stats.bodypart==part), 'dof'].unique()[0]+1}")
        ax.margins(x=0.25)
        ax.spines[['top', 'right']].set_visible(False)
        
        subset = choice_vs_context_data[(choice_vs_context_data.bodypart==part)]# & (choice_vs_context_data.context==c)]
        subset['legend'] = subset.apply(lambda x: f"{x.context} - {'correct' if x.correct_choice==1 else 'incorrect'}", axis=1)
        subset['color'] = subset['legend'].map(palette)

        g = sns.pointplot(subset,
                        x='legend', 
                        y='value', 
                        hue='legend',
                        palette=palette,
                        estimator='mean',
                        errorbar=('ci', 95),
                        markers='o',
                        scale=1.3,
                        join=False,
                        dodge=False,
                        ax = ax
                        )
        ax.get_legend().set_visible(False)
        ax.set_xlabel('')
        ax.set_xticklabels([])
        for i, c in enumerate(choice_vs_context_data.context.unique()):
            if i==1:
                i=2
            pivoted = subset.loc[subset.context==c].pivot(index='mouse_id', columns=['correct_choice'], values='value')
            pivoted = pivoted.dropna()
            for _, row in pivoted.iterrows():
                ax.plot([i+0.1, i+0.9], row.values, color='gray', alpha=0.4, linewidth=3)
            
            if stats.loc[stats.bodypart==part, 'significant'].any():
                star_loc = max(ax.get_ylim())
                ax.scatter([i+0.5], stats.loc[(stats.bodypart==part) & (stats.context==c), 'significant'].map({True: 1}).to_numpy()*star_loc*0.9, marker='*', s=100, c='k')

    fig.tight_layout()
    fig.savefig(os.path.join(save_path,'context_choice_mixed_effect.png'))
    fig.savefig(os.path.join(save_path,'context_choice_mixed_effect.svg'))


    lick_vs_context_data = data.groupby(by=['mouse_id', 'session_id', 'context', 'lick', 'bodypart']).agg('mean').reset_index()
    lick_vs_context_data = lick_vs_context_data.groupby(by=['mouse_id', 'context', 'lick', 'bodypart']).agg('mean').reset_index()
    lick_vs_context_data['legend'] = lick_vs_context_data.apply(lambda x: f"{x.context} - {'lick' if x.lick else 'no-lick'}", axis=1)
    stats = []
    for name, group in lick_vs_context_data.groupby(by=['bodypart', 'context']):
        correct= group.loc[group.lick==True].dropna()
        incorrect = group.loc[group.lick==False].dropna()
        if correct.shape[0] != incorrect.shape[0]:
            correct = correct[correct.mouse_id.isin(incorrect.mouse_id)]

        t, p = ttest_rel(correct['value'].values, incorrect['value'].values)
        results = {
         'bodypart': name[0],
         'context': name[1],
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
         'd_prime': abs((correct['value'].mean()-incorrect['value'].mean()))/np.std(correct['value'].to_numpy()-incorrect['value'].to_numpy())
         }
        stats+=[results]
    stats= pd.DataFrame(stats)
    stats.to_csv(os.path.join(save_path, 'stats_context_vs_lick_mixed_effects.csv'))

    palette = {'non-rewarded - no-lick': '#C5A2D0',
               'non-rewarded - lick': '#6E188A',
                'rewarded - no-lick': '#ADD0A2',
                'rewarded - lick': '#348A18'}
    fig, axes = plt.subplots(1, len(lick_vs_context_data.bodypart.unique()), figsize=(12,3))
    for ax, part in zip(axes.flat, lick_vs_context_data.bodypart.unique()):
        ax.set_title(f"{part} \nn = {stats.loc[(stats.bodypart==part), 'dof'].unique()[0]+1}")
        ax.margins(x=0.25)
        ax.spines[['top', 'right']].set_visible(False)
        
        subset = lick_vs_context_data[(lick_vs_context_data.bodypart==part)]# & (choice_vs_context_data.context==c)]
        subset['legend'] = subset.apply(lambda x: f"{x.context} - {'lick' if x.lick==1 else 'no-lick'}", axis=1)
        subset['color'] = subset['legend'].map(palette)

        g = sns.pointplot(subset,
                        x='legend', 
                        y='value', 
                        hue='legend',
                        palette=palette,
                        estimator='mean',
                        errorbar=('ci', 95),
                        markers='o',
                        scale=1.3,
                        join=False,
                        dodge=False,
                        ax = ax
                        )
        ax.get_legend().set_visible(False)
        ax.set_xlabel('')
        ax.set_xticklabels([])
        for i, c in enumerate(lick_vs_context_data.context.unique()):
            if i==1:
                i=2
            pivoted = subset.loc[subset.context==c].pivot(index='mouse_id', columns=['lick'], values='value')
            pivoted = pivoted.dropna()
            for _, row in pivoted.iterrows():
                ax.plot([i+0.1, i+0.9], row.values, color='gray', alpha=0.4, linewidth=3)
            
            if stats.loc[stats.bodypart==part, 'significant'].any():
                star_loc = max(ax.get_ylim())
                ax.scatter([i+0.5], stats.loc[(stats.bodypart==part) & (stats.context==c), 'significant'].map({True: 1}).to_numpy()*star_loc*0.9, marker='*', s=100, c='k')

    fig.tight_layout()
    fig.savefig(os.path.join(save_path,'context_lick_mixed_effect.png'))
    fig.savefig(os.path.join(save_path,'context_lick_mixed_effect.svg'))

    reference = 'non-rewarded - no-lick'
    norm_df = []
    for i, row in lick_vs_context_data.iterrows():
        mouse_id = row.mouse_id
        bodypart = row.bodypart
        row.value = row.value - lick_vs_context_data.loc[(lick_vs_context_data.mouse_id==mouse_id) & (lick_vs_context_data.bodypart==bodypart) & (lick_vs_context_data.legend==reference), 'value'].to_numpy()[0]
        norm_df += [row]
    norm_df = pd.DataFrame(norm_df)
    
    lick_vs_context_data = norm_df
    fig, axes = plt.subplots(1, len(lick_vs_context_data.bodypart.unique()), figsize=(12,3))
    for ax, part in zip(axes.flat, lick_vs_context_data.bodypart.unique()):
        ax.set_title(f"{part} \nn = {stats.loc[(stats.bodypart==part), 'dof'].unique()[0]+1}")
        ax.margins(x=0.25)
        ax.spines[['top', 'right']].set_visible(False)
        
        subset = lick_vs_context_data[(lick_vs_context_data.bodypart==part)]# & (choice_vs_context_data.context==c)]
        subset['legend'] = subset.apply(lambda x: f"{x.context} - {'lick' if x.lick==1 else 'no-lick'}", axis=1)
        subset['color'] = subset['legend'].map(palette)

        g = sns.pointplot(subset,
                        x='legend', 
                        y='value', 
                        hue='legend',
                        palette=palette,
                        estimator='mean',
                        errorbar=('ci', 95),
                        markers='o',
                        scale=1.3,
                        join=False,
                        dodge=False,
                        ax = ax
                        )
        ax.get_legend().set_visible(False)
        ax.set_xlabel('')
        ax.set_xticklabels([])
        for i, c in enumerate(lick_vs_context_data.context.unique()):
            if i==1:
                i=2
            pivoted = subset.loc[subset.context==c].pivot(index='mouse_id', columns=['lick'], values='value')
            pivoted = pivoted.dropna()
            for _, row in pivoted.iterrows():
                ax.plot([i+0.1, i+0.9], row.values, color='gray', alpha=0.4, linewidth=3)
            
            if stats.loc[stats.bodypart==part, 'significant'].any():
                star_loc = max(ax.get_ylim())
                ax.scatter([i+0.5], stats.loc[(stats.bodypart==part) & (stats.context==c), 'significant'].map({True: 1}).to_numpy()*star_loc*0.9, marker='*', s=100, c='k')

    fig.tight_layout()
    fig.savefig(os.path.join(save_path,'context_lick_mixed_effect_norm.png'))
    fig.savefig(os.path.join(save_path,'context_lick_mixed_effect_norm.svg'))

    sub_df = lick_vs_context_data.groupby(by=['mouse_id', 'context', 'bodypart'], as_index=False, sort=False).apply(
        lambda x: x.loc[x.lick==1, 'value'].to_numpy()[0] - x.loc[x.lick==0, 'value'].to_numpy()[0]).rename(columns={None:'value'})
    fig, axes = plt.subplots(1, len(sub_df.bodypart.unique()), figsize=(10,4))
    for ax, part in zip(axes.flat, lick_vs_context_data.bodypart.unique()):
        ax.set_title(f"{part} \nn = {stats.loc[(stats.bodypart==part), 'dof'].unique()[0]+1}")
        ax.margins(x=0.25)
        ax.spines[['top', 'right']].set_visible(False)
        
        subset = sub_df[(sub_df.bodypart==part)].dropna()# & (choice_vs_context_data.context==c)]
        subset['color'] = subset['context'].map({'rewarded':'green', 'non-rewarded':'purple'})

        g = sns.boxplot(subset,
                        x='context', 
                        y='value', 
                        hue='context',
                        palette=['purple', 'green'],
                        whis=(2.5, 97.5),
                        showfliers=False,
                        linewidth=1,
                        saturation=0.5,
                        dodge=False,
                        ax = ax
                        )
        sns.stripplot(ax=ax, data=subset, x='context', y='value', hue='context', palette=['purple', 'green'],
              dodge=True, jitter=0.05, zorder=0)
        ax.get_legend().set_visible(False)
        ax.set_xlabel('')
        ax.set_xticklabels([])
        # for i, c in enumerate(sub_df.context.unique()):
        #     if i==1:
        #         i=2
        #     if stats.loc[stats.bodypart==part, 'significant'].any():
        #         star_loc = max(ax.get_ylim())
        #         ax.scatter([i+0.5], stats.loc[(stats.bodypart==part) & (stats.context==c), 'significant'].map({True: 1}).to_numpy()*star_loc*0.9, marker='*', s=100, c='k')

    fig.tight_layout()
    fig.savefig(os.path.join(save_path,'context_lick_mixed_effect_norm_diff.png'))
    fig.savefig(os.path.join(save_path,'context_lick_mixed_effect_norm_diff.svg'))


def plot_example_traces():
    nwb_file = '/mnt/lsens-analysis/Pol_Bech/NWB/PB185_20240823_102701.nwb'
    keys = ['behavior', 'BehavioralTimeSeries']
    timestamps = nwb_read.get_dlc_timestamps(nwb_file, keys=['behavior', 'BehavioralTimeSeries'])

    trial_table = nwb_read.get_trial_table(nwb_file)
    trial_table['context'] = trial_table['context'].map({0: 'non-rewarded', 1: 'rewarded'})    
    
    # dlc_data = pd.DataFrame(columns=['whisker_angle', 'jaw_angle', 'pupil_area'])
    # dlc_data = pd.DataFrame(columns=['whisker_angle', 'jaw_distance', 'pupil_area'])
    dlc_data = pd.DataFrame(columns=['whisker_angle', 'jaw_y', 'pupil_area'])

    # for part in ['whisker_angle', 'jaw_angle', 'pupil_area']:
    for part in ['whisker_angle', 'jaw_y', 'pupil_area']:
        dlc_data[part] = get_likelihood_filtered_bodypart(nwb_file, keys, part, threshold=0.5)
    dlc_data['time'] = timestamps[0]

    time = (75, 115)
    # g = sns.FacetGrid(dlc_data.loc[(dlc_data.time>=time[0])&(dlc_data.time<time[1])].melt(id_vars='time', 
    #                                                                             value_vars=['whisker_angle', 'jaw_angle', 'pupil_area'], 
    #                                                                             var_name='part', 
    #                                                                             value_name='movement'), 
    #                                 row='part', sharey=False, height=1, aspect=3)    
    g = sns.FacetGrid(dlc_data.loc[(dlc_data.time>=time[0])&(dlc_data.time<time[1])].melt(id_vars='time', 
                                                                                value_vars=['whisker_angle', 'jaw_y', 'pupil_area'], 
                                                                                var_name='part', 
                                                                                value_name='movement'), 
                                    row='part', sharey=False, height=1, aspect=3)
    g.map_dataframe(sns.lineplot, x='time', y='movement', color='k', linewidth=1)
    for ax in g.axes:
        trials = trial_table.loc[(trial_table.start_time>=time[0]) & (trial_table.start_time<time[1]), ['start_time', 'trial_type']]
        trials['color'] = trials.trial_type.map({'no_stim_trial':'gray', 'whisker_trial':'orange', 'auditory_trial':'blue'})
        for i, trial in trials.iterrows():
            ax[0].axvline(trial.start_time, c=trial.color, linestyle='--', linewidth=1)

    g.figure.savefig(os.path.join(output_path,'dlc_example_traces.png'))
    g.figure.savefig(os.path.join(output_path,'dlc_example_traces.svg'))


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
        output_path = os.path.join(f'{utils_io.get_experimenter_saving_folder_root("PB")}', 'Pop_results', 'Context_behaviour', 'combined_dlc_results', 'adaptive_threshold', dtype)
        output_path = haas_pathfun(output_path.replace("\\", "/"))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        if recompute_data:
            print("ATTENTION: Preprocessing dlc data")
            combined_side_data, combined_top_data, uncentered_combined_side_data, uncentered_combined_top_data = compute_dlc_data(nwb_files, output_path)

        # data_path = glob.glob(os.path.join(output_path, '*.csv'))
        # main(data_path, output_path=os.path.join(output_path, 'results'))
        
    # output_path = os.path.join(f'{utils_io.get_experimenter_saving_folder_root("PB")}', 'Pop_results', 'Context_behaviour', 'combined_dlc_results', 'likelihood_70')
    output_path = os.path.join(f'{utils_io.get_experimenter_saving_folder_root("PB")}', 'Pop_results', 'Context_behaviour', 'combined_dlc_results', 'adaptive_threshold')
    output_path = haas_pathfun(output_path.replace("\\", "/"))

    data_path = glob.glob(os.path.join(output_path, '**', '*.csv'))
    main(data_path, output_path=os.path.join(output_path, 'results_jaw_y'))