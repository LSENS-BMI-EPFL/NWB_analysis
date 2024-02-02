import numpy as np
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import pandas as pd
import os
import behavior_analysis_utils as bhv_utils

from plotting_utils import lighten_color, remove_top_right_frame, remove_bottom_right_frame, save_figure_to_files


def plot_single_mouse_across_days(combine_bhv_data, color_palette, saving_path):
    mice_list = np.unique(combine_bhv_data['mouse_id'].values[:])
    n_mice = len(mice_list)
    print(f" ")
    print(f"Plot average across days for {n_mice} mice")
    for mouse_id in mice_list:
        print(f"Mouse : {mouse_id}")
        mouse_table = bhv_utils.get_single_mouse_table(combine_bhv_data, mouse=mouse_id)

        # Keep only Auditory and Whisker days
        mouse_table = mouse_table[mouse_table.behavior.isin(('auditory', 'whisker', 'whisker_psy'))]

        # Select columns for plot
        cols = ['outcome_a', 'outcome_w', 'outcome_n', 'day']
        df = mouse_table.loc[mouse_table.early_lick == 0, cols]

        # Compute hit rates. Use transform to propagate hit rate to all entries.
        df['hr_w'] = df.groupby(['day'], as_index=False)['outcome_w'].transform(np.nanmean)
        df['hr_a'] = df.groupby(['day'], as_index=False)['outcome_a'].transform(np.nanmean)
        df['hr_n'] = df.groupby(['day'], as_index=False)['outcome_n'].transform(np.nanmean)

        # Average by day for this mouse
        df_by_day = df.groupby(['day'], as_index=False).agg(np.nanmean)

        # Do the plot
        figsize = (4, 6)
        figure, ax = plt.subplots(1, 1, figsize=figsize)
        sns.lineplot(data=df_by_day, x='day', y='hr_n', color='k', ax=ax, marker='o')
        sns.lineplot(data=df_by_day, x='day', y='hr_a', color=color_palette[0], ax=ax, marker='o')
        if max(df_by_day['day'].values) >= 0:  # This means there's one whisker training day at least
            sns.lineplot(data=df_by_day, x='day', y='hr_w', color=color_palette[2], ax=ax, marker='o')

        ax.set_ylim([-0.1, 1.05])
        ax.set_xlabel('Day')
        ax.set_ylabel('Lick probability')
        ax.set_title(f"{mouse_id}")
        sns.despine()

        save_formats = ['pdf', 'png', 'svg']
        for save_format in save_formats:
            figure.savefig(os.path.join(f'{saving_path}', f'{mouse_id}_fast_learning.{save_format}'),
                           format=f"{save_format}")

        plt.close()


def plot_single_mouse_weight_across_days(combine_bhv_data, color_palette, saving_path):
    mice_list = np.unique(combine_bhv_data['mouse_id'].values[:])
    n_mice = len(mice_list)
    print(f" ")
    print(f"Plot normalized mouse weight across days for {n_mice} mice")

    for mouse_id in mice_list:
        print(f"Mouse : {mouse_id}")
        mouse_table = bhv_utils.get_single_mouse_table(combine_bhv_data, mouse=mouse_id)

        # Keep only Auditory and Whisker days
        mouse_table = mouse_table[mouse_table.behavior.isin(('auditory', 'whisker', 'whisker_psy'))]

        # Select columns for plot
        cols = ['normalized_weight', 'day']
        df = mouse_table.loc[mouse_table.early_lick == 0, cols]
        df.normalized_weight = df.normalized_weight * 100

        # Do the plot
        figsize = (4, 2)
        figure, ax = plt.subplots(1, 1, figsize=figsize)
        sns.lineplot(data=df, x='day', y='normalized_weight', color='grey', ax=ax, marker='o')

        ax.set_ylim([70, 100])
        ax.set_yticks([70, 80, 90, 100])
        ax.set_xlabel('Day')
        ax.set_ylabel('Normalized weight [%]')
        ax.set_title(f"{mouse_id}")
        sns.despine()

        save_formats = ['pdf', 'png', 'svg']
        for save_format in save_formats:
            figure.savefig(os.path.join(f'{saving_path}', f'{mouse_id}_weight.{save_format}'),
                           format=f"{save_format}",
                           bbox_inches='tight'
                           )

        plt.close()


def plot_single_session_piezo_raster(combine_bhv_data, color_palette, saving_path, whisker_only=False):
    """
    Plot raster of piezo licks for each session.
    :param combine_bhv_data:
    :param color_palette:
    :param saving_path:
    :return:
    """
    sessions_list = np.unique(combine_bhv_data['session_id'].values[:])
    n_sessions = len(sessions_list)
    print(f"N sessions : {n_sessions}")
    for session_id in sessions_list:
        session_table, _, _ = bhv_utils.get_standard_single_session_table(combine_bhv_data,
                                                                        session=session_id)


        if session_table['behavior'].values[0] not in ['auditory', 'whisker'] and not whisker_only:
            continue
        elif session_table['behavior'].values[0] not in ['whisker'] and whisker_only:
            continue

        # Keep lick trials only, reverse trials from early to late
        session_table = session_table[session_table.lick_flag == 1]
        session_table = session_table.iloc[::-1]
        if whisker_only:
            session_table = session_table[session_table.trial_type == 'whisker_trial']
        lick_times_aligned = session_table['piezo_lick_times']
        session_table['quiet_window'] = session_table['start_time'] - session_table['abort_window_start_time']

        # Make plot
        fig, ax = plt.subplots(1, 1, figsize=(4, 6), dpi=300)
        remove_bottom_right_frame(ax)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False,
                       axis='both', which='major', labelsize=12)
        ax.invert_yaxis()
        max_time = 5
        ax.set_xlim(-0.2, max_time)
        ax.set_xticks(np.arange(0, max_time+0.1, 1.0))
        ax.set_ylim(len(lick_times_aligned) - 0.5, -0.5)
        ax.axvline(x=0, color='k', lw=1, ls='--')


        # Plot lick raster from top to bottom
        trial_collection = ax.eventplot(lick_times_aligned)

        # Make trial color map
        reward_group = session_table['reward_group'].values[0]
        wh_color = 'forestgreen' if reward_group == 1 else 'crimson'
        color_dict = {
            'auditory_trial': 'mediumblue',
            'whisker_trial': wh_color,
            'no_stim_trial': 'k',
        }
        cmap = [color_dict[trial_type] for trial_type in session_table['trial_type']]
        for idx, col in enumerate(trial_collection):
            # Set color per trial
            trial_color = cmap[idx]
            col.set_colors(trial_color)

            # Plot quiet windows
            #offset = col.get_lineoffset() # get dims
            #offsets.append(offset)
            #lw = col.get_linewidth() * 3
            #lh = col.get_linelength()
            #quiet_window = session_table['quiet_window'].values[idx]
            # print(offset, offset+lw, -quiet_window, 0)
            #ax.hlines(  # ymin=offset,
            #    # ymax=offset+lw,
            #    y=offset,
            #    xmin=-quiet_window,
            #    xmax=0,
            #    color='dimgrey',
            #    lw=lw,
            #    alpha=0.2)

        # Add legend and title
        ax.set_xlabel('Time [s]', fontsize=15)
        ax.set_ylabel('Lick trials', fontsize=15)
        figure_title = f"{session_table.mouse_id.values[0]}, {session_table.behavior.values[0]} " \
                       f"{session_table.day.values[0]}"
        ax.set_title(figure_title)

        fig.tight_layout()
        plt.show()

        # Save figure
        save_formats = ['pdf', 'png', 'svg']
        figure_name = f"{session_table.mouse_id.values[0]}_{session_table.behavior.values[0]}_" \
                      f"{session_table.day.values[0]}_piezo_raster"
        if whisker_only:
            figure_name = figure_name + '_whisker'

        session_saving_path = os.path.join(saving_path,
                                           f'{session_table.behavior.values[0]}_{session_table.day.values[0]}')
        if not os.path.exists(session_saving_path):
            os.makedirs(session_saving_path)
        for save_format in save_formats:
            fig.savefig(os.path.join(f'{session_saving_path}', f'{figure_name}.{save_format}'),
                           format=f"{save_format}")
    return

def plot_single_mouse_piezo_lick_probability(combine_bhv_data, color_palette, saving_path):
    """
    Plot time-binned piezo lick probability for each session.
    :param combine_bhv_data:
    :param color_palette:
    :param saving_path:
    :return:
    """
    all_session_data = []

    sessions_list = np.unique(combine_bhv_data['session_id'].values[:])
    n_sessions = len(sessions_list)
    print(f"N sessions : {n_sessions}")
    for session_id in sessions_list:

        session_table, _, _ = bhv_utils.get_standard_single_session_table(combine_bhv_data,
                                                                          session=session_id)
        if session_table['behavior'].values[0] not in ['auditory', 'whisker']:
            continue

        # Plotting parameters
        bin_size_sec = 0.1
        reward_group = session_table['reward_group'].values[0]
        wh_color = 'forestgreen' if reward_group == 1 else 'crimson'
        trial_type_dict = {'auditory_trial': 'mediumblue', 'whisker_trial': wh_color, 'no_stim_trial': 'k'}

        # Make plot
        fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)
        remove_top_right_frame(ax)
        ax.tick_params(axis='both', which='major', labelsize=12)

        for trial_type in trial_type_dict.keys():
            print(f"Trial type : {trial_type}")
            session_table_sub = session_table[session_table.trial_type == trial_type]

            session_dict = dict.fromkeys(
                ['mouse_id', 'behavior', 'day', 'lick_count_mean', 'bins', 'trial_type', 'reward_group', 'session_id'])

            # Count number of licks per bin
            lick_count = []
            for idx, trial in session_table_sub.iterrows():
                counts = np.histogram(trial['piezo_lick_times'], bins=np.arange(0, 5+bin_size_sec, bin_size_sec), density=False)[0]
                lick_count.append(counts)

            lick_count = np.asarray(lick_count)
            lick_count[np.where(lick_count>1)] = 1 # count presence of at least one lick
            lick_count_mean = np.mean(lick_count, axis=0)

            # Plot average lick count bin-wise, for each trial type
            ax.plot(lick_count_mean, c=trial_type_dict[trial_type], lw=2)

            if np.isnan(lick_count_mean).any(): # for whisker trials when absent
                lick_count_mean = np.empty(int(1/bin_size_sec))
                lick_count_mean[:] = np.nan

            # Append trial-specific session data to container, for each bin
            for bin in range(len(lick_count_mean)):
                session_dict = dict.fromkeys(
                    ['mouse_id', 'behavior', 'day', 'lick_count_mean', 'bin_number', 'trial_type', 'reward_group', 'session_id'])
                session_dict['mouse_id'] = session_table.mouse_id.values[0]
                session_dict['behavior'] = session_table.behavior.values[0]
                session_dict['day'] = session_table.day.values[0]
                session_dict['lick_count_mean'] = lick_count_mean[bin]
                session_dict['bin_number'] = bin
                session_dict['trial_type'] = trial_type
                session_dict['reward_group'] = reward_group
                session_dict['session_id'] = session_table.session_id.values[0]
                all_session_data.append(session_dict)

        # Make axes and legend
        ax.set_xlabel('Time [s]', fontsize=15)
        ax.set_ylabel('Lick probability', fontsize=15)
        ax.set_ylim(0, 1.2)
        #max_x = int(1 / bin_size_sec)
        ax.set_xticks(ticks=[0,4,10],labels=[0,0.5,1.0])
        ax.set_yticks(ticks=[0,0.5,1.0], labels= [0,0.5,1.0])

        title = f"{session_table.mouse_id.values[0]}, {session_table.behavior.values[0]} " \
                        f"{session_table.day.values[0]}"
        ax.set_title(title)
        fig.tight_layout()
        plt.close()

        # Save figure
        if saving_path is None:
            continue
        else:
            save_formats = ['pdf', 'png', 'svg']
            figure_name = f"{session_table.mouse_id.values[0]}_{session_table.behavior.values[0]}_" \
                          f"{session_table.day.values[0]}_piezo_lick_proba"

            session_saving_path = os.path.join(saving_path,
                                               f'{session_table.behavior.values[0]}_{session_table.day.values[0]}')
            if not os.path.exists(session_saving_path):
                os.makedirs(session_saving_path)
            for save_format in save_formats:
                fig.savefig(os.path.join(f'{session_saving_path}', f'{figure_name}.{save_format}'),
                            format=f"{save_format}")

    all_session_data = pd.DataFrame(all_session_data)
    return all_session_data

def plot_multiple_mouse_piezo_lick_probability(all_session_data, color_palette, saving_path, whisker_only=False):
    """
    Plot time-binned piezo lick probability for all mice.
    :param all_session_data:
    :param color_palette:
    :param saving_path:
    :return:
    """

    # Plot only 1 second after reward window start time
    all_session_data = all_session_data[all_session_data.bin_number <= 10]

    # Whisker only: make one plot
    if whisker_only:

        # Select data
        all_session_data = all_session_data[(all_session_data.day == 0) & (all_session_data.trial_type == 'whisker_trial')]
        all_session_data = all_session_data.reset_index(drop=True)
        max_y_lim = 0.6

        # Make plot
        fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)
        remove_top_right_frame(ax)
        ax.tick_params(axis='both', which='major', labelsize=12)

        sns.lineplot(data=all_session_data,
                     x='bin_number',
                     y='lick_count_mean',
                     estimator='mean',
                     errorbar=('ci', 95),
                     n_boot=1000,
                     seed=42,
                     hue='reward_group',
                     hue_order=[1,0],
                     palette=['forestgreen', 'crimson'],
                     legend=False
                     )

        # Make axes and legend
        ax.set_xlabel('Time [s]', fontsize=15)
        ax.set_ylabel('Lick probability', fontsize=15)
        ax.set_ylim(0, max_y_lim)
        ax.set_xticks(ticks=[0, 5, 10], labels=[0, 0.5, 1.0])
        ax.set_yticks(ticks=[0, 0.25, 0.5], labels=[0, 0.25, 0.5])

        # Make title
        n_rewarded = all_session_data[all_session_data.reward_group == 1].mouse_id.nunique()
        n_unrewarded = all_session_data[all_session_data.reward_group == 0].mouse_id.nunique()
        title = r"$n_{R+}"+"={}$".format(n_rewarded) + ", " + r"$n_{R-}"+"={}$".format(n_unrewarded)
        ax.set_title(title, fontsize=12)
        fig.tight_layout()
        plt.show()

    else: # Make three plots
        # Select data
        all_session_data = all_session_data[(all_session_data.day == 0)]
        all_session_data = all_session_data.reset_index(drop=True)
        max_y_lim = 1.1

        # Make plot
        fig, axs = plt.subplots(1, 3, figsize=(9, 3), dpi=300, sharey=False)
        for ax in axs.flat:
            remove_top_right_frame(ax)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.set_xlabel('Time [s]', fontsize=15)
            ax.set_ylabel('Lick probability', fontsize=15)
            ax.set_ylim(0, max_y_lim)
            ax.set_xticks(ticks=[0, 5, 9], labels=[0, 0.5, 1.0])
            ax.set_yticks(ticks=[0, 0.5, 1.0], labels=[0, 0.5, 1.0])

        # Plot whisker trials
        whisker_data = all_session_data[all_session_data.trial_type == 'whisker_trial']
        sns.lineplot(data=whisker_data,
                     x='bin_number',
                     y='lick_count_mean',
                     estimator='mean',
                     errorbar=('ci', 95),
                     n_boot=1000,
                     seed=42,
                     hue='reward_group',
                     hue_order=[1,0],
                     palette=['forestgreen', 'crimson'],
                     legend=False,
                     ax=axs[0]
                     )

        # Plot auditory trials
        auditory_data = all_session_data[all_session_data.trial_type == 'auditory_trial']
        sns.lineplot(data=auditory_data,
                     x='bin_number',
                     y='lick_count_mean',
                     estimator='mean',
                     errorbar=('ci', 95),
                     n_boot=1000,
                     seed=42,
                     hue='reward_group',
                     hue_order=[1,0],
                     palette=['mediumblue', 'lightblue'],
                     legend=False,
                     ax=axs[1]
                     )

        # Plot no-stim trials
        no_stim_data = all_session_data[all_session_data.trial_type == 'no_stim_trial']
        sns.lineplot(data=no_stim_data,
                     x='bin_number',
                     y='lick_count_mean',
                     estimator='mean',
                     errorbar=('ci', 95),
                     n_boot=1000,
                     seed=42,
                     hue='reward_group',
                     hue_order=[1,0],
                     palette=['k', 'lightgrey'],
                     legend=False,
                     ax=axs[2]
                     )

        fig.tight_layout()
        plt.show()

    # Save figure
    if saving_path is None:
        return
    else:
        save_formats = ['pdf', 'png', 'svg']
        if whisker_only:
            figure_name = f"all_mice_piezo_lick_proba_whisker"
        else:
            figure_name = f"all_mice_piezo_lick_proba"
        for save_format in save_formats:
            fig.savefig(os.path.join(f'{saving_path}', f'{figure_name}.{save_format}'),
                        format=f"{save_format}")

    return


def categorical_context_boxplot(data, hue, palette, mouse_id, saving_path, figname):
    figsize = (18, 9)
    figure, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=figsize)

    sns.boxplot(data=data, x='day', y='hr_n', hue=hue, palette=palette['catch_palette'], ax=ax0)
    sns.boxplot(data=data, x='day', y='hr_a', hue=hue, palette=palette['aud_palette'], ax=ax1)
    sns.boxplot(data=data, x='day', y='hr_w', hue=hue, palette=palette['wh_palette'], ax=ax2)

    for ax in [ax0, ax1, ax2]:
        ax.set_ylim([-0.1, 1.05])
        ax.set_xlabel('Day')
        ax.set_ylabel('Lick probability')
        sns.despine()

        sns.move_legend(
            ax, "lower center",
            bbox_to_anchor=(0.5, 1), ncol=1, title=f"{mouse_id}", frameon=False,
        )
    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        figure.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                       format=f"{save_format}")
    plt.close()


def categorical_context_stripplot(data, hue, palette, mouse_id, saving_path, figname):
    figsize = (18, 9)
    figure, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=figsize)

    sns.stripplot(data=data, x='day', y='hr_n', hue=hue, palette=palette['catch_palette'], dodge=True,
                  jitter=0.2, ax=ax0)
    sns.stripplot(data=data, x='day', y='hr_a', hue=hue, palette=palette['aud_palette'], dodge=True,
                  jitter=0.2, ax=ax1)
    sns.stripplot(data=data, x='day', y='hr_w', hue=hue, palette=palette['wh_palette'], dodge=True,
                  jitter=0.2, ax=ax2)
    for ax in [ax0, ax1, ax2]:
        ax.set_ylim([-0.1, 1.05])
        ax.set_xlabel('Day')
        ax.set_ylabel('Lick probability')
        sns.despine()

        sns.move_legend(
            ax, "lower center",
            bbox_to_anchor=(0.5, 1), ncol=1, title=f"{mouse_id}", frameon=False,
        )
    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        figure.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                       format=f"{save_format}")
    plt.close()

def plot_single_mouse_reaction_time_across_days(combine_bhv_data, color_palette, saving_path):
    mice_list = np.unique(combine_bhv_data['mouse_id'].values[:])
    n_mice = len(mice_list)
    print(f" ")
    print(f"Plot reaction time across days for {n_mice} mice")
    for mouse_id in mice_list:
        print(f"Mouse : {mouse_id}")
        mouse_table = bhv_utils.get_single_mouse_table(combine_bhv_data, mouse=mouse_id)

        # Keep only Auditory and Whisker days
        mouse_table = mouse_table[mouse_table.behavior.isin(('auditory', 'whisker', 'context'))]

        # Select columns for plot
        cols = ['start_time', 'stop_time', 'lick_time', 'trial_type', 'lick_flag', 'early_lick', 'context',
                'day', 'response_window_start_time']

        # first df with only rewarded context as no trial stop ttl in non-rewarded context: compute reaction time
        df = mouse_table.loc[(mouse_table.early_lick == 0) & (mouse_table.lick_flag == 1) &
                             (mouse_table.context == 1), cols]
        df['computed_reaction_time'] = df['lick_time'] - df['response_window_start_time']
        df = df.replace({'context': {1: 'Rewarded', 0: 'Non-Rewarded'}})

        # second df with only rewarded context as no trial stop ttl in non-rewarded context: compute reaction time
        df_2 = mouse_table.loc[(mouse_table.early_lick == 0) & (mouse_table.lick_flag == 1), cols]
        df_2['computed_reaction_time'] = df_2['lick_time'] - df_2['response_window_start_time']
        df_2 = df_2.replace({'context': {1: 'Rewarded', 0: 'Non-Rewarded'}})

        trial_types = np.sort(list(np.unique(mouse_table.trial_type.values[:])))
        colors = [color_palette[0], color_palette[4], color_palette[2]]
        context_reward_palette = {
            'auditory_trial': {'Rewarded': 'mediumblue', 'Non-Rewarded': 'cornflowerblue'},
            'no_stim_trial': {'Rewarded': 'black', 'Non-Rewarded': 'darkgray'},
            'whisker_trial': {'Rewarded': 'green', 'Non-Rewarded': 'firebrick'},
        }

        # Do the plot with all trials
        figsize = (18, 18)
        figure, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=figsize)
        figname = f"{mouse_id}_reaction_time"

        for index, ax in enumerate([ax0, ax1, ax2]):

            sns.boxenplot(df.loc[df.trial_type == trial_types[index]], x='day', y='computed_reaction_time',
                          color=colors[index], ax=ax)

            ax.set_ylim([-0.1, 1.25])
            ax.set_xlabel('Day')
            ax.set_ylabel(f'{trial_types[index].capitalize()} \n Reaction time (s)')
            if index == 0:
                ax.set_title(f"{mouse_id}")
            sns.despine()

        save_formats = ['pdf', 'png', 'svg']
        for save_format in save_formats:
            figure.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                           format=f"{save_format}")
        plt.close()

        # Do the plot by context
        figsize = (18, 18)
        figure, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=figsize)
        figname = f"{mouse_id}_reaction_time_context"

        for index, ax in enumerate([ax0, ax1, ax2]):

            sns.boxenplot(df_2.loc[df_2.trial_type == trial_types[index]], x='day', y='computed_reaction_time',
                          hue='context', palette=context_reward_palette.get(trial_types[index]), ax=ax)

            ax.set_ylim([-0.1, 1.25])
            ax.set_xlabel('Day')
            ax.set_ylabel(f'{trial_types[index].capitalize()} \n Reaction time (s)')
            if index == 0:
                ax.set_title(f"{mouse_id}")
            sns.despine()

        save_formats = ['pdf', 'png', 'svg']
        for save_format in save_formats:
            figure.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                           format=f"{save_format}")
        plt.close()


def plot_single_mouse_psychometrics_across_days(combine_bhv_data, saving_path):
    mice_list = np.unique(combine_bhv_data['mouse_id'].values[:])
    n_mice = len(mice_list)

    for mouse_id in mice_list:
        print(f"Mouse : {mouse_id}")
        mouse_table = bhv_utils.get_single_mouse_table(combine_bhv_data, mouse=mouse_id)

        # Keep only whisker psychophysical days
        mouse_table = mouse_table[mouse_table.behavior.isin(['whisker_psy'])]

        # Remap individual whisker stim amplitude to 5 levels including no_stim_trials (level 0)
        for day_idx in mouse_table['day'].unique():
            print(mouse_id, 'day', day_idx)
            mouse_table_day = mouse_table.loc[mouse_table['day'] == day_idx]
            wh_stim_amp_mapper = {k: idx for idx, k in
                                  enumerate(np.unique(mouse_table_day['whisker_stim_amplitude'].values))}

            mouse_table.loc[mouse_table['day'] == day_idx, 'whisker_stim_levels'] = mouse_table.loc[
                mouse_table['day'] == day_idx, 'whisker_stim_amplitude'].map(wh_stim_amp_mapper)

        stim_amplitude_levels = np.unique(mouse_table['whisker_stim_levels'].values)
        print('Stim levels', stim_amplitude_levels)

        # Plot psychometric curve
        g = sns.FacetGrid(mouse_table, col='day', col_wrap=4, height=4, aspect=1, sharey=True, sharex=True)
        figname = f"{mouse_id}_psychometric_curve"
        g.map(sns.pointplot,
              'whisker_stim_levels',
              'lick_flag',
              order=sorted(stim_amplitude_levels),
              estimator='mean',
              errorbar=('ci', 95),
              n_boot=1000,
              seed=42,
              color='forestgreen'
              )

    g.set_axis_labels('Stimulus amplitude [mT]', 'P(lick)')
    g.set(xticks=range(5), xticklabels=[0, 10, 20, 25, 30])
    g.set(ylim=(0, 1.1))


    # Save figures
    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        g.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                  format=f"{save_format}",
                  bbox_inches='tight')
    plt.close()
    return


def plot_behavior(nwb_list, output_folder, plots, info_path):

    # Get combined behavior data
    #bhv_data = bhv_utils.build_standard_behavior_table(nwb_list)

    # Get combined behavior data with processed timestamps
    bhv_data = bhv_utils.build_standard_behavior_event_table(nwb_list)

    if info_path is not None:
        mouse_info_df = pd.read_excel(os.path.join(info_path, 'mouse_reference_weight.xlsx'))
        mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
        mouse_info_df['reward_group'] = mouse_info_df['reward_group'].map({'R+': 1,
                                                                           'R-': 0,
                                                                           'R+proba': 2})
        bhv_data = bhv_data.merge(mouse_info_df[['mouse_id', 'reward_group']], on='mouse_id', how='left')


    # Plot all single session figures
    colors = ['#225ea8', '#00FFFF', '#238443', '#d51a1c', '#cccccc']
    if 'single_session' in plots:
        print('Plotting single sessions')
        plot_single_session(combine_bhv_data=bhv_data, color_palette=colors, saving_path=output_folder)

    if 'across_days' in plots:
        plot_single_mouse_across_days(combine_bhv_data=bhv_data, color_palette=colors, saving_path=output_folder)

    if 'psycho' in plots:
        plot_single_mouse_psychometrics_across_days(combine_bhv_data=bhv_data, color_palette=colors,
                                                    saving_path=output_folder)
    if 'reaction_time' in plots:
        plot_single_mouse_reaction_time_across_days(combine_bhv_data=bhv_data, color_palette=colors,
                                                    saving_path=output_folder)
    if 'across_context_days' in plots:
        print('Plotting across_context_days')
        plot_single_mouse_across_context_days(combine_bhv_data=bhv_data, saving_path=output_folder)

    if 'context_switch' in plots:
        print('Plotting context_switch')
        get_single_session_time_to_switch(combine_bhv_data=bhv_data, do_single_session_plot=True)

    if 'multiple' in plots:
        plot_multiple_mice_training(bhv_data, saving_path=output_folder, reward_group_hue=False)

    if 'opto_grid' in plots:
        plot_single_mouse_opto_grid(bhv_data, saving_path=output_folder)

    if 'opto_grid_multiple' in plots:
        plot_multiple_mice_opto_grid(bhv_data, saving_path=output_folder)

    if 'history' in plots:
        plot_single_mouse_history(combine_bhv_data=bhv_data, saving_path=output_folder)

    if 'piezo_raster' in plots:
        plot_single_session_piezo_raster(combine_bhv_data=bhv_data, color_palette=colors, saving_path=output_folder, whisker_only=False)
        plot_single_session_piezo_raster(combine_bhv_data=bhv_data, color_palette=colors, saving_path=output_folder, whisker_only=True)

    if 'piezo_proba' in plots:
        plot_single_mouse_piezo_lick_probability(bhv_data, colors, saving_path=output_folder)

    return

def plot_group_behavior(nwb_list, plots, info_path):

    # Exclude mouse and format info
    if info_path is not None:
        mouse_info_df = pd.read_excel(os.path.join(info_path, 'mouse_reference_weight.xlsx'))
        mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
        mouse_list = mouse_info_df['mouse_id'].values[:]
        mouse_info_df['reward_group'] = mouse_info_df['reward_group'].map({'R+': 1,
                                                                           'R-': 0,
                                                                           'R+proba': 2})

    # Get combined behavior data
    nwb_list = [nwb_file for nwb_file in nwb_list if any(mouse in nwb_file for mouse in mouse_list)]
    bhv_data = bhv_utils.build_standard_behavior_event_table(nwb_list)
    bhv_data = bhv_data.merge(mouse_info_df[['mouse_id', 'reward_group']], on='mouse_id', how='left')

    print('Number of mice: ', bhv_data['mouse_id'].nunique())

    # Plot group figures
    colors = ['forestgreen', 'crimson', 'k', 'dimgrey', 'mediumblue', 'lightblue']

    #if 'single_session' in plots:
    #    plot_single_session(combine_bhv_data=bhv_data, color_palette=colors, saving_path=output_folder)
    #if 'across_days' in plots:
    #    plot_single_mouse_across_days(combine_bhv_data=bhv_data, color_palette=colors, saving_path=output_folder)
    # if 'training' in plots:
    #    plot_multiple_mice_training(bhv_data, colors, reward_group_hue=False)
    # if 'training_stats' in plots:
    #    plot_multiple_mouse_training_stats(bhv_data, colors, reward_group_hue=False)

    if 'history' in plots:
        plot_multiple_mice_history(combine_bhv_data=bhv_data)

    if 'piezo_proba' in plots:
        all_mice_data = []
        output_folder = r'M:\analysis\Axel_Bisi\results\piezo_lick_proba'

        for m_name in bhv_data['mouse_id'].unique():

            # Keep mice with continuous logging available
            if int(m_name[2:]) < 68:
                continue

            bhv_data_mouse = bhv_data[bhv_data['mouse_id']==m_name]
            mouse_data = plot_single_mouse_piezo_lick_probability(bhv_data_mouse, colors, saving_path=None)
            all_mice_data.append(mouse_data)

        all_mice_data = pd.concat(all_mice_data)
        plot_multiple_mouse_piezo_lick_probability(all_mice_data, colors, saving_path=output_folder, whisker_only=True)
        plot_multiple_mouse_piezo_lick_probability(all_mice_data, colors, saving_path=output_folder, whisker_only=False)

    return

def plot_multiple_mice_history(bhv_data):


    # Call single mouse function
    _, all_mouse_df = plot_single_mouse_history(bhv_data, saving_path=None)

    # Figure content
    trial_types_to_plot = ['whisker_trial', 'auditory_trial']
    var_as_col = 'history'

    save_formats = ['pdf', 'png', 'svg']
    saving_path = r'M:\analysis\Axel_Bisi\results\history'

    for group in all_mouse_df.reward_group.unique():
        figname = 'all_mice_trial_history_0-150'

        if group==0:
            figname = figname + '_nonrewarded'
            order = ['Miss', 'Hit']
            palette = [lighten_color('crimson', 0.8),
                       lighten_color('mediumblue', 0.8),
                       ]
        elif group==1:
            figname = figname + '_rewarded'
            order = ['Miss', 'Hit']
            palette = [lighten_color('forestgreen', 0.8),
                       lighten_color('mediumblue', 0.8),
                       ]
        elif group==2:
            figname = figname + '_rewarded_proba'
            order = ['Miss', 'Hit R+', 'Hit R-']
            palette = [lighten_color('forestgreen', 0.8),
                        lighten_color('mediumblue', 0.8),
                        ]

        # Subset of data
        all_mouse_df_sub = all_mouse_df[all_mouse_df.reward_group==group]

        # Plot: facet columns are different trial types
        if var_as_col == 'trial_type':
            figname = figname + '_col_trial_type'
            order = ['n-3', 'n-2', 'n-1', 'n+0']
            col_wrap = len(trial_types_to_plot)

            # Plot single mouse data points
            g = sns.catplot(
                data=all_mouse_df_sub[all_mouse_df_sub.outcome.isin(['Hit', 'Hit R+', 'Hit R-'])],
                x='history',
                y='value',
                kind='strip',
                estimator='mean',
                seed=42,
                orient='v',
                order=order,
                col='trial_type',
                col_wrap=col_wrap,
                col_order=trial_types_to_plot,
                height=4,
                aspect=0.6,
                sharey=True,
                sharex=True,
                hue='trial_type',
                hue_order=trial_types_to_plot,
                palette=palette,
                legend=False,
            )

            # Axes
            g.set_axis_labels('Trial', r'P(lick) | Hit at trial $n$')
            g.set(ylim=(-0.05, 1.1))

            # Save
            for save_format in save_formats:
                g.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                          format=f"{save_format}",
                          bbox_inches='tight'
                          )

            # Plot mean over mice
            figname = figname + '_mean'
            g = sns.catplot(
                data=all_mouse_df_sub[all_mouse_df_sub.outcome.isin(['Hit', 'Hit R+'])],
                x='history',
                y='value',
                kind='bar',
                seed=42,
                orient='v',
                order=order,
                col='trial_type',
                col_wrap=col_wrap,
                col_order=trial_types_to_plot,
                height=4,
                aspect=0.6,
                sharey=True,
                sharex=True,
                hue='trial_type',
                hue_order=trial_types_to_plot,
                palette=palette,
                legend=False,
            )

            # Axes
            g.set_axis_labels('Trial', r'P(lick) | Hit at trial $n$')
            g.set(ylim=(-0.05, 1.1))

            # Save
            for save_format in save_formats:
                g.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                          format=f"{save_format}",
                          bbox_inches='tight'
                          )


        elif var_as_col == 'history':
            figname = figname + '_col_history'
            col_order = ['n-1', 'n-2', 'n-3'] # specify order

            # Plot single mouse data points
            g = sns.catplot(
                data=all_mouse_df_sub,
                x='outcome',
                y='value',
                kind='strip',
                estimator='mean',
                seed=42,
                orient='v',
                order=order,
                col='history',
                col_order=col_order,
                col_wrap=3,
                height=4,
                aspect=0.6,
                sharey=True,
                sharex=True,
                hue='trial_type',
                hue_order=trial_types_to_plot,
                palette=palette,
                legend=False,
            )

            # Axes
            g.set_axis_labels('Outcome', r'P(lick) | Hit at trial $n$')
            g.set(ylim=(-0.05, 1.1))

            # Save
            for save_format in save_formats:
                g.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                          format=f"{save_format}",
                          bbox_inches='tight'
                          )

            # Plot mean over mice
            figname = figname + '_mean'

            g = sns.catplot(
                data=all_mouse_df_sub,
                x='outcome',
                y='value',
                kind='bar',
                seed=42,
                orient='v',
                order=order,
                col='history',
                col_wrap=3,
                col_order=col_order,
                height=4,
                aspect=0.6,
                sharey=True,
                sharex=True,
                hue='trial_type',
                hue_order=trial_types_to_plot,
                palette=palette,
                legend=False,
            )

            # Axes
            g.set_axis_labels('Outcome', r'P(lick) | Hit at trial $n$')
            g.set(ylim=(-0.05, 1.1))

            # Save
            for save_format in save_formats:
                g.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                          format=f"{save_format}",
                          bbox_inches='tight'
                          )

    return

def plot_multiple_mouse_training_stats(bhv_data, colors, reward_group_hue=False):
    mice_list = np.unique(bhv_data['mouse_id'].values[:])
    n_mice = len(mice_list)
    print(f" ")
    print(f"Plot training statistics for {n_mice} mice")

    # Keep only Auditory and Whisker days
    bhv_data = bhv_data[bhv_data.behavior.isin(('auditory', 'whisker'))]
    # Select columns for plot
    cols = ['mouse_id', 'reward_group', 'outcome_a', 'outcome_w', 'outcome_n', 'day']
    df = bhv_data.loc[bhv_data.early_lick == 0, cols]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
    remove_top_right_frame(ax)
    sns.set_style('whitegrid', {'grid.color': '.6', 'grid.linestyle': ':'})

    # Save
    save_formats = ['pdf', 'png', 'svg']
    saving_path = r'M:\analysis\Axel_Bisi\results\training'
    for save_format in save_formats:
        if reward_group_hue:
            filename = r'training_performance_reward_group.{}'.format(save_format)
        else:
            filename = r'training_performance.{}'.format(save_format)
        fig.savefig(os.path.join(f'{saving_path}', filename),
                    format=f"{save_format}",
                    bbox_inches='tight')

    plt.close()
    return fig

def plot_single_mouse_history(combine_bhv_data, saving_path=None):


    # Keep only first trials
    N = range(0,151)

    # Init. container for all mouse figure
    all_mouse_df = []

    reward_groups_to_plot = [0,1,2] #R-, R+, R+proba
    trial_types_to_plot = ['whisker_trial', 'auditory_trial']
    max_ref_trial = 3
    #comparison_trials = range(-max_ref_trial, max_ref_trial+1)
    comparison_trials = range(-max_ref_trial, 1)
    comparison_trial_txt = ['n{}'.format(n) if n<0 else 'n+{}'.format(n) for n in comparison_trials]

    var_as_col = 'history'

    # For each mouse_id, find frequency of pairs of outcomes
    mice_list = np.unique(combine_bhv_data['mouse_id'].values[:])
    n_mice = len(mice_list)
    print(f" ")
    print(f"Plot trial history for {n_mice} mice")

    for mouse_id in mice_list:

        # Copy needed here for execution
        mouse_table = combine_bhv_data.copy()
        # Data subset for mouse_id
        mouse_table = bhv_utils.get_single_mouse_table(mouse_table, mouse_id)
        # Keep only whisker day 0
        if mouse_id in ['AB050', 'AB051', 'AB052','AB053']:
            mouse_table = mouse_table[mouse_table.day==1]
        else:
            mouse_table = mouse_table[mouse_table.day==0]
        mouse_table = mouse_table.reset_index(drop=True)
        mouse_table = mouse_table[mouse_table.index.isin(N)]
        mouse_table = mouse_table[mouse_table.trial_type.isin(trial_types_to_plot)]

        # Check reward group
        reward_group = mouse_table.reward_group.unique()[0]
        if reward_group not in reward_groups_to_plot:
            continue
        elif reward_group == 0:
            reward_group_hue = 'R-'
            w_color = lighten_color('crimson', 0.8)
        elif reward_group == 1:
            reward_group_hue = 'R+'
            w_color = lighten_color('forestgreen', 0.8)
        elif reward_group == 2:
            reward_group_hue = 'R+proba'
            w_color = lighten_color('forestgreen', 0.5)


        # Init. dataframe to store pairs
        pairs_df = pd.DataFrame(columns=['trial_type',
                                         'outcome',
                                         'history',
                                         'trial_index'])
        # Iterate over trial types
        for trial_type in trial_types_to_plot:
            # Get lick trials for trial type
            bhv_data_type = mouse_table[mouse_table.trial_type == trial_type]
            bhv_data_type.reset_index(inplace=True)
            lick_trials = bhv_data_type[bhv_data_type.lick_flag == 1]

            # Iterate over range of trial histories
            for ref_trial, txt_label in zip(comparison_trials, comparison_trial_txt):

                # Ignore trials with which no comparisons are possible
                if ref_trial <= 0:
                    lick_trials_ids = lick_trials.index[abs(ref_trial):]
                else:
                    lick_trials_ids = lick_trials.index[:-ref_trial]

                # Iterate over lick trials
                for lick_idx in lick_trials_ids:

                    ref_idx = lick_idx + ref_trial

                    if reward_group_hue in ['R+', 'R-']:
                        outcome = bhv_data_type.iloc[ref_idx, :]['perf']
                        trial_perf_map = {0: 'Miss', 2: 'Hit', 1: 'Miss', 3: 'Hit'}

                    elif reward_group_hue == 'R+proba':
                        outcome = bhv_data_type.iloc[ref_idx, :]['perf']
                        reward_available = bhv_data_type.iloc[ref_idx, :]['reward_available']

                        # Note: this assumes only whisker has reward proba
                        if outcome == 2 and reward_available == 1:
                            outcome = outcome
                        elif outcome == 2 and reward_available == 0:
                            outcome = 20

                        trial_perf_map = {0: 'Miss', 2: 'Hit R+', 1: 'Miss', 3: 'Hit R+',
                                          20: 'Hit R-'}

                    entry_dict = {'trial_type': trial_type,
                                  'outcome': outcome,
                                  'history': txt_label,
                                  'trial_index': lick_idx}
                    pairs_df = pd.concat([pairs_df, pd.DataFrame(entry_dict, index=[lick_idx])],
                                         ignore_index=False)

        # Format columns for plotting
        pairs_df['outcome'] = pairs_df['outcome'].map(trial_perf_map)


        # Calculate lick rate for each trial type, per history values
        if reward_group_hue in ['R+', 'R-']:
            # Map to licks
            lick_map = {'Miss': 0, 'Hit': 1}
            pairs_df['lick_flag'] = pairs_df['outcome'].map(lick_map)

            # Get rate per conditions -> this could be improved, see for R+proba
            hit_groupby_df = pairs_df.groupby(['trial_type', 'history'])['lick_flag'].mean().unstack()
            hit_groupby_df['outcome'] = 'Hit'
            hit_groupby_df.reset_index(inplace=True)
            miss_groupby_df = 1 - pairs_df.groupby(['trial_type', 'history'])['lick_flag'].mean().unstack()
            miss_groupby_df['outcome'] = 'Miss'
            miss_groupby_df.reset_index(inplace=True)
            history_rate_df = pd.concat([hit_groupby_df, miss_groupby_df], axis=0)
            history_rate_df = history_rate_df.melt(id_vars=['trial_type', 'outcome'],
                                                   var_name='history')
            x_order = ['Miss', 'Hit']
        elif reward_group_hue == 'R+proba':

            history_rate_df = pairs_df.groupby(['trial_type', 'history', 'outcome',]).size().reset_index(name='lick_count') # get pair counts per group and outcome
            sum_df = history_rate_df.groupby(['trial_type', 'history'])['lick_count'].transform('sum') # get sum per group
            history_rate_df['value'] = history_rate_df['lick_count'].div(sum_df) # get proportions per group
            x_order = ['Miss', 'Hit R-', 'Hit R+']

        # Append to all mouse df
        history_rate_df['mouse_id'] = mouse_id
        history_rate_df['reward_group'] = mouse_table['reward_group'].values[0]
        all_mouse_df.append(history_rate_df)

        if saving_path:
            # Plot based on trial type at single mouse_id level
            figname = f"{mouse_id}_trial_history_0-150trials"

            if var_as_col=='trial_type':
                figname = f"{mouse_id}_trial_history_col_trial_type"
                col_wrap = len(trial_types_to_plot)

                g = sns.catplot(
                    data=history_rate_df[history_rate_df.outcome=='Hit'],
                    x='history',
                    y='value',
                    kind='bar',
                    orient='v',
                    order=comparison_trial_txt,
                    col='trial_type',
                    col_wrap=col_wrap,
                    col_order = trial_types_to_plot,
                    height=4,
                    aspect=0.6,
                    sharey=True,
                    sharex=True,
                    hue='trial_type',
                    hue_order=trial_types_to_plot,
                    palette=[w_color,
                             lighten_color('mediumblue', 0.8),
                    #         # lighten_color('k', 0.8),
                             ],
                    legend=False,
                    dodge=False,
                )

                # Axes
                g.set_axis_labels('Trial', r'P(lick) | Hit at trial $n$')
                g.set(ylim=(-0.05, 1.1))

            # Plot type: grid where subplots show different histories
            elif var_as_col=='history':

                g = sns.catplot(
                    data=history_rate_df[history_rate_df.history!='n+0'],
                    x='outcome',
                    y='value',
                    kind='bar',
                    orient='v',
                    order=x_order,
                    col='history',
                    col_wrap=3,
                    height=4,
                    aspect=0.6,
                    sharey=True,
                    sharex=True,
                    hue='trial_type',
                    hue_order=trial_types_to_plot,
                    palette=[lighten_color('forestgreen', 0.8),
                             lighten_color('mediumblue', 0.8),
                            # lighten_color('k', 0.8),
                             ],
                    legend=False,
                )

                # Axes
                g.set_axis_labels('Outcome', r'P(lick) | Hit at trial $n$')
                g.set(ylim=(-0.05, 1.1))

            plt.show()

            save_formats = ['pdf', 'png', 'svg']
            session_saving_path = os.path.join(saving_path,
                                               f'{mouse_table.behavior.values[0]}_{mouse_table.day.values[0]}')
            if not os.path.exists(session_saving_path):
                os.makedirs(session_saving_path)
            for save_format in save_formats:
                g.savefig(os.path.join(f'{session_saving_path}', f'{mouse_id}_{figname}.{save_format}'),
                               format=f"{save_format}",
                               bbox_inches='tight'
                               )

            plt.close()

        else:
            g=None
            pass

    return g, pd.concat(all_mouse_df)

def plot_single_session(combine_bhv_data, color_palette, saving_path):
    sessions_list = np.unique(combine_bhv_data['session_id'].values[:])
    n_sessions = len(sessions_list)
    print(f"N sessions : {n_sessions}")
    for session_id in sessions_list:
        session_table, switches, block_size = bhv_utils.get_standard_single_session_table(combine_bhv_data, session=session_id)
        if session_table['behavior'].values[0] == 'free_licking':
            print(f"No plot for {session_table['behavior'].values[0]} sessions")
            continue

        # Set plot parameters.
        raster_marker = 2
        marker_width = 2
        figsize = (12, 4)
        figure, ax = plt.subplots(1, 1, figsize=figsize)

        d = session_table.loc[session_table.early_lick == 0][int(block_size / 2)::block_size]
        marker = itertools.cycle(['o', 's'])
        markers = [next(marker) for i in d["opto_stim"].unique()]

        # Fix for NWB opto stim is 0 and not NaN todo: remove this condition with new NWBs as opto_stim is 0 not NaN
        if pd.isnull(d.opto_stim).all():
            d['opto_stim'].values[:] = 0
        # Remove legend if not necessary
        if (d['opto_stim'] == 0).all():
            plot_legend = False
        else:
            plot_legend = 'brief'

        # Plot the lines :
        sns.lineplot(data=d, x='trial', y='hr_n', hue="opto_stim", style="opto_stim", palette=['k', 'k'], ax=ax,
                     markers=markers, legend=plot_legend)

        if 'hr_w' in list(d.columns) and (not np.isnan(d.hr_w.values[:]).all()):
            sns.lineplot(data=d, x='trial', y='hr_w', hue="opto_stim", style="opto_stim",
                         palette=[color_palette[2], color_palette[2]],
                         ax=ax, markers=markers, legend=plot_legend)
        if 'hr_a' in list(d.columns) and (not np.isnan(d.hr_a.values[:]).all()):
            sns.lineplot(data=d, x='trial', y='hr_a', hue="opto_stim", style="opto_stim",
                         palette=[color_palette[0], color_palette[0]],
                         ax=ax, markers=markers, legend=plot_legend)

        if session_table['behavior'].values[0] in ['context', 'whisker_context']:
            rewarded_bloc_bool = list(d.context.values[:])
            bloc_limites = np.arange(start=0, stop=len(session_table.index), step=block_size)
            bloc_area_color = ['green' if i == 1 else 'firebrick' for i in rewarded_bloc_bool]
            if bloc_limites[-1] < len(session_table.index):
                bloc_area = [(bloc_limites[i], bloc_limites[i + 1]) for i in range(len(bloc_limites) - 1)]
                bloc_area.append((bloc_limites[-1], len(session_table.index)))
                if len(bloc_area) > len(bloc_area_color):
                    bloc_area = bloc_area[0: len(bloc_area_color)]
                for index, coords in enumerate(bloc_area):
                    color = bloc_area_color[index]
                    ax.axvspan(coords[0], coords[1], alpha=0.25, facecolor=color, zorder=1)

        # Plot the trials :
        ax.scatter(x=session_table.loc[session_table.lick_flag == 0]['trial'],
                   y=session_table.loc[session_table.lick_flag == 0]['outcome_n'] - 0.1,
                   color=color_palette[4], marker=raster_marker, linewidths=marker_width)
        ax.scatter(x=session_table.loc[session_table.lick_flag == 1]['trial'],
                   y=session_table.loc[session_table.lick_flag == 1]['outcome_n'] - 1.1,
                   color='k', marker=raster_marker, linewidths=marker_width)

        ax.scatter(x=session_table.loc[session_table.lick_flag == 0]['trial'],
                   y=session_table.loc[session_table.lick_flag == 0]['outcome_a'] - 0.15,
                   color=color_palette[1], marker=raster_marker, linewidths=marker_width)
        ax.scatter(x=session_table.loc[session_table.lick_flag == 1]['trial'],
                   y=session_table.loc[session_table.lick_flag == 1]['outcome_a'] - 1.15,
                   color=color_palette[0], marker=raster_marker, linewidths=marker_width)

        if 'hr_w' in list(d.columns) and (not np.isnan(d.hr_w.values[:]).all()):
            ax.scatter(x=session_table.loc[session_table.lick_flag == 0]['trial'],
                       y=session_table.loc[session_table.lick_flag == 0]['outcome_w'] - 0.2,
                       color=color_palette[3], marker=raster_marker, linewidths=marker_width)
            ax.scatter(x=session_table.loc[session_table.lick_flag == 1]['trial'],
                       y=session_table.loc[session_table.lick_flag == 1]['outcome_w'] - 1.2,
                       color=color_palette[2], marker=raster_marker, linewidths=marker_width)

        ax.set_ylim([-0.2, 1.05])
        ax.set_xlabel('Trial number')
        ax.set_ylabel('Lick probability')
        figure_title = f"{session_table.mouse_id.values[0]}, {session_table.behavior.values[0]} " \
                       f"{session_table.day.values[0]}"
        ax.set_title(figure_title)
        sns.despine()

        save_formats = ['pdf', 'png', 'svg']
        figure_name = f"{session_table.mouse_id.values[0]}_{session_table.behavior.values[0]}_" \
                      f"{session_table.day.values[0]}"
        session_saving_path = os.path.join(saving_path,
                                           f'{session_table.behavior.values[0]}_{session_table.day.values[0]}')
        if not os.path.exists(session_saving_path):
            os.makedirs(session_saving_path)
        for save_format in save_formats:
            figure.savefig(os.path.join(f'{session_saving_path}', f'{figure_name}.{save_format}'),
                           format=f"{save_format}")

        plt.close()


def categorical_context_lineplot(data, hue, palette, mouse_id, saving_path, figname):
    figsize = (6, 9)
    figure, ax0 = plt.subplots(1, 1, figsize=figsize)

    sns.pointplot(data.loc[data['opto_stim'] == 0], x='day', y='hr_n', hue=hue, palette=palette['catch_palette'], ax=ax0, markers='o')
    sns.pointplot(data.loc[data['opto_stim'] == 0], x='day', y='hr_a', hue=hue, palette=palette['aud_palette'], ax=ax0, markers='o')
    sns.pointplot(data.loc[data['opto_stim'] == 0], x='day', y='hr_w', hue=hue, palette=palette['wh_palette'], ax=ax0, markers='o')

    ax0.set_ylim([-0.1, 1.05])
    ax0.set_xlabel('Day')
    ax0.set_ylabel('Lick probability')
    sns.despine()

    sns.move_legend(
        ax0, "lower center",
        bbox_to_anchor=(0.5, 1), ncol=3, title=f"{mouse_id}", frameon=False,
    )
    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        figure.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                       format=f"{save_format}")
    plt.close()

def categorical_context_pointplot(data, hue, palette, mouse_id, saving_path, figname):
    figsize = (18, 9)
    figure, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=figsize)

    sns.pointplot(data=data.loc[data['opto_stim'] == 0], x='day', y='hr_n', hue=hue, palette=palette['catch_palette'], dodge=True,
                  estimator='mean', errorbar=('ci', 95), n_boot=1000, ax=ax0)
    sns.pointplot(data=data.loc[data['opto_stim'] == 0], x='day', y='hr_a', hue=hue, palette=palette['aud_palette'], dodge=True,
                  estimator='mean', errorbar=('ci', 95), n_boot=1000, ax=ax1)
    sns.pointplot(data=data.loc[data['opto_stim'] == 0], x='day', y='hr_w', hue=hue, palette=palette['wh_palette'], dodge=True,
                  estimator='mean', errorbar=('ci', 95), n_boot=1000, ax=ax2)

    for ax in [ax0, ax1, ax2]:
        ax.set_ylim([-0.1, 1.05])
        ax.set_xlabel('Day')
        ax.set_ylabel('Lick probability')
        sns.despine()

        sns.move_legend(
            ax, "lower center",
            bbox_to_anchor=(0.5, 1), ncol=1, title=f"{mouse_id}", frameon=False,
        )
    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        figure.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                       format=f"{save_format}")
    plt.close()


def categorical_context_opto(data, hue, palette, mouse_id, saving_path, figname):

    data = data.groupby('day').filter(lambda x: x['opto_stim'].sum()>0)

    figsize = (18, 9)
    figure, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=figsize)

    sns.pointplot(data=data.loc[data['opto_stim'] == 0], x='day', y='hr_n', hue=hue, palette=palette['catch_palette'], dodge=True,
                  estimator='mean', errorbar=('ci', 95), n_boot=1000, ax=ax0)
    sns.pointplot(data=data.loc[data['opto_stim'] == 0], x='day', y='hr_a', hue=hue, palette=palette['aud_palette'], dodge=True,
                  estimator='mean', errorbar=('ci', 95), n_boot=1000, ax=ax1)
    sns.pointplot(data=data.loc[data['opto_stim'] == 0], x='day', y='hr_w', hue=hue, palette=palette['wh_palette'], dodge=True,
                  estimator='mean', errorbar=('ci', 95), n_boot=1000, ax=ax2)

    sns.pointplot(data=data.loc[data['opto_stim'] == 1], x='day', y='hr_n', hue=hue, palette=palette['catch_palette'], dodge=True,
                  estimator='mean', errorbar=('ci', 95), n_boot=1000, ax=ax0, markers='*', linestyles='dashed')
    sns.pointplot(data=data.loc[data['opto_stim'] == 1], x='day', y='hr_a', hue=hue, palette=palette['aud_palette'], dodge=True,
                  estimator='mean', errorbar=('ci', 95), n_boot=1000, ax=ax1, markers='*', linestyles='dashed')
    sns.pointplot(data=data.loc[data['opto_stim'] == 1], x='day', y='hr_w', hue=hue, palette=palette['wh_palette'], dodge=True,
                  estimator='mean', errorbar=('ci', 95), n_boot=1000, ax=ax2, markers='*', linestyles='dashed')

    for ax in [ax0, ax1, ax2]:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(title=f'{mouse_id}', handles=handles, labels=['Non Rewarded', 'Rewarded', 'Non Rewarded opto', 'Rewarded opto'])
        ax.set_ylim([-0.1, 1.05])
        ax.set_xlabel('Day')
        ax.set_ylabel('Lick probability')
        sns.despine()

        sns.move_legend(
            ax, "lower center",
            bbox_to_anchor=(0.5, 1), ncol=1, title=f"{mouse_id}", frameon=False,
        )
    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        figure.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                       format=f"{save_format}")
    plt.close()


def categorical_context_opto_avg(data, hue, palette, mouse_id, saving_path, figname):

    data = data.groupby('day').filter(lambda x: x['opto_stim'].sum()>0)

    figsize = (18, 9)
    figure, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=figsize)

    sns.pointplot(data=data, x='context_rwd_str', y='hr_n', hue=hue,
                  palette=palette['catch_palette'], dodge=True,
                  estimator='mean', errorbar=('ci', 95), n_boot=1000, ax=ax0)
    sns.pointplot(data=data, x='context_rwd_str', y='hr_a', hue=hue, palette=palette['aud_palette'],
                  dodge=True,
                  estimator='mean', errorbar=('ci', 95), n_boot=1000, ax=ax1)
    sns.pointplot(data=data, x='context_rwd_str', y='hr_w', hue=hue, palette=palette['wh_palette'],
                  dodge=True,
                  estimator='mean', errorbar=('ci', 95), n_boot=1000, ax=ax2)

    for ax in [ax0, ax1, ax2]:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(title=f'{mouse_id}', handles=handles,
                  labels=['No opto', 'Opto'])
        ax.set_ylim([-0.1, 1.05])
        ax.set_xlabel('Day')
        ax.set_ylabel('Lick probability')
        sns.despine()

        sns.move_legend(
            ax, "lower center",
            bbox_to_anchor=(0.5, 1), ncol=1, title=f"{mouse_id}", frameon=False,
        )
    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        figure.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                       format=f"{save_format}")
    plt.close()


def plot_single_mouse_across_context_days(combine_bhv_data, saving_path):
    mice_list = np.unique(combine_bhv_data['mouse_id'].values[:])
    n_mice = len(mice_list)
    print(f" ")
    print(f"Plot average across context days for {n_mice} mice")
    for mouse_id in mice_list:
        print(f"Mouse : {mouse_id}")
        mouse_table = bhv_utils.get_single_mouse_table(combine_bhv_data, mouse=mouse_id)

        # Keep only Context days
        mouse_table = mouse_table[mouse_table.behavior.isin(['context', 'whisker_context'])]
        mouse_table = mouse_table.reset_index(drop=True)
        if mouse_table.empty:
            print(f"No context day: return")
            return

        # Add column with string for rewarded and non-rewarded context
        mouse_table['context_rwd_str'] = mouse_table['context']
        mouse_table = mouse_table.replace({'context_rwd_str': {1: 'Rewarded', 0: 'Non-Rewarded'}})

        # Select columns for the first plot
        cols = ['outcome_a', 'outcome_w', 'outcome_n', 'day', 'context', 'context_background', 'context_rwd_str', 'opto_stim']
        df = mouse_table.loc[mouse_table.early_lick == 0, cols]

        # Compute hit rates. Use transform to propagate hit rate to all entries.
        df['hr_w'] = df.groupby(['day', 'context', 'context_rwd_str', 'opto_stim'], as_index=False)['outcome_w'] \
            .transform(np.nanmean)
        df['hr_a'] = df.groupby(['day', 'context', 'context_rwd_str', 'opto_stim'], as_index=False)['outcome_a'] \
            .transform(np.nanmean)
        df['hr_n'] = df.groupby(['day', 'context', 'context_rwd_str', 'opto_stim'], as_index=False)['outcome_n'] \
            .transform(np.nanmean)

        # Average by day and context blocks for this mouse
        df_by_day = df.groupby(['day', 'context', 'context_rwd_str', 'context_background', 'opto_stim'], as_index=False).agg(np.nanmean)

        # Look at the mean difference in Lick probability between rewarded and non-rewarded context
        df_by_day_diff = df_by_day.sort_values(by=['day', 'context_rwd_str'], ascending=True)
        df_by_day_diff['hr_w_diff'] = df_by_day_diff.loc[df_by_day_diff['opto_stim'] == 0].groupby('day')['hr_w'].diff()
        df_by_day_diff['hr_a_diff'] = df_by_day_diff.loc[df_by_day_diff['opto_stim'] == 0].groupby('day')['hr_a'].diff()
        df_by_day_diff['hr_n_diff'] = df_by_day_diff.loc[df_by_day_diff['opto_stim'] == 0].groupby('day')['hr_n'].diff()

        df_by_day_diff['hr_w_diff_opto'] = df_by_day_diff.groupby(['day', 'context'])['hr_w'].diff()
        df_by_day_diff['hr_a_diff_opto'] = df_by_day_diff.groupby(['day', 'context'])['hr_a'].diff()
        df_by_day_diff['hr_n_diff_opto'] = df_by_day_diff.groupby(['day', 'context'])['hr_n'].diff()

        # Plot the diff
        figsize = (6, 9)
        figure, ax0 = plt.subplots(1, 1, figsize=figsize)

        sns.pointplot(df_by_day_diff.dropna(subset=['hr_w_diff']), x='day', y='hr_n_diff', color='black', ax=ax0, markers='o')
        sns.pointplot(df_by_day_diff.dropna(subset=['hr_w_diff']), x='day', y='hr_a_diff', color='mediumblue', ax=ax0, markers='o')
        sns.pointplot(df_by_day_diff.dropna(subset=['hr_w_diff']), x='day', y='hr_w_diff', color='green', ax=ax0, markers='o')

        ax0.set_ylim([-0.2, 1.05])

        ax0.set_xlabel('Day')
        ax0.set_ylabel("\u0394 Lick probability")
        ax0.set_title(f"{mouse_id}")
        sns.despine()

        save_formats = ['pdf', 'png', 'svg']
        figname = f"{mouse_id}_context_diff"
        for save_format in save_formats:
            figure.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                           format=f"{save_format}")
        plt.close()

        # Get each session to look at each individual
        mouse_session_list = list(np.unique(mouse_table['session_id'].values[:]))
        by_block_data = []
        for mouse_session in mouse_session_list:
            session_table, switches, block_size = bhv_utils.get_standard_single_session_table(mouse_table, session=mouse_session,
                                                                                     verbose=False)
            session_table = session_table.loc[session_table.early_lick == 0][int(block_size / 2)::block_size]
            by_block_data.append(session_table)
        by_block_data = pd.concat(by_block_data, ignore_index=True)
        by_block_data['context_rwd_str'] = by_block_data['context']
        by_block_data = by_block_data.replace({'context_rwd_str': {1: 'Rewarded', 0: 'Non-Rewarded'}})

        # Define the two colort palette
        context_name_palette = {
            'catch_palette': {'brown': 'black', 'pink': 'darkgray'},
            'wh_palette': {'brown': 'green', 'pink': 'limegreen'},
            'aud_palette': {'brown': 'mediumblue', 'pink': 'cornflowerblue'}
        }
        context_reward_palette = {
            'catch_palette': {'Rewarded': 'black', 'Non-Rewarded': 'darkgray'},
            'wh_palette': {'Rewarded': 'green', 'Non-Rewarded': 'firebrick'},
            'aud_palette': {'Rewarded': 'mediumblue', 'Non-Rewarded': 'cornflowerblue'}
        }

        # Do the plots  : one point per day per context
        # Keep the context by background
        # categorical_context_lineplot(data=df_by_day, hue='context_background', palette=context_name_palette,
        #                              mouse_id=mouse_id, saving_path=saving_path, figname=f"{mouse_id}_context_name")

        # Keep the context by rewarded
        categorical_context_lineplot(data=df_by_day, hue='context_rwd_str', palette=context_reward_palette,
                                     mouse_id=mouse_id, saving_path=saving_path, figname=f"{mouse_id}_context_reward")

        # Do the plots : with context block distribution for each day
        # # Boxplots
        # categorical_context_boxplot(data=by_block_data, hue='context_background', palette=context_name_palette,
        #                             mouse_id=mouse_id, saving_path=saving_path,
        #                             figname=f"{mouse_id}_box_context_name_bloc")
        #
        # categorical_context_boxplot(data=by_block_data, hue='context_rwd_str', palette=context_reward_palette,
        #                             mouse_id=mouse_id, saving_path=saving_path,
        #                             figname=f"{mouse_id}_box_context_reward")
        #
        # # Stripplots
        # categorical_context_stripplot(data=by_block_data, hue='context_background', palette=context_name_palette,
        #                               mouse_id=mouse_id, saving_path=saving_path,
        #                               figname=f"{mouse_id}_strip_context_name_bloc")
        #
        # categorical_context_stripplot(data=by_block_data, hue='context_rwd_str', palette=context_reward_palette,
        #                               mouse_id=mouse_id, saving_path=saving_path,
        #                               figname=f"{mouse_id}_strip_context_reward")

        # Point-plots
        # categorical_context_pointplot(data=by_block_data, hue='context_background', palette=context_name_palette,
        #                               mouse_id=mouse_id, saving_path=saving_path,
        #                               figname=f"{mouse_id}_point_context_name_bloc")

        categorical_context_pointplot(data=by_block_data, hue='context_rwd_str', palette=context_reward_palette,
                                      mouse_id=mouse_id, saving_path=saving_path,
                                      figname=f"{mouse_id}_point_context_reward")

        if not df_by_day_diff['hr_w_diff_opto'].dropna().empty:
            categorical_context_opto(data=by_block_data, hue='context_rwd_str', palette=context_reward_palette,
                                      mouse_id=mouse_id, saving_path=saving_path,
                                      figname=f"{mouse_id}_point_context_opto")
            opto_diff_by_day(data=df_by_day_diff, mouse_id=mouse_id, palette=context_reward_palette, saving_path=saving_path,
                             figname=f"{mouse_id}_context_opto_diff")

            context_opto_palette = {
                'catch_palette': {0: 'black', 1: 'darkgray'},
                'wh_palette': {0: 'green', 1: 'firebrick'},
                'aud_palette': {0: 'mediumblue', 1: 'cornflowerblue'}
            }
            categorical_context_opto_avg(data=by_block_data, hue='opto_stim', palette=context_opto_palette,
                                      mouse_id=mouse_id, saving_path=saving_path,
                                      figname=f"{mouse_id}_point_context_opto_avg")
            opto_diff_avg(data=df_by_day_diff, mouse_id=mouse_id, palette=context_reward_palette, saving_path=saving_path,
                             figname=f"{mouse_id}_context_opto_diff_avg")

def opto_diff_by_day(data, mouse_id, palette, saving_path, figname):
    figsize = (6, 9)
    figure, ax0 = plt.subplots(1, 1, figsize=figsize)

    sns.pointplot(data, x='day', y='hr_n_diff_opto', hue='context_rwd_str', palette=palette['catch_palette'], ax=ax0, markers='o')
    sns.pointplot(data, x='day', y='hr_a_diff_opto', hue='context_rwd_str', palette=palette['aud_palette'], ax=ax0, markers='o')
    sns.pointplot(data, x='day', y='hr_w_diff_opto', hue='context_rwd_str', palette=palette['wh_palette'], ax=ax0, markers='o')
    ax0.set_ylim([-1.05, 1.05])
    # else:

    ax0.set_xlabel('Day')
    ax0.set_ylabel("\u0394 Lick probability")
    ax0.set_title(f"{mouse_id} opto")
    sns.despine()

    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        figure.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                       format=f"{save_format}")
    plt.close()


def opto_diff_avg(data, mouse_id, palette, saving_path, figname):
    figsize = (6, 9)
    figure, ax0 = plt.subplots(1, 1, figsize=figsize)

    sns.pointplot(data, x='context_rwd_str', y='hr_n_diff_opto', palette=palette['catch_palette'], ax=ax0, markers='o')
    sns.pointplot(data, x='context_rwd_str', y='hr_a_diff_opto', palette=palette['aud_palette'], ax=ax0, markers='o')
    sns.pointplot(data, x='context_rwd_str', y='hr_w_diff_opto', palette=palette['wh_palette'], ax=ax0, markers='o')
    ax0.set_ylim([-0.55, 0.55])
    # else:

    ax0.set_xlabel('Day')
    ax0.set_ylabel("\u0394 Lick probability")
    ax0.set_title(f"{mouse_id} opto")
    sns.despine()

    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        figure.savefig(os.path.join(f'{saving_path}', f'{figname}.{save_format}'),
                       format=f"{save_format}")
    plt.close()


def get_single_session_time_to_switch(combine_bhv_data, do_single_session_plot=False):
    sessions_list = np.unique(combine_bhv_data['session_id'].values[:])
    n_sessions = len(sessions_list)
    print(f"N sessions : {n_sessions}")
    to_rewarded_transitions_prob = dict()
    to_non_rewarded_transitions_prob = dict()
    for session_id in sessions_list:
        session_table, switches, block_size = bhv_utils.get_standard_single_session_table(combine_bhv_data, session=session_id)

        # Keep only the session with context
        if session_table['behavior'].values[0] not in ["context", 'whisker_context']:
            continue

        # Keep only the whisker trials
        whisker_session_table = session_table.loc[session_table.trial_type == 'whisker_trial']
        whisker_session_table = whisker_session_table.reset_index(drop=True)

        # extract licks array
        licks = whisker_session_table.outcome_w.values[:]

        # Extract transitions rwd to non rwd and opposite
        rewarded_transitions = np.where(np.diff(whisker_session_table.context.values[:]) == 1)[0]
        non_rewarded_transitions = np.where(np.diff(whisker_session_table.context.values[:]) == -1)[0]

        # Build rewarded transitions matrix from trial -3 to trial +3
        wh_switches = np.where(np.diff(whisker_session_table.context.values[:]))[0]
        n_trials_around = min(np.diff(wh_switches))
        trials_above = n_trials_around + 1
        trials_below = n_trials_around - 1
        rewarded_transitions_mat = np.zeros((len(rewarded_transitions), 2 * n_trials_around))
        for index, transition in enumerate(list(rewarded_transitions)):
            if transition + trials_above > len(licks):
                rewarded_transitions_mat = rewarded_transitions_mat[0: len(rewarded_transitions) - 1, :]
                continue
            else:
                rewarded_transitions_mat[index, :] = licks[np.arange(transition - trials_below, transition + trials_above)]
        rewarded_transition_prob = np.mean(rewarded_transitions_mat, axis=0)
        to_rewarded_transitions_prob[session_id] = rewarded_transition_prob

        # Build non_rewarded transitions matrix from trial -3 to trial +3
        non_rewarded_transitions_mat = np.zeros((len(non_rewarded_transitions), 2 * n_trials_around))
        for index, transition in enumerate(list(non_rewarded_transitions)):
            if transition + trials_above > len(licks):
                non_rewarded_transitions_mat = non_rewarded_transitions_mat[0: len(non_rewarded_transitions) - 1, :]
                continue
            else:
                non_rewarded_transitions_mat[index, :] = licks[np.arange(transition - trials_below, transition + trials_above)]
        non_rewarded_transition_prob = np.mean(non_rewarded_transitions_mat, axis=0)
        to_non_rewarded_transitions_prob[session_id] = non_rewarded_transition_prob

        # Do single session plot
        if do_single_session_plot:
            figsize = (6, 4)
            figure, ax = plt.subplots(1, 1, figsize=figsize)
            scale = np.arange(-n_trials_around, n_trials_around + 1)
            scale = np.delete(scale, n_trials_around)
            before_switch = scale[np.where(scale < 0)[0]]
            after_switch = scale[np.where(scale > 0)[0]]
            # ax.plot(scale, rewarded_transition_prob, '--go')
            # ax.plot(scale, non_rewarded_transition_prob, '--ro')
            ax.plot(before_switch, rewarded_transition_prob[0: len(before_switch)], '--ro')
            ax.plot(before_switch, non_rewarded_transition_prob[0: len(before_switch)], '--go')
            ax.plot(after_switch, rewarded_transition_prob[len(before_switch):], '--go')
            ax.plot(after_switch, non_rewarded_transition_prob[len(before_switch):], '--ro')
            ax.plot(([-1, 1]), ([rewarded_transition_prob[len(before_switch) - 1], rewarded_transition_prob[len(before_switch)]]),
                    color='grey', zorder=0)
            ax.plot(([-1, 1]), ([non_rewarded_transition_prob[len(before_switch) - 1], non_rewarded_transition_prob[len(before_switch)]]),
                    color='grey', zorder=1)
            ax.set_xlabel('Trial number')
            ax.set_ylabel('Lick probability')
            figure_title = f"{session_table.mouse_id.values[0]}, {session_table.behavior.values[0]} " \
                           f"{session_table.day.values[0]}"
            ax.set_title(figure_title)
            ax.spines[['right', 'top']].set_visible(False)
            ax.set_ylim([-0.1, 1.05])
            plt.xticks(range(-n_trials_around, n_trials_around + 1))
            plt.show()

    return to_rewarded_transitions_prob, to_non_rewarded_transitions_prob

def plot_multiple_mice_training(bhv_data, saving_path, reward_group_hue=False):
    mice_list = np.unique(bhv_data['mouse_id'].values[:])
    n_mice = len(mice_list)
    print(f" ")
    print(f"Plot average across days for {n_mice} mice")

    # Keep only Auditory and Whisker days
    bhv_data = bhv_data[bhv_data.behavior.isin(('auditory', 'whisker'))]
    # Select columns for plot
    # cols = ['mouse_id', 'reward_group', 'outcome_a', 'outcome_w', 'outcome_n', 'day']
    cols = ['mouse_id', 'outcome_a', 'outcome_w', 'outcome_n', 'day']
    df = bhv_data.loc[bhv_data.early_lick == 0, cols]

    # Compute hit rates. Use transform to propagate hit rate to all entries.
    df['hr_w'] = df.groupby(['mouse_id', 'day'], as_index=False)['outcome_w'].transform(np.nanmean)
    df['hr_a'] = df.groupby(['mouse_id', 'day'], as_index=False)['outcome_a'].transform(np.nanmean)
    df['hr_n'] = df.groupby(['mouse_id', 'day'], as_index=False)['outcome_n'].transform(np.nanmean)

    # Average by day per mouse
    df_by_day = df.groupby(['mouse_id', 'day'], as_index=False).agg(np.nanmean)

    # Select days to show
    first_before_wday = 6
    last_after_wday = 2
    df_by_day = df_by_day[(df_by_day['day'] >= -first_before_wday) &
                                  (df_by_day['day'] <= last_after_wday)]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
    remove_top_right_frame(ax)
    sns.set_style('whitegrid', {'grid.color': '.6', 'grid.linestyle': ':'})

    if reward_group_hue:
        reward_group_hue = 'reward_group'
        hue_order=[1,0]
    else:
        reward_group_hue = None
        hue_order=None

    # Plot all mice
    sns.lineplot(data=df_by_day,
                 ax=ax,
                 x='day',
                 y='hr_n',
                 style='mouse_id',
                 hue=reward_group_hue,
                 hue_order=hue_order, # 1 first in case of reward_group_hue is False
                 palette=['k', 'dimgrey'],
                 dashes=False,
                 lw=1,
                 alpha=0.5,
                 legend=False
                 )

    sns.lineplot(data=df_by_day,
                 ax=ax,
                 x='day',
                 y='hr_a',
                 style='mouse_id',
                 hue=reward_group_hue,
                 hue_order=hue_order,
                 palette=['blue', 'lightblue'],
                 dashes=False,
                 lw=1,
                 alpha=0.5,
                 legend=False
                 )

    sns.lineplot(data=df_by_day[df_by_day['day'].isin(range(0,last_after_wday+1))],
                 ax=ax,
                 x='day',
                 y='hr_w',
                 style='mouse_id',
                 hue=reward_group_hue,
                 hue_order=hue_order,
                 palette=['forestgreen', 'crimson'],
                 dashes=False,
                 lw=1,
                 alpha=0.5,
                 legend=False
                 )


    # Plot mean over mice
    sns.lineplot(data=df_by_day,
                 x='day',
                 y='hr_n',
                 marker='o',
                 hue=reward_group_hue,
                 hue_order=hue_order,
                 palette=['k', 'dimgrey'],
                 mew=0,
                 estimator='mean',
                 err_style='bars',
                 errorbar='se',
                 lw=2.5,
                 alpha=1.0)

    sns.lineplot(data=df_by_day,
                 x='day',
                 y='hr_a',
                 marker='o',
                 hue=reward_group_hue,
                 hue_order=hue_order,
                 palette=['lightblue', 'mediumblue'],
                 mew=0,
                 estimator='mean',
                 err_style='bars',
                 errorbar='se',
                 lw=2.5,
                 alpha=1.0)

    sns.lineplot(data=df_by_day,
                 x='day',
                 y='hr_w',
                 marker='o',
                 hue=reward_group_hue,
                 hue_order=hue_order,
                 palette=['forestgreen', 'crimson'],
                 mew=0,
                 estimator='mean',
                 err_style='bars',
                 errorbar='se',
                 lw=2.5,
                 alpha=1.0)


    # Axes
    ax.set_xlabel('Days to whisker learning', fontsize=20)
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel('P(lick)', fontsize=20)
    ax.tick_params(axis='both', labelsize=15)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # ensures integers i.e. days
    fig.tight_layout()
    plt.title(r'Training performance, $n={}$'.format(n_mice), fontsize=20)

    plt.show()

    # Save
    save_formats = ['pdf', 'png', 'svg']
    saving_path = saving_path
    for save_format in save_formats:
        if reward_group_hue:
            filename = r'training_performance_reward_group.{}'.format(save_format)
        else:
            filename = r'training_performance.{}'.format(save_format)
        fig.savefig(os.path.join(f'{saving_path}', filename),
                    format=f"{save_format}",
                    bbox_inches='tight')

    plt.close()
    return fig

def plot_single_mouse_opto_grid(data, saving_path):
    fig, ax = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
    fig.suptitle('Opto grid performance')

    fig1, ax1 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
    fig1.suptitle('Opto grid trial density')

    data_stim = data.loc[data.opto_stim == 1].drop_duplicates()

    for name, group in data_stim.groupby(by=['context_background', 'trial_type']):
        group['opto_grid_no_global'] = group.groupby(by=['session_id', 'opto_grid_no']).ngroup()
        if 'whisker_trial' in name:
            outcome = 'outcome_w'
            col = 2
        elif 'auditory_trial' in name:
            outcome = 'outcome_a'
            col = 1
        else:
            outcome = 'outcome_n'
            col = 0

        row = group.context.unique()[0]-1
        grid = group.groupby(by=['opto_grid_ml', 'opto_grid_ap'])[outcome].apply(np.nanmean).reset_index()
        sns.heatmap(grid.pivot(index='opto_grid_ap', columns='opto_grid_ml', values=outcome), vmin=0, vmax=1,
                    ax=ax[row, col])
        ax[row, col].invert_yaxis()
        ax[row, col].invert_xaxis()

        grid = group.groupby(by=['opto_grid_ml', 'opto_grid_ap'])[outcome].count().reset_index()
        sns.heatmap(grid.pivot(index='opto_grid_ap', columns='opto_grid_ml', values=outcome), vmin=0,
                    ax=ax1[row, col])
        ax1[row, col].invert_yaxis()
        ax1[row, col].invert_xaxis()

    cols = ['No stim', 'Auditory', 'Whisker']
    rows = ['Rewarded', 'No rewarded']
    for a, col in zip(ax[0], cols):
        a.set_title(col)
    for a, row in zip(ax[:, 0], rows):
        a.set_ylabel(row)

    fig.tight_layout()
    fig.show()
    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        fig.savefig(os.path.join(f'{saving_path}', f'{data.mouse_id.unique()[0]}_opto_grid_performance.{save_format}'),
                       format=f"{save_format}")

    for a, col in zip(ax1[0], cols):
        a.set_title(col)
    for a, row in zip(ax1[:, 0], rows):
        a.set_ylabel(row)

    fig1.tight_layout()
    fig1.show()
    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        fig1.savefig(os.path.join(f'{saving_path}', f'{data.mouse_id.unique()[0]}_opto_grid_trial_density.{save_format}'),
                    format=f"{save_format}")

    return

def plot_multiple_mice_opto_grid(data, saving_path):
    fig, ax = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
    fig.suptitle('Pop opto grid performance')

    fig1, ax1 = plt.subplots(2, 3, figsize=(8, 6), dpi=300)
    fig1.suptitle('Pop opto grid trial density')
    data_stim = data.loc[data.opto_stim == 1].drop_duplicates()

    for name, group in data_stim.groupby(by=['context_background', 'trial_type']):

        group['opto_grid_no_global'] = group.groupby(by=['session_id', 'opto_grid_no']).ngroup()

        if 'whisker_trial' in name:
            outcome = 'outcome_w'
            col = 2
        elif 'auditory_trial' in name:
            outcome = 'outcome_a'
            col = 1
        else:
            outcome = 'outcome_n'
            col = 0

        row = group.context.unique()[0] - 1

        grid = group.groupby(by=['mouse_id', 'opto_grid_ml', 'opto_grid_ap'])[outcome].apply(np.nanmean).groupby(['opto_grid_ml', 'opto_grid_ap']).apply(np.nanmean).reset_index()
        sns.heatmap(grid.pivot(index='opto_grid_ap', columns='opto_grid_ml', values=outcome), vmin=0, vmax=1,
                    ax=ax[row, col])
        ax[row, col].invert_yaxis()
        ax[row, col].invert_xaxis()

        grid = group.groupby(by=['opto_grid_ml', 'opto_grid_ap'])[outcome].count().reset_index()
        sns.heatmap(grid.pivot(index='opto_grid_ap', columns='opto_grid_ml', values=outcome), vmin=0,
                    ax=ax1[row, col])
        ax1[row, col].invert_yaxis()
        ax1[row, col].invert_xaxis()

    cols = ['No stim', 'Auditory', 'Whisker']
    rows = ['Rewarded', 'No rewarded']
    for a, col in zip(ax[0], cols):
        a.set_title(col)
    for a, row in zip(ax[:, 0], rows):
        a.set_ylabel(row)

    fig.tight_layout()
    fig.show()
    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        fig.savefig(os.path.join(f'{saving_path}', 'Pop_opto_grid_trial_density.{save_format}'),
                       format=f"{save_format}")

    for a, col in zip(ax1[0], cols):
        a.set_title(col)
    for a, row in zip(ax1[:, 0], rows):
        a.set_ylabel(row)

    fig1.tight_layout()
    fig1.show()
    save_formats = ['pdf', 'png', 'svg']
    for save_format in save_formats:
        fig1.savefig(os.path.join(f'{saving_path}', 'Pop_opto_grid_trial_density.{save_format}'),
                    format=f"{save_format}")

    return



if __name__ == '__main__':

    # Use the functions to do the plots
    experimenter = 'Axel_Bisi'

    info_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'mice_info')
    root_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'NWBFull')
    output_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'results')
    all_nwb_names = os.listdir(root_path)

    # Load mouse table
    mouse_info_df = pd.read_excel(os.path.join(info_path, 'mouse_reference_weight.xlsx'))
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)
    mouse_info_df = mouse_info_df[mouse_info_df['exclude'] == 0] # exclude mice
    subject_ids = mouse_info_df['mouse_id'].unique()
    #subject_ids = ['AB082', 'AB093', 'AB094', 'AB095']
    #subject_ids = ['AB0{}'.format(i) for i in subject_ids]

    plots_to_do = ['piezo_raster', 'single_session', 'across_days', 'history']
    #plots_to_do = ['piezo_raster']
    plots_to_do = ['piezo_proba']

    for subject_id in subject_ids:
        print(" ")
        print(f"Subject ID : {subject_id}")
        nwb_names = [name for name in all_nwb_names if subject_id in name]
        nwb_files = [os.path.join(root_path, name) for name in nwb_names]
        results_path = os.path.join(output_path, subject_id)
        if not os.path.exists(results_path):
            os.makedirs(results_path)
#
        #plot_behavior(nwb_list=nwb_files, output_folder=results_path, plots=plots_to_do,
        #              info_path=info_path)


    # Plot group behaviour
    nwb_list = [os.path.join(root_path, name) for name in all_nwb_names]
    nwb_list = [nwb_file for nwb_file in nwb_list if any(mouse in nwb_file for mouse in subject_ids)]
    plot_group_behavior(nwb_list=nwb_list, plots=plots_to_do, info_path=info_path)