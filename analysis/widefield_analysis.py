import math
import os

# import dask.array as da
import numpy as np
import nwb_wrappers.nwb_reader_functions as nwb_read
from nwb_utils import server_path, utils_misc, utils_behavior
from analysis.psth_analysis import return_events_aligned_data_table
import tifffile as tiff
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml


def plot_wf_activity(nwb_files, output_path):
    print(f"Create WF average movies")
    for nwb_file in nwb_files:
        mouse_id = nwb_read.get_mouse_id(nwb_file)
        session_id = nwb_read.get_session_id(nwb_file)
        session_type = nwb_read.get_session_type(nwb_file)
        if 'wf' not in session_type:
            print(f"{session_id} is not a widefield session")
            continue

        wf_timestamps = nwb_read.get_widefield_timestamps(nwb_file, ['ophys', 'dff0'])
        epochs = nwb_read.get_behavioral_epochs_names(nwb_file)

        if len(epochs) > 0:
            for epoch in nwb_read.get_behavioral_epochs_names(nwb_file):
                epoch_times = nwb_read.get_behavioral_epochs_times(nwb_file, epoch)
                for trial_type in nwb_read.get_behavioral_events_names(nwb_file):
                    print(f"Trial type : {trial_type}")
                    trials = nwb_read.get_behavioral_events_times(nwb_file, trial_type)[0]
                    print(f"Total of {len(trials)} trials")
                    trials_kept = utils_behavior.filter_events_based_on_epochs(events_ts=trials, epochs=epoch_times)
                    print(f"Total of {len(trials_kept)} trials in {epoch} epoch")
                    frames = []
                    for tstamp in trials:
                        frame = utils_misc.find_nearest(wf_timestamps, tstamp)
                        data = nwb_read.get_widefield_dff0(nwb_file, ['ophys', 'dff0'], frame-200, frame+200)
                        frames.append(data)

                    data_frames = np.array(frames)
                    data_frames = np.stack(data_frames, axis=0)
                    avg_data = np.nanmean(data_frames, axis=0)
                    save_path = os.path.join(output_path, f"{mouse_id}", f"{session_id}")
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    tiff.imwrite(os.path.join(save_path, f'{trial_type}_{epoch}.tiff'), avg_data)

                frames = []
                for tstamp in epoch_times[0]:
                    if tstamp < 10:
                        continue
                    frame = utils_misc.find_nearest(wf_timestamps, tstamp)
                    data = nwb_read.get_widefield_dff0(nwb_file, ['ophys', 'dff0'], frame - 200, frame + 200)
                    frames.append(data)

                data_frames = np.array(frames)
                data_frames = np.stack(data_frames, axis=0)
                avg_data = np.nanmean(data_frames, axis=0)
                save_path = os.path.join(output_path, f"{mouse_id}", f"{session_id}")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                tiff.imwrite(os.path.join(save_path, f'to_{epoch}.tiff'), avg_data)
        else:
            for trial_type in nwb_read.get_behavioral_events_names(nwb_file):
                print(f"Trial type : {trial_type}")
                trials = nwb_read.get_behavioral_events_times(nwb_file, trial_type)[0]
                print(f"Total of {len(trials)} trials")
                frames = []
                for tstamp in trials:
                    frame = utils_misc.find_nearest(wf_timestamps, tstamp)
                    data = nwb_read.get_widefield_dff0(nwb_file, ['ophys', 'dff0'], frame - 200, frame + 200)
                    frames.append(data)

                data_frames = np.array(frames)
                data_frames = np.stack(data_frames, axis=0)
                avg_data = np.nanmean(data_frames, axis=0)
                save_path = os.path.join(output_path, f"{mouse_id}", f"{session_id}")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                tiff.imwrite(os.path.join(save_path, f'{trial_type}.tiff'), avg_data)


def return_events_aligned_wf_table(nwb_files, rrs_keys, trials_dict, trial_names, epochs, time_range):
    """

    :param nwb_files: list of path to nwb files to analyse
    :param rrs_keys: list of keys to access traces from different brain regions in nwb file
    :param trials_dict: list of dictionaries describing the trial to get from table
    :param trial_names: list of trial names
    :param epochs: list of epochs
    :param time_range: time range for psth
    :return: a dataframe with activity aligned and trial info
    """

    full_df = []
    for index, trial_dict in enumerate(trials_dict):
        for epoch_index, epoch in enumerate(epochs):
            print(f" ")
            print(f"Trial selection : {trials_dict[index]} (Trial name : {trial_names[index]})")
            print(f"Epoch : {epoch}")
            data_table = return_events_aligned_data_table(nwb_list=nwb_files,
                                                          rrs_keys=rrs_keys,
                                                          time_range=time_range,
                                                          trial_selection=trials_dict[index],
                                                          epoch=epoch)
            data_table['trial_type'] = trial_names[index]
            data_table['epoch'] = epochs[epoch_index]
            full_df.append(data_table)
    full_df = pd.concat(full_df, ignore_index=True)

    return full_df


if __name__ == "__main__":
    experimenter_initials = "RD"

    root_path = server_path.get_experimenter_nwb_folder(experimenter_initials)

    output_path = os.path.join(f'{server_path.get_experimenter_saving_folder_root(experimenter_initials)}',
                               'Pop_results', 'Context_behaviour')
    all_nwb_names = os.listdir(root_path)

    # Sessions
    # session_to_do = [
    #
    #     "RD039_20240208_143129", "RD039_20240209_162220",
    #     "RD039_20240210_140338"
    # ]

    # Selection of sessions with no wf frames missing
    # session_to_do = [
    #     "RD039_20240124_142334", "RD039_20240125_142517",
    #     "RD039_20240206_134324",
    #     "RD039_20240208_143129", "RD039_20240209_162220",
    #     "RD039_20240210_140338", "RD039_20240212_135702",
    #     "RD039_20240213_161938", "RD039_20240214_164330",
    #     "RD039_20240215_142858"
    # ]

    # Selection of sessions with no WF frames missing and 'good' behavior
    session_to_do = [
        "RD039_20240124_142334", "RD039_20240125_142517",

        "RD039_20240209_162220",
        "RD039_20240210_140338", "RD039_20240212_135702",
        "RD039_20240213_161938",
        "RD039_20240215_142858"
    ]

    # To do single session
    # session_to_do = ["RD043_20240214_105456"]

    # Get list of mouse ID from list of session to do
    subject_ids = list(np.unique([session[0:5] for session in session_to_do]))

    # Decide what to do :
    do_wf_movies_average = False
    do_psths = True

    # ---------------------------------------------------------------------------------------------------------- #
    if do_wf_movies_average:
        for subject_id in subject_ids:
            nwb_names = [name for name in all_nwb_names if subject_id in name]
            nwb_files = []
            for session in session_to_do:
                nwb_files += [os.path.join(root_path, name) for name in nwb_names if session in name]

            print(f"nwb_files : {nwb_files}")

            plot_wf_activity(nwb_files, output_path)

    # ---------------------------------------------------------------------------------------------------------- #
    if do_psths:
        # Build one dataframe with all mice
        df = []
        for subject_id in subject_ids:
            nwb_names = [name for name in all_nwb_names if subject_id in name]
            nwb_files = []
            for session in session_to_do:
                nwb_files += [os.path.join(root_path, name) for name in nwb_names if session in name]

            print(f"nwb_files : {nwb_files}")

            # Create dataframe of traces aligned to events
            trials_dict = [{'whisker_stim': [1], 'lick_flag': [1]},
                           {'whisker_stim': [1], 'lick_flag': [0]},
                           {'auditory_stim': [1], 'lick_flag': [1]},
                           {'auditory_stim': [1], 'lick_flag': [0]}]

            trial_names = ['whisker_hit',
                           'whisker_miss',
                           'auditory_hit',
                           'auditory_miss']

            epochs = ['rewarded', 'non-rewarded']

            t_range = (0.8, 1.5)

            mouse_df = return_events_aligned_wf_table(nwb_files=nwb_files,
                                                      rrs_keys=['ophys', 'brain_area_fluorescence', 'dff0_traces'],
                                                      trials_dict=trials_dict,
                                                      trial_names=trial_names,
                                                      epochs=epochs,
                                                      time_range=t_range)
            df.append(mouse_df)
        df = pd.concat(df, ignore_index=True)

        # ---------------------------------------------------------------------------------------------------------- #
        # Group data by sessions
        session_avg_data = df.groupby(["mouse_id", "session_id", "trial_type", "epoch",
                                      "behavior_day", "behavior_type", "roi", "cell_type", "time"],
                                      as_index=False).agg(np.nanmean)
        # Group session data by mice
        mice_avg_data = session_avg_data.drop(['session_id', 'behavior_day'], axis=1)
        mice_avg_data = mice_avg_data.groupby(["mouse_id", "trial_type", "epoch",
                                               "behavior_type", "roi", "cell_type", "time"],
                                              as_index=False).agg(np.nanmean)

        # --------------------------------------------------------------------------------------------------------- #
        print('Do some plots')
        # DO SOME PLOTS #
        figsize = (10, 10)

        # -------------------------------- Plot general average --------------------------------------------------- #
        # Plot all area to see successive activation
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=figsize)
        rwd_data_to_plot = mice_avg_data.loc[(mice_avg_data.trial_type.isin(['whisker_hit'])) &
                                             (mice_avg_data.cell_type.isin(
                                                 ['tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                             (mice_avg_data.epoch == 'rewarded')]
        sns.lineplot(data=rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[0, 0])
        rwd_data_to_plot = mice_avg_data.loc[(mice_avg_data.trial_type.isin(['whisker_hit'])) &
                                             (mice_avg_data.cell_type.isin(
                                                 ['tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                             (mice_avg_data.epoch == 'non-rewarded')]
        sns.lineplot(data=rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[1, 0])
        axs[0, 0].set_title('Whisker hit Rewarded context')
        axs[1, 0].set_title('Whisker hit Non rewarded context')

        nn_rwd_data_to_plot = mice_avg_data.loc[(mice_avg_data.trial_type.isin(['whisker_miss'])) &
                                                (mice_avg_data.cell_type.isin(
                                                       ['tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                                (mice_avg_data.epoch == 'rewarded')]
        sns.lineplot(data=nn_rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[0, 1])
        nn_rwd_data_to_plot = mice_avg_data.loc[(mice_avg_data.trial_type.isin(['whisker_miss'])) &
                                                (mice_avg_data.cell_type.isin(
                                                    ['tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                                (mice_avg_data.epoch == 'non-rewarded')]
        sns.lineplot(data=nn_rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[1, 1])
        axs[0, 1].set_title('Whisker miss Rewarded context')
        axs[1, 1].set_title('Whisker miss Non rewarded context')
        plt.suptitle(f'Whisker trials average from {len(subject_ids)} mice ({subject_ids})')
        # plt.show()
        saving_folder = os.path.join(output_path)
        if not os.path.exists(saving_folder):
            os.makedirs(saving_folder)
        fig.savefig(os.path.join(saving_folder, 'whisker_trials_average.pdf'))
        plt.close()

        # Plot by area
        areas = ['A1', 'wS1', 'wS2', 'wM1', 'wM2', 'tjM1']
        for area in areas:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            data_to_plot = mice_avg_data.loc[(mice_avg_data.cell_type == area) &
                                             (mice_avg_data.trial_type.isin(['whisker_hit', 'whisker_miss']))]
            sns.lineplot(data=data_to_plot, x='time', y='activity', hue='epoch', style='trial_type', ax=ax)
            plt.suptitle(f"{area} response to whisker trials average from {len(subject_ids)} mice ({subject_ids})")
            # plt.show()
            saving_folder = os.path.join(output_path)
            if not os.path.exists(saving_folder):
                os.makedirs(saving_folder)
            fig.savefig(os.path.join(saving_folder, f'whisker_trials_average_{area}.pdf'))
            plt.close()

        # ------------------------------------ Plot by mouse ----------------------------------------------------- #
        for subject_id in subject_ids:
            # List subject sessions
            subject_sessions = [session for session in session_to_do if subject_id in session]
            # Average per session : Plot with one point per time per session:
            session_avg_data = session_avg_data.loc[session_avg_data.mouse_id == subject_id]
            # Plot all area to see successive activation
            fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=figsize)
            rwd_data_to_plot = session_avg_data.loc[(session_avg_data.trial_type.isin(['whisker_hit'])) &
                                                    (session_avg_data.cell_type.isin(
                                                        ['tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                                    (session_avg_data.epoch == 'rewarded')]
            sns.lineplot(data=rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[0, 0])
            rwd_data_to_plot = session_avg_data.loc[(session_avg_data.trial_type.isin(['whisker_hit'])) &
                                                    (session_avg_data.cell_type.isin(
                                                        ['tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                                    (session_avg_data.epoch == 'non-rewarded')]
            sns.lineplot(data=rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[1, 0])
            axs[0, 0].set_title('Whisker hit Rewarded context')
            axs[1, 0].set_title('Whisker hit Non rewarded context')

            nn_rwd_data_to_plot = session_avg_data.loc[(session_avg_data.trial_type.isin(['whisker_miss'])) &
                                                       (session_avg_data.cell_type.isin(
                                                           ['tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                                       (session_avg_data.epoch == 'rewarded')]
            sns.lineplot(data=nn_rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[0, 1])
            nn_rwd_data_to_plot = session_avg_data.loc[(session_avg_data.trial_type.isin(['whisker_miss'])) &
                                                       (session_avg_data.cell_type.isin(
                                                           ['tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                                       (session_avg_data.epoch == 'non-rewarded')]
            sns.lineplot(data=nn_rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[1, 1])
            axs[0, 1].set_title('Whisker miss Rewarded context')
            axs[1, 1].set_title('Whisker miss Non rewarded context')
            plt.suptitle(f'{subject_id} : average from {len(subject_sessions)} sessions')
            # plt.show()
            saving_folder = os.path.join(output_path, f"{subject_id}")
            if not os.path.exists(saving_folder):
                os.makedirs(saving_folder)
            fig.savefig(os.path.join(saving_folder, f"{subject_id}_whisker_trials_average.pdf"))
            plt.close()

            # Plot per area to compare the two contexts in each:
            areas = ['A1', 'wS1', 'wS2', 'wM1', 'wM2', 'tjM1']
            for area in areas:
                fig, ax = plt.subplots(1, 1, figsize=figsize)
                data_to_plot = session_avg_data.loc[(session_avg_data.cell_type == area) &
                                                    (session_avg_data.trial_type.isin(
                                                        ['whisker_hit', 'whisker_miss']))]
                sns.lineplot(data=data_to_plot, x='time', y='activity', hue='epoch', style='trial_type', ax=ax)
                plt.suptitle(f"{area} response to whisker trials")
                # plt.show()
                saving_folder = os.path.join(output_path, f"{subject_id}")
                if not os.path.exists(saving_folder):
                    os.makedirs(saving_folder)
                fig.savefig(os.path.join(saving_folder, f'{subject_id}_whisker_trials_average_{area}.pdf'))
                plt.close()

            # Plot with single session
            for session in subject_sessions:
                # Plot with all areas
                fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=figsize)
                rwd_data_to_plot = df.loc[(df.trial_type.isin(['whisker_hit'])) &
                                          (df.cell_type.isin(
                                                            ['tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                          (df.epoch == 'rewarded') &
                                          (df.session_id == session)]
                sns.lineplot(data=rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[0, 0])
                rwd_data_to_plot = df.loc[(df.trial_type.isin(['whisker_hit'])) &
                                          (df.cell_type.isin(
                                                            ['tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                          (df.epoch == 'non-rewarded') &
                                          (df.session_id == session)]
                sns.lineplot(data=rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[1, 0])
                axs[0, 0].set_title('Whisker hit Rewarded context')
                axs[1, 0].set_title('Whisker hit Non rewarded context')

                nn_rwd_data_to_plot = df.loc[(df.trial_type.isin(['whisker_miss'])) &
                                             (df.cell_type.isin(
                                                               ['tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                             (df.epoch == 'rewarded') &
                                             (df.session_id == session)]
                sns.lineplot(data=nn_rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[0, 1])
                nn_rwd_data_to_plot = df.loc[(df.trial_type.isin(['whisker_miss'])) &
                                             (df.cell_type.isin(
                                                               ['tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                             (df.epoch == 'non-rewarded') &
                                             (df.session_id == session)]
                sns.lineplot(data=nn_rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[1, 1])
                axs[0, 1].set_title('Whisker miss Rewarded context')
                axs[1, 1].set_title('Whisker miss Non rewarded context')
                plt.suptitle(f"{session}, whisker trials")
                # plt.show()
                saving_folder = os.path.join(output_path, f"{session[0:5]}", f"{session}")
                if not os.path.exists(saving_folder):
                    os.makedirs(saving_folder)
                fig.savefig(os.path.join(saving_folder, f"{session}_whisker_trials.pdf"))
                plt.close()

                # Plot by area
                areas = ['A1', 'wS1', 'wS2', 'wM1', 'wM2', 'tjM1']
                for area in areas:
                    sub_data_to_plot = df.loc[(df.cell_type == area) &
                                              (df.trial_type.isin(['whisker_hit', 'whisker_miss'])) &
                                              (df.session_id == session)]
                    fig, ax = plt.subplots(1, 1, figsize=figsize)
                    sns.lineplot(data=sub_data_to_plot, x='time', y='activity', hue='epoch', style='trial_type', ax=ax)

                    plt.suptitle(f"{session}, {area} response to whisker trials")

                    # plt.show()
                    saving_folder = os.path.join(output_path, f"{session[0:5]}", f"{session}")
                    if not os.path.exists(saving_folder):
                        os.makedirs(saving_folder)
                    fig.savefig(os.path.join(saving_folder, f"{session}_whisker_trials_{area}.pdf"))
                    plt.close()
