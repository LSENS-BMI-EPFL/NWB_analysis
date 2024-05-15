import nwb_wrappers.nwb_reader_functions as nwb_read
import numpy as np
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
import statsmodels.api as sm
import os

pd.options.mode.chained_assignment = None  # default='warn'


def plot_first_whisker_outcome_against_time(nwb_files, save_path):
    rwd_wh_table = []
    nn_rwd_wh_table = []
    last_rwd_wh_table = []
    last_nn_rwd_wh_table = []
    for nwb_file in nwb_files:
        print(" ")
        session_id = nwb_read.get_session_id(nwb_file)
        mouse_id = session_id[0:5]
        print(f"Session : {session_id}")
        trial_table = nwb_read.get_trial_table(nwb_file)
        rewarded_context_timestamps = nwb_read.get_behavioral_epochs_times(nwb_file, epoch_name='rewarded')
        non_rewarded_context_timestamps = nwb_read.get_behavioral_epochs_times(nwb_file, epoch_name='non-rewarded')

        whisker_table = trial_table.loc[trial_table.trial_type == 'whisker_trial']
        whisker_table = whisker_table.reset_index(drop=True)

        transitions = list(np.diff(whisker_table['context']))
        transitions.insert(0, 0)
        whisker_table['transitions'] = transitions
        cols = ['start_time', 'lick_flag', 'context', 'transitions']
        filter_wh_table = whisker_table[cols]

        # Transitions to rewarded
        rwd_filter_wh_table = filter_wh_table.loc[filter_wh_table.transitions == 1]
        if rewarded_context_timestamps[0][0] == 0:
            if len(rewarded_context_timestamps[0][1:]) == len(rwd_filter_wh_table) + 1:
                rwd_filter_wh_table['reward_epoch_start'] = rewarded_context_timestamps[0][1:-1]
            else:
                rwd_filter_wh_table['reward_epoch_start'] = rewarded_context_timestamps[0][1:]
        else:
            if len(rewarded_context_timestamps[0]) == len(rwd_filter_wh_table) + 1:
                rwd_filter_wh_table['reward_epoch_start'] = rewarded_context_timestamps[0][:-1]
            else:
                rwd_filter_wh_table['reward_epoch_start'] = rewarded_context_timestamps[0]

        rwd_filter_wh_table['time_in_reward'] = rwd_filter_wh_table['start_time'] - rwd_filter_wh_table[
            'reward_epoch_start']
        rwd_filter_wh_table['session_id'] = session_id
        rwd_filter_wh_table['mouse_id'] = mouse_id
        rwd_wh_table.append(rwd_filter_wh_table)

        # Transitions to non-rewarded
        nn_rwd_filter_wh_table = filter_wh_table.loc[filter_wh_table.transitions == -1]
        if non_rewarded_context_timestamps[0][0] == 0:
            if len(non_rewarded_context_timestamps[0][1:]) == len(nn_rwd_filter_wh_table) + 1:
                nn_rwd_filter_wh_table['non-reward_epoch_start'] = non_rewarded_context_timestamps[0][1:-1]
            else:
                nn_rwd_filter_wh_table['non-reward_epoch_start'] = non_rewarded_context_timestamps[0][1:]
        else:
            if len(non_rewarded_context_timestamps[0]) == len(nn_rwd_filter_wh_table) + 1:
                nn_rwd_filter_wh_table['non-reward_epoch_start'] = non_rewarded_context_timestamps[0][:-1]
            else:
                nn_rwd_filter_wh_table['non-reward_epoch_start'] = non_rewarded_context_timestamps[0]

        nn_rwd_filter_wh_table['time_in_non_reward'] = nn_rwd_filter_wh_table['start_time'] - nn_rwd_filter_wh_table[
            'non-reward_epoch_start']
        nn_rwd_filter_wh_table['session_id'] = session_id
        nn_rwd_filter_wh_table['mouse_id'] = mouse_id
        nn_rwd_wh_table.append(nn_rwd_filter_wh_table)

        # Last non-rewarded whisker trial
        last_non_rwd_whisker = (np.where(filter_wh_table.transitions == 1)[0] - 1).tolist()
        last_non_rwd_whisker_table = filter_wh_table.loc[last_non_rwd_whisker]
        if rewarded_context_timestamps[0][0] == 0:
            if len(rewarded_context_timestamps[0][1:]) == len(last_non_rwd_whisker_table) + 1:
                last_non_rwd_whisker_table['next_reward_epoch_start'] = rewarded_context_timestamps[0][1:-1]
            else:
                last_non_rwd_whisker_table['next_reward_epoch_start'] = rewarded_context_timestamps[0][1:]
        else:
            if len(rewarded_context_timestamps[0]) == len(last_non_rwd_whisker_table) + 1:
                last_non_rwd_whisker_table['next_reward_epoch_start'] = rewarded_context_timestamps[0][:-1]
            else:
                last_non_rwd_whisker_table['next_reward_epoch_start'] = rewarded_context_timestamps[0]

        last_non_rwd_whisker_table['time_to_reward'] = (last_non_rwd_whisker_table['start_time'] -
                                                        last_non_rwd_whisker_table['next_reward_epoch_start'])
        last_non_rwd_whisker_table['session_id'] = session_id
        last_non_rwd_whisker_table['mouse_id'] = mouse_id
        last_nn_rwd_wh_table.append(last_non_rwd_whisker_table)

        # Last rewarded whisker trial
        last_rwd_whisker = (np.where(filter_wh_table.transitions == -1)[0] - 1).tolist()
        last_rwd_whisker_table = filter_wh_table.loc[last_rwd_whisker]
        if non_rewarded_context_timestamps[0][0] == 0:
            if len(non_rewarded_context_timestamps[0][1:]) == len(last_rwd_whisker_table) + 1:
                last_rwd_whisker_table['next_non-reward_epoch_start'] = non_rewarded_context_timestamps[0][1:-1]
            else:
                last_rwd_whisker_table['next_non-reward_epoch_start'] = non_rewarded_context_timestamps[0][1:]
        else:
            if len(non_rewarded_context_timestamps[0]) == len(last_rwd_whisker_table) + 1:
                last_rwd_whisker_table['next_non-reward_epoch_start'] = non_rewarded_context_timestamps[0][:-1]
            else:
                last_rwd_whisker_table['next_non-reward_epoch_start'] = non_rewarded_context_timestamps[0]

        last_rwd_whisker_table['time_to_non_reward'] = (last_rwd_whisker_table['start_time'] -
                                                        last_rwd_whisker_table['next_non-reward_epoch_start'])
        last_rwd_whisker_table['session_id'] = session_id
        last_rwd_whisker_table['mouse_id'] = mouse_id
        last_rwd_wh_table.append(last_rwd_whisker_table)

    rwd_wh_table = pd.concat(rwd_wh_table)
    nn_rwd_wh_table = pd.concat(nn_rwd_wh_table)
    last_rwd_wh_table = pd.concat(last_rwd_wh_table)
    last_nn_rwd_wh_table = pd.concat(last_nn_rwd_wh_table)

    # ------------------------------- LOOK AT FIRST WHISKER TRIAL AFTER TRANSITION --------------------------------- #
    # Figure 1 : distribution of first whisker trial time
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 8), sharey=True)
    sns.boxplot(data=rwd_wh_table, y='time_in_reward', color='green', ax=ax0)
    sns.despine(top=True, right=True)
    sns.boxplot(data=nn_rwd_wh_table, y='time_in_non_reward', color='red', ax=ax1)
    ax0.set_yticks(np.arange(0, 100, 10))
    ax0.set_ylabel('Time after context transition (s)')
    ax0.set_xlabel('To rewarded context')
    ax1.set_yticks(np.arange(0, 100, 10))
    ax1.set_ylabel('Time after context transition (s)')
    ax1.set_xlabel('To non-rewarded context')
    plt.show()
    for ext in ['.png', '.svg']:
        fig.savefig(os.path.join(save_path, f"first_wh_trial_dist{ext}"))

    # Figure 2 : first whisker with separated hit and miss
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 8), sharey=True)
    sns.stripplot(data=rwd_wh_table, x='lick_flag', y='time_in_reward', color='black', ax=ax0)
    sns.pointplot(data=rwd_wh_table, x='lick_flag', y='time_in_reward', color='green',
                  estimator=np.mean, errorbar=('ci', 95), n_boot=1000, ax=ax0)
    sns.despine(top=True, right=True)
    ax0.set_ylabel('Time after transition')
    ax0.set_title('To rewarded context')
    ax0.set_xlabel('Outcome of first rewarded whisker trial')
    ax0.set_xticklabels(['NO LICK', 'LICK'])

    sns.stripplot(data=nn_rwd_wh_table, x='lick_flag', y='time_in_non_reward', color='black', ax=ax1)
    sns.pointplot(data=nn_rwd_wh_table, x='lick_flag', y='time_in_non_reward', color='red',
                  estimator=np.mean, errorbar=('ci', 95), n_boot=1000, ax=ax1)
    sns.despine(top=True, right=True)
    ax1.set_ylabel('Time after transition')
    ax1.set_title('To non-rewarded context')
    ax1.set_xlabel('Outcome of first non-rewarded whisker trial')
    ax1.set_xticklabels(['NO LICK', 'LICK'])
    plt.show()
    for ext in ['.png', '.svg']:
        fig.savefig(os.path.join(save_path, f"first_wh_trial_hit_miss{ext}"))

    # Figure 3 : first whisker lick probability by bin
    rwd_times = rwd_wh_table.time_in_reward.values[:]
    rwd_wh_table['time_bin'] = np.digitize(rwd_times, bins=np.arange(0, 100, 10))
    cols = ['time_bin', 'lick_flag']
    bin_averaged_data_rwd = rwd_wh_table[cols].groupby('time_bin', as_index=False).agg(np.mean)

    nn_rwd_times = nn_rwd_wh_table.time_in_non_reward.values[:]
    nn_rwd_wh_table['time_bin'] = np.digitize(nn_rwd_times, bins=np.arange(0, 100, 10))
    cols = ['time_bin', 'lick_flag']
    bin_averaged_data_nn_rwd = nn_rwd_wh_table[cols].groupby('time_bin', as_index=False).agg(np.mean)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
    xlabel_dict = {1: '0-10', 2: '10-20', 3: '20-30', 4: '30-40',
                   5: '40-50', 6: '50-60', 7: '60-70', 8: '70-80', 9: '80-90', 10: '90-100'}

    sns.pointplot(data=bin_averaged_data_rwd, x='time_bin', y='lick_flag', color='green', ax=ax0)
    sns.despine(top=True, right=True)
    ax0.set_xlabel('Time after transition')
    ax0.set_title('To rewarded context')
    ax0.set_ylabel('Lick probability')
    ax0.set_ylim(-0.05, 1.05)
    ax0.set_xlim(-0.05, 5.05)
    new_label = [xlabel_dict[int(i.get_text())] for i in ax0.get_xticklabels()]
    ax0.set_xticklabels(new_label)

    sns.pointplot(data=bin_averaged_data_nn_rwd, x='time_bin', y='lick_flag', color='red', ax=ax1)
    ax1.set_xlabel('Time after transition')
    ax1.set_title('To non-rewarded context')
    ax1.set_ylabel('Lick probability')
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(-0.05, 5.05)
    new_label = [xlabel_dict[int(i.get_text())] for i in ax1.get_xticklabels()]
    ax1.set_xticklabels(new_label)
    plt.show()
    for ext in ['.png', '.svg']:
        fig.savefig(os.path.join(save_path, f"first_wh_trial_prob_split{ext}"))

    # Figure 4 :
    bin_averaged_data_rwd['Context transition'] = 'To rewarded'
    bin_averaged_data_nn_rwd['Context transition'] = 'To non-rewarded'
    bin_averaged_data = pd.concat([bin_averaged_data_rwd, bin_averaged_data_nn_rwd])
    fig, ax0 = plt.subplots(1, 1, figsize=(8, 8))
    sns.pointplot(data=bin_averaged_data, x='time_bin', y='lick_flag', hue='Context transition',
                  palette=['green', 'red'],
                  ax=ax0)
    xlabel_dict = {1: '0-10', 2: '10-20', 3: '20-30', 4: '30-40',
                   5: '40-50', 6: '50-60', 7: '60-70', 8: '70-80', 9: '80-90', 10: '90-100'}
    new_label = [xlabel_dict[int(i.get_text())] for i in ax0.get_xticklabels()]
    ax0.set_xticklabels(new_label)
    ax0.set_xlabel('Time after transition')
    ax0.set_ylabel('Lick probability')
    ax0.axhline(y=0.5, xmin=0, xmax=1, color='k', linestyle='--')
    sns.despine(top=True, right=True)
    ax0.set_ylim(-0.05, 1.05)
    ax0.set_xlim(-0.05, 5.05)
    plt.show()
    for ext in ['.png', '.svg']:
        fig.savefig(os.path.join(save_path, f"first_wh_trial_prob_overlay{ext}"))
    # ------------------------------- LOOK AT LAST WHISKER TRIAL BEFORE TRANSITION --------------------------------- #
    # Figure 1b : distribution of last whisker trial time
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 8), sharey=True)
    sns.boxplot(data=last_rwd_wh_table, y='time_to_non_reward', color='green', ax=ax0)
    sns.despine(top=True, right=True)
    sns.boxplot(data=last_nn_rwd_wh_table, y='time_to_reward', color='red', ax=ax1)
    ax0.set_yticks(np.arange(0, 100, 10) * (-1))
    ax0.set_ylabel('Time before context transition (s)')
    ax0.set_xlabel('To non-rewarded context')
    ax1.set_yticks(np.arange(0, 100, 10) * (-1))
    ax1.set_ylabel('Time before context transition (s)')
    ax1.set_xlabel('To rewarded context')
    plt.show()
    for ext in ['.png', '.svg']:
        fig.savefig(os.path.join(save_path, f"last_wh_trial_dist{ext}"))

    # Figure 2b : last whisker with separated hit and miss
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 8), sharey=True)
    sns.stripplot(data=last_rwd_wh_table, x='lick_flag', y='time_to_non_reward', color='black', ax=ax0)
    sns.pointplot(data=last_rwd_wh_table, x='lick_flag', y='time_to_non_reward', color='green',
                  estimator=np.mean, errorbar=('ci', 95), n_boot=1000, ax=ax0)
    sns.despine(top=True, right=True)
    ax0.set_ylabel('Time before transition')
    ax0.set_title('To non-rewarded context')
    ax0.set_xlabel('Outcome of last rewarded whisker trial')
    ax0.set_xticklabels(['NO LICK', 'LICK'])

    sns.stripplot(data=last_nn_rwd_wh_table, x='lick_flag', y='time_to_reward', color='black', ax=ax1)
    sns.pointplot(data=last_nn_rwd_wh_table, x='lick_flag', y='time_to_reward', color='red',
                  estimator=np.mean, errorbar=('ci', 95), n_boot=1000, ax=ax1)
    sns.despine(top=True, right=True)
    ax1.set_ylabel('Time before transition')
    ax1.set_title('To rewarded context')
    ax1.set_xlabel('Outcome of last non-rewarded whisker trial')
    ax1.set_xticklabels(['NO LICK', 'LICK'])
    plt.show()
    for ext in ['.png', '.svg']:
        fig.savefig(os.path.join(save_path, f"last_wh_trial_hit_miss{ext}"))

    # Figure 3b : last lick probability by bin
    rwd_times = last_rwd_wh_table.time_to_non_reward.values[:]
    last_rwd_wh_table['time_bin'] = np.digitize(rwd_times, bins=np.arange(0, 100, 10) * (-1))
    cols = ['time_bin', 'lick_flag']
    bin_averaged_data_rwd_last = last_rwd_wh_table[cols].groupby('time_bin', as_index=False).agg(np.mean)

    nn_rwd_times = last_nn_rwd_wh_table.time_to_reward.values[:]
    last_nn_rwd_wh_table['time_bin'] = np.digitize(nn_rwd_times, bins=np.arange(0, 100, 10) * (-1))
    cols = ['time_bin', 'lick_flag']
    bin_averaged_data_nn_rwd_last = last_nn_rwd_wh_table[cols].groupby('time_bin', as_index=False).agg(np.mean)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
    sns.pointplot(data=bin_averaged_data_rwd_last, x='time_bin', y='lick_flag', color='green', ax=ax0)
    sns.despine(top=True, right=True)
    ax0.set_xlabel('Time before transition')
    ax0.set_title('To non-rewarded context')
    ax0.set_ylabel('Lick probability')
    ax0.set_ylim(-0.05, 1.05)
    ax0.set_xlim(-0.05, 5.05)
    xlabel_dict_pre = {1: '10-0', 2: '20-10', 3: '30-20', 4: '40-30',
                       5: '50-40', 6: '60-50', 7: '70-60', 8: '80-70', 9: '90-80', 10: '90-100'}
    new_label = [xlabel_dict_pre[int(i.get_text())] for i in ax0.get_xticklabels()]
    ax0.set_xticklabels(new_label)
    ax0.invert_xaxis()

    sns.pointplot(data=bin_averaged_data_nn_rwd_last, x='time_bin', y='lick_flag', color='red', ax=ax1)
    ax1.set_xlabel('Time before transition')
    ax1.set_title('To rewarded context')
    ax1.set_ylabel('Lick probability')
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(-0.05, 5.05)
    ax1.invert_xaxis()
    new_label = [xlabel_dict_pre[int(i.get_text())] for i in ax1.get_xticklabels()]
    ax1.set_xticklabels(new_label)
    plt.show()
    for ext in ['.png', '.svg']:
        fig.savefig(os.path.join(save_path, f"last_wh_trial_prob_split{ext}"))

    # PLOT ONE FIGURE WITH TIME AROUND CONTEXT TRANSITION
    fig, axs = plt.subplots(2, 2, figsize=(15, 9), sharey=True)

    xlabel_dict = {1: '0-10', 2: '10-20', 3: '20-30', 4: '30-40',
                   5: '40-50', 6: '50-60', 7: '60-70', 8: '70-80', 9: '80-90', 10: '90-100'}

    xlabel_dict_pre = {1: '10-0', 2: '20-10', 3: '30-20', 4: '40-30',
                   5: '50-40', 6: '60-50', 7: '70-60', 8: '80-70', 9: '90-80', 10: '100-90'}

    # ax0 : lick probability at last non-rewarded whisker trial
    sns.pointplot(data=bin_averaged_data_nn_rwd_last.loc[bin_averaged_data_nn_rwd_last.time_bin < 6],
                  x='time_bin', y='lick_flag', color='red', ax=axs[0, 0])
    axs[0, 0].set_xlabel('Time before transition')
    axs[0, 0].set_title('In non-rewarded context')
    axs[0, 0].set_ylabel('Lick probability')
    axs[0, 0].set_ylim(-0.05, 1.05)
    axs[0, 0].set_xlim(-0.05, 5.05)
    axs[0, 0].invert_xaxis()
    axs[0, 0].get_xaxis().set_visible(False)
    new_label = [xlabel_dict_pre[int(i.get_text())] for i in axs[0, 0].get_xticklabels()]
    axs[0, 0].set_xticklabels(new_label)

    # ax1 : lick probability at first rewarded whisker trial
    sns.pointplot(data=bin_averaged_data_rwd.loc[bin_averaged_data_rwd.time_bin < 6],
                  x='time_bin', y='lick_flag', color='green', ax=axs[0, 1])
    sns.despine(top=True, right=True)
    axs[0, 1].set_xlabel('Time after transition')
    axs[0, 1].set_title('In rewarded context')
    axs[0, 1].set_ylabel('Lick probability')
    axs[0, 1].set_ylim(-0.05, 1.05)
    axs[0, 1].set_xlim(-0.05, 5.05)
    axs[0, 1].get_xaxis().set_visible(False)
    new_label = [xlabel_dict[int(i.get_text())] for i in axs[0, 1].get_xticklabels()]
    axs[0, 1].set_xticklabels(new_label)

    # ax2 : lick probability at last rewarded whisker trial
    sns.pointplot(data=bin_averaged_data_rwd_last.loc[bin_averaged_data_rwd_last.time_bin < 6],
                  x='time_bin', y='lick_flag', color='green', ax=axs[1, 0])
    sns.despine(top=True, right=True)
    axs[1, 0].set_xlabel('Time before transition')
    axs[1, 0].set_title('In rewarded context')
    axs[1, 0].set_ylabel('Lick probability')
    axs[1, 0].set_ylim(-0.05, 1.05)
    axs[1, 0].set_xlim(-0.05, 5.05)
    new_label = [xlabel_dict_pre[int(i.get_text())] for i in axs[1, 0].get_xticklabels()]
    axs[1, 0].set_xticklabels(new_label)
    axs[1, 0].invert_xaxis()

    # ax3 : lick probability at first non-rewarded whisker trial
    sns.pointplot(data=bin_averaged_data_nn_rwd.loc[bin_averaged_data_nn_rwd.time_bin < 6],
                  x='time_bin', y='lick_flag', color='red', ax=axs[1, 1])
    axs[1, 1].set_xlabel('Time after transition')
    axs[1, 1].set_title('In non-rewarded context')
    axs[1, 1].set_ylabel('Lick probability')
    axs[1, 1].set_ylim(-0.05, 1.05)
    axs[1, 1].set_xlim(-0.05, 5.05)
    new_label = [xlabel_dict[int(i.get_text())] for i in axs[1, 1].get_xticklabels()]
    axs[1, 1].set_xticklabels(new_label)

    plt.show()
    for ext in ['.png', '.svg']:
        fig.savefig(os.path.join(save_path, f"first_last_wh_trial_prob_split{ext}"))
    
    # Figure all in one
    fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
    sns.pointplot(data=bin_averaged_data_nn_rwd_last.loc[bin_averaged_data_nn_rwd_last.time_bin < 6],
                  x='time_bin', y='lick_flag', color='red', ax=ax0)
    sns.pointplot(data=bin_averaged_data_rwd_last.loc[bin_averaged_data_rwd_last.time_bin < 6],
                  x='time_bin', y='lick_flag', color='green', ax=ax0)
    ax0.set_xlabel('Time before transition')
    ax0.set_title('Block n-1')
    ax0.set_ylabel('Lick probability')
    ax0.set_ylim(-0.05, 1.05)
    ax0.set_xlim(-0.1, 4.5)
    ax0.axhline(y=0.5, xmin=0, xmax=1, linestyle='--', color='gray')
    new_label = [xlabel_dict_pre[int(i.get_text())] for i in ax0.get_xticklabels()]
    ax0.set_xticklabels(new_label)
    ax0.invert_xaxis()
    sns.pointplot(data=bin_averaged_data_rwd.loc[bin_averaged_data_rwd.time_bin < 6],
                  x='time_bin', y='lick_flag', color='green', ax=ax1)
    sns.pointplot(data=bin_averaged_data_nn_rwd.loc[bin_averaged_data_nn_rwd.time_bin < 6],
                  x='time_bin', y='lick_flag', color='red', ax=ax1)
    ax1.set_xlabel('Time after transition')
    ax1.set_title('Block n')
    ax1.set_ylabel('Lick probability')
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(-0.1, 4.5)
    ax1.axhline(y=0.5, xmin=0, xmax=1, linestyle='--', color='gray')
    ax1.get_yaxis().set_visible(False)
    new_label = [xlabel_dict[int(i.get_text())] for i in ax1.get_xticklabels()]
    ax1.set_xticklabels(new_label)
    sns.despine(top=True, right=True)
    ax1.spines[['left']].set_visible(False)
    plt.show()
    for ext in ['.png', '.svg']:
        fig.savefig(os.path.join(save_path, f"first_last_wh_trial_prob_combined{ext}"))

def model_first_whisker_outcome(nwb_files, mode):
    rwd_wh_table = []
    nn_rwd_wh_table = []
    for nwb_file in nwb_files:
        print(" ")
        print(f"Session : {nwb_read.get_session_id(nwb_file)}")

        # Get trial table
        trial_table = nwb_read.get_trial_table(nwb_file)
        # Add block index
        if nwb_read.get_bhv_type_and_training_day_index(nwb_file)[0] in ["context", "whisker_context"]:
            switches = np.where(np.diff(trial_table.context.values[:]))[0]
            if len(switches) <= 1:
                block_length = switches[0] + 1
            else:
                block_length = min(np.diff(switches))
        else:
            switches = None
            block_length = 20
        trial_table['trial'] = trial_table.index
        trial_table['block'] = trial_table.loc[trial_table.early_lick == 0, 'trial'].transform(
            lambda x: x // block_length)

        # Get time spent in context after transition at first whisker trial
        rewarded_context_timestamps = nwb_read.get_behavioral_epochs_times(nwb_file, epoch_name='rewarded')
        non_rewarded_context_timestamps = nwb_read.get_behavioral_epochs_times(nwb_file, epoch_name='non-rewarded')

        # Get whisker table
        whisker_table = trial_table.loc[trial_table.trial_type == 'whisker_trial']
        whisker_table = whisker_table.reset_index(drop=True)

        # Add context transition column
        transitions = list(np.diff(whisker_table['context']))
        transitions.insert(0, 0)
        whisker_table['transitions'] = transitions

        # Keep only useful columns
        cols = ['start_time', 'lick_flag', 'context', 'transitions']
        filter_wh_table = whisker_table[cols]

        # Transitions to rewarded context
        rwd_filter_wh_table = filter_wh_table.loc[filter_wh_table.transitions == 1]
        if rewarded_context_timestamps[0][0] == 0:
            if len(rewarded_context_timestamps[0][1:]) == len(rwd_filter_wh_table) + 1:
                rwd_filter_wh_table['reward_epoch_start'] = rewarded_context_timestamps[0][1:-1]
            else:
                rwd_filter_wh_table['reward_epoch_start'] = rewarded_context_timestamps[0][1:]
        else:
            rwd_filter_wh_table['reward_epoch_start'] = rewarded_context_timestamps[0]
        rwd_filter_wh_table['time_in_reward'] = rwd_filter_wh_table['start_time'] - rwd_filter_wh_table[
            'reward_epoch_start']
        rwd_filter_wh_table['session_id'] = nwb_read.get_session_id(nwb_file)
        rwd_filter_wh_table['mouse_id'] = nwb_read.get_mouse_id(nwb_file)

        # Transitions to non-rewarded context
        nn_rwd_filter_wh_table = filter_wh_table.loc[filter_wh_table.transitions == -1]
        if non_rewarded_context_timestamps[0][0] == 0:
            if len(non_rewarded_context_timestamps[0][1:]) == len(nn_rwd_filter_wh_table) + 1:
                nn_rwd_filter_wh_table['non-reward_epoch_start'] = non_rewarded_context_timestamps[0][1:-1]
            else:
                nn_rwd_filter_wh_table['non-reward_epoch_start'] = non_rewarded_context_timestamps[0][1:]
        else:
            nn_rwd_filter_wh_table['non-reward_epoch_start'] = non_rewarded_context_timestamps[0]
        nn_rwd_filter_wh_table['time_in_non_reward'] = nn_rwd_filter_wh_table['start_time'] - nn_rwd_filter_wh_table[
            'non-reward_epoch_start']
        nn_rwd_filter_wh_table['session_id'] = nwb_read.get_session_id(nwb_file)
        nn_rwd_filter_wh_table['mouse_id'] = nwb_read.get_mouse_id(nwb_file)

        # Get regressors : number of each trial type before first whisker and time spent in new context
        trial_table_cols = ['trial_type', 'whisker_stim', 'auditory_stim', 'no_stim',
                            'reward_available', 'lick_flag', 'block', 'context', 'context_background']
        trial_table = trial_table[trial_table_cols]
        trial_table = trial_table.loc[trial_table.block > 0]
        rewarded_trial_table = trial_table.loc[trial_table.context == 1]
        non_rewarded_trial_table = trial_table.loc[trial_table.context == 0]

        trial_types = ['auditory_trial', 'no_stim_trial']
        lick_flags = [1, 0]
        comb = list(itertools.product(trial_types, lick_flags))
        n_rwd_blocks = len(rewarded_trial_table.block.unique())
        rwd_regressors_mat = np.zeros((n_rwd_blocks, len(comb)))
        removed_blocks = []
        for block_index, block in enumerate(rewarded_trial_table.block.unique()):
            block_df = rewarded_trial_table.loc[rewarded_trial_table.block == block]
            if 'whisker_trial' not in block_df.trial_type.unique():
                removed_blocks.append(block_index)
                continue
            first_whisker_index = block_df.index[block_df.whisker_stim == 1][0]
            ind_list = np.arange(block_df.index[0], first_whisker_index).tolist()
            ind_list = list(set(ind_list) & set(block_df.index))

            before_whisker_df = block_df.loc[ind_list]
            for trial_type_index, trial in enumerate(comb):
                n_trials = len(before_whisker_df.loc[(before_whisker_df.trial_type == trial[0]) & (before_whisker_df.lick_flag == trial[1])])
                rwd_regressors_mat[block_index, trial_type_index] = n_trials
        rwd_regressors_mat = np.delete(rwd_regressors_mat, removed_blocks, 0)

        rwd_filter_wh_table['n_auditory_hits'] = rwd_regressors_mat[:, 0].astype(int).tolist()
        rwd_filter_wh_table['n_auditory_miss'] = rwd_regressors_mat[:, 1].astype(int).tolist()
        rwd_filter_wh_table['n_false_alarms'] = rwd_regressors_mat[:, 2].astype(int).tolist()
        rwd_filter_wh_table['n_correct_rejections'] = rwd_regressors_mat[:, 3].astype(int).tolist()
        rwd_filter_wh_table['n_trials'] = (rwd_filter_wh_table['n_auditory_hits'] +
                                           rwd_filter_wh_table['n_auditory_miss'] +
                                           rwd_filter_wh_table['n_false_alarms'] +
                                           rwd_filter_wh_table['n_correct_rejections'])

        n_non_rwd_blocks = len(non_rewarded_trial_table.block.unique())
        non_rwd_regressors_mat = np.zeros((n_non_rwd_blocks, len(comb)))
        removed_blocks = []
        for block_index, block in enumerate(non_rewarded_trial_table.block.unique()):
            block_df = non_rewarded_trial_table.loc[non_rewarded_trial_table.block == block]
            if 'whisker_trial' not in block_df.trial_type.unique():
                removed_blocks.append(block_index)
                continue
            first_whisker_index = block_df.index[block_df.whisker_stim == 1][0]
            ind_list = np.arange(block_df.index[0], first_whisker_index).tolist()
            ind_list = list(set(ind_list) & set(block_df.index))

            before_whisker_df = block_df.loc[ind_list]

            for index, trial in enumerate(comb):
                n_trials = len(before_whisker_df.loc[(before_whisker_df.trial_type == trial[0]) & (
                            before_whisker_df.lick_flag == trial[1])])
                non_rwd_regressors_mat[block_index, index] = n_trials
        non_rwd_regressors_mat = np.delete(non_rwd_regressors_mat, removed_blocks, 0)

        nn_rwd_filter_wh_table['n_auditory_hits'] = non_rwd_regressors_mat[:, 0].astype(int).tolist()
        nn_rwd_filter_wh_table['n_auditory_miss'] = non_rwd_regressors_mat[:, 1].astype(int).tolist()
        nn_rwd_filter_wh_table['n_false_alarms'] = non_rwd_regressors_mat[:, 2].astype(int).tolist()
        nn_rwd_filter_wh_table['n_correct_rejections'] = non_rwd_regressors_mat[:, 3].astype(int).tolist()
        nn_rwd_filter_wh_table['n_trials'] = (nn_rwd_filter_wh_table['n_auditory_hits'] +
                                              nn_rwd_filter_wh_table['n_auditory_miss'] +
                                              nn_rwd_filter_wh_table['n_false_alarms'] +
                                              nn_rwd_filter_wh_table['n_correct_rejections'])

        rwd_wh_table.append(rwd_filter_wh_table)
        nn_rwd_wh_table.append(nn_rwd_filter_wh_table)

    rwd_wh_table = pd.concat(rwd_wh_table)
    nn_rwd_wh_table = pd.concat(nn_rwd_wh_table)

    # Add time bins
    rwd_times = rwd_wh_table.time_in_reward.values[:]
    rwd_wh_table['time_bin'] = np.digitize(rwd_times, bins=np.arange(0, 100, 10))
    nn_rwd_times = nn_rwd_wh_table.time_in_non_reward.values[:]
    nn_rwd_wh_table['time_bin'] = np.digitize(nn_rwd_times, bins=np.arange(0, 100, 10))

    # Add categorized time bins
    rwd_wh_table['whisker_period'] = ['early' if rwd_wh_table.time_bin.values[i] < 3 else 'late' for i in
                                      range(len(rwd_wh_table))]
    nn_rwd_wh_table['whisker_period'] = ['early' if nn_rwd_wh_table.time_bin.values[i] < 3 else 'late' for i in
                                         range(len(nn_rwd_wh_table))]

    # Add simpler flag for auditory hits
    rwd_wh_table['auditory_hits'] = rwd_wh_table['n_auditory_hits'] > 0
    nn_rwd_wh_table['auditory_hits'] = nn_rwd_wh_table['n_auditory_hits'] > 0

    # Keep only up to bin 5 (40-50sec)
    rwd_wh_table = rwd_wh_table.loc[rwd_wh_table.time_bin < 6]
    nn_rwd_wh_table = nn_rwd_wh_table.loc[nn_rwd_wh_table.time_bin < 6]

    # ------------------------------------------- GLM PART ------------------------------------------------------ #
    if mode == 'GLM':
        # Choose regressors
        # regressors = ['time_bin', 'n_auditory_hits', 'n_auditory_miss', 'n_false_alarms', 'n_correct_rejections',
        #               'n_trials']
        # regressors = ['time_bin', 'n_trials']
        regressors = ['whisker_period', 'auditory_hits']

        # Build regressors table
        rwd_wh_table_regressors = rwd_wh_table[regressors]
        rwd_wh_table_regressors = sm.add_constant(rwd_wh_table_regressors)
        nn_rwd_wh_table_regressors = nn_rwd_wh_table[regressors]
        nn_rwd_wh_table_regressors = sm.add_constant(nn_rwd_wh_table_regressors)

        # Fit the models
        binomial_model_rwd_wh = sm.GLM(rwd_wh_table.lick_flag.values[:], rwd_wh_table_regressors,
                                       family=sm.families.Binomial(link=sm.families.links.Logit()))
        binomial_model_rwd_wh_results = binomial_model_rwd_wh.fit()

        binomial_model_nn_rwd_wh = sm.GLM(nn_rwd_wh_table.lick_flag.values[:], nn_rwd_wh_table_regressors,
                                          family=sm.families.Binomial(link=sm.families.links.Logit()))
        binomial_model_nn_rwd_wh_results = binomial_model_nn_rwd_wh.fit()

        # Figures
        fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 6, sharey=True)
        sns.lineplot(data=rwd_wh_table, x='time_bin', y='lick_flag', ax=ax0)
        sns.lineplot(data=rwd_wh_table, x='n_auditory_hits', y='lick_flag', ax=ax1)
        sns.lineplot(data=rwd_wh_table, x='n_auditory_miss', y='lick_flag', ax=ax2)
        sns.lineplot(data=rwd_wh_table, x='n_false_alarms', y='lick_flag', ax=ax3)
        sns.lineplot(data=rwd_wh_table, x='n_correct_rejections', y='lick_flag', ax=ax4)
        sns.lineplot(data=rwd_wh_table, x='n_trials', y='lick_flag', ax=ax5)
        sns.despine(top=True, right=True)

        fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 6, sharey=True)
        sns.lineplot(data=nn_rwd_wh_table, x='time_bin', y='lick_flag', ax=ax0)
        sns.lineplot(data=nn_rwd_wh_table, x='n_auditory_hits', y='lick_flag', ax=ax1)
        sns.lineplot(data=nn_rwd_wh_table, x='n_auditory_miss', y='lick_flag', ax=ax2)
        sns.lineplot(data=nn_rwd_wh_table, x='n_false_alarms', y='lick_flag', ax=ax3)
        sns.lineplot(data=nn_rwd_wh_table, x='n_correct_rejections', y='lick_flag', ax=ax4)
        sns.lineplot(data=nn_rwd_wh_table, x='n_trials', y='lick_flag', ax=ax5)
        sns.despine(top=True, right=True)

        test_df = rwd_wh_table[['lick_flag', 'n_auditory_hits', 'time_bin']]
        test_df = test_df.groupby(['n_auditory_hits', 'time_bin']).agg(np.sum)
        test_df = test_df.reset_index()
        heatmap_data = test_df.pivot(index="n_auditory_hits", columns="time_bin", values="lick_flag")
        ax = sns.heatmap(heatmap_data)
        ax.invert_yaxis()

    # ------------------------------------------- ANOVA PART ------------------------------------------------------ #
    if mode == 'ANOVA':
        fig, axes = plt.subplot(2, 2, figsize=(10, 10))
        sns.lmplot(data=rwd_wh_table, x="auditory_hits", y="lick_flag", hue="whisker_period", logistic=True)
        sns.lmplot(data=rwd_wh_table, x="auditory_hits", y="lick_flag", logistic=True)
        sns.lmplot(data=rwd_wh_table, x='time_in_reward', hue="auditory_hits", y="lick_flag", logistic=True)
        sns.lmplot(data=nn_rwd_wh_table, x='time_in_non_reward', hue="auditory_hits", y="lick_flag", logistic=True)

        full_transition_table = pd.concat((rwd_wh_table, nn_rwd_wh_table))
        full_transition_table['time_after_transition'] = full_transition_table.time_in_reward.fillna(
            0) + full_transition_table.time_in_non_reward.fillna(0)
        sns.lmplot(data=full_transition_table, y="lick_flag", x='time_after_transition',
                   col='context', logistic=True)
        sns.lmplot(data=full_transition_table, y="lick_flag", x='time_after_transition', hue="auditory_hits",
                   col='context', logistic=True)

        fig, ax0 = plt.subplots(1, 1)
        sns.regplot(data=rwd_wh_table, x="time_in_reward", y="lick_flag", logistic=True, ax=ax0, color='green')
        sns.regplot(data=nn_rwd_wh_table, x="time_in_non_reward", y="lick_flag", logistic=True, ax=ax0, color='red')
        sns.despine()

        fig, ax0 = plt.subplots(1, 1)
        sns.regplot(data=rwd_wh_table, x="auditory_hits", y="lick_flag", logistic=True, ax=ax0, color='green')
        sns.regplot(data=nn_rwd_wh_table, x="auditory_hits", y="lick_flag", logistic=True, ax=ax0, color='red')
        sns.despine()

if __name__ == "__main__":

    config_file = "//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Sessions_list/context_contrast_expert_sessions_path.yaml"
    with open(config_file, 'r', encoding='utf8') as stream:
        config_dict = yaml.safe_load(stream)
    # sessions = config_dict['NWB_CI_LSENS']['Context_expert_sessions']
    # sessions = config_dict['NWB_CI_LSENS']['Context_contrast_expert']
    # files = [session[1] for session in sessions]
    save_path =  "//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/Context_behaviour_analysis_20240502/first_last_whisker_analysis"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    sessions = config_dict['Sessions path']
    # Analysis
    plot_first_whisker_outcome_against_time(nwb_files=sessions, save_path=save_path)
    # model_first_whisker_outcome(nwb_files=sessions, mode='ANOVA')


