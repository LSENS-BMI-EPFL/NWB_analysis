import nwb_wrappers.nwb_reader_functions as nwb_read
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yaml

pd.options.mode.chained_assignment = None  # default='warn'


def plot_first_whisker_outcome_against_time(nwb_files):
    rwd_wh_table = []
    nn_rwd_wh_table = []
    for nwb_file in nwb_files:
        print(" ")
        print(f"Session : {nwb_read.get_session_id(nwb_file)}")
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
            rwd_filter_wh_table['reward_epoch_start'] = rewarded_context_timestamps[0]
        rwd_filter_wh_table['time_in_reward'] = rwd_filter_wh_table['start_time'] - rwd_filter_wh_table[
            'reward_epoch_start']
        rwd_filter_wh_table['session_id'] = nwb_read.get_session_id(nwb_file)
        rwd_filter_wh_table['mouse_id'] = nwb_read.get_mouse_id(nwb_file)

        # Transitions to non-rewarded
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

        rwd_wh_table.append(rwd_filter_wh_table)
        nn_rwd_wh_table.append(nn_rwd_filter_wh_table)

    rwd_wh_table = pd.concat(rwd_wh_table)
    nn_rwd_wh_table = pd.concat(nn_rwd_wh_table)

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

    # Figure 2 : with separated hit and miss
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 8))
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

    # Figure 3 : lick probability by bin
    rwd_times = rwd_wh_table.time_in_reward.values[:]
    rwd_wh_table['time_bin'] = np.digitize(rwd_times, bins=np.arange(0, 100, 10))
    cols = ['time_bin', 'lick_flag']
    bin_averaged_data_rwd = rwd_wh_table[cols].groupby('time_bin', as_index=False).agg(np.mean)

    nn_rwd_times = nn_rwd_wh_table.time_in_non_reward.values[:]
    nn_rwd_wh_table['time_bin'] = np.digitize(nn_rwd_times, bins=np.arange(0, 100, 10))
    cols = ['time_bin', 'lick_flag']
    bin_averaged_data_nn_rwd = nn_rwd_wh_table[cols].groupby('time_bin', as_index=False).agg(np.mean)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 8))
    sns.pointplot(data=bin_averaged_data_rwd, x='time_bin', y='lick_flag', color='green', ax=ax0)
    sns.despine(top=True, right=True)
    ax0.set_xlabel('Time after transition')
    ax0.set_title('To rewarded context')
    ax0.set_ylabel('Lick probability')
    ax0.set_ylim(-0.05, 1.05)

    xlabel_dict = {1: '0-10', 2: '10-20', 3: '20-30', 4: '30-40',
                   5: '40-50', 6: '50-60', 7: '60-70', 8: '70-80', 9: '80-90'}
    new_label = [xlabel_dict[int(i.get_text())] for i in ax0.get_xticklabels()]
    ax0.set_xticklabels(new_label)

    sns.pointplot(data=bin_averaged_data_nn_rwd, x='time_bin', y='lick_flag', color='red', ax=ax1)
    ax1.set_xlabel('Time after transition')
    ax1.set_title('To non-rewarded context')
    ax1.set_ylabel('Lick probability')
    ax1.set_ylim(-0.05, 1.05)

    new_label = [xlabel_dict[int(i.get_text())] for i in ax1.get_xticklabels()]
    ax1.set_xticklabels(new_label)

    plt.show()

    # Figure 4 :
    bin_averaged_data_rwd['Context transition'] = 'To rewarded'
    bin_averaged_data_nn_rwd['Context transition'] = 'To non-rewarded'
    bin_averaged_data = pd.concat([bin_averaged_data_rwd, bin_averaged_data_nn_rwd])
    fig, ax0 = plt.subplots(1, 1, figsize=(8, 8))
    sns.pointplot(data=bin_averaged_data, x='time_bin', y='lick_flag', hue='Context transition',
                  palette=['green', 'red'],
                  ax=ax0)
    xlabel_dict = {1: '0-10', 2: '10-20', 3: '20-30', 4: '30-40',
                   5: '40-50', 6: '50-60', 7: '60-70', 8: '70-80', 9: '80-90'}
    new_label = [xlabel_dict[int(i.get_text())] for i in ax0.get_xticklabels()]
    ax0.set_xticklabels(new_label)
    ax0.set_xlabel('Time after transition')
    ax0.set_ylabel('Lick probability')
    ax0.axhline(y=0.5, xmin=0, xmax=1, color='k', linestyle='--')
    sns.despine(top=True, right=True)
    ax0.set_ylim(-0.05, 1.05)

    plt.show()


config_file = "C:/Users/rdard/Documents/Codes/Python_Codes/CICADA_gitlab/cicada/src/cicada/config/group.yaml"
with open(config_file, 'r', encoding='utf8') as stream:
    config_dict = yaml.safe_load(stream)
sessions = config_dict['NWB_CI_LSENS']['Context_expert_sessions']
files = [session[1] for session in sessions]
plot_first_whisker_outcome_against_time(nwb_files=files)


