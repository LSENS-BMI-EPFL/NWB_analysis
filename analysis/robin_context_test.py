import nwb_wrappers.nwb_reader_functions as nwb_read
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yaml


def plot_first_whisker_outcome_against_time(nwb_files):
    first_whisker_table = []
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
        filter_wh_table = filter_wh_table.loc[filter_wh_table.transitions == 1]
        if rewarded_context_timestamps[0][0] == 0:
            if len(rewarded_context_timestamps[0][1:]) == len(filter_wh_table) + 1:
                filter_wh_table['reward_epoch_start'] = rewarded_context_timestamps[0][1:-1]
            else:
                filter_wh_table['reward_epoch_start'] = rewarded_context_timestamps[0][1:]
        else:
            filter_wh_table['reward_epoch_start'] = rewarded_context_timestamps[0]
        filter_wh_table['time_in_reward'] = filter_wh_table['start_time'] - filter_wh_table['reward_epoch_start']
        filter_wh_table['session_id'] = nwb_read.get_session_id(nwb_file)
        filter_wh_table['mouse_id'] = nwb_read.get_mouse_id(nwb_file)

        # Transitions to non-rewarded
        # filter_wh_table = filter_wh_table.loc[filter_wh_table.transitions == -1]
        # if non_rewarded_context_timestamps[0][0] == 0:
        #     if len(non_rewarded_context_timestamps[0][1:]) == len(filter_wh_table) + 1:
        #         filter_wh_table['non-reward_epoch_start'] = non_rewarded_context_timestamps[0][1:-1]
        #     else:
        #         filter_wh_table['non-reward_epoch_start'] = non_rewarded_context_timestamps[0][1:]
        # else:
        #     filter_wh_table['non-reward_epoch_start'] = non_rewarded_context_timestamps[0]
        # filter_wh_table['time_in_non_reward'] = filter_wh_table['start_time'] - filter_wh_table['non-reward_epoch_start']
        # filter_wh_table['session_id'] = nwb_read.get_session_id(nwb_file)
        # filter_wh_table['mouse_id'] = nwb_read.get_mouse_id(nwb_file)

        first_whisker_table.append(filter_wh_table)

    first_whisker_table = pd.concat(first_whisker_table)

    # times = first_whisker_table.time_in_non_reward.values[:]
    # first_whisker_table['time_bin'] = np.digitize(times, bins=np.arange(0, 100, 10))
    # cols = ['time_bin', 'lick_flag']
    # bin_averaged_data = first_whisker_table[cols].groupby('time_bin').agg(np.mean)

    # print(f"first_whisker_table: {first_whisker_table}")
    # avg_time_table = first_whisker_table.groupby('lick_flag').agg(np.sum)
    # print(f"avg_time_table: {avg_time_table}")
    # print(f"n miss : {len(first_whisker_table.loc[first_whisker_table.lick_flag == 0])}")
    # print(f"n hit : {len(first_whisker_table.loc[first_whisker_table.lick_flag == 1])}")

    # fig, ax0 = plt.subplots(1, 1,  figsize=(8, 8))
    # sns.stripplot(data=first_whisker_table, x='lick_flag', y='time_in_non_reward', color='black', ax=ax0)
    # sns.pointplot(data=first_whisker_table, x='lick_flag', y='time_in_non_reward', color='red',
    #               estimator=np.mean, errorbar=('ci', 95), n_boot=1000,
    #               ax=ax0)
    # sns.despine(top=True, right=True)
    # ax0.set_ylabel('Time after transition to non-rewarded context (s)')
    # ax0.set_xlabel('Outcome of first non-rewarded whisker trial')
    # plt.show()

    # fig, ax0 = plt.subplots(1, 1, figsize=(8, 8))
    # sns.pointplot(data=bin_averaged_data, x='time_bin', y='lick_flag', color='red',
    #               ax=ax0)
    # sns.despine(top=True, right=True)
    # ax0.set_xlabel('Time bin after transition to non-rewarded context at first whisker trial (s)')
    # ax0.set_ylabel('Lick probability')
    # ax0.set_ylim(-0.05, 1.05)
    # xlabel_dict = {1: '0-10', 2: '10-20', 3: '20-30', 4: '30-40',
    #                5: '40-50', 6: '50-60', 7: '60-70', 8: '70-80', 9: '80-90'}
    # new_label = [xlabel_dict[int(i.get_text())] for i in ax0.get_xticklabels()]
    # ax0.set_xticklabels(new_label)
    # plt.show()

    fig, ax0 = plt.subplots(1, 1,  figsize=(8, 8))
    sns.boxplot(data=first_whisker_table, y='time_in_reward', color='green', ax=ax0)
    sns.despine(top=True, right=True)
    ax0.set_yticks(np.arange(0, 90, 10))
    ax0.set_ylabel('Time after context transition (s)')
    ax0.set_xlabel('Transition to rewarded context')
    plt.show()


config_file = "C:/Users/rdard/Documents/python_repos/CICADA/cicada/src/cicada/config/group.yaml"
with open(config_file, 'r', encoding='utf8') as stream:
    config_dict = yaml.safe_load(stream)
sessions = config_dict['NWB_CI_LSENS']['Context_expert_sessions']
files = [session[1] for session in sessions]
plot_first_whisker_outcome_against_time(nwb_files=files)


