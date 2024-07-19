import os
import yaml
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
from matplotlib.cm import get_cmap
from skimage.transform import rescale
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap


def ols_statistics(coefficients, confidence=0.95):
    """
    Calculate OLS statistics.
    """
    lower_percentile = (1 - confidence) / 2
    upper_percentile = 1 - lower_percentile

    # Calculate mean and standard error of coefficients
    coef_mean = np.mean(coefficients, axis=0)

    coef_std_error = np.std(coefficients, axis=0, ddof=1)

    # Calculate confidence intervals
    if coefficients.shape[0]==1:
        lower_bound = np.percentile(coefficients, lower_percentile * 100, axis=1)
        upper_bound = np.percentile(coefficients, upper_percentile * 100, axis=1)
    else:
        lower_bound = np.percentile(coefficients, lower_percentile * 100, axis=0)
        upper_bound = np.percentile(coefficients, upper_percentile * 100, axis=0)

    return coef_mean, coef_std_error, lower_bound, upper_bound


def plot_contrast_matrix(result_path):
    labels = ['A1', 'ALM', 'tjM1', 'tjS1', 'wM1', 'wM2', 'wS1', 'wS2', 'motor', 'sensory']
    matrix = np.ones([len(labels) - 2, len(labels) - 2])
    np.fill_diagonal(matrix, 0)
    sensory = np.zeros_like(matrix[0])
    sensory[[0, 6, 7]] = 1
    motor = np.where(sensory != 1, np.ones_like(sensory), 0)
    matrix = np.vstack([matrix, sensory, motor])
    fig, ax = plt.subplots()
    ax.imshow(matrix, cmap='Greys_r')
    ax.set_yticks(np.arange(matrix.shape[0]), labels, rotation=30)
    ax.set_xticks(np.arange(matrix.shape[1]), labels[:-2], rotation=30)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    fig.savefig(os.path.join(result_path, "contrast_matrix.png"))

def plot_single_session(result_path, decode, save_path, plot=False):

    norm = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5)
    data = pd.read_json(os.path.join(result_path, "results.json"))

    fig = plt.figure(figsize=(15, 7))
    fig.suptitle("Decoding over time")

    xlabel_dict = [f"{data.loc[i, 'start_frame']}-{data.loc[i, 'stop_frame']}" for i, row in data.iterrows()]
    xticks = list(range(0, 11))
    # xticks.append(11)
    data[['accuracy_mean', 'accuracy_std', 'accuracy_lb', 'accuracy_hb']] = data.apply(lambda x: ols_statistics(np.asarray(x['accuracy_shuffle'])), axis='columns', result_type='expand')
    data[['precision_mean', 'precision_std', 'precision_lb', 'precision_hb']] = data.apply(lambda x: ols_statistics(np.asarray(x['precision_shuffle'])), axis='columns', result_type='expand')
    data.loc[data['stop_frame']==-1, 'stop_frame'] = data['stop_frame'].max()
    data['chunk'] = data[['start_frame', 'stop_frame']].astype(str).agg('-'.join, axis=1)

    if decode == 'baseline':
        chunk_order = {'0-20': 0, '20-40': 1, '40-60': 2, '60-80': 3, '80-100': 4, '100-120': 5, '120-140': 6,
                       '140-160': 7, '160-180': 8, '180-200': 9, '0-200': 10}
    elif decode == 'stim':
        chunk_order = {'0-2': 0, '2-4': 1, '4-6': 2, '6-8': 3, '8-10': 4, '10-12': 5, '12-14': 6,
                       '14-16': 7, '16-18': 8, '18-20': 9, '0-20': 10}


    data['order'] = data['chunk'].apply(lambda x: chunk_order[x])
    labels = ['A1', 'ALM', 'tjM1', 'tjS1', 'wM1', 'wM2', 'wS1', 'wS2']

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i, metric in enumerate(['accuracy', 'precision']):
        ax[i].plot(data.loc[0:9, metric], marker='o', c='r')
        ax[i].plot(np.mean(np.stack(data.loc[0:9, f'{metric}_shuffle']), axis=1), marker='o', c='k')
        ax[i].fill_between(range(10), data.loc[0:9, f'{metric}_lb'], data.loc[0:9, f'{metric}_hb'], color='grey',
                           alpha=0.25)

        ax[i].scatter(11, data.loc[10, metric], marker='o', c='r')
        ax[i].scatter(11, np.mean(data.loc[10, f'{metric}_shuffle']), marker='o', c='k')
        ax[i].errorbar(11, np.mean(data.loc[10, f'{metric}_shuffle']),
                       yerr=[[np.mean(data.loc[10, f'{metric}_shuffle']) - data.loc[10, f'{metric}_lb']],
                             [data.loc[10, f'{metric}_hb'] - np.mean(data.loc[10, f'{metric}_shuffle'])]], ecolor='k')
        ax[i].set_xticks(xticks, xlabel_dict, rotation=30)
        ax[i].set_ylim(-0.05, 1.05)
        ax[i].set_ylabel(metric)
        ax[i].set_xlabel("Time bins")

    coefficients = np.stack(data['coefficients']).squeeze()
    im = ax[2].pcolor(coefficients.T, cmap='seismic', norm=norm)
    ax[2].set_xticks(np.arange(coefficients.shape[0]) + 0.5, xlabel_dict, rotation=30)
    ax[2].set_yticks(np.arange(coefficients.shape[1]) + 0.5)
    ax[2].set_yticklabels(labels)
    fig.colorbar(im, label='Coefficients')

    for ext in ['.png', '.svg']:
        fig.savefig(os.path.join(save_path, f"accuracy_and_precision_over_time{ext}"))
    if plot:
        fig.show()

    labels += ['motor', 'sensory']

    for item in labels:
        data[f"{item}_accuracy_delta"] = data[f'{item}_accuracy'] - data['accuracy']
        data[f"{item}_precision_delta"] = data[f'{item}_precision'] - data['precision']

    cm = plt.cm.get_cmap('rainbow', len(labels))
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for i, metric in enumerate(['accuracy', 'precision']):
        for j, item in enumerate(labels):
            sns.pointplot(data=data, x='chunk', y=f"{item}_{metric}", markers='o', scale=0.8, ax=ax[i], color=cm(j), label=item if i == 1 else None)
        ax[i].plot(np.mean(np.stack(data.loc[:, f'{metric}_shuffle']), axis=1), marker='o', c='k', label='shuffle' if i == 1 else None)
        ax[i].fill_between(range(11), data.loc[:, f'{metric}_lb'], data.loc[:, f'{metric}_hb'], color='grey',
                           alpha=0.25)
        ax[i].set_xticks(xticks, xlabel_dict, rotation=30)
        ax[i].set_ylim(-0.05, 1.05)
        ax[i].set_ylabel(metric)
        ax[i].set_xlabel("Time bins")
    fig.legend()

    for ext in ['.png', '.svg']:
        fig.savefig(os.path.join(save_path, f"accuracy_and_precision_leave_one_out{ext}"))
    if plot:
        fig.show()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for i, metric in enumerate(['accuracy', 'precision']):
        for j, item in enumerate(labels):
            sns.pointplot(data=data, x='chunk', y=f"{item}_{metric}_delta", markers='o', scale=0.8, ax=ax[i], color=cm(j), label=f"$\Delta${item}" if i == 1 else None)
        ax[i].set_xticks(xticks, xlabel_dict, rotation=30)
        ax[i].set_ylim(-0.55, 0.55)
        ax[i].set_ylabel(metric)
        ax[i].set_xlabel("Time bins")
    fig.legend()
    # fig.show()

    for ext in ['.png', '.svg']:
        fig.savefig(os.path.join(save_path, f"accuracy_and_precision_delta{ext}"))
    if plot:
        fig.show()

    return 0


def plot_multi_sessions(data, decode, result_path, classify_by, plot):

    norm = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5)
    save_path = os.path.join(result_path, classify_by)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    data.loc[data['stop_frame']==-1, 'stop_frame'] = data['stop_frame'].max()
    data['chunk'] = data[['start_frame', 'stop_frame']].astype(str).agg('-'.join, axis=1)

    if decode == 'baseline':
        chunk_order = {'0-20': 0, '20-40': 1, '40-60': 2, '60-80': 3, '80-100': 4, '100-120': 5, '120-140': 6,
                       '140-160': 7, '160-180': 8, '180-200': 9, '0-200': 10}
    elif decode == 'stim':
        chunk_order = {'0-2': 0, '2-4': 1, '4-6': 2, '6-8': 3, '8-10': 4, '10-12': 5, '12-14': 6,
                       '14-16': 7, '16-18': 8, '18-20': 9, '0-20': 10}

    data['order'] = data['chunk'].apply(lambda x: chunk_order[x])

    group_df = data.groupby(['mouse_id', 'chunk']).agg('mean')
    group_df['precision_mean'] = data.groupby(['mouse_id', 'chunk']).apply(lambda x: np.mean(np.stack(x[f'precision_shuffle'])))
    group_df['accuracy_mean'] = data.groupby(['mouse_id', 'chunk']).apply(lambda x: np.mean(np.stack(x[f'accuracy_shuffle'])))
    group_df['coefficients_mean'] = data.groupby(['mouse_id', 'chunk']).apply(lambda x: np.mean(np.stack(x[f'coefficients_shuffle'])))
    group_df = group_df.reset_index()
    group_df = group_df.sort_values('order')
    coefficients_group = []

    labels = ['A1', 'ALM', 'tjM1', 'tjS1', 'wM1', 'wM2', 'wS1', 'wS2', 'motor', 'sensory']
    cm = plt.cm.get_cmap('rainbow', len(labels))

    for mouse, group in data.groupby('mouse_id'):

        if not os.path.exists(os.path.join(save_path, mouse)):
            os.makedirs(os.path.join(save_path, mouse), exist_ok=True)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Accuracy and precision for mouse {mouse}")
        textstr = f"n = {group.session_id.unique().shape[0]} sessions"
        fig.text(0.8, 0.9, textstr, fontsize=14)

        fig1, ax1 = plt.subplots(1, 2, figsize=(10, 5))
        fig1.suptitle(f"Leave one out accuracy and precision for mouse {mouse}")
        textstr = f"n = {group.session_id.unique().shape[0]} sessions"
        fig1.text(0.8, 0.9, textstr, fontsize=14)

        fig2, ax2 = plt.subplots(1, 2, figsize=(10, 5))
        fig2.suptitle(f"Delta accuracy and precision for mouse {mouse}")
        textstr = f"n = {group.session_id.unique().shape[0]} sessions"
        fig2.text(0.8, 0.9, textstr, fontsize=14)

        group = group.sort_values('order')

        for i, metric in enumerate(['accuracy', 'precision']):
            group[f'{metric}_mean'] = group.apply(lambda x: np.mean(x[f'{metric}_shuffle']), axis=1, result_type='expand')

            sns.pointplot(data=group, x='chunk', y=metric, estimator='mean', errorbar=('ci', 95),
                          n_boot=1000, label='full model' if i==1 else None,
                          color='r', ax=ax[i])
            sns.pointplot(data=group, x='chunk', y=f"{metric}_mean", estimator='mean', errorbar=('ci', 95),
                          n_boot=1000, label='shuffle' if i==1 else None,
                          color='k', ax=ax[i])

            ax[i].set_ylim([-0.05, 1.05])
            plt.setp(ax[i].xaxis.get_majorticklabels(), rotation=30)

            for j, contrast in enumerate(labels):
                group[f"{contrast}_{metric}_delta"] = group[f'{contrast}_{metric}'] - group[metric]

                sns.pointplot(data=group, x='chunk', y=f"{contrast}_{metric}", estimator='mean', errorbar=('ci', 95),
                              n_boot=1000, label=contrast if i == 1 else None,
                              color=cm(j), ax=ax1[i])
                sns.pointplot(data=group, x='chunk', y=f"{metric}_mean", estimator='mean', errorbar=('ci', 95),
                              n_boot=1000, label='shuffle' if i == 0 and j == 0 else None,
                              color='k', ax=ax1[i])

                sns.pointplot(data=group, x='chunk', y=f"{contrast}_{metric}_delta", estimator='mean', errorbar=('ci', 95),
                              n_boot=1000, label=f"$\Delta${contrast}" if i == 1 else None,
                              color=cm(j), ax=ax2[i])

            ax1[i].set_ylim([-0.05, 1.05])
            plt.setp(ax1[i].xaxis.get_majorticklabels(), rotation=30)
            if metric == 'precision':
                ax2[i].set_ylim([-0.7, 0.7])
            else:
                ax2[i].set_ylim([-0.3, 0.3])

            plt.setp(ax2[i].xaxis.get_majorticklabels(), rotation=30)

        coefficients_mouse = group.groupby('chunk')['coefficients'].apply(lambda x: np.mean(np.stack(x), axis=0)).reset_index()
        coefficients_mouse['order'] = coefficients_mouse['chunk'].apply(lambda x: chunk_order[x])
        coefficients_mouse = coefficients_mouse.sort_values('order')
        coefficients = np.stack(coefficients_mouse['coefficients']).squeeze()
        coefficients_group += [coefficients]

        im = ax[2].pcolor(coefficients.T, cmap='seismic', norm=norm)
        ax[2].set_xticks(np.arange(coefficients.shape[0]) + 0.5, coefficients_mouse.chunk, rotation=30)
        ax[2].set_yticks(np.arange(coefficients.shape[1]) + 0.5)
        ax[2].set_yticklabels(labels[:-2])
        fig.colorbar(im, label='Coefficients')
        fig.legend()
        for ext in ['.png', '.svg']:
            fig.savefig(os.path.join(save_path, mouse, f"accuracy_and_precision{ext}"))
        if plot:
            fig.show()

        fig1.legend()
        for ext in ['.png', '.svg']:
            fig1.savefig(os.path.join(save_path, mouse, f"accuracy_and_precision_delta{ext}"))
        if plot:
            fig1.show()

        fig2.legend()
        for ext in ['.png', '.svg']:
            fig2.savefig(os.path.join(save_path, mouse, f"accuracy_and_precision_delta{ext}"))
        if plot:
            fig2.show()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Average accuracy and precision")
    textstr = f"n = {data.mouse_id.unique().shape[0]} mice\nn = {data.session_id.unique().shape[0]} sessions"
    fig.text(0.8, 0.9, textstr, fontsize=14)

    fig1, ax1 = plt.subplots(1, 2, figsize=(10, 5))
    fig1.suptitle(f"Leave one out accuracy and precision")
    fig1.text(0.8, 0.9, textstr, fontsize=14)

    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 5))
    fig2.suptitle(f"Delta accuracy and precision")
    fig2.text(0.8, 0.9, textstr, fontsize=14)

    for i, metric in enumerate(['accuracy', 'precision']):

        sns.pointplot(data=group_df, x='chunk', y=metric, estimator='mean', errorbar=('ci', 95),
                      n_boot=1000,
                      color='r', ax=ax[i])
        sns.pointplot(data=group_df, x='chunk', y=f"{metric}_mean", estimator='mean', errorbar=('ci', 95),
                      n_boot=1000,
                      color='k', ax=ax[i])

        ax[i].set_ylim([-0.05, 1.05])
        plt.setp(ax[i].xaxis.get_majorticklabels(), rotation=30)

        for j, contrast in enumerate(labels):
            group_df[f"{contrast}_{metric}_delta"] = group_df[f'{contrast}_{metric}'] - group_df[metric]

            sns.pointplot(data=group_df, x='chunk', y=f"{contrast}_{metric}", estimator='mean', errorbar=('ci', 95),
                          n_boot=1000, label=contrast if i == 0 else None,
                          color=cm(j), ax=ax1[i])
            sns.pointplot(data=group_df, x='chunk', y=f"{metric}_mean", estimator='mean', errorbar=('ci', 95),
                          n_boot=1000, label='shuffle' if i == 0 and j == 0 else None,
                          color='k', ax=ax1[i])

            sns.pointplot(data=group_df, x='chunk', y=f"{contrast}_{metric}_delta", estimator='mean', errorbar=('ci', 95),
                          n_boot=1000, label=f"$\Delta${contrast}" if i == 0 else None,
                          color=cm(j), ax=ax2[i])

        ax1[i].set_ylim([-0.05, 1.05])
        plt.setp(ax1[i].xaxis.get_majorticklabels(), rotation=30)

        if metric == 'precision':
            ax2[i].set_ylim([-0.7, 0.7])
        else:
            ax2[i].set_ylim([-0.3, 0.3])

        plt.setp(ax2[i].xaxis.get_majorticklabels(), rotation=30)

    coefficients_group = np.stack(coefficients_group)
    im = ax[2].pcolor(np.mean(coefficients_group, axis=0).T, cmap='seismic', norm=norm)
    ax[2].set_xticks(np.arange(coefficients_group.shape[1]) + 0.5, coefficients_mouse.chunk, rotation=30)
    ax[2].set_yticks(np.arange(coefficients_group.shape[2]) + 0.5)
    ax[2].set_yticklabels(labels[:-2])
    fig.colorbar(im, label='Coefficients')
    for ext in ['.png', '.svg']:
        fig.savefig(os.path.join(save_path, f"accuracy_and_precision{ext}"))
    if plot:
        fig.show()

    fig1.legend()
    for ext in ['.png', '.svg']:
        fig1.savefig(os.path.join(save_path, f"accuracy_and_precision_leave_one_out{ext}"))
    if plot:
        fig1.show()

    fig2.legend()
    for ext in ['.png', '.svg']:
        fig2.savefig(os.path.join(save_path, f"accuracy_and_precision_delta{ext}"))
    if plot:
        fig2.show()

    return group_df


def main(config_dict, classify_by, decode, result_folder):
    plot_contrast_matrix(result_folder)
    for nwb_path in config_dict['Session path']:
        session = nwb_path.split("\\")[-1].split(".")[0]
        print(f"Session: {session}")
        animal_id = session.split("_")[0]
        result_path = os.path.join(result_folder, decode, animal_id, session, f"{classify_by}_decoding")
        save_path = os.path.join(result_path, classify_by)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # plot_single_session(result_path,
        #                     decode=decode,
        #                     save_path=save_path,
        #                     plot=False)

    all_files = glob.glob(os.path.join(result_folder, decode, "**", "*", f"{classify_by}_decoding", "results.json").replace("\\", "/"))
    data = []
    for file in all_files:
        df = pd.read_json(file)
        data.append(df)
    data = pd.concat(data, axis=0, ignore_index=True)
    group_df = plot_multi_sessions(data,
                                   decode=decode,
                                   result_path=os.path.join(result_folder, decode),
                                   classify_by=classify_by,
                                   plot=False)

    group_df.to_json(os.path.join(result_folder, f'{classify_by}_combined_results.json'))


if __name__ == "__main__":

    group = "context_contrast_widefield"
    decode = 'stim'
    result_folder = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/widefield_decoding_area_gcamp_experts"
    config_file = f"//sv-nas1.rcp.epfl.ch/Petersen-Lab/z_LSENS/Share/Pol_Bech/Session_list/context_sessions_gcamp_expert.yaml"
    # config_file = r"M:\analysis\Pol_Bech\Sessions_list\context_contrast_expert_widefield_sessions_path.yaml"

    with open(config_file, 'r', encoding='utf8') as stream:
        config_dict = yaml.safe_load(stream)
    for decode in ['stim']:
        for classify_by in ['lick']:
            print(f"Processing {classify_by}")
            main(config_dict, classify_by, decode, result_folder)