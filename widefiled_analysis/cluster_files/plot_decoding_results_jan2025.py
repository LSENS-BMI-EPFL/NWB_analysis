import os
import sys
sys.path.append(os.getcwd())
import yaml
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from PIL import Image
from matplotlib.cm import get_cmap
from itertools import combinations
from skimage.transform import rescale
from scipy.stats import ttest_rel, sem, norm
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
import nwb_utils.utils_behavior as bhv_utils


def transition(row, classify_by):
    if classify_by == 'context':
        if row.context == 1 and row.trial_count in [-1, -2, -3, -4]:
            return 'To non-rewarded'
        elif row.context == 0 and row.trial_count in [-1, -2, -3, -4]:
            return 'To rewarded'
        elif row.context == 1 and row.trial_count in [1, 2, 3, 4]:
            return 'To rewarded'
        elif row.context == 0 and row.trial_count in [1, 2, 3, 4]:
            return 'To non-rewarded'

    elif classify_by == 'tone':
        if row.context_background == 'pink' and row.trial_count in [-1, -2, -3, -4]:
            return 'To brown'
        elif row.context_background == 'brown' and row.trial_count in [-1, -2, -3, -4]:
            return 'To pink'
        elif row.context_background == 'pink' and row.trial_count in [1, 2, 3, 4]:
            return 'To pink'
        elif row.context_background == 'brown' and row.trial_count in [1, 2, 3, 4]:
            return 'To brown'


def get_wf_scalebar(scale = 1, plot=False, savepath=None):
    file = r"M:\analysis\Pol_Bech\Parameters\Widefield\wf_scalebars\reference_grid_20240314.tif"
    grid = Image.open(file)
    im = np.array(grid)
    im = im.reshape(int(im.shape[0] / 2), 2, int(im.shape[1] / 2), 2).mean(axis=1).mean(axis=2) # like in wf preprocessing
    x = [62*scale, 167*scale]
    y = [162*scale, 152*scale]
    fig, ax = plt.subplots()
    ax.imshow(rescale(im, scale, anti_aliasing=False))
    ax.plot(x, y, c='r')
    ax.plot(x, [y[0], y[0]], c='k')
    ax.plot([x[1], x[1]], y, c='k')
    ax.text(x[0] + int((x[1] - x[0]) / 2), 175*scale, f"{x[1] - x[0]} px")
    ax.text(170*scale, 168*scale, f"{np.abs(y[1] - y[0])} px")
    c = np.sqrt((x[1] - x[0]) ** 2 + (y[0] - y[1]) ** 2)
    ax.text(100*scale, 145*scale, f"{round(c)} px")
    ax.text(200*scale, 25*scale, f"{round(c / 6)} px/mm", color="r")
    if plot:
        fig.show()
    if savepath:
        fig.savefig(savepath+rf'\wf_scalebar_scale{scale}.png')
    return round(c / 6)


def get_allen_ccf(bregma = (528, 315), root=r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Robin_Dard\Parameters\Widefield\allen_brain"):
    """Find in utils the AllenSDK file to generate the npy files"""

     ## all images aligned to 240,175 at widefield video alignment, after expanding image, goes to this. Set manually.
    iso_mask = np.load(root + r"\allen_isocortex_tilted_500x640.npy")
    atlas_mask = np.load(root + r"\allen_brain_tilted_500x640.npy")
    bregma_coords = np.load(root + r"\allen_bregma_tilted_500x640.npy")

    displacement_x = int(bregma[0] - np.round(bregma_coords[0] + 20))
    displacement_y = int(bregma[1] - np.round(bregma_coords[1]))

    margin_y = atlas_mask.shape[0]-np.abs(displacement_y)
    margin_x = atlas_mask.shape[1]-np.abs(displacement_x)

    if displacement_y >= 0 and displacement_x >= 0:
        atlas_mask[displacement_y:, displacement_x:] = atlas_mask[:margin_y, :margin_x]
        atlas_mask[:displacement_y, :] *= 0
        atlas_mask[:, :displacement_x] *= 0

        iso_mask[displacement_y:, displacement_x:] = iso_mask[:margin_y, :margin_x]
        iso_mask[:displacement_y, :] *= 0
        iso_mask[:, :displacement_x] *= 0

    elif displacement_y < 0 and displacement_x>=0:
        atlas_mask[:displacement_y, displacement_x:] = atlas_mask[-margin_y:, :margin_x]
        atlas_mask[displacement_y:, :] *= 0
        atlas_mask[:, :displacement_x] *= 0

        iso_mask[:displacement_y, displacement_x:] = iso_mask[-margin_y:, :margin_x]
        iso_mask[displacement_y:, :] *= 0
        iso_mask[:, :displacement_x] *= 0

    elif displacement_y >= 0 and displacement_x<0:
        atlas_mask[displacement_y:, :displacement_x] = atlas_mask[:margin_y, -margin_x:]
        atlas_mask[:displacement_y, :] *= 0
        atlas_mask[:, displacement_x:] *= 0

        iso_mask[displacement_y:, :displacement_x] = iso_mask[:margin_y, -margin_x:]
        iso_mask[:displacement_y, :] *= 0
        iso_mask[:, displacement_x:] *= 0

    else:
        atlas_mask[:displacement_y, :displacement_x] = atlas_mask[-margin_y:, -margin_x:]
        atlas_mask[displacement_y:, :] *= 0
        atlas_mask[:, displacement_x:] *= 0

        iso_mask[:displacement_y, :displacement_x] = iso_mask[-margin_y:, -margin_x:]
        iso_mask[displacement_y:, :] *= 0
        iso_mask[:, displacement_x:] *= 0

    return iso_mask, atlas_mask, bregma_coords


def get_colormap(cmap='hotcold'):
    hotcold = ['#aefdff', '#60fdfa', '#2adef6', '#2593ff', '#2d47f9', '#3810dc', '#3d019d',
               '#313131',
               '#97023d', '#d90d39', '#f8432d', '#ff8e25', '#f7da29', '#fafd5b', '#fffda9']

    cyanmagenta = ['#00FFFF', '#FFFFFF', '#FF00FF']

    if cmap == 'cyanmagenta':
        cmap = LinearSegmentedColormap.from_list("Custom", cyanmagenta)

    elif cmap == 'whitemagenta':
        cmap = LinearSegmentedColormap.from_list("Custom", ['#FFFFFF', '#FF00FF'])

    elif cmap == 'hotcold':
        cmap = LinearSegmentedColormap.from_list("Custom", hotcold)

    elif cmap == 'grays':
        cmap = get_cmap('Greys')

    elif cmap == 'viridis':
        cmap = get_cmap('viridis')

    elif cmap == 'blues':
        cmap = get_cmap('Blues')

    elif cmap == 'magma':
        cmap = get_cmap('magma')

    elif cmap == 'seismic':
        cmap = get_cmap('seismic')

    else:
        cmap = get_cmap(cmap)

    cmap.set_bad(color='k', alpha=0.1)

    return cmap


def plot_single_frame(data, title, fig=None, ax=None, norm=True, colormap='seismic', save_path=None, vmin=-0.5, vmax=0.5, show=False):
    bregma = (488, 290)
    scale = 4
    scalebar = get_wf_scalebar(scale=scale)
    iso_mask, atlas_mask, allen_bregma = get_allen_ccf(bregma)

    if fig is None and ax is None:
        fig, ax = plt.subplots(1, figsize=(7, 7))
        fig.suptitle(title)
        new_fig = True
    else:
        new_fig = False

    cmap = get_colormap(colormap)
    cmap.set_bad(color='white')

    if norm:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm= Normalize(vmin=vmin, vmax=vmax)

    single_frame = np.rot90(rescale(data, scale, anti_aliasing=False))
    single_frame = np.pad(single_frame, [(0, 650 - single_frame.shape[0]), (0, 510 - single_frame.shape[1])],
                          mode='constant', constant_values=np.nan)

    mask = np.pad(iso_mask, [(0, 650 - iso_mask.shape[0]), (0, 510 - iso_mask.shape[1])], mode='constant',
                  constant_values=np.nan)
    single_frame = np.where(mask > 0, single_frame, np.nan)

    im = ax.imshow(single_frame, norm=norm, cmap=cmap)
    ax.contour(atlas_mask, levels=np.unique(atlas_mask), colors='gray',
                       linewidths=1)
    ax.contour(iso_mask, levels=np.unique(np.round(iso_mask)), colors='black',
                       linewidths=2, zorder=2)
    ax.scatter(bregma[0], bregma[1], marker='+', c='k', s=100, linewidths=2,
                       zorder=3)
    ax.hlines(25, 25, 25 + scalebar * 3, linewidth=2, colors='k')
    # ax.text(50, 100, "3 mm", size=10)
    ax.set_title(f"{title}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.axes[1].set(ylabel="Coefficients")
    fig.tight_layout()

    if new_fig and save_path is not None:
        fig.savefig(save_path + ".png")
        fig.savefig(save_path + ".svg")
    if show:
        fig.show()
    if new_fig == False:
        return fig, ax


def plot_trialbased_accuracy(data, bhv_data, classify_by, result_path):

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    data['match'] = data.prediction.eq(data.true)
    data = data.groupby(by=['state', 'mouse_id', 'session_id', 'data', 'trial', 'true']).match.mean().round(2).reset_index()
    df = data.pivot(index=['state', 'mouse_id', 'session_id', 'trial'], columns=['data'])['match'].reset_index()
    df = df.sort_values(['state', 'session_id', 'trial']).reset_index(drop=True)
    bhv_data = bhv_data.sort_values(['state', 'session_id', 'trial_id']).reset_index(drop=True)

    if df.session_id.eq(bhv_data.session_id).sum() != df.shape[0] & df.trial.eq(bhv_data.trial_id).sum() != df.shape[0]:
        print('Error')
        return

    bhv_data['block_id'] = bhv_data.groupby('session_id', sort=False)['context'].apply(lambda x: np.abs(np.diff(x, prepend=x.iloc[0])).cumsum()).explode('context')
    df['trial_type'] = bhv_data.trial_type
    df['context'] = bhv_data.context
    df['block_id'] = bhv_data.block_id
    df['context_background'] = bhv_data.context_background

    df['trial_count'] = np.empty(len(df), dtype=int)
    df.loc[df.trial_type == 'whisker_trial', 'trial_count'] = df.loc[df.trial_type == 'whisker_trial'].groupby(
        ['session_id', 'block_id'], sort=False).cumcount()
    df.loc[df.trial_type == 'auditory_trial', 'trial_count'] = df.loc[df.trial_type == 'auditory_trial'].groupby(
        ['session_id', 'block_id'], sort=False).cumcount()
    df.loc[df.trial_type == 'no_stim_trial', 'trial_count'] = df.loc[df.trial_type == 'no_stim_trial'].groupby(
        ['session_id', 'block_id'], sort=False).cumcount()

    value_to_avg = 'context_background' if classify_by=='tone' else 'context'

    whisker_trials = df.loc[df.trial_type=='whisker_trial'].melt(id_vars=['mouse_id', 'session_id', 'state', value_to_avg, 'block_id', 'trial_count'],
            value_vars=['all_trial_shuffle', 'block_shuffle', 'data', 'within_block_shuffle'])
    avg_whisker = whisker_trials.groupby(by=['state', 'mouse_id', 'session_id', value_to_avg, 'trial_count', 'data'])['value'].agg(
        'mean').reset_index()

    total_avg = avg_whisker.groupby(by=['state', 'mouse_id', value_to_avg, 'trial_count', 'data'])['value'].agg(
        'mean').reset_index()

    total_avg['trial_count'] = total_avg.trial_count.map({0: 1, 1: 2, 2: 3, 3: 4, 4: -4, 5: -3, 6: -2, 7: -1})
    total_avg['transition'] = total_avg.apply(lambda x: transition(x, classify_by), axis=1)

    hue_order = ['To rewarded', 'To non-rewarded'] if classify_by == 'context' else ['To pink', 'To brown']
    palette = ['green', 'red'] if classify_by == 'context' else ['#ffd6e9', '#946656']

    for shuffle in ['all_trial_shuffle', 'block_shuffle', 'within_block_shuffle']:
        results_accuracy = []
        results_v_shuffle = []

        fig, ax = plt.subplots(1,2, figsize=(7,5))
        fig.suptitle("Accuracy aligned to transition")
        for i, state in enumerate(['naive', 'expert']):
            sns.pointplot(total_avg.loc[(total_avg.data == 'data') & (total_avg.state == state)], x='trial_count', y='value', hue='transition', hue_order=hue_order,
                          palette=palette, estimator='mean', errorbar=('ci', 95), ax=ax[i])
            sns.pointplot(total_avg.loc[(total_avg.data == shuffle) & (total_avg.state == state)], x='trial_count', y='value', hue='transition', hue_order=hue_order,
                          palette=['gray', 'black'], estimator='mean', errorbar=('ci', 95), ax=ax[i])
            ax[i].set_title(state)
            ax[i].vlines(3.5, 0, 1, 'gray', '--')
            ax[i].set_ylabel("Accuracy")
            ax[i].set_ylim([-0.05, 1.05])
            ax[i].set_xlabel('Whisker trials')
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            if i == 0:
                ax[i].legend().set_visible(False)

            for trial in total_avg.trial_count.unique():
                for context_change in total_avg.transition.unique():
                    sample_1 = total_avg.loc[(total_avg.data == 'data') & (total_avg.state == state) & (total_avg.trial_count == trial) & (total_avg.transition == context_change), 'value']
                    sample_2 = total_avg.loc[(total_avg.data == shuffle) & (total_avg.state == state) & (total_avg.trial_count == trial) & (total_avg.transition == context_change), 'value']
                    t, p = ttest_rel(sample_1, sample_2)
                    p_corr = p * total_avg.trial_count.unique().shape[0]
                    results_partial = {
                        'state': state,
                        'transition': context_change,
                        'combination': f"data vs {shuffle}",
                        'trial': trial,
                        'df': sample_1.shape[0] - 1,
                        'mean_1': sample_1.mean(),
                        'std_1': sample_1.std(),
                        'mean_2': sample_2.mean(),
                        'std_2': sample_2.std(),
                        't': t,
                        'p': p,
                        'p_corr': p_corr,
                        'd': t / np.sqrt(sample_1.shape[0]),
                        'significant': "True" if p_corr < 0.05 else "False"
                    }
                    results_v_shuffle += [results_partial]

        for ext in ['.png', '.svg']:
            fig.savefig(os.path.join(result_path, f"{shuffle}_trialbased_decoding{ext}"))


        fig, ax = plt.subplots(1, 2, figsize=(5, 5))
        fig.suptitle("Last vs First whisker trial")
        for i, state in enumerate(['naive', 'expert']):
            sns.pointplot(
                total_avg.loc[
                    (total_avg.data == 'data') & (total_avg.state == state) & (total_avg.trial_count.isin([1, -1]))],
                x='trial_count', y='value', hue='transition', hue_order=hue_order,
                palette=palette, estimator='mean', errorbar=('ci', 95), ax=ax[i])
            sns.pointplot(
                total_avg.loc[
                    (total_avg.data == shuffle) & (total_avg.state == state) & (total_avg.trial_count.isin([1, -1]))],
                x='trial_count', y='value', hue='transition', hue_order=hue_order,
                palette=['gray', 'black'], estimator='mean', errorbar=('ci', 95), ax=ax[i])
            ax[i].set_title(state)
            ax[i].vlines(0.5, 0, 1, 'gray', '--')
            ax[i].set_ylabel("Accuracy")
            ax[i].set_ylim([-0.05, 1.05])
            ax[i].set_xlim([-0.25, 1.25])
            ax[i].set_xlabel('Whisker trials')
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            if i == 0:
                ax[i].legend().set_visible(False)

            for k, change in enumerate(total_avg.transition.unique()):
                for d_type in ['data', shuffle]:
                    sample_1 = total_avg.loc[(total_avg.data == d_type) & (total_avg.state == state) & (total_avg.trial_count == -1) & (total_avg.transition == change), 'value']
                    sample_2 = total_avg.loc[(total_avg.data == d_type) & (total_avg.state == state) & (total_avg.trial_count == 1) & (total_avg.transition == change), 'value']

                    t, p = ttest_rel(sample_1, sample_2)
                    p_corr = p * 4
                    results_partial = {
                        'state': state,
                        'data_type': d_type,
                        'combination': f"{change} last vs {change} first",
                        'df': sample_1.shape[0] - 1,
                        'mean_1': sample_1.mean(),
                        'std_1': sample_1.std(),
                        'mean_2': sample_2.mean(),
                        'std_2': sample_2.std(),
                        't': t,
                        'p': p,
                        'p_corr': p_corr,
                        'd': t / np.sqrt(sample_1.shape[0]),
                        'significant': "True" if p_corr < 0.05 else "False"
                    }

                    results_accuracy += [results_partial]

        results_v_shuffle = pd.DataFrame(results_v_shuffle)
        results_v_shuffle.to_csv(os.path.join(result_path, f'accuracy_data_vs_{shuffle}.csv'))

        results_accuracy = pd.DataFrame(results_accuracy)
        results_accuracy.to_csv(os.path.join(result_path, f'accuracy_firstvslast_{shuffle}.csv'))

        fig.tight_layout()
        for ext in ['.png', '.svg']:
            fig.savefig(os.path.join(result_path, f"{shuffle}_trialbased_decoding_first_v_last{ext}"))

    if classify_by == 'context':
        total_avg.loc[total_avg.context == 0, 'value'] = 1-total_avg.loc[total_avg.context == 0, 'value']
    elif classify_by == 'tone':
        total_avg.loc[total_avg.context_background == 'brown', 'value'] = 1 - total_avg.loc[total_avg.context_background == 'brown', 'value']

    for n, shuffle in enumerate(['all_trial_shuffle', 'block_shuffle', 'within_block_shuffle']):
        results_p_data_shuffle = []
        results_p_firstlast = []

        fig, ax = plt.subplots(1,2, figsize=(7,5))
        fig.suptitle("Accuracy aligned to transition")
        for i, state in enumerate(['naive', 'expert']):
            sns.pointplot(total_avg.loc[(total_avg.data == 'data')&(total_avg.state == state)], x='trial_count', y='value', hue='transition', hue_order=hue_order,
                          palette=palette, estimator='mean', errorbar=('ci', 95), ax=ax[i])
            sns.pointplot(total_avg.loc[(total_avg.data == shuffle)&(total_avg.state == state)], x='trial_count', y='value', hue='transition', hue_order=hue_order,
                          palette=['gray', 'black'], estimator='mean', errorbar=('ci', 95), ax=ax[i])
            ax[i].set_title(state)
            ax[i].vlines(3.5, 0, 1, 'gray', '--')
            ax[i].set_ylabel("P Rewarded" if classify_by == 'context' else "P Pink")
            ax[i].set_ylim([-0.05, 1.05])
            ax[i].set_xlabel('Whisker trials')
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            if i == 0:
                ax[i].legend().set_visible(False)

            for trial in total_avg.trial_count.unique():
                for context_change in total_avg.transition.unique():
                    sample_1 = total_avg.loc[(total_avg.data == 'data') & (total_avg.state == state) & (
                                total_avg.trial_count == trial) & (total_avg.transition == context_change), 'value']
                    sample_2 = total_avg.loc[(total_avg.data == shuffle) & (total_avg.state == state) & (
                                total_avg.trial_count == trial) & (total_avg.transition == context_change), 'value']
                    t, p = ttest_rel(sample_1, sample_2)
                    p_corr = p * total_avg.trial_count.unique().shape[0]
                    results_partial = {
                        'state': state,
                        'transition': context_change,
                        'combination': f"data vs {shuffle}",
                        'trial': trial,
                        'df': sample_1.shape[0] - 1,
                        'mean_1': sample_1.mean(),
                        'std_1': sample_1.std(),
                        'mean_2': sample_2.mean(),
                        'std_2': sample_2.std(),
                        't': t,
                        'p': p,
                        'p_corr': p_corr,
                        'd': t / np.sqrt(sample_1.shape[0]),
                        'significant': "True" if p_corr < 0.05 else "False"
                    }
                    results_p_data_shuffle += [results_partial]

        for ext in ['.png', '.svg']:
            fig.savefig(os.path.join(result_path, f"{shuffle}_trialbased_p_rewarded{ext}"))

        fig, ax = plt.subplots(1, 2, figsize=(5, 5))
        fig.suptitle("Last vs First whisker trial")
        for i, state in enumerate(['naive', 'expert']):
            sns.pointplot(
                total_avg.loc[
                    (total_avg.data == 'data') & (total_avg.state == state) & (total_avg.trial_count.isin([1, -1]))],
                x='trial_count', y='value', hue='transition', hue_order=hue_order,
                palette=palette, estimator='mean', errorbar=('ci', 95), ax=ax[i])
            sns.pointplot(
                total_avg.loc[
                    (total_avg.data == shuffle) & (total_avg.state == state) & (total_avg.trial_count.isin([1, -1]))],
                x='trial_count', y='value', hue='transition', hue_order=hue_order,
                palette=['gray', 'black'], estimator='mean', errorbar=('ci', 95), ax=ax[i])
            ax[i].set_title(state)
            ax[i].vlines(0.5, 0, 1, 'gray', '--')
            ax[i].set_ylabel("P Rewarded" if classify_by == 'context' else "P Pink")
            ax[i].set_ylim([-0.05, 1.05])
            ax[i].set_xlim([-0.25, 1.25])
            ax[i].set_xlabel('Whisker trials')
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            if i == 0:
                ax[i].legend().set_visible(False)

            possible_changes = total_avg.transition.unique()
            for k, change in enumerate(possible_changes):
                for d_type in ['data', shuffle]:
                    sample_1 = total_avg.loc[(total_avg.data == d_type) & (total_avg.state == state) & (total_avg.trial_count == -1) & (total_avg.transition == change), 'value']
                    alternative_transition = possible_changes[0] if k == 1 else possible_changes[1]
                    sample_2 = total_avg.loc[(total_avg.data == d_type) & (total_avg.state == state) & (total_avg.trial_count == 1) & (total_avg.transition == alternative_transition), 'value']

                    t, p = ttest_rel(sample_1, sample_2)
                    p_corr = p * 4
                    results_partial = {
                        'state': state,
                        'data_type': d_type,
                        'combination': f"{change} vs {alternative_transition}",
                        'df': sample_1.shape[0] - 1,
                        'mean_1': sample_1.mean(),
                        'std_1': sample_1.std(),
                        'mean_2': sample_2.mean(),
                        'std_2': sample_2.std(),
                        't': t,
                        'p': p,
                        'p_corr': p_corr,
                        'd': t / np.sqrt(sample_1.shape[0]),
                        'significant': "True" if p_corr < 0.05 else "False"
                    }

                    results_p_firstlast += [results_partial]

        results_p_data_shuffle = pd.DataFrame(results_p_data_shuffle)
        results_p_data_shuffle.to_csv(os.path.join(result_path, f'p_data_vs_{shuffle}.csv'))

        results_p_firstlast = pd.DataFrame(results_p_firstlast)
        results_p_firstlast.to_csv(os.path.join(result_path, f'p_firstvslast_{shuffle}.csv'))

        fig.tight_layout()
        for ext in ['.png', '.svg']:
            fig.savefig(os.path.join(result_path, f"{shuffle}_trialbased_p_rewarded_first_v_last{ext}"))

    return


def load_avg_results(data_folder):
    model_data = []
    null_data = []
    avg_result_files = glob.glob(os.path.join(data_folder, decode, "**", "*", f"{classify_by}_decoding", "model_results.json"))
    avg_null_files = glob.glob(os.path.join(data_folder, decode, "**", "*", f"{classify_by}_decoding", "null_results.json"))

    for file in avg_result_files:
        mouse_name = file.replace("/", "\\").split("\\")[-4]
        session = file.replace("/", "\\").split("\\")[-3]

        df = pd.read_json(file.replace("\\", "/"))
        df['state'] = state
        df['mouse_name']= mouse_name
        df['session'] = session
        model_data+=[df]
    model_data = pd.concat(model_data)

    for file in avg_null_files:
        mouse_name = file.replace("/", "\\").split("\\")[-4]
        session = file.replace("/", "\\").split("\\")[-3]

        df = pd.read_json(file.replace("\\", "/"))
        df['state'] = state
        df['mouse_name'] = mouse_name
        df['session'] = session
        null_data.append(df)
    null_data = pd.concat(null_data)
    null_data['accuracy_mean'] = null_data.apply(lambda x: np.nanmean(x['accuracy']), axis=1)

    return model_data, null_data


def load_trial_based_results(data_folder):
    model_data = []
    null_data = []
    trial_result_files = glob.glob(os.path.join(data_folder, decode, "**", "*", f"{classify_by}_decoding", "model_trial_based_scores.json"))
    trial_null_files = glob.glob(os.path.join(data_folder, decode, "**", "*", f"{classify_by}_decoding", "null_trial_based_scores.json"))

    for file in trial_result_files:
        mouse_name = file.replace("/", "\\").split("\\")[-4]
        session = file.replace("/", "\\").split("\\")[-3]

        df = pd.read_json(file.replace("\\", "/"))
        df['state'] = state
        df['mouse_name']= mouse_name
        df['session'] = session
        model_data+=[df]
    model_data = pd.concat(model_data)

    for file in trial_null_files:
        mouse_name = file.replace("/", "\\").split("\\")[-4]
        session = file.replace("/", "\\").split("\\")[-3]

        df = pd.read_json(file.replace("\\", "/"))
        df['state'] = state
        df['mouse_name'] = mouse_name
        df['session'] = session
        null_data.append(df)
    null_data = pd.concat(null_data)
    null_data['accuracy_mean'] = null_data.apply(lambda x: np.nanmean(x['accuracy']), axis=1)

    return model_data, null_data


def plot_avg_accuracy(model_data, null_data, result_path, plot=False, save=False):
                
                ## Mouse based
                mouse_mean = model_data.groupby('mouse_name').accuracy.mean().reset_index()
                mouse_mean['data'] = 'model'
                mouse_percentile = []
                for mouse, group in null_data.groupby('mouse_name'):
                    mouse_percentile += [{'mouse_name': mouse,
                                          'percentile': (mouse_mean.loc[mouse_mean['mouse_name'] == mouse, 'accuracy'].values[0] > group.accuracy_mean).sum() / group.accuracy_mean.count(),
                                          'pvalue': 1-(mouse_mean.loc[mouse_mean['mouse_name'] == mouse, 'accuracy'].values[0] > group.accuracy_mean).sum() / group.accuracy_mean.count()}]

                mouse_percentile = pd.DataFrame(mouse_percentile)

                fig, ax = plt.subplots(1, 3, figsize=(7, 4), gridspec_kw={'width_ratios': [4, 1, 1]})
                ax[0].spines[['right', 'top']].set_visible(False)
                ax[1].spines[['right', 'top']].set_visible(False)
                ax[2].spines[['right', 'top']].set_visible(False)

                percentile = (mouse_mean.accuracy.mean() > null_data.accuracy_mean).sum() / null_data.accuracy_mean.count()
                y, x, _ = ax[0].hist(null_data.accuracy_mean * 100, bins=20, color='gray', alpha=0.5)
                ax[0].axvline(x=mouse_mean.accuracy.mean() * 100, c='#FF8D21', linewidth=2)
                ax[0].text(70, y.max(), f'Cross-validated accuracy:', fontdict={'fontname': 'Arial', 'color': '#FF8D21'})
                ax[0].text(70, y.max()*0.95,
                           f'Mean $\pm$ std: {round(mouse_mean.accuracy.mean(), 2)} $\pm$ {round(mouse_mean.accuracy.std(), 2)}',
                           fontdict={'fontname': 'Arial', 'color': '#FF8D21'})
                ax[0].text(70, y.max()*0.9, f'Percentile: {round(percentile, 2)}',
                           fontdict={'fontname': 'Arial', 'color': '#FF8D21'})
                ax[0].set_xlim([20, 100])
                ax[0].set_xlabel('Decoding accuracy (%)')
                ax[0].set_ylabel('Shuffle Counts')

                sns.pointplot(null_data, x='data', y='accuracy_mean', color='k', estimator='mean', errorbar=('ci', 95),
                              ax=ax[1])
                sns.pointplot(mouse_mean, x='data', y='accuracy', color='#FF8D21', estimator='mean',
                              errorbar=('ci', 95), ax=ax[1])
                sns.stripplot(mouse_mean, x='data', y='accuracy', color='#FF8D21', alpha=0.75, ax=ax[1])
                ax[1].set_yticklabels([40, 50, 60, 70, 80, 90, 100])
                ax[1].set_ylim([0.4, 1])
                ax[1].set_ylabel('Decoding accuracy (%)')

                g = sns.stripplot(mouse_percentile, y='pvalue', color='#FF8D21', ax=ax[2])
                ax[2].set_yscale('log')
                # ax[2].get_legend().set_visible(False)
                ax[2].axhline(y=0.05, color='red', linestyle='dashed')
                ax[2].set_ylabel("P-value")
                ax[2].set_ylim([0.5, 0.001])
                fig.tight_layout()
                if plot:
                    fig.show()
                if save:
                    for ext in ['.png', '.svg']:
                        fig.savefig(os.path.join(result_path, f"mouse_classificaition_res{ext}"))

                ## Session based
                session_mean = model_data.groupby('session').accuracy.mean().reset_index()
                session_mean['data'] = 'model'
                sess_percentile = []
                for sess, group in null_data.groupby('session'):
                    sess_percentile += [{'mouse_name': sess.split("_")[0],
                                         'session': sess,
                                         'percentile': (session_mean.loc[session_mean['session']==sess, 'accuracy'].values[0] > group.accuracy_mean).sum() / group.accuracy_mean.count(),
                                         'pvalue': 1-(session_mean.loc[session_mean['session']==sess, 'accuracy'].values[0] > group.accuracy_mean).sum() / group.accuracy_mean.count()}]

                sess_percentile = pd.DataFrame(sess_percentile)

                fig, ax = plt.subplots(1, 3, figsize=(7, 4), gridspec_kw={'width_ratios': [4, 1, 1]})
                ax[0].spines[['right', 'top']].set_visible(False)
                ax[1].spines[['right', 'top']].set_visible(False)
                ax[2].spines[['right', 'top']].set_visible(False)
                percentile = (session_mean.accuracy.mean() > null_data.accuracy_mean).sum() / null_data.accuracy_mean.count()

                y, x, _ = ax[0].hist(null_data.accuracy_mean * 100, bins=20, color='gray', alpha=0.5)
                ax[0].axvline(x=session_mean.accuracy.mean() * 100, c='#FF8D21', linewidth=2)
                ax[0].text(70, y.max(), f'Cross-validated accuracy:', fontdict={'fontname': 'Arial', 'color': '#FF8D21'})
                ax[0].text(70, y.max()*0.95,
                           f'Mean $\pm$ std: {round(session_mean.accuracy.mean(), 2)} $\pm$ {round(session_mean.accuracy.std(), 2)}',
                           fontdict={'fontname': 'Arial', 'color': '#FF8D21'})
                ax[0].text(70, y.max()*0.9, f'Percentile: {round(percentile, 2)}',
                           fontdict={'fontname': 'Arial', 'color': '#FF8D21'})
                ax[0].set_xlim([20, 100])
                ax[0].set_xlabel('Decoding accuracy (%)')
                ax[0].set_ylabel('Shuffle Counts')

                sns.pointplot(null_data, x='data', y='accuracy_mean', color='k', estimator='mean', errorbar=('ci', 95),
                              ax=ax[1])
                sns.pointplot(session_mean, x='data', y='accuracy', color='#FF8D21', estimator='mean',
                              errorbar=('ci', 95), ax=ax[1])
                sns.stripplot(session_mean, x='data', y='accuracy', color='#FF8D21', alpha=0.75, ax=ax[1])
                ax[1].set_yticklabels([40, 50, 60, 70, 80, 90, 100])
                ax[1].set_ylim([0.4, 1])
                ax[1].set_ylabel('Decoding accuracy (%)')

                g = sns.stripplot(sess_percentile, y='pvalue', color='#FF8D21', ax=ax[2])
                ax[2].set_yscale('log')
                # ax[2].get_legend().set_visible(False)
                ax[2].axhline(y=0.05, color='red', linestyle='dashed')
                ax[2].set_ylabel("P-value")
                ax[2].set_ylim([0.5, 0.001])
                fig.tight_layout()
                if plot:
                    fig.show()
                if save:
                    for ext in ['.png', '.svg']:
                        fig.savefig(os.path.join(result_path, f"session_classificaition_res{ext}"))


if __name__ == "__main__":

    root_folder = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/widefield_decoding_20250227/"
    for folder in os.listdir(root_folder):
        result_folder = fr"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/widefield_decoding_20250227/{folder}/results"
        for decode in ['baseline']:
            for classify_by in ['context']:#, 'lick', 'tone']:
                for state in ['expert']:
                    result_path = Path(result_folder, f"{classify_by}_results", state)
                    if not os.path.exists(result_path):
                        os.makedirs(result_path, exist_ok=True)

                    if 'gcamp' in folder:
                        c_file = f"context_sessions_gcamp_{state}"
                    elif 'gfp' in folder:
                        c_file = f"context_sessions_controls_gfp_{state}"
                    elif 'jrgeco' in folder:
                        c_file = f"context_sessions_jrgeco_{state}"
                    elif 'tdtomato' in folder:
                        c_file = f"context_sessions_controls_tdtomato_{state}"
                    
                    config_file = Path(f"M:\z_LSENS\Share\Pol_Bech\Session_list\{c_file}.yaml")

                    with open(config_file, 'r', encoding='utf8') as stream:
                        config_dict = yaml.safe_load(stream)

                    # Choose session from dict wit keys
                    nwb_files = config_dict['Session path']
                    bhv_data = bhv_utils.build_standard_behavior_table(nwb_files)
                    data_folder = Path(fr"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/widefield_decoding_20250227/{folder}")

                    model_data, null_data = load_avg_results(data_folder=data_folder)
                    plot_avg_accuracy(model_data, null_data, result_path=result_path, plot=False, save=True)

                    trial_based_model, trial_based_null = load_trial_based_results(data_folder=data_folder) 

                    # plot_trialbased_accuracy()

                    # coefficient_files = glob.glob(Path(data_folder, decode, "**", "*", f"{classify_by}_decoding", "model_coefficients.npy"))
                    
                    # coefficients = []
                    # for file in coefficient_files:
                    #     coefficients += [np.load(file).mean(axis=0).squeeze()]
                    # coefficients_mean = np.nanmean(np.stack(coefficients), axis=0)
                    
                    # mice = [a.replace("/", "\\").split("\\")[-4] for a in coefficient_files]
                    # coef_df = pd.DataFrame({'mouse_name': mice, 'coefficients': coefficients})
                    # mouse_avg = coef_df.groupby('mouse_name').coefficients.apply(lambda x: np.nanmean(np.stack(x), axis=0))
                    # fig1, ax1 = plt.subplots(figsize=(7, 7))
                    # plot_single_frame(np.mean(np.stack(mouse_avg.values), axis=0).reshape(125, -1), title='Decoding Coefficients',
                    #                   fig=fig1, ax=ax1,
                    #                   colormap='seismic',
                    #                   vmin=-0.004,
                    #                   vmax=0.004)
                    # fig1.show()
                    # fig1.savefig(os.path.join(result_path, f"model_coefficients.png"))
