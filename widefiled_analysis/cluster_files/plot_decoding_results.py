import os
import yaml
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms

import warnings
warnings.filterwarnings("ignore")

from PIL import Image
from matplotlib.cm import get_cmap
from skimage.transform import rescale
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
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


def plot_image_stats(image, y_binary, classify_by, save_path):
    cat_a = image[np.where(y_binary == 1)[0]]
    cat_b = image[np.where(y_binary == 0)[0]]

    fig, ax = plt.subplots(1, 2, figsize=(10,7))
    ax[0].scatter(np.nanmean(cat_a, axis=0).flatten(), np.nanmean(cat_b, axis=0).flatten(), c='k', alpha=0.5, s=2)
    ax[0].set_xlabel('Lick' if classify_by == 'lick' else 'Rewarded')
    ax[0].set_ylabel('No Lick' if classify_by == 'lick' else 'Non-Rewarded')
    ax[0].plot(ax[0].get_xlim(), ax[0].get_ylim(), ls='--', c='r')
    ax[0].set_box_aspect(1)

    ax[1].hist(np.nanmean(cat_a, axis=0).flatten(), 100, alpha=0.5, label='Lick' if classify_by == 'lick' else 'Rewarded')
    ax[1].hist(np.nanmean(cat_b, axis=0).flatten(), 100, alpha=0.5, label='No Lick' if classify_by == 'lick' else 'Non-Rewarded')
    ax[1].set_xlabel("MinMax Scores")
    ax[1].set_ylabel("Counts")
    ax[1].set_box_aspect(1)
    fig.legend()
    fig.tight_layout()

    for ext in ['png', 'svg']:
        fig.savefig(save_path + f".{ext}")


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

def plot_single_session(result_path, decode, save_path, plot=False):
    data = pd.read_json(os.path.join(result_path, "results.json"))

    if decode == 'baseline':
        # chunk_order = {'0-40': 0, '40-80': 1, '80-120': 2, '120-160': 3, '160-200': 4, '-200-0': 5}
        chunk_order = {'-200--160': 0, '-160--120': 1, '-120--80': 2, '-80--40': 3, '-40-0': 4, '-200-0': 5}
    else:
        chunk_order = {'0-4': 0, '4-8': 1, '8-12': 2, '12-16': 3, '16-20': 4, '0-20': 5}

    data['chunk'] = data[['start_frame', 'stop_frame']].astype(str).agg('-'.join, axis=1)
    data['order'] = data['chunk'].apply(lambda x: chunk_order[x])
    data = data.sort_values('order')

    fig = plt.figure(figsize=(15, 7))
    fig.suptitle("Decoding over time")
    gs = fig.add_gridspec(nrows=2, ncols=5, left=0.1, bottom=0.25, right=0.95, top=0.95,
        wspace=0.05, hspace=0., width_ratios=np.ones(5))

    for i, row in data.iterrows():
        coefs = np.asarray(data.loc[i, 'coefficients']).squeeze().reshape(125,-1)
        CI_out = (np.asarray(data.loc[i, 'coefficients']) > data.loc[i, 'upper_bound']) | (np.asarray(data.loc[i, 'coefficients']) < data.loc[i, 'lower_bound'])
        CI_out = CI_out.squeeze().reshape(125, -1)

        if row.order < 5:
            ax = fig.add_subplot(gs[0, i])
            plot_single_frame(coefs, title=f'{data.loc[i, "start_frame"]} - {data.loc[i, "stop_frame"]}',
                              fig=fig, ax=ax,
                              colormap='seismic',
                              vmin=-0.5,
                              vmax=0.5)

            ax = fig.add_subplot(gs[1, i])
            plot_single_frame(CI_out, title='',
                              fig=fig, ax=ax,
                              norm=False,
                              colormap='Greys_r',
                              vmin=0,
                              vmax=0.5)

        else:
            fig1, ax1 = plt.subplots(1, 2, figsize=(15,7))
            plot_single_frame(coefs, title=f'{data.loc[i, "start_frame"]} - {data.loc[i, "stop_frame"]}',
                              fig=fig1, ax=ax1[0],
                              colormap='seismic',
                              vmin=-0.25,
                              vmax=0.25)
            plot_single_frame(CI_out, title='',
                              fig=fig1, ax=ax1[1],
                              norm=False,
                              colormap='Greys_r',
                              vmin=0,
                              vmax=0.5)

    fig.tight_layout()
    fig1.tight_layout()
    for ext in ['.png']:#, '.svg']:
        fig.savefig(os.path.join(save_path, f"coefficient_image_over_time{ext}"))
        fig1.savefig(os.path.join(save_path, f"coefficient_image_total{ext}"))

    if plot:
        fig.show()
        fig1.show()

    xticks = list(range(0, 6))

    data[['accuracy_mean', 'accuracy_std', 'accuracy_lb', 'accuracy_hb']] = data.apply(
        lambda x: ols_statistics(np.asarray(x['accuracy_shuffle'])), axis='columns', result_type='expand')
    data[['precision_mean', 'precision_std', 'precision_lb', 'precision_hb']] = data.apply(
        lambda x: ols_statistics(np.asarray(x['precision_shuffle'])), axis='columns', result_type='expand')

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    for i, metric in enumerate(['accuracy', 'precision']):
        sns.pointplot(data=data, x='chunk', y=metric, markers='o', scale=0.8, ax=ax[i], color='r',
                      label='full_model' if i == 1 else None)
        ax[i].plot(np.mean(np.stack(data.loc[:, f'{metric}_shuffle']), axis=1), marker='o', c='k',
                   label='shuffle' if i == 1 else None)
        ax[i].fill_between(range(6), data.loc[:, f'{metric}_lb'], data.loc[:, f'{metric}_hb'], color='grey',
                           alpha=0.25)
        ax[i].set_xticks(xticks, data.chunk, rotation=30)
        ax[i].set_ylim(-0.05, 1.05)
        ax[i].set_ylabel(metric)
        ax[i].set_xlabel("Time bins")

    for ext in ['.png', '.svg']:
        fig.savefig(os.path.join(save_path, f"accuracy_and_precision_over_time{ext}"))
    if plot:
      fig.show()

    return 0


def plot_multi_sessions(data, decode, result_path, classify_by, plot):

    for i, metric in enumerate(['accuracy', 'precision']):
        data[f"{metric}_shuffle_mean"] = data[f'{metric}_shuffle'].apply(lambda x: np.mean(x))

    if decode == 'baseline':
        # chunk_order = {'0-40': 0, '40-80': 1, '80-120': 2, '120-160': 3, '160-200': 4, '-200-0': 5}
        chunk_order = {'-200--160': 0, '-160--120': 1, '-120--80': 2, '-80--40': 3, '-40-0': 4, '-200-0': 5}

    else:
        chunk_order = {'0-4': 0, '4-8': 1, '8-12': 2, '12-16': 3, '16-20': 4, '0-20': 5}


    data['chunk'] = data[['start_frame', 'stop_frame']].astype(str).agg('-'.join, axis=1)
    data['order'] = data['chunk'].apply(lambda x: chunk_order[x])

    for name, group in data.groupby('mouse_id'):

        save_path = os.path.join(result_path, name, classify_by)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if decode == 'stim':

            fig = plt.figure(figsize=(15, 7))
            fig.suptitle(f"{name} Decoding over time")
            gs = fig.add_gridspec(nrows=2, ncols=5, left=0.1, bottom=0.25, right=0.95, top=0.95,
                                  wspace=0.05, hspace=0., width_ratios=np.ones(5))

            for i, (chunk, time_group) in enumerate(group.groupby(['start_frame', 'stop_frame'], sort=False)):
                print(f"Mouse {name}, frames {time_group.chunk}")

                if len(group.session_id.unique()) == 1:
                    coefs = np.stack(time_group['coefficients'].to_numpy()).squeeze()
                else:
                    coefs = np.stack(time_group['coefficients'].to_numpy()).squeeze().mean(axis=0)

                CI_out = (coefs < np.mean(np.stack(time_group['lower_bound']))) | (coefs > np.mean(np.stack(time_group['upper_bound'])))

                if time_group.order.unique()[0] < 5:
                    if coefs.shape[0] != 20000:
                        continue
                    ax = fig.add_subplot(gs[0, i])
                    plot_single_frame(coefs.reshape(125,-1), title=f'{chunk[0]} - {chunk[1]}',
                                      fig=fig, ax=ax,
                                      colormap='seismic',
                                      vmin=-0.5,
                                      vmax=0.5)

                    ax = fig.add_subplot(gs[1, i])
                    plot_single_frame(CI_out.reshape(125,-1), title='',
                                      fig=fig, ax=ax,
                                      norm=False,
                                      colormap='Greys_r',
                                      vmin=0,
                                      vmax=0.5)

                else:
                    fig1, ax1 = plt.subplots(1, 2, figsize=(15, 7))
                    plot_single_frame(coefs.reshape(125,-1), title=f'{chunk[0]} - {chunk[1]}',
                                      fig=fig1, ax=ax1[0],
                                      colormap='seismic',
                                      vmin=-0.25,
                                      vmax=0.25)
                    plot_single_frame(CI_out.reshape(125,-1), title='',
                                      fig=fig1, ax=ax1[1],
                                      norm=False,
                                      colormap='Greys_r',
                                      vmin=0,
                                      vmax=0.5)

            for ext in ['.png']:#, '.svg']:
                fig.savefig(os.path.join(save_path, f"{classify_by}_coefficient_image_over_time{ext}"))
                fig1.savefig(os.path.join(save_path, f"{classify_by}_coefficient_image_total{ext}"))

            if plot:
                fig.show()
                fig1.show()

    xticks = list(range(0, 6))
    for name, subgroup in data.groupby(['mouse_id']):
        save_path = os.path.join(result_path, name, classify_by)
        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        for i, metric in enumerate(['accuracy', 'precision']):

            sns.pointplot(data=subgroup, x='chunk', y=metric, markers='o', scale=0.8, ax=ax[i], color='r',
                          label='full_model' if i == 1 else None, estimator='mean', errorbar=('ci', 95))
            sns.pointplot(data=subgroup, x='chunk', y=f"{metric}_shuffle_mean", markers='o', scale=0.8, ax=ax[i], color='k',
                          label='full_model' if i == 1 else None, estimator='mean', errorbar=('ci', 95))
            ax[i].set_xticks(xticks, subgroup.chunk.unique(), rotation=30)
            ax[i].set_ylim(-0.05, 1.05)
            ax[i].set_ylabel(metric)
            ax[i].set_xlabel("Time bins")

        for ext in ['.png', '.svg']:
            fig.savefig(os.path.join(save_path, f"{classify_by}_accuracy_and_precision_over_time{ext}"))
        if plot:
            fig.show()

    subgroup = data.groupby(['mouse_id', 'chunk']).agg('mean')
    subgroup['coefficients'] = data.groupby(['mouse_id', 'chunk']).apply(
        lambda x: np.stack(x['coefficients']).mean(axis=0))
    # subgroup['coefs_shuffle'] = data.groupby(['mouse_id', 'chunk']).apply(
    #     lambda x: np.stack(x['coefficients_shuffle']))
    subgroup['upper_bound'] = data.groupby(['mouse_id', 'chunk']).apply(
        lambda x: np.stack(x['upper_bound']).mean(axis=0))
    subgroup['lower_bound'] = data.groupby(['mouse_id', 'chunk']).apply(
        lambda x: np.stack(x['lower_bound']).mean(axis=0))

    subgroup = subgroup.reset_index().sort_values('order')

    fig = plt.figure(figsize=(15, 7))
    fig.suptitle(f"All mice decoding over time")
    gs = fig.add_gridspec(nrows=2, ncols=5, left=0.1, bottom=0.25, right=0.95, top=0.95,
                          wspace=0.05, hspace=0., width_ratios=np.ones(5))
    for i, (chunk, group) in enumerate(subgroup.groupby('chunk', sort=False)):
        print(f"All mice, frames {chunk}")

        coefs = np.stack(group['coefficients'].to_numpy()).squeeze().mean(axis=0)
        # coefs_shuffle = np.vstack(group['coefs_shuffle']).reshape(-1, 20000)

        # empirical_p_values = (np.sum(
        #     np.abs(coefs_shuffle) >= np.abs(coefs), axis=0) + 1) / (coefs_shuffle.shape[0] + 1)

        CI_out = (coefs < np.mean(np.stack(group['lower_bound']))) | (coefs > np.mean(np.stack(group['upper_bound'])))

        if group.order.unique()[0] < 5:
            if decode == 'baseline':
                continue

            ax = fig.add_subplot(gs[0, i])
            plot_single_frame(coefs.reshape(125,-1), title=f'{chunk[0]} - {chunk[1]}',
                              fig=fig, ax=ax,
                              colormap='seismic',
                              vmin=-0.5,
                              vmax=0.5)
            ax.set_axis_off()

            ax = fig.add_subplot(gs[1, i])
            plot_single_frame(CI_out.reshape(125,-1), title='',
                              fig=fig, ax=ax,
                              norm=False,
                              colormap='Greys_r',
                              vmin=0,
                              vmax=0.5)
            ax.set_axis_off()


        else:
            fig1, ax1 = plt.subplots(1, 2, figsize=(15, 7))
            plot_single_frame(coefs.reshape(125,-1), title=f'{chunk[0]} - {chunk[1]}',
                              fig=fig1, ax=ax1[0],
                              colormap='seismic',
                              vmin=-0.25,
                              vmax=0.25)
            ax1[0].set_axis_off()
            plot_single_frame(CI_out.reshape(125,-1), title='',
                              fig=fig1, ax=ax1[1],
                              norm=False,
                              colormap='Greys_r',
                              vmin=0,
                              vmax=0.5)
            ax1[1].set_axis_off()


    for ext in ['.png']:#, '.svg']:
        # fig.savefig(os.path.join(result_path, decode, f"{classify_by}_coefficient_image_over_time{ext}"))
        fig1.savefig(os.path.join(result_path, f"{classify_by}_coefficient_image_total{ext}"))

    if plot:
        fig.show()
        fig1.show()

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    for i, metric in enumerate(['accuracy', 'precision']):

        sns.pointplot(data=subgroup, x='chunk', y=metric, markers='o', scale=0.8, ax=ax[i], color='r',
                      label='full_model' if i == 1 else None, estimator='mean', errorbar=('ci', 95))
        sns.pointplot(data=subgroup, x='chunk', y=f"{metric}_shuffle_mean", markers='o', scale=0.8, ax=ax[i], color='k',
                      label='full_model' if i == 1 else None, estimator='mean', errorbar=('ci', 95))
        ax[i].set_xticks(xticks, subgroup.chunk.unique(), rotation=30)
        ax[i].set_ylim(-0.05, 1.05)
        ax[i].set_ylabel(metric)
        ax[i].set_xlabel("Time bins")

    for ext in ['.png', '.svg']:
        fig.savefig(os.path.join(result_path, f"{classify_by}_accuracy_and_precision_over_time{ext}"))
    if plot:
        fig.show()

    return 0


def plot_baseline_results_total(data, result_path):

    stat_table = []

    agg_data = data.loc[(data.start_frame == -200) & (data.stop_frame == 0)].groupby('state').agg('sum')
    confusion_matrix_naive = np.array([[agg_data.loc['naive', 'conf_mat_tn'], agg_data.loc['naive', 'conf_mat_fp']],
                                       [agg_data.loc['naive', 'conf_mat_fn'], agg_data.loc['naive', 'conf_mat_tp']]])
    confusion_matrix_expert = np.array([[agg_data.loc['expert', 'conf_mat_tn'], agg_data.loc['expert', 'conf_mat_fp']],
                                        [agg_data.loc['expert', 'conf_mat_fn'], agg_data.loc['expert', 'conf_mat_tp']]])

    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    sns.heatmap(confusion_matrix_naive, annot=True, fmt='g', ax=ax[0], cbar=False)
    ax[0].set_title('Naive')
    ax[0].set_aspect('equal')
    sns.heatmap(confusion_matrix_expert, annot=True, fmt='g', ax=ax[1], cbar=False)
    ax[1].set_title('Expert')
    ax[1].set_aspect('equal')

    for axi in ax:
        axi.set_aspect('equal')
        axi.set_ylabel('True')
        axi.set_xlabel('Predicted')
    fig.tight_layout()

    for ext in ['.png', '.svg']:
        fig.savefig(os.path.join(result_path, f"{classify_by}_confusion_matrix{ext}"))

    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(4, 5))
    for i, metric in enumerate(['accuracy', 'precision']):
        data[f"{metric}_shuffle_mean"] = data[f'{metric}_shuffle'].apply(lambda x: np.mean(x))

        subset = data.loc[(data.start_frame == -200) & (data.stop_frame == 0)].groupby(by=['mouse_id', 'state']).agg(
            'mean').reset_index().melt(id_vars=['mouse_id', 'state'],
                                       value_vars=[f'{metric}',
                                                   f'{metric}_shuffle_mean'])

        sns.pointplot(subset.loc[(subset.variable == f'{metric}') | (subset.variable == f'{metric}_shuffle_mean')],
                      x='variable', y='value', estimator='mean', errorbar=('ci', 95), ax=ax[i], hue='state',
                      hue_order=['naive', 'expert'], palette=['#ffa345', '#624185'])

        ax[i].set_ylim([-0.05, 1.05])
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].set_ylabel(metric)
        ax[i].set_xticks([0, 1], ['Data', 'Shuffle'])
        ax[i].legend().set_visible(False)

        t_n, p_n = ttest_rel(subset.loc[(subset.state == 'naive') & (subset.variable == f'{metric}'), 'value'],
                             subset.loc[
                                 (subset.state == 'naive') & (subset.variable == f'{metric}_shuffle_mean'), 'value'])
        t_e, p_e = ttest_rel(subset.loc[(subset.state == 'expert') & (subset.variable == f'{metric}'), 'value'],
                             subset.loc[
                                 (subset.state == 'expert') & (subset.variable == f'{metric}_shuffle_mean'), 'value'])

        ax[i].text(0.5, 1.05, f'p naive={np.round(p_n, 4)}', verticalalignment='top', horizontalalignment='center', color='#ffa345')
        ax[i].text(0.5, 1, f'p expert={np.round(p_e, 4)}', verticalalignment='top', horizontalalignment='center', color ='#624185')

        res_naive = {'state': 'naive',
                     'metric': metric,
                     'df': subset.mouse_id.unique().shape[0] -1,
                     'mean': subset.loc[(subset.state == 'naive') & (subset.variable == f'{metric}'), 'value'].mean(),
                     'std': subset.loc[(subset.state == 'naive') & (subset.variable == f'{metric}'), 'value'].std(),
                     'mean_shuffle':subset.loc[(subset.state == 'naive') & (subset.variable == f'{metric}_shuffle_mean'), 'value'].mean(),
                     'std_shuffle': subset.loc[(subset.state == 'naive') & (subset.variable == f'{metric}_shuffle_mean'), 'value'].std(),
                     't': t_n,
                     'p': p_n,
                     'd': t_n/np.sqrt(subset.mouse_id.unique().shape[0]),
                     'significant': "True" if p_n < 0.05 else "False"}

        res_expert = {'state': 'expert',
                     'metric': metric,
                     'df': subset.mouse_id.unique().shape[0] -1,
                     'mean': subset.loc[(subset.state == 'expert') & (subset.variable == f'{metric}'), 'value'].mean(),
                     'std': subset.loc[(subset.state == 'expert') & (subset.variable == f'{metric}'), 'value'].std(),
                     'mean_shuffle':subset.loc[(subset.state == 'expert') & (subset.variable == f'{metric}_shuffle_mean'), 'value'].mean(),
                     'std_shuffle': subset.loc[(subset.state == 'expert') & (subset.variable == f'{metric}_shuffle_mean'), 'value'].std(),
                     't': t_e,
                     'p': p_e,
                     'd': t_e/np.sqrt(subset.mouse_id.unique().shape[0]),
                     'significant': "True" if p_e < 0.05 else "False"}

        stat_table += [res_naive]
        stat_table += [res_expert]

    stat_table = pd.DataFrame(stat_table)
    stat_table.to_excel(os.path.join(result_path, f"{classify_by}_decoding_statistics.xlsx"))

    for ext in ['.png', '.svg']:
        fig.savefig(os.path.join(result_path, f"{classify_by}_accuracy_and_precision{ext}"))


def plot_stim_results_total(data, result_path):

    data['chunk'] = data.groupby(by=['start_frame', 'stop_frame']).ngroup()
    data['chunk'] = data.chunk.map({0: 0, 1: 5, 2: 1, 3: 2, 4: 3, 5: 4})

    xticks = [f'{start * 10}-{end * 10}' for start, end in
              zip(data['start_frame'].unique(), data['stop_frame'].unique())]

    stat_table = []

    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(7, 5))

    for i, metric in enumerate(['accuracy', 'precision']):

        data[f"{metric}_shuffle_mean"] = data[f'{metric}_shuffle'].apply(lambda x: np.mean(x))

        subset = data.groupby(by=['mouse_id', 'state', 'chunk']).agg('mean').reset_index().melt(
            id_vars=['mouse_id', 'state', 'chunk'],
            value_vars=[f'{metric}',
                        f'{metric}_shuffle_mean'])
        subset = subset.loc[subset.chunk != 5]

        naive = subset.loc[subset.state == 'naive']
        sns.pointplot(naive.loc[(naive.variable == f'{metric}') | (naive.variable == f'{metric}_shuffle_mean')],
                      x='chunk', y='value', estimator='mean', errorbar=('ci', 95), ax=ax[i], hue='variable',
                      hue_order=[f'{metric}', f'{metric}_shuffle_mean'], palette=['#ffa345', '#fe5803'])

        expert = subset.loc[subset.state == 'expert']
        sns.pointplot(expert.loc[(expert.variable == f'{metric}') | (expert.variable == f'{metric}_shuffle_mean')],
                      x='chunk', y='value', estimator='mean', errorbar=('ci', 95), ax=ax[i], hue='variable',
                      hue_order=[f'{metric}', f'{metric}_shuffle_mean'], palette=['#9A77CF', '#3b2747'])

        ax[i].set_ylim([-0.05, 1.05])
        ax[i].set_xlim([-0.5, 4.5])
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].hlines(0.5, 0, 5, color='gray', linestyle='dashed')
        ax[i].set_ylabel(f'{metric}')
        ax[i].set_xticks(np.arange(5), xticks, rotation=30)
        ax[i].legend().set_visible(False)

        for tick in ax[i].xaxis.get_majorticklabels():
            tick.set_horizontalalignment("right")

        for j, chunk in enumerate(naive.chunk.unique()):
            t_n, p_n = ttest_rel(naive.loc[(naive.chunk == chunk) & (naive.variable == f'{metric}'), 'value'],
                                 naive.loc[
                                     (naive.chunk == chunk) & (naive.variable == f'{metric}_shuffle_mean'), 'value'])
            t_e, p_e = ttest_rel(expert.loc[(expert.chunk == chunk) & (expert.variable == f'{metric}'), 'value'],
                                 expert.loc[
                                     (expert.chunk == chunk) & (expert.variable == f'{metric}_shuffle_mean'), 'value'])

            p_n_corr = p_n * len(naive.chunk.unique())
            p_e_corr = p_e * len(naive.chunk.unique())

            ax[i].text(j, 1.05, f'p={np.round(p_n_corr, 4)}', verticalalignment='top', horizontalalignment='center',
                       color='#ffa345', fontsize='x-small')
            ax[i].text(j, 1, f'p={np.round(p_e_corr, 4)}', verticalalignment='top', horizontalalignment='center',
                       color='#9A77CF', fontsize='x-small')

            res_naive = {'state': 'naive',
                         'chunk': xticks[j],
                         'metric': metric,
                         'df': naive.mouse_id.unique().shape[0] - 1,
                         'mean': naive.loc[(naive.chunk == chunk) & (naive.variable == f'{metric}'), 'value'].mean(),
                         'std': naive.loc[(naive.chunk == chunk) & (naive.variable == f'{metric}'), 'value'].std(),
                         'mean_shuffle': naive.loc[
                             (naive.chunk == chunk) & (naive.variable == f'{metric}_shuffle_mean'), 'value'].mean(),
                         'std_shuffle': naive.loc[
                             (naive.chunk == chunk) & (naive.variable == f'{metric}_shuffle_mean'), 'value'].std(),
                         't': t_n,
                         'p': p_n,
                         'p_corr': p_n_corr,
                         'd': t_n / np.sqrt(subset.mouse_id.unique().shape[0]),
                         'significant': "True" if p_n_corr < 0.05 else "False"}
            res_expert = {'state': 'expert',
                          'chunk': xticks[j],
                          'metric': metric,
                          'df': expert.mouse_id.unique().shape[0] - 1,
                          'mean': expert.loc[
                              (expert.chunk == chunk) & (expert.variable == f'{metric}'), 'value'].mean(),
                          'std': expert.loc[(expert.chunk == chunk) & (expert.variable == f'{metric}'), 'value'].std(),
                          'mean_shuffle': expert.loc[
                              (expert.chunk == chunk) & (expert.variable == f'{metric}_shuffle_mean'), 'value'].mean(),
                          'std_shuffle': expert.loc[
                              (expert.chunk == chunk) & (expert.variable == f'{metric}_shuffle_mean'), 'value'].std(),
                          't': t_e,
                          'p': p_e,
                         'p_corr': p_e_corr,
                          'd': t_e / np.sqrt(subset.mouse_id.unique().shape[0]),
                          'significant': "True" if p_e_corr < 0.05 else "False"}
            stat_table += [res_naive]
            stat_table += [res_expert]

    stat_table = pd.DataFrame(stat_table)
    stat_table = stat_table.sort_values(by=['state'])

    stat_table.to_excel(os.path.join(result_path, f"{classify_by}_decoding_statistics.xlsx"))

    for ext in ['.png', '.svg']:
        fig.savefig(os.path.join(result_path, f"{classify_by}_accuracy_and_precision{ext}"))


if __name__ == "__main__":

    # group = "context_contrast_widefield"
    # result_folder = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/widefield_decoding_cluster_gcamp_naive"
    # config_file = f"//sv-nas1.rcp.epfl.ch/Petersen-Lab/z_LSENS/Share/Pol_Bech/Session_list/context_sessions_gcamp_naive.yaml"
    # with open(config_file, 'r', encoding='utf8') as stream:
    #     config_dict = yaml.safe_load(stream)

    # for decode in ['baseline', 'stim']:
    #     for classify_by in ['context', 'lick', 'tone']:
    #         print(f"Processing {decode} {classify_by}")
    #
    #         for i, nwb_path in enumerate(config_dict['Session path']):
    #             session = config_dict['Session id'][i]
    #             print(f"Session: {session}")
    #             # if classify_by == 'context':
    #             #     continue
    #             animal_id = session.split("_")[0]
    #             result_path = os.path.join(result_folder, decode, animal_id, session, f"{classify_by}_decoding",)
    #             save_path = os.path.join(result_path, 'results')
    #             if not os.path.exists(save_path):
    #                 os.makedirs(save_path)
    #
    #             # plot_single_session(result_path,
    #             #                     decode=decode,
    #             #                     save_path=save_path,
    #             #                     plot=False)
    #
    #         # all_files = glob.glob(os.path.join(result_folder, f"{classify_by}_decoding", "**", "*", "results.json").replace("\\", "/"))
    #         all_files = glob.glob(os.path.join(result_folder, decode, "**", "*", f"{classify_by}_decoding", "results.json").replace("\\", "/"))
    #         data = []
    #         for file in all_files:
    #             df = pd.read_json(file)
    #             data.append(df)
    #         data = pd.concat(data, axis=0, ignore_index=True)
    #         group_df = plot_multi_sessions(data,
    #                                        decode=decode,
    #                                        result_path=result_folder,
    #                                        classify_by=classify_by,
    #                                        plot=False)

    for decode in ['baseline']:
        result_folder = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/widefield_decoding_cluster_gcamp_naive_vs_expert_elasticnet_alltrials"
        if not os.path.exists(os.path.join(result_folder, decode)):
            os.makedirs(os.path.join(result_folder, decode), exist_ok=True)

        for classify_by in ['context']:#, 'lick', 'tone']:
            data_folder = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/widefield_decoding_cluster_gcamp_naive_elasticnet_alltrials"
            all_files = glob.glob(os.path.join(data_folder, decode, "**", "*", f"{classify_by}_decoding", "results.json").replace("\\", "/"))
            data_naive = []
            for file in all_files:
                df = pd.read_json(file)
                data_naive.append(df)
            data_naive = pd.concat(data_naive, axis=0, ignore_index=True)
            data_naive['state'] = 'naive'

            data_folder = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/widefield_decoding_cluster_gcamp_expert_elasticnet_alltrials"
            all_files = glob.glob(os.path.join(data_folder, decode, "**", "*", f"{classify_by}_decoding", "results.json").replace("\\", "/"))
            data_expert = []
            for file in all_files:
                df = pd.read_json(file)
                data_expert.append(df)
            data_expert = pd.concat(data_expert, axis=0, ignore_index=True)
            data_expert['state'] = 'expert'

            data = pd.concat([data_naive, data_expert], ignore_index=True)

            if decode == 'baseline':
                plot_baseline_results_total(data, result_path=os.path.join(result_folder, decode))

            elif decode == 'stim':
                plot_stim_results_total(data, result_path=os.path.join(result_folder, decode))

            group_df = plot_multi_sessions(data.loc[data.state == 'naive'],
                                           decode=decode,
                                           result_path=os.path.join(result_folder, decode, 'naive'),
                                           classify_by=classify_by,
                                           plot=False)

            group_df = plot_multi_sessions(data.loc[data.state == 'expert'],
                                           decode=decode,
                                           result_path=os.path.join(result_folder, decode, 'expert'),
                                           classify_by=classify_by,
                                           plot=False)
