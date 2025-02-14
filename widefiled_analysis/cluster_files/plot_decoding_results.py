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
from itertools import combinations
from skimage.transform import rescale
from scipy.stats import ttest_rel, sem, norm
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
import nwb_utils.utils_behavior as bhv_utils


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


def plot_stim_coefficients(mouse, group, save_path):

    xticks = list(range(0, 5))

    if mouse != 'average':
        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        for i, metric in enumerate(['accuracy', 'precision']):

            group[f"{metric}_shuffle_mean"] = group[f'{metric}_shuffle'].apply(lambda x: np.mean(x))
            sns.pointplot(data=group, x='chunk', y=metric, markers='o', scale=0.8, ax=ax[i], color='r',
                          label='data' if i == 1 else None, estimator='mean', errorbar=('ci', 95))
            sns.pointplot(data=group, x='chunk', y=f"{metric}_shuffle_mean", markers='o', scale=0.8, ax=ax[i], color='k',
                          label='shuffle' if i == 1 else None, estimator='mean', errorbar=('ci', 95))
            ax[i].set_xticks(xticks, group.chunk.unique(), rotation=30)
            ax[i].set_ylim(-0.05, 1.05)
            ax[i].set_ylabel(metric)
            ax[i].set_xlabel("Time bins")

        for ext in ['.png', '.svg']:
            fig.savefig(os.path.join(save_path, f"{classify_by}_accuracy_and_precision_over_time{ext}"))


    fig = plt.figure(figsize=(15, 7))
    fig.suptitle(f"{mouse} Decoding over time")
    gs = fig.add_gridspec(nrows=2, ncols=5)

    for i, (chunk, time_group) in enumerate(group.groupby(['start_frame', 'stop_frame'], sort=True)):
        print(f"Mouse {mouse}, frames {time_group.chunk.unique()}")

        if mouse != 'average':
            if len(group.session_id.unique()) == 1:
                coefs = np.stack(time_group['coefficients'].to_numpy()).squeeze()
            else:
                coefs = np.stack(time_group['coefficients'].to_numpy()).squeeze().mean(axis=0)
        else:
            coefs = np.stack(time_group['coefficients'].to_numpy()).squeeze().mean(axis=0)

        CI_out = (coefs < np.mean(np.stack(time_group['lower_bound']))) | (
                coefs > np.mean(np.stack(time_group['upper_bound'])))

        if coefs.shape[0] != 20000:
            continue

        ax = fig.add_subplot(gs[0, i])
        plot_single_frame(coefs.reshape(125, -1), title=f'{chunk[0]*10} - {chunk[1]*10}',
                          fig=fig, ax=ax,
                          colormap='seismic',
                          vmin=-0.1,
                          vmax=0.1)
        ax.set_axis_off()

        ax = fig.add_subplot(gs[1, i])
        plot_single_frame(CI_out.reshape(125, -1), title='',
                          fig=fig, ax=ax,
                          norm=False,
                          colormap='Greys_r',
                          vmin=0,
                          vmax=0.5)

        ax.set_axis_off()

    for ext in ['.png']:
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, f"{classify_by}_coefficient_image{ext}"))


def plot_stim_coefficients_naive_vs_expert(avg, save_path):

    fig = plt.figure(figsize=(15, 7))
    fig.suptitle(f"Naive vs expert decoding over time")
    gs = fig.add_gridspec(nrows=3, ncols=5)

    for i, (chunk, time_group) in enumerate(avg.groupby(by=['start_frame', 'stop_frame'], sort=True)):

        coefs_naive = np.stack(time_group.loc[time_group.state=='naive', 'coefficients'].to_numpy()).squeeze().mean(axis=0)
        coefs_expert = np.stack(time_group.loc[time_group.state=='expert', 'coefficients'].to_numpy()).squeeze().mean(axis=0)

        ax = fig.add_subplot(gs[0, i])
        plot_single_frame(coefs_naive.reshape(125, -1), title=f'{chunk[0]*10} - {chunk[1]*10}',
                          fig=fig, ax=ax,
                          colormap='seismic',
                          vmin=-0.1,
                          vmax=0.1)
        ax.set_axis_off()

        ax = fig.add_subplot(gs[1, i])
        plot_single_frame(coefs_expert.reshape(125, -1), title=f'',
                          fig=fig, ax=ax,
                          colormap='seismic',
                          vmin=-0.1,
                          vmax=0.1)
        ax.set_axis_off()

        ax = fig.add_subplot(gs[2, i])
        plot_single_frame((coefs_expert-coefs_naive).reshape(125, -1), title='',
                          fig=fig, ax=ax,
                          norm=False,
                          colormap='PiYG_r',
                          vmin=-0.05,
                          vmax=0.05)

        ax.set_axis_off()

    for ext in ['.png']:
        fig.savefig(os.path.join(save_path, f"{classify_by}_coefficient_image_naive_vs_expert{ext}"))


def plot_baseline_coefficients(mouse, group, save_path):

    print(f"Plotting {mouse} baseline coefficients")
    if mouse != 'average':
        if len(group.session_id.unique()) == 1:
            coefs = np.stack(group['coefficients'].to_numpy()).squeeze()
            shuffle = np.stack(group['shuffle_mean'].to_numpy()).squeeze()
            CI_out = (coefs < np.mean(np.stack(group['lower_bound']))) | (coefs > np.mean(np.stack(group['upper_bound'])))

        else:
            coefs = np.stack(group['coefficients'].to_numpy()).squeeze().mean(axis=0)
            shuffle = np.stack(group['shuffle_mean'].to_numpy()).squeeze().mean(axis=0)
            CI_out = (coefs < np.mean(np.stack(group['lower_bound']), axis=0)) | (coefs > np.mean(np.stack(group['upper_bound']), axis=0))

    else:
        coefs = np.stack(group['coefficients'].to_numpy()).squeeze().mean(axis=0)
        shuffle = np.stack(group['shuffle_mean'].to_numpy()).squeeze().mean(axis=0)
        CI_out = (coefs < np.mean(np.stack(group['lower_bound']), axis=0)) | (
                    coefs > np.mean(np.stack(group['upper_bound']), axis=0))

    # Z = (coefs-shuffle)/np.sqrt(sem(coefs)**2+sem(shuffle)**2)
    # pvals = 2 * (1.0 - norm.cdf(Z))
    #
    # CI_out = np.where(pvals < 0.05/len(pvals), np.ones_like(pvals), 0)

    fig1, ax1 = plt.subplots(1, 2, figsize=(15, 7))
    plot_single_frame(coefs.reshape(125, -1), title=f'{str(group.start_frame.unique())} - {str(group.stop_frame.unique())}',
                      fig=fig1, ax=ax1[0],
                      colormap='seismic',
                      vmin=-0.1,
                      vmax=0.1)
    ax1[0].set_axis_off()
    plot_single_frame(CI_out.reshape(125, -1), title='',
                      fig=fig1, ax=ax1[1],
                      norm=False,
                      colormap='Greys_r',
                      vmin=0,
                      vmax=0.5)
    ax1[1].set_axis_off()

    for ext in ['.png']:
        fig1.savefig(os.path.join(save_path, f"{classify_by}_coefficient_image{ext}"))


def plot_baseline_coefficients_naive_vs_expert(avg, save_path):
    coefs_naive = np.stack(avg.loc[avg.state == 'naive', 'coefficients'].to_numpy()).squeeze().mean(axis=0)

    coefs_expert = np.stack(avg.loc[avg.state == 'expert', 'coefficients'].to_numpy()).squeeze().mean(axis=0)

    fig1, ax1 = plt.subplots(1, 3, figsize=(15, 7))
    plot_single_frame(coefs_naive.reshape(125, -1), title=f'naive',
                      fig=fig1, ax=ax1[0],
                      colormap='seismic',
                      vmin=-0.1,
                      vmax=0.1)
    ax1[0].set_axis_off()
    plot_single_frame(coefs_expert.reshape(125, -1), title=f'expert',
                      fig=fig1, ax=ax1[1],
                      colormap='seismic',
                      vmin=-0.1,
                      vmax=0.1)
    ax1[1].set_axis_off()
    plot_single_frame((coefs_expert - coefs_naive).reshape(125, -1), title='expert-naive',
                      fig=fig1, ax=ax1[2],
                      colormap='PiYG_r',
                      vmin=-0.05,
                      vmax=0.05)
    ax1[2].set_axis_off()

    for ext in ['.png']:
        fig1.savefig(os.path.join(save_path, f"{classify_by}_coefficient_image_naive_vs_expert{ext}"))

def plot_decoding_coefficients_mouse(data, decode, result_path):

    for name, group in data.groupby('mouse_id'):

        save_path = os.path.join(result_path, name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if decode == 'stim':
            group['chunk'] = group[['start_frame', 'stop_frame']].transform(lambda x: x * 10).astype(str).agg('-'.join, axis=1)
            group.drop(group.loc[(group.start_frame == 0) & (group.stop_frame == 20)].index, axis=0, inplace=True)
            plot_stim_coefficients(name, group, save_path)

        if decode == 'baseline':
            group = group.loc[(group.start_frame==group.start_frame.min()) & (group.stop_frame==group.stop_frame.max())]
            plot_baseline_coefficients(name, group, save_path)

def plot_decoding_coefficients_avg(data, decode, result_path):

    data['chunk'] = data[['start_frame', 'stop_frame']].astype(str).agg('-'.join, axis=1)

    avg = data.groupby(['state', 'mouse_id', 'chunk', 'start_frame', 'stop_frame']).apply(lambda x: np.stack(x['coefficients']).mean(axis=0)).reset_index()
    avg = avg.rename(columns={avg[0].name: 'coefficients'})
    avg['upper_bound'] = data.groupby(['state', 'mouse_id', 'chunk', 'start_frame', 'stop_frame']).apply(lambda x: np.stack(x['upper_bound']).mean(axis=0)).reset_index()[0]
    avg['lower_bound'] = data.groupby(['state', 'mouse_id', 'chunk', 'start_frame', 'stop_frame']).apply(lambda x: np.stack(x['lower_bound']).mean(axis=0)).reset_index()[0]
    avg['shuffle_mean'] = data.groupby(['state', 'mouse_id', 'chunk', 'start_frame', 'stop_frame']).apply(lambda x: np.stack(x['shuffle_mean']).mean(axis=0)).reset_index()[0]

    if decode == 'stim':
        avg.drop(avg.loc[(avg.start_frame == 0) & (avg.stop_frame == 20)].index, axis=0, inplace=True)
        plot_stim_coefficients_naive_vs_expert(avg, result_path)

    if decode == 'baseline':
        avg = avg.reset_index()
        avg = avg.loc[(avg.start_frame == avg.start_frame.min()) & (avg.stop_frame == avg.stop_frame.max())]
        plot_baseline_coefficients_naive_vs_expert(avg, result_path)

    return 0


def plot_baseline_results_mouse(data, classify_by, result_path):

    stat_table = []

    agg_data = data.loc[(data.start_frame == data.start_frame.min()) & (data.stop_frame == data.stop_frame.max())].groupby('state').agg('sum')
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
                      hue_order=['naive', 'expert'], palette=['#43B3AE', '#F88378'])

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

        ax[i].text(0.5, 1.05, f'p naive={np.round(p_n, 4)}', verticalalignment='top', horizontalalignment='center', color='#026402')
        ax[i].text(0.5, 1, f'p expert={np.round(p_e, 4)}', verticalalignment='top', horizontalalignment='center', color='#FFB7C6')

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


def plot_baseline_results_total(data, classify_by, result_path):

    stat_table = []

    agg_data = data.loc[(data.start_frame == data.start_frame.min()) & (data.stop_frame == data.stop_frame.max())].groupby('state').agg('sum')
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

        subset = data.loc[(data.start_frame == data.start_frame.min()) & (data.stop_frame == data.stop_frame.max()), ['mouse_id', 'state', f'{metric}', f'{metric}_shuffle_mean']]
        subset = subset.groupby(by=['mouse_id', 'state']).agg('mean').reset_index().melt(id_vars=['mouse_id', 'state'], value_vars=[f'{metric}', f'{metric}_shuffle_mean'])

        sns.pointplot(subset.loc[(subset.variable == f'{metric}') | (subset.variable == f'{metric}_shuffle_mean')],
                      x='variable', y='value', estimator='mean', errorbar=('ci', 95), ax=ax[i], hue='state',
                      hue_order=['naive', 'expert'], palette=['#43B3AE', '#F88378'])

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

        ax[i].text(0.5, 1.05, f'p naive={np.round(p_n, 4)}', verticalalignment='top', horizontalalignment='center', color='#026402')
        ax[i].text(0.5, 1, f'p expert={np.round(p_e, 4)}', verticalalignment='top', horizontalalignment='center', color ='#FFB7C6')

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


def plot_corr_vs_incorr(data, result_path):

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    subset = data.melt(id_vars=['mouse_id', 'state'],
                       value_vars=['accuracy', 'precision', 'accuracy_correct_trial_model',
                                   'precision_correct_trial_model', 'accuracy_correct_vs_incorrect',
                                   'precision_correct_vs_incorrect', 'accuracy_shuffle_mean',
                                   'precision_shuffle_mean'])

    subset = subset.groupby(by=['mouse_id', 'state', 'variable']).agg('mean').reset_index()

    fig, ax = plt.subplots(1, 2, figsize=(6, 5))
    sns.pointplot(subset[subset.variable.str.contains('accuracy')], x='variable', y='value', hue='state',
                  hue_order=['naive', 'expert'], palette=['#43B3AE', '#F88378'], estimator='mean', errorbar=('ci', 95),
                  ax=ax[0], dodge=0.4, linestyles='')
    sns.pointplot(subset[subset.variable.str.contains('precision')], x='variable', y='value', hue='state',
                  hue_order=['naive', 'expert'], palette=['#032e34', '#a31e21'], estimator='mean', errorbar=('ci', 95),
                  ax=ax[1], dodge=0.4, linestyles='')

    for axi, metric in zip(ax, ['Accuracy', 'Precision']):
        axi.set_ylim([0.45, 1.05])
        axi.set_ylabel(metric)
        axi.tick_params(axis='x', rotation=45)
        for tick in axi.xaxis.get_majorticklabels():
            tick.set_horizontalalignment("right")
    fig.tight_layout()
    for ext in['.png', '.svg']:
        fig.savefig(os.path.join(result_path, f"correct_vs_incorrect_trials{ext}"))

    pairs = list(combinations(subset.loc[subset.variable.str.contains('accuracy'), 'variable'].unique(), 2))
    results = []
    for state in ['naive', 'expert']:
        for metric in ['accuracy', 'precision']:
            for a, b in combinations(subset.loc[subset.variable.str.contains(f'{metric}'), 'variable'].unique(), 2):
                sample_1 = subset.loc[(subset.state == state) & (subset.variable == a), 'value']
                sample_2 = subset.loc[(subset.state == state) & (subset.variable == b), 'value']
                t, p = ttest_rel(sample_1, sample_2)
                p_corr = p * len(pairs)
                results_partial = {
                    'state': state,
                    'test': f"{a} vs {b}",
                    'df': subset.loc[(subset.state == state), 'mouse_id'].unique().shape[0] - 1,
                    'mean_1': sample_1.mean(),
                    'std_1': sample_1.std(),
                    'mean_2': sample_2.mean(),
                    'std_2': sample_2.std(),
                    't': t,
                    'p': p,
                    'p_corr': p_corr,
                    'd': t / np.sqrt(subset.loc[(subset.state == state), 'mouse_id'].unique().shape[0]),
                    'significant': "True" if p_corr < 0.05 else "False"
                }
                results += [results_partial]

    results = pd.DataFrame(results)
    results.to_csv(os.path.join(result_path, 'corr_vs_incorr_stats.csv'))

    return


def plot_stim_results_total(data, result_path):

    data['chunk'] = data[['start_frame', 'stop_frame']].transform(lambda x: x*10).astype(str).agg('-'.join, axis=1)
    data.drop(data.loc[data.chunk == '0-200'].index, axis=0, inplace=True)

    stat_table = []

    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(7, 5))

    for i, metric in enumerate(['accuracy', 'precision']):

        data[f"{metric}_shuffle_mean"] = data[f'{metric}_shuffle'].apply(lambda x: np.mean(x))

        subset = data.groupby(by=['mouse_id', 'state', 'chunk', 'start_frame'])[[f'{metric}', f'{metric}_shuffle_mean']].agg('mean').reset_index().melt(id_vars=['mouse_id', 'state', 'chunk', 'start_frame'],
                                                                                                                                         value_vars=[f'{metric}', f'{metric}_shuffle_mean'])
        subset = subset.sort_values(['state', 'start_frame']).reset_index(drop=True)

        naive = subset.loc[subset.state == 'naive']
        sns.pointplot(naive.loc[(naive.variable == f'{metric}') | (naive.variable == f'{metric}_shuffle_mean')],
                      x='chunk', y='value', estimator='mean', errorbar=('ci', 95), ax=ax[i], hue='variable',
                      hue_order=[f'{metric}', f'{metric}_shuffle_mean'], palette=['#43B3AE', '#F88378'])

        expert = subset.loc[subset.state == 'expert']
        sns.pointplot(expert.loc[(expert.variable == f'{metric}') | (expert.variable == f'{metric}_shuffle_mean')],
                      x='chunk', y='value', estimator='mean', errorbar=('ci', 95), ax=ax[i], hue='variable',
                      hue_order=[f'{metric}', f'{metric}_shuffle_mean'], palette=['#032e34', '#a31e21'])

        ax[i].set_ylim([-0.05, 1.05])
        ax[i].set_xlim([-0.5, 4.5])
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].hlines(0.5, 0, 5, color='gray', linestyle='dashed')
        ax[i].set_xlabel('Time bins (ms)')
        ax[i].set_xticklabels(expert.chunk.unique(), rotation=30, ha='right')
        ax[i].set_ylabel(f'{metric}')
        ax[i].legend().set_visible(False)

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
                       color='#026402', fontsize='x-small')
            ax[i].text(j, 1, f'p={np.round(p_e_corr, 4)}', verticalalignment='top', horizontalalignment='center',
                       color='#FFB7C6', fontsize='x-small')

            res_naive = {'state': 'naive',
                         'chunk': chunk,
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
                          'chunk': chunk,
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


    for decode in ['baseline']:
        result_folder = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/widefield_decoding_cluster_gcamp_naive_vs_expert_trialbased_2s_final"

        for classify_by in ['context', 'tone', 'lick']:#, 'lick', 'tone']:
            result_path = os.path.join(result_folder, decode, classify_by)
            if not os.path.exists(result_path):
                os.makedirs(result_path, exist_ok=True)

            bhv_data = []
            data = []
            trial_data = []
            for state in ['naive', 'expert']:
                data_folder = fr"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/widefield_decoding_cluster_gcamp_{state}_saga_elasticnet_001_final"
                json_files = glob.glob(os.path.join(data_folder, decode, "**", "*", f"{classify_by}_decoding", "results.json"))
                for file in json_files:
                    df = pd.read_json(file.replace("\\", "/"))
                    df['state'] = state
                    data.append(df)

                config_file = fr"M:\z_LSENS\Share\Pol_Bech\Session_list\context_sessions_gcamp_{state}.yaml"

                with open(config_file, 'r', encoding='utf8') as stream:
                    config_dict = yaml.safe_load(stream)

                # Choose session from dict wit keys
                nwb_files = config_dict['Session path']
                if decode == 'baseline' and classify_by != 'lick':
                    trial_files = glob.glob(os.path.join(data_folder, decode, "**", "*", f"{classify_by}_decoding", "trial_based_scores.csv"))
                    for file in trial_files:
                        trial_df = pd.read_csv(file.replace("\\", "/"))
                        trial_df['mouse_id'] = file.replace("\\", "/").split("/")[-4]
                        trial_df['session_id'] = file.replace("\\", "/").split("/")[-3]
                        trial_df['state'] = state
                        trial_data.append(trial_df)

                    bhv = bhv_utils.build_standard_behavior_table(nwb_files)
                    bhv['state'] = state
                    bhv_data.append(bhv)

            data = pd.concat(data, ignore_index=True)

            if decode == 'baseline':

                plot_baseline_results_total(data, classify_by, result_path=result_path)

                if classify_by != 'lick':
                    trial_data = pd.concat(trial_data, ignore_index=True)
                    bhv_data = pd.concat(bhv_data, ignore_index=True)
                    bhv_data.trial_id = bhv_data.groupby('session_id').cumcount()

                    plot_corr_vs_incorr(data, result_path=os.path.join(result_path, "corr_vs_incorr"))
                    plot_trialbased_accuracy(trial_data, bhv_data, classify_by, result_path=os.path.join(result_path, "trial_based"))

            elif decode == 'stim':
                plot_stim_results_total(data, result_path=os.path.join(result_folder, decode))

            # plot_decoding_coefficients_mouse(data.loc[data.state == 'naive'], decode, os.path.join(result_folder, decode, classify_by, 'naive'))
            # plot_decoding_coefficients_mouse(data.loc[data.state == 'expert'], decode, os.path.join(result_folder, decode, classify_by, 'expert'))

            plot_decoding_coefficients_avg(data,
                                       decode=decode,
                                       result_path=os.path.join(result_folder, decode, classify_by))
