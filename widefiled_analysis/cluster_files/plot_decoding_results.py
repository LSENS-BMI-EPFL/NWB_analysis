import os
import yaml
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms

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
    ax.text(50, 100, "3 mm", size=10)
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

def plot_single_session(result_path, save_path, plot=False):
    data = pd.read_json(os.path.join(result_path, "results.json"))

    fig = plt.figure(figsize=(15, 7))
    fig.suptitle("Decoding over time")
    gs = fig.add_gridspec(nrows=2, ncols=10, left=0.1, bottom=0.25, right=0.95, top=0.95,
        wspace=0.05, hspace=0., width_ratios=np.ones(10))

    for i, row in data.iterrows():
        coefs = np.asarray(data.loc[i, 'coefficients']).squeeze().reshape(125,-1)
        CI_out = (np.asarray(data.loc[i, 'coefficients']) > data.loc[i, 'upper_bound']) | (np.asarray(data.loc[i, 'coefficients']) < data.loc[i, 'lower_bound'])
        CI_out = CI_out.squeeze().reshape(125, -1)

        if i < 10:
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
    for ext in ['.png', '.svg']:
        fig.savefig(os.path.join(save_path, f"coefficient_image_over_time{ext}"))
        fig1.savefig(os.path.join(save_path, f"coefficient_image_total{ext}"))

    if plot:
        fig.show()
        fig1.show()

    xlabel_dict = [f"{data.loc[i, 'start_frame']}-{data.loc[i, 'stop_frame']}" for i, row in data.iterrows()]
    xticks = list(range(0, 10))
    xticks.append(11)
    data[['accuracy_lb', 'accuracy_hb']] = data.apply(lambda x: sms.DescrStatsW(x.accuracy_shuffle).tconfint_mean(),
                                                      axis='columns', result_type='expand')
    data[['precision_lb', 'precision_hb']] = data.apply(lambda x: sms.DescrStatsW(x.precision_shuffle).tconfint_mean(),
                                                      axis='columns', result_type='expand')

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    for i, metric in enumerate(['accuracy', 'precision']):
        ax[i].plot(data.loc[0:9, metric], marker='o', c='r')
        ax[i].plot(np.mean(np.stack(data.loc[0:9, f'{metric}_shuffle']), axis=1), marker='o', c='k')
        ax[i].fill_between(range(10), data.loc[0:9, f'{metric}_lb'], data.loc[0:9, f'{metric}_hb'], color='grey', alpha=0.25)

        ax[i].scatter(11, data.loc[10, metric], marker='o', c='r')
        ax[i].scatter(11,  np.mean(data.loc[10, f'{metric}_shuffle']), marker='o', c='k')
        ax[i].errorbar(11, np.mean(data.loc[10, f'{metric}_shuffle']),
                     yerr=[[np.mean(data.loc[10, f'{metric}_shuffle']) - data.loc[10, f'{metric}_lb']], [data.loc[10, f'{metric}_hb']- np.mean(data.loc[10, f'{metric}_shuffle'])]], ecolor='k')
        ax[i].set_xticks(xticks, xlabel_dict, rotation=30)
        ax[i].set_ylim(-0.05, 1.05)
        ax[i].set_ylabel(metric)
        ax[i].set_xlabel("Time bins")

    for ext in ['.png', '.svg']:
        fig.savefig(os.path.join(save_path, f"accuracy_and_precision_over_time{ext}"))
    if plot:
      fig.show()

    return 0


def plot_multi_sessions(data, result_path, classify_by, plot):

    group_df = pd.DataFrame()
    for name, group in data.groupby('mouse_id'):
        fig = plt.figure(figsize=(15, 7))
        fig.suptitle(f"{name} Decoding over time")
        gs = fig.add_gridspec(nrows=2, ncols=10, left=0.1, bottom=0.25, right=0.95, top=0.95,
                              wspace=0.05, hspace=0., width_ratios=np.ones(10))
        save_path = os.path.join(result_path, name, classify_by)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i, (chunk, time_group) in enumerate(group.groupby(['start_frame', 'stop_frame'], sort=False)):
            print(f"Mouse {name}, frames {chunk}")
            mouse_results = {}
            mouse_results = {'mouse_id': name, 'chunk': chunk}
            coefs = np.stack(time_group['coefficients'].to_numpy()).squeeze().mean(axis=0)
            coef_mean, coef_std_error, lower_bound, upper_bound = ols_statistics(np.vstack(time_group['coefficients_shuffle']), confidence=0.95)
            CI_out = np.asarray((coefs > upper_bound) | (coefs < lower_bound))

            mouse_results['coefs'] = coefs
            mouse_results['accuracy'] = np.stack(time_group['accuracy'].to_numpy()).mean(axis=0)
            mouse_results['accuracy_std'] = np.stack(time_group['accuracy'].to_numpy()).std(axis=0)
            mouse_results['precision'] = np.stack(time_group['precision'].to_numpy()).mean(axis=0)
            mouse_results['precision_std'] = np.stack(time_group['precision'].to_numpy()).std(axis=0)
            mouse_results['accuracy_shuffle_mean'] = np.mean(np.hstack(time_group[f'accuracy_shuffle']))
            mouse_results['precision_shuffle_mean'] = np.mean(np.hstack(time_group[f'precision_shuffle']))
            mouse_results['accuracy_lb'], mouse_results['accuracy_hb'] = sms.DescrStatsW(np.hstack(time_group.accuracy_shuffle)).tconfint_mean()
            mouse_results['precision_lb'], mouse_results['precision_hb'] = sms.DescrStatsW(np.hstack(time_group.precision_shuffle)).tconfint_mean()

            group_df = group_df.append(mouse_results, ignore_index=True)

            if chunk != (0, 200):
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
        for ext in ['.png', '.svg']:
            fig.savefig(os.path.join(save_path, f"{classify_by}_coefficient_image_over_time{ext}"))
            fig1.savefig(os.path.join(save_path, f"{classify_by}_coefficient_image_total{ext}"))

        if plot:
            fig.show()
            fig1.show()

    # group_df.to_json(os.path.join(result_path, 'combined_results.json'))
    # return

        chunk_order = {(0, 20): 0, (20, 40): 1, (40, 60): 2, (60, 80): 3, (80, 100): 4, (100, 120): 5, (120, 140): 6,
                       (140, 160): 7, (160, 180): 8, (180, 200): 9, (0, 200): 10}
        subgroup = group_df[group_df['mouse_id'] == name].reset_index()
        subgroup['order'] = subgroup['chunk'].apply(lambda x: chunk_order[x])
        subgroup = subgroup.sort_values('order')

        xlabel_dict = [f"{frame[0]} - {frame[1]}" for frame in subgroup['chunk']]
        # xlabel_dict = [f"{subgroup.loc[i, 'start_frame']}-{subgroup.loc[i, 'stop_frame']}" for i, row in subgroup.iterrows()]
        # xlabel_dict = np.unique(xlabel_dict).tolist()
        # xticks = list(range(0, 10))
        # xticks.append(11)
        # xlabel_dict = [f"{frame[0]} - {frame[1]}" for frame in subgroup['chunk']]

        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        for i, metric in enumerate(['accuracy', 'precision']):
            ax[i].plot(subgroup.loc[0:9, metric], c='r', marker='o', mfc='r', mec='k')
            ax[i].errorbar(range(10), subgroup.loc[0:9, metric],
                           yerr=subgroup.loc[0:9, f"{metric}_std"],
                           ecolor='k')
            ax[i].plot(subgroup.loc[0:9, f"{metric}_shuffle_mean"], marker='o', c='k')
            ax[i].fill_between(range(10), subgroup.loc[0:9, f"{metric}_lb"], subgroup.loc[0:9, f"{metric}_hb"], color='grey',
                               alpha=0.25)

            ax[i].scatter(11, subgroup.loc[10, metric], marker='o', c='r')
            ax[i].errorbar(11, subgroup.loc[10, metric],
                           yerr=subgroup.loc[10, f"{metric}_std"],
                           ecolor='k')
            ax[i].scatter(11, subgroup.loc[10, f"{metric}_shuffle_mean"], marker='o', c='k')
            ax[i].errorbar(11, subgroup.loc[10, f"{metric}_shuffle_mean"],
                           yerr=[[subgroup.loc[10, f"{metric}_shuffle_mean"] - subgroup.loc[10, f"{metric}_lb"]],
                         [subgroup.loc[10, f"{metric}_hb"] - subgroup.loc[10, f"{metric}_shuffle_mean"]]],
                           ecolor='k')

            ax[i].set_xticks(range(11), xlabel_dict, rotation=30)
            ax[i].set_ylim(-0.05, 1.05)
            ax[i].set_ylabel(metric)
            ax[i].set_xlabel("Time bins")

        for ext in ['.png', '.svg']:
            fig.savefig(os.path.join(save_path, f"{classify_by}_accuracy_and_precision_over_time{ext}"))
        if plot:
            fig.show()

    return group_df

if __name__ == "__main__":

    group = "context_contrast_widefield"
    classify_by = 'context'
    result_folder = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/widefield_decoding_cluster_PCA"
    config_file = f"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/group.yaml"
    with open(config_file, 'r', encoding='utf8') as stream:
        config_dict = yaml.safe_load(stream)
    for classify_by in ['context', 'lick']:
        print(f"Processing {classify_by}")

        for session, nwb_path in config_dict['NWB_CI_LSENS'][group]:
            print(f"Session: {session}")
            animal_id = session.split("_")[0]
            result_path = os.path.join(result_folder, animal_id, session, f"{classify_by}_decoding")
            save_path = os.path.join(result_path, 'results', classify_by)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # plot_single_session(result_path,
            #                     save_path=save_path,
            #                     plot=False)

        all_files = glob.glob(os.path.join(result_folder, "**", "*", f"{classify_by}_decoding", "results.json").replace("\\", "/"))
        data = []
        for file in all_files:
            df = pd.read_json(file)
            data.append(df)
        data = pd.concat(data, axis=0, ignore_index=True)
        group_df = plot_multi_sessions(data,
                                       result_path=result_folder,
                                       classify_by=classify_by,
                                       plot=False)

        group_df.to_json(os.path.join(result_folder, f'{classify_by}_combined_results.json'))