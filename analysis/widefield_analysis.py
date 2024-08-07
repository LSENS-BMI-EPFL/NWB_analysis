import os
import yaml
import math
import itertools
import numpy as np
import h5py
import pandas as pd
import seaborn as sns
import gc
import imageio as iio
import tifffile as tiff
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import nwb_wrappers.nwb_reader_functions as nwb_read

from PIL import Image
from matplotlib.cm import get_cmap
from skimage.transform import rescale
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from nwb_utils import server_path, utils_misc, utils_behavior
from analysis.psth_analysis import make_events_aligned_data_table


def get_wf_scalebar(scale=1, plot=False, savepath=None):
    file = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Robin_Dard\Parameters\Widefield\wf_scalebars\reference_grid_20240314.tif"
    grid = Image.open(file)
    im = np.array(grid)
    im = im.reshape(int(im.shape[0] / 2), 2, int(im.shape[1] / 2), 2).mean(axis=1).mean(axis=2) # like in wf preprocessing
    x = [62*scale, 167*scale]
    y = [162*scale, 152*scale]
    c = np.sqrt((x[1] - x[0]) ** 2 + (y[0] - y[1]) ** 2)
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(rescale(im, scale, anti_aliasing=False))
        ax.plot(x, y, c='r')
        ax.plot(x, [y[0], y[0]], c='k')
        ax.plot([x[1], x[1]], y, c='k')
        ax.text(x[0] + int((x[1] - x[0]) / 2), 175*scale, f"{x[1] - x[0]} px")
        ax.text(170*scale, 168*scale, f"{np.abs(y[1] - y[0])} px")
        ax.text(100*scale, 145*scale, f"{round(c)} px")
        ax.text(200*scale, 25*scale, f"{round(c / 6)} px/mm", color="r")
        fig.show()
        if savepath:
            fig.savefig(savepath+rf'\wf_scalebar_scale{scale}.png')
    return round(c / 6)


def make_wf_movies(nwb_files, output_path):
    print(f"Create WF average movies")
    for nwb_file in nwb_files:
        mouse_id = nwb_read.get_mouse_id(nwb_file)
        session_id = nwb_read.get_session_id(nwb_file)
        print(f"Analyzing session {session_id}")
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
                    if len(trials_kept) == 0:
                        print("No trials in this condition, skipping")
                        continue

                    frames = []
                    for tstamp in trials_kept:
                        frame = utils_misc.find_nearest(wf_timestamps, tstamp)
                        data = nwb_read.get_widefield_dff0(nwb_file, ['ophys', 'dff0'], frame-200, frame+200)
                        if data.shape != (400, 125, 160):
                            continue
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
                if len(trials) == 0:
                    print("No trials in this condition, skipping")
                    continue

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
                tiff.imwrite(os.path.join(save_path, f'{session_id}_{trial_type}.tiff'), avg_data)


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


def get_frames_by_type_epoch(nwb_file, trials, wf_timestamps):
    frames = []
    for tstamp in trials:
        frame = utils_misc.find_nearest(wf_timestamps, tstamp)
        data = nwb_read.get_widefield_dff0(nwb_file, ['ophys', 'dff0'], frame - 200, frame + 200)
        if data.shape != (400, 125, 160):
            continue
        frames.append(data)

    data_frames = np.array(frames)
    data_frames = np.stack(data_frames, axis=0)
    return data_frames


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

    else:
        cmap = get_cmap(cmap)

    cmap.set_bad(color='k', alpha=0.1)

    return cmap


def plot_wf_timecourses(avg_data, title, save_path, vmin=-0.005, vmax=0.035):

    bregma = (488, 290)
    scale = 4
    scalebar = get_wf_scalebar(scale=scale)
    iso_mask, atlas_mask, allen_bregma = get_allen_ccf(bregma)

    fig, ax = plt.subplots(5, 5, figsize=(15, 15))
    fig.suptitle(title)
    cmap = get_colormap('hotcold')
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    for i, frame in enumerate(range(200, 224)):
        single_frame = np.rot90(rescale(avg_data[frame], scale, anti_aliasing=False))
        single_frame = np.pad(single_frame, [(0, 650 - single_frame.shape[0]), (0, 510 - single_frame.shape[1])],
                              mode='constant', constant_values=np.nan)

        mask = np.pad(iso_mask, [(0, 650 - iso_mask.shape[0]), (0, 510 - iso_mask.shape[1])], mode='constant',
                      constant_values=np.nan)
        single_frame = np.where(mask > 0, single_frame, np.nan)

        im = ax.flat[i].imshow(single_frame, norm=norm, cmap=cmap)
        ax.flat[i].contour(atlas_mask, levels=np.unique(atlas_mask), colors='gray',
                           linewidths=1)
        ax.flat[i].contour(iso_mask, levels=np.unique(np.round(iso_mask)), colors='black',
                           linewidths=2, zorder=2)
        ax.flat[i].scatter(bregma[0], bregma[1], marker='+', c='k', s=100, linewidths=2,
                           zorder=3)
        ax.flat[i].hlines(25, 25, 25+scalebar*3, linewidth=2, colors='k')
        ax.flat[i].text(50, 100, "3 mm", size=10)
        ax.flat[i].set_title(f"{i * 10} ms")

    fig.colorbar(im, cax=ax.flat[i + 1])
    fig.tight_layout()
    fig.savefig(save_path + ".png")
    # fig.savefig(save_path + ".svg")
    plt.close()


def plot_wf_single_frame(frame, title, fig, ax_to_plot, colormap='hotcold', vmin=-0.005, vmax=0.035,
                         halfrange=0.04, cbar_shrink=1.0):

    bregma = (488, 290)
    scale = 4
    scalebar = get_wf_scalebar(scale=scale)
    iso_mask, atlas_mask, allen_bregma = get_allen_ccf(bregma)

    fig, ax = plt.subplots(1, figsize=(4, 4))
    fig.suptitle(title)
    cmap = get_colormap(colormap)
    cmap.set_bad(color="white")

    single_frame = np.rot90(rescale(frame, scale, anti_aliasing=False))
    single_frame = np.pad(single_frame, [(0, 650 - single_frame.shape[0]), (0, 510 - single_frame.shape[1])],
                          mode='constant', constant_values=np.nan)

    mask = np.pad(iso_mask, [(0, 650 - iso_mask.shape[0]), (0, 510 - iso_mask.shape[1])], mode='constant',
                  constant_values=np.nan)
    single_frame = np.where(mask > 0, single_frame, np.nan)

    if colormap == 'hotcold':
        cmap = get_colormap('hotcold')
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        im = ax_to_plot.imshow(single_frame, norm=norm, cmap=cmap)
    else:
        norm = colors.CenteredNorm(halfrange=halfrange)
        im = ax_to_plot.imshow(single_frame, cmap="seismic", norm=norm)

    ax_to_plot.contour(atlas_mask, levels=np.unique(atlas_mask), colors='gray',
                       linewidths=1)
    ax_to_plot.contour(iso_mask, levels=np.unique(np.round(iso_mask)), colors='black',
                       linewidths=2, zorder=2)
    ax_to_plot.scatter(bregma[0], bregma[1], marker='+', c='k', s=100, linewidths=2,
                       zorder=3)
    ax_to_plot.hlines(25, 25, 25+scalebar*3, linewidth=2, colors='k')
    ax_to_plot.text(50, 100, "3 mm", size=10)
    ax_to_plot.set_title(title)
    fig.colorbar(im, ax=ax_to_plot, location='right', shrink=cbar_shrink)

    plt.close()


def plot_wf_activity(nwb_files, trials_to_do, epochs_to_do, output_path):
    print(f"Plot wf timecourses")

    for nwb_file in nwb_files:
        mouse_id = nwb_read.get_mouse_id(nwb_file)
        session_id = nwb_read.get_session_id(nwb_file)
        print(" ")
        print(f"Analyzing session {session_id}")
        session_type = nwb_read.get_session_type(nwb_file)
        if 'wf' not in session_type:
            print(f"{session_id} is not a widefield session")
            continue

        save_path = os.path.join(output_path, f"{mouse_id}", f"{session_id}", "timecourses")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        wf_timestamps = nwb_read.get_widefield_timestamps(nwb_file, ['ophys', 'dff0'])
        epochs = nwb_read.get_behavioral_epochs_names(nwb_file)

        if len(epochs) > 0:
            for epoch in nwb_read.get_behavioral_epochs_names(nwb_file):
                if epoch not in epochs_to_do:
                    continue
                epoch_times = nwb_read.get_behavioral_epochs_times(nwb_file, epoch)
                for trial_type in nwb_read.get_behavioral_events_names(nwb_file):
                    print(f"Trial type : {trial_type}")
                    if trial_type not in trials_to_do:
                        print("Skip this trial type")
                        continue
                    trials = nwb_read.get_behavioral_events_times(nwb_file, trial_type)[0]
                    print(f"Total of {len(trials)} trials")
                    trials_kept = utils_behavior.filter_events_based_on_epochs(events_ts=trials, epochs=epoch_times)
                    print(f"Total of {len(trials_kept)} trials in {epoch} epoch")
                    if len(trials_kept) == 0:
                        print("No trials in this condition, skipping")
                        continue

                    data_frames = get_frames_by_type_epoch(nwb_file, trials_kept, wf_timestamps)
                    avg_data = np.nanmean(data_frames, axis=0)
                    avg_data = avg_data - np.nanmean(avg_data[174:199], axis=0)
                    plot_wf_timecourses(avg_data, f"{trial_type}_{epoch} timecourse", os.path.join(save_path, f"{trial_type}_{epoch}_wf_timecourse"))

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
                plot_wf_timecourses(avg_data, f"to_{epoch} timecourse", os.path.join(save_path, f"to_{epoch}_wf_timecourse"), vmin=-0.005, vmax=0.05)

        else:
            for trial_type in nwb_read.get_behavioral_events_names(nwb_file):
                print(f"Trial type : {trial_type}")
                trials = nwb_read.get_behavioral_events_times(nwb_file, trial_type)[0]
                print(f"Total of {len(trials)} trials")
                if len(trials) == 0:
                    print("No trials in this condition, skipping")
                    continue

                frames = []
                for tstamp in trials:
                    frame = utils_misc.find_nearest(wf_timestamps, tstamp)
                    data = nwb_read.get_widefield_dff0(nwb_file, ['ophys', 'dff0'], frame - 200, frame + 200)
                    frames.append(data)

                data_frames = np.array(frames)
                data_frames = np.stack(data_frames, axis=0)
                avg_data = np.nanmean(data_frames, axis=0)
                avg_data = avg_data - np.nanmean(avg_data[174:199], axis=0)

                plot_wf_timecourses(avg_data, f"{trial_type} wf timecourse", os.path.join(save_path, f"{trial_type}_wf_timecourse"))


def plot_wf_activity_mouse_average(nwb_files, mouse_id, trials_list, epochs_to_do, output_path):
    nwb_files = [nwb_file for nwb_file in nwb_files if mouse_id in nwb_file]
    save_path = os.path.join(output_path, f"{mouse_id}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f'nwb_files : {nwb_files}')
    mouse_trial_avg_data = dict()
    all_mice_data = []
    for nwb_index, nwb_file in enumerate(nwb_files):
        session_id = nwb_read.get_session_id(nwb_file)
        print(" ")
        print(f"Analyzing session {session_id}")
        session_type = nwb_read.get_session_type(nwb_file)
        if 'wf' not in session_type:
            print(f"{session_id} is not a widefield session")
            continue

        wf_timestamps = nwb_read.get_widefield_timestamps(nwb_file, ['ophys', 'dff0'])
        epochs = nwb_read.get_behavioral_epochs_names(nwb_file)
        epochs = [epoch for epoch in epochs if epoch in epochs_to_do]
        trial_types = nwb_read.get_behavioral_events_names(nwb_file)
        trial_types = [trial for trial in trial_types if trial in trials_list]

        if len(epochs) > 0:
            epoch_trial_permutations = list(itertools.product(epochs, trial_types))
            for epoch_trial in epoch_trial_permutations:

                print(f"Epoch : {epoch_trial[0]}, Trials : {epoch_trial[1]}")
                if nwb_index == 0:
                    mouse_trial_avg_data[f'{epoch_trial[0]}_{epoch_trial[1]}'] = []
                epoch_times = nwb_read.get_behavioral_epochs_times(nwb_file, epoch_trial[0])
                trials = nwb_read.get_behavioral_events_times(nwb_file, epoch_trial[1])[0]
                trials_kept = utils_behavior.filter_events_based_on_epochs(events_ts=trials, epochs=epoch_times)
                print(f"Total of {len(trials_kept)} trials in {epoch_trial[0]} epoch")
                if len(trials_kept) == 0:
                    print("No trials in this condition, skipping")
                    continue
                data_frames = get_frames_by_type_epoch(nwb_file, trials_kept, wf_timestamps)
                avg_data = np.nanmean(data_frames, axis=0)
                avg_data = avg_data - np.nanmean(avg_data[174:199], axis=0)
                mouse_trial_avg_data[f'{epoch_trial[0]}_{epoch_trial[1]}'].append(avg_data)

            for epoch in epochs:
                print(' ')
                print(f"Epoch : {epoch}")
                epoch_transitions = nwb_read.get_behavioral_epochs_times(nwb_file, epoch)[0]
                print(f"Number of transitions : {len(epoch_transitions)}")
                if nwb_index == 0:
                    mouse_trial_avg_data[epoch] = []
                frames = []
                for tstamp in epoch_transitions:
                    if tstamp < 10:
                        continue
                    frame = utils_misc.find_nearest(wf_timestamps, tstamp)
                    data = nwb_read.get_widefield_dff0(nwb_file, ['ophys', 'dff0'], frame - 200, frame + 200)
                    frames.append(data)

                data_frames = np.array(frames)
                data_frames = np.stack(data_frames, axis=0)
                avg_data = np.nanmean(data_frames, axis=0)
                avg_data = avg_data - np.nanmean(avg_data[174:199], axis=0)
                mouse_trial_avg_data[epoch].append(avg_data)

        else:
            for trial_type in trial_types:
                if nwb_index == 0:
                    mouse_trial_avg_data[trial_type] = []
                print(f"Trial type : {trial_type}")
                trials = nwb_read.get_behavioral_events_times(nwb_file, trial_type)[0]
                print(f"Total of {len(trials)} trials")
                if len(trials) == 0:
                    print("No trials in this condition, skipping")
                    continue

                frames = []
                for tstamp in trials:
                    frame = utils_misc.find_nearest(wf_timestamps, tstamp)
                    data = nwb_read.get_widefield_dff0(nwb_file, ['ophys', 'dff0'], frame - 200, frame + 200)
                    frames.append(data)

                data_frames = np.array(frames)
                data_frames = np.stack(data_frames, axis=0)
                avg_data = np.nanmean(data_frames, axis=0)
                avg_data = avg_data - np.nanmean(avg_data[174:199], axis=0)

                mouse_trial_avg_data[trial_type].append(avg_data)

    avg_mice_data = dict.fromkeys(mouse_trial_avg_data.keys(), [])
    # Average across sessions and do figures
    for key, data in mouse_trial_avg_data.items():
        print(' ')
        print('Do the plots')
        print(f"Key: {key}, Data shape : {len(data)} sessions, with {data[0].shape} shape")
        data = np.stack(data)
        avg_data = np.nanmean(data, axis=0)
        avg_mice_data[key] = avg_data
    #     # plot_wf_timecourses(avg_data, f" {mouse_id} {key} wf timecourse",
    #     #                     os.path.join(save_path, f'{mouse_id}_{key}'))
    #
        path = os.path.join(save_path, f"{key}_single_frames")
        if not os.path.exists(path):
            os.makedirs(path)
    #
    #     for start in range(195,220,5):
    #         fig,ax =plt.subplots(figsize=(7,7))
    #         plot_wf_single_frame(np.nanmean(avg_data[start:start+5],axis=0),
    #                              title=f'{key} {start}-{start+5}',
    #                              fig=fig,
    #                              ax_to_plot=ax,
    #                              colormap='hotcold',
    #                              vmin=-0.005,
    #                              vmax=0.02,
    #                              cbar_shrink=0.75)
    #         fig.savefig(os.path.join(path, f'{mouse_id}_{key}_{start}-{start+5}_avg.png'))
    #         fig.savefig(os.path.join(path, f'{mouse_id}_{key}_{start}-{start+5}_avg.svg'))

    return avg_mice_data


def return_events_aligned_wf_table(nwb_files, rrs_keys, trials_dict, trial_names, epochs, time_range, subtract_baseline):
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
            data_table = make_events_aligned_data_table(nwb_list=nwb_files,
                                                        rrs_keys=rrs_keys,
                                                        time_range=time_range,
                                                        trial_selection=trials_dict[index],
                                                        epoch=epoch,
                                                        subtract_baseline=subtract_baseline)
            data_table['trial_type'] = trial_names[index]
            data_table['epoch'] = epochs[epoch_index]
            full_df.append(data_table)
    full_df = pd.concat(full_df, ignore_index=True)

    return full_df


def save_f0_image(nwb_files):
    for nwb_file in nwb_files:
        session_id = nwb_read.get_session_id(nwb_file)
        print(f"Analyzing session {session_id}")
        raw_f_file = nwb_read.get_widefield_raw_acquisition_path(nwb_file, acquisition_name='F')[0]
        print(f"Find F file : {raw_f_file}")
        print("Open F file to compute F0 on full recording ... ")
        F_file = h5py.File(raw_f_file, 'r')
        F = F_file['F'][:]
        print("Compute F0")
        winsize = F.shape[0]
        f0 = np.nanpercentile(F[:winsize], 5, axis=0)
        print(f"Save F0 image in {os.path.dirname(raw_f_file)}")
        iio.imwrite(os.path.join(os.path.dirname(raw_f_file), 'F0.tiff'), f0)
        F_file.close()
        del F
        gc.collect()
    return


def compare_quiet_windows_across_context(nwb_files, saving_path, only_correct_trials=False):
    print(f"{len(nwb_files)} sessions")
    mouse_quiet_windows_img_dict = dict()
    mouse_quiet_windows_context_dict = {'Non-Rewarded': [], 'Rewarded': []}
    easy_names_dict = dict()
    mouse_quiet_windows_img_diff_dict = dict()
    easy_diff_name_dict = dict()
    for nwb_idx, nwb_file in enumerate(nwb_files):
        session_id = nwb_read.get_session_id(nwb_file)
        print(' ')
        print(f"Session: {session_id}")
        mouse_id = session_id[0:5]

        # Get trial table
        trial_table = nwb_read.get_trial_table(nwb_file)
        trial_table['correct_trials'] = trial_table.lick_flag == trial_table.reward_available
        if only_correct_trials:
            print('Use only correct trials')
            trial_table = trial_table.loc[trial_table.correct_trials == True]
        trial_table['context'] = trial_table['context'].map({0: "Non-Rewarded", 1: "Rewarded"})

        # Get wf timestamps
        wf_timestamps = nwb_read.get_widefield_timestamps(nwb_file, ['ophys', 'dff0'])

        # Deal with all names
        quiet_windows_img_dict = dict()
        diff_keys_dict = dict()
        context_avg_dict = {'Non-Rewarded': [], 'Rewarded': []}

        trial_names = list(itertools.product(trial_table.context.unique(),
                                     trial_table.trial_type.unique(),
                                     trial_table.lick_flag.unique()))

        for trial_name in trial_names:
            quiet_windows_img_dict[f'{trial_name[0]}_{trial_name[1]}_{trial_name[2]}'] = []
            name_0 = trial_name[0]
            if trial_name[1].split('_')[0] in ['whisker', 'auditory']:
                name_1 = trial_name[1].split('_')[0]
                if trial_name[2] == 1:
                    easy_names_dict[f'{trial_name[0]}_{trial_name[1]}_{trial_name[2]}'] = f'{name_0} {name_1} hit'
                else:
                    easy_names_dict[f'{trial_name[0]}_{trial_name[1]}_{trial_name[2]}'] = f'{name_0} {name_1} miss'
            else:
                if trial_name[2] == 1:
                    easy_names_dict[f'{trial_name[0]}_{trial_name[1]}_{trial_name[2]}'] = f'{name_0} false alarm'
                else:
                    easy_names_dict[f'{trial_name[0]}_{trial_name[1]}_{trial_name[2]}'] = f'{name_0} correct rejection'

            b = [((trial_name[1] in trial) & (trial_name[2] in trial)) for trial in trial_names]
            diff_keys_dict[f'{trial_name[1]}_{trial_name[2]}'] = np.array(trial_names)[np.where(b)[0]].tolist()

        for key in list(diff_keys_dict.keys()):
            diff_keys_dict[key] = [
                f"{diff_keys_dict[key][i][0]}_{diff_keys_dict[key][i][1]}_{diff_keys_dict[key][i][2]}" for i in [0, 1]]

            if key.split('_')[0] in ['whisker', 'auditory']:
                if key.split('_')[2] == '1':
                    easy_diff_name_dict[key] = f'{key.split("_")[0]} hit'
                else:
                    easy_diff_name_dict[key] = f'{key.split("_")[0]} miss'
            else:
                if key.split('_')[3] == '1':
                    easy_diff_name_dict[key] = f'false alarm'
                else:
                    easy_diff_name_dict[key] = f'correct rejection'

        print(f'Extract imaging data from quiet window in all trials ({len(trial_table)})')
        for trial_index in range(len(trial_table)):
            trial_type = (f"{trial_table.iloc[trial_index].context}_"
                          f"{trial_table.iloc[trial_index].trial_type}_"
                          f"{trial_table.iloc[trial_index].lick_flag}")
            quiet_start = trial_table.iloc[trial_index].abort_window_start_time
            stim_time = trial_table.iloc[trial_index].start_time
            start_frame = utils_misc.find_nearest(wf_timestamps, quiet_start)
            end_frame = utils_misc.find_nearest(wf_timestamps, stim_time)
            data = nwb_read.get_widefield_dff0(nwb_file, ['ophys', 'dff0'], start_frame, end_frame)
            avg_data = np.mean(data, axis=0)
            quiet_windows_img_dict[trial_type].append(avg_data)
            if 'Non-Rewarded' in trial_type:
                context_avg_dict['Non-Rewarded'].append(avg_data)
            else:
                context_avg_dict['Rewarded'].append(avg_data)

        # --------------------------------------------------------------------------------------------------- #
        print('Do the session plots')
        # Figure 1 : by session one subplot for each trial type
        print('Plot average baseline image for all trial type')
        fig, axes = plt.subplots(2, 6, figsize=(18, 8), sharey=True, sharex=True)
        ax = 0
        for key, data in quiet_windows_img_dict.items():
            if nwb_idx == 0:
                mouse_quiet_windows_img_dict[key] = []
            if len(data) == 0:
                print(f'No {easy_names_dict[key]} trial')
                ax += 1
                continue
            print(f"Name: {easy_names_dict[key]}, Data shape : {len(data)} trials")
            data = np.stack(data)
            avg_data = np.nanmean(data, axis=0)
            mouse_quiet_windows_img_dict[key].append(avg_data)
            plot_wf_single_frame(frame=avg_data, title=f'{len(data)} {easy_names_dict[key]}',
                                 fig=fig, ax_to_plot=axes.flatten()[ax],
                                 colormap='hotcold', vmin=-0.005, vmax=0.04, cbar_shrink=0.65)
            ax += 1
        fig.suptitle(f'Session: {session_id}')
        fig.tight_layout()
        save_formats = ['png']
        if not os.path.exists(os.path.join(f'{saving_path}', f'{mouse_id}', f'{session_id}')):
            os.makedirs(os.path.join(f'{saving_path}', f'{mouse_id}', f'{session_id}'))
        for save_format in save_formats:
            fig.savefig(os.path.join(f'{saving_path}', f'{mouse_id}', f'{session_id}',
                                     f'{session_id}_trial_type_images.{save_format}'),
                        format=f"{save_format}")
        plt.close()

        # Figure 1b: by session one subplot for each trial type difference
        print('Plot average baseline image difference between contexts for all trial type')
        fig, axes = plt.subplots(1, 6, figsize=(18, 4), sharey=True, sharex=True)
        ax = 0
        for key, data in diff_keys_dict.items():
            if nwb_idx == 0:
                mouse_quiet_windows_img_diff_dict[key] = []
            rwd_key = [key for key in data if 'Non' not in key][0]
            nn_rwd_key = [key for key in data if 'Non' in key][0]
            rwd_data = quiet_windows_img_dict[rwd_key]
            nn_rwd_data = quiet_windows_img_dict[nn_rwd_key]
            if len(rwd_data) == 0 or len(nn_rwd_data) == 0:
                ax += 1
                continue
            print(f"Name: {easy_diff_name_dict[key]}, Data shape : {len(rwd_data)} and {len(nn_rwd_data)} trials")
            avg_rwd_data = np.nanmean(rwd_data, axis=0)
            avg_nn_rwd_data = np.nanmean(nn_rwd_data, axis=0)
            avg_data = avg_rwd_data - avg_nn_rwd_data
            mouse_quiet_windows_img_diff_dict[key].append(avg_data)
            plot_wf_single_frame(frame=avg_data, title=f'{easy_diff_name_dict[key]}',
                                 fig=fig, ax_to_plot=axes.flatten()[ax],
                                 colormap='seismic', vmin=-0.025, vmax=0.02, halfrange=0.04, cbar_shrink=0.7)
            ax += 1
        fig.suptitle(f'Session: {session_id},  RWD - NN-RWD')
        fig.tight_layout()
        save_formats = ['png']
        if not os.path.exists(os.path.join(f'{saving_path}', f'{mouse_id}', f'{session_id}')):
            os.makedirs(os.path.join(f'{saving_path}', f'{mouse_id}', f'{session_id}'))
        for save_format in save_formats:
            fig.savefig(os.path.join(f'{saving_path}', f'{mouse_id}', f'{session_id}',
                                     f'{session_id}_trial_type_diff_images.{save_format}'),
                        format=f"{save_format}")
        plt.close()

        # Figure 2 : by session average across the two contexts
        print('Plot average baseline image and difference for contexts combining all trial type')
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
        ax = 0
        for key, data in context_avg_dict.items():
            if len(data) == 0:
                print(f'No {key} context')
                continue
            print(f"Key: {key} context, Data shape : {len(data)} trials")
            data = np.stack(data)
            avg_data = np.nanmean(data, axis=0)
            if ax == 0:
                context_avg_data = np.zeros(avg_data.shape)
            mouse_quiet_windows_context_dict[key].append(avg_data)
            plot_wf_single_frame(frame=avg_data, title=f'{key} context ({len(data)} trials)',
                                 fig=fig, ax_to_plot=axes.flatten()[ax],
                                 colormap='hotcold', vmin=-0.005, vmax=0.04, cbar_shrink=1)
            if 'Non' in key:
                context_avg_data -= avg_data
            else:
                context_avg_data += avg_data
            ax += 1
            if ax == 2:
                plot_wf_single_frame(frame=context_avg_data, title=f'Difference',
                                     fig=fig, ax_to_plot=axes.flatten()[ax],
                                     colormap='seismic', vmin=-0.025, vmax=0.04, halfrange=0.018, cbar_shrink=1)
        fig.suptitle(f'Session: {session_id}')
        fig.tight_layout()
        save_formats = ['png']
        if not os.path.exists(os.path.join(f'{saving_path}', f'{mouse_id}', f'{session_id}')):
            os.makedirs(os.path.join(f'{saving_path}', f'{mouse_id}', f'{session_id}'))
        for save_format in save_formats:
            fig.savefig(os.path.join(f'{saving_path}', f'{mouse_id}', f'{session_id}',
                                     f'{session_id}_context_images.{save_format}'),
                        format=f"{save_format}")
        plt.close()
        print('Figures saved')

    print(' ')
    print(f'Do the mouse plots')
    # Figure 3: group sessions for each trial type
    print('Plot average baseline image for all trial type combining all sessions')
    fig, axes = plt.subplots(2, 6, figsize=(18, 8), sharey=True, sharex=True)
    ax = 0
    for key, data in mouse_quiet_windows_img_dict.items():
        if len(data) == 0:
            ax += 1
            continue
        print(f"Name: {easy_names_dict[key]}, Data shape : {len(data)} sessions")
        data = np.stack(data)
        avg_data = np.nanmean(data, axis=0)
        plot_wf_single_frame(frame=avg_data, title=easy_names_dict[key], fig=fig, ax_to_plot=axes.flatten()[ax],
                             colormap='hotcold', vmin=-0.005, vmax=0.04, cbar_shrink=0.65)
        ax += 1
    fig.suptitle(f'{mouse_id}: average from {len(nwb_files)} sessions')
    fig.tight_layout()
    save_formats = ['png']
    if not os.path.exists(os.path.join(f'{saving_path}', f'{mouse_id}')):
        os.makedirs(os.path.join(f'{saving_path}', f'{mouse_id}'))
    for save_format in save_formats:
        fig.savefig(os.path.join(f'{saving_path}', f'{mouse_id}',
                                 f'{mouse_id}_trial_type_images.{save_format}'),
                    format=f"{save_format}")
    plt.close()

    # Figure 3b: group sessions difference for each trial type
    fig, axes = plt.subplots(1, 6, figsize=(18, 4), sharey=True, sharex=True)
    ax = 0
    for key, data in mouse_quiet_windows_img_diff_dict.items():
        if len(data) == 0:
            ax += 1
            continue
        print(f"Name: {easy_diff_name_dict[key]}, Data shape : {len(data)} sessions")
        data = np.stack(data)
        avg_data = np.nanmean(data, axis=0)
        plot_wf_single_frame(frame=avg_data, title=easy_diff_name_dict[key], fig=fig, ax_to_plot=axes.flatten()[ax],
                             colormap='seismic', vmin=-0.025, vmax=0.04, halfrange=0.018, cbar_shrink=0.65)
        ax += 1
    fig.suptitle(f'{mouse_id}: average from {len(nwb_files)} sessions')
    fig.tight_layout()
    save_formats = ['png']
    if not os.path.exists(os.path.join(f'{saving_path}', f'{mouse_id}')):
        os.makedirs(os.path.join(f'{saving_path}', f'{mouse_id}'))
    for save_format in save_formats:
        fig.savefig(os.path.join(f'{saving_path}', f'{mouse_id}',
                                 f'{mouse_id}_trial_type_diff_images.{save_format}'),
                    format=f"{save_format}")
    plt.close()

    # Figure 4 : average context baseline across contexts
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
    ax = 0
    for key, data in mouse_quiet_windows_context_dict.items():
        if len(data) == 0:
            print(f'No {key} context')
            continue
        print(f"Key: {key} context, Data shape : {len(data)} sessions")
        data = np.stack(data)
        avg_data = np.nanmean(data, axis=0)
        if ax == 0:
            context_avg_data = np.zeros(avg_data.shape)
        plot_wf_single_frame(frame=avg_data, title=f'{key} context ({len(data)} sessions)',
                             fig=fig, ax_to_plot=axes.flatten()[ax],
                             colormap='hotcold', vmin=-0.005, vmax=0.04, cbar_shrink=1)
        if 'Non' in key:
            context_avg_data -= avg_data
        else:
            context_avg_data += avg_data
        ax += 1
        if ax == 2:
            plot_wf_single_frame(frame=context_avg_data, title=f'Difference',
                                 fig=fig, ax_to_plot=axes.flatten()[ax],
                                 colormap='seismic', vmin=-0.025, vmax=0.04, halfrange=0.018, cbar_shrink=1)
    fig.suptitle(f'{mouse_id}: average from {len(nwb_files)} sessions')
    fig.tight_layout()
    save_formats = ['png']
    if not os.path.exists(os.path.join(f'{saving_path}', f'{mouse_id}')):
        os.makedirs(os.path.join(f'{saving_path}', f'{mouse_id}'))
    for save_format in save_formats:
        fig.savefig(os.path.join(f'{saving_path}', f'{mouse_id}',
                                 f'{mouse_id}_context_images.{save_format}'),
                    format=f"{save_format}")
    plt.close()

    print('Figures saved')


def plot_example_traces(nwb_files, mouse_id, output_path):
    nwb_files = [nwb_file for nwb_file in nwb_files if mouse_id in nwb_file]
    save_path = os.path.join(output_path, f"{mouse_id}")
    top_parts = ['top_nose_tip_x', 'top_nose_tip_y', 'top_nose_tip_likelihood', 'top_nose_base_x',
                 'top_nose_base_y', 'top_nose_base_likelihood', 'whisker_base_x',
                 'whisker_base_y', 'whisker_base_likelihood', 'whisker_tip_x',
                 'whisker_tip_y', 'whisker_tip_likelihood', 'top_particle_x', 'top_particle_y',
                 'top_particle_likelihood', 'whisker_angle', 'whisker_velocity',
                 'top_nose_angle', 'top_nose_distance', 'top_nose_velocity']

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f'nwb_files : {nwb_files}')
    mouse_trial_avg_data = dict()
    all_mice_data = []
    for nwb_index, nwb_file in enumerate(nwb_files):
        session_id = nwb_read.get_session_id(nwb_file)
        print(" ")
        print(f"Analyzing session {session_id}")
        session_type = nwb_read.get_session_type(nwb_file)
        if 'wf' not in session_type:
            print(f"{session_id} is not a widefield session")
            continue

        wf_timestamps = nwb_read.get_widefield_timestamps(nwb_file, ['ophys', 'dff0'])
        epochs = nwb_read.get_behavioral_epochs_names(nwb_file)
        epochs = [epoch for epoch in epochs if epoch in ['rewarded', 'non-rewarded']]
        epoch = 'rewarded'
        epoch_times = nwb_read.get_behavioral_epochs_times(nwb_file, epoch)
        trial_types = nwb_read.get_behavioral_events_names(nwb_file)
        trial_types = [trial for trial in trial_types if trial not in ['jaw_dlc_licks', 'tongue_dlc_licks']]

        trial_type = 'whisker_hit_trial'
        trials = nwb_read.get_behavioral_events_times(nwb_file, trial_type)[0]
        print(f"Total of {len(trials)} trials")
        trials_kept = utils_behavior.filter_events_based_on_epochs(events_ts=trials, epochs=epoch_times)

        trial =trials_kept[0]
        dlc_keys = ["whisker_angle", 'jaw_angle', 'tongue_y', 'nose_angle', "pupil_area"]

        data_side = pd.DataFrame()
        data_top = pd.DataFrame()
        data_side['timestamps'], data_top['timestamps'] = nwb_read.get_dlc_timestamps(nwb_file, ['behavior', 'BehavioralTimeSeries'])

        for i, part in enumerate(dlc_keys):
            data = nwb_read.get_dlc_data(nwb_file, ['behavior', 'BehavioralTimeSeries'], part)
            print(i, f"{part}", len(data))
            if part not in top_parts:
                data_side[part] = data
            else:
                data_top[part] = data
        side_start = utils_misc.find_nearest(data_side['timestamps'].to_numpy(), trials_kept[0])
        top_start = utils_misc.find_nearest(data_top['timestamps'].to_numpy(), trials_kept[0])
        wf_start = utils_misc.find_nearest(wf_timestamps, trials_kept[0])
        wf_data = pd.DataFrame()
        for key in ['A1', 'ALM', 'tjM1', 'tjS1', 'wM1', 'wM2', 'wS1', 'wS2']:
            wf_data[key] = nwb_read.get_widefield_dff0_traces(nwb_file, key)


        fig, ax = plt.subplots(len(dlc_keys), 1, figsize=(15, 4), sharex=True)
        for i, part in enumerate(dlc_keys):
            if part == 'whisker_angle':
                ax[i].plot(np.where(data_top[f'{part.split("_")[0]}_tip_likelihood'] > 0.6, data_top[part], np.nan)[
                           top_start - 10000:top_start + 10000],
                           c='k')
            else:
                ax[i].plot(np.where(data_side[f'{part.split("_")[0]}_likelihood'] > 0.6, data_side[part], np.nan)[
                           side_start - 10000:side_start + 10000],
                           c='k', label=part)
            ax[i].set_ylabel(part, rotation="horizontal")
            ax[i].yaxis.set_label_coords(-0.075, 0.25)
            ax[i].spines[['right', 'top', 'bottom']].set_visible(False)
            ticks = np.arange(0, 25000, 5000)
            labels = np.arange(trials_kept[0] - 10000 / 200, trials_kept[0] + 15000 / 200, 30)
            labels = [round(label, 2) for label in labels]
            ax[i].set_xticks(ticks, labels)
            ax[i].set_xlabel("Time (s)")
        fig.tight_layout
        fig.savefig(os.path.join(output_path, 'dlc_traces.png'))
        wf_keys = ['A1', 'ALM', 'tjM1', 'tjS1', 'wM1', 'wM2', 'wS1', 'wS2']
        fig, ax = plt.subplots(len(wf_keys), 1, figsize=(15, 4), sharex=True)
        for i, part in enumerate(wf_keys):
            ax[i].plot(wf_data[part][wf_start - 5000:wf_start + 5000], c='r')
            ax[i].set_ylim(-0.03, 0.05)
            ax[i].set_ylabel(part, rotation="horizontal")
            ax[i].yaxis.set_label_coords(-0.075, 0.25)
            ax[i].spines[['right', 'top', 'bottom']].set_visible(False)
            ticks = np.arange(wf_start - 5000, wf_start + 7500, 2500)
            labels = np.arange(trials_kept[0] - 5000 / 100, trials_kept[0] + 7500 / 100, 30)
            labels = [round(label, 2) for label in labels]
            ax[i].set_xticks(ticks, labels)
            ax[i].set_xlabel("Time (s)")
        fig.savefig(os.path.join(output_path, 'wf_traces.png'))


if __name__ == "__main__":
    # Sessions to do
    # session_to_do = ["RD039_20240124_142334", "RD039_20240125_142517",
    #                  "RD039_20240215_142858", "RD039_20240222_145509",
    #                  "RD039_20240228_182245", "RD039_20240229_182734"]
    #
    # session_to_do += ["RD043_20240229_145751", "RD043_20240301_104556",
    #                   "RD043_20240304_143401", "RD043_20240306_175640"]
    #
    # session_to_do += ["RD045_20240227_183215", "RD045_20240228_171641",
    #                   "RD045_20240229_172110", "RD045_20240301_141157"]

    # Selection of sessions with no WF frames missing and 'good' behavior
    # session_to_do = [
    #     "PB173_20240222_103437",
    #     "PB173_20240220_113617", #meh
    #     "PB173_20240221_091701", #meh
    #     "PB173_20240308_151920",
    #     "PB174_20240220_130214",
    #     "PB174_20240221_104529",
    #     "PB174_20240222_120146",
    #     "PB174_20240308_125107",
    #     "PB175_20240307_124909",
    #     "PB175_20240311_170817",
    # ]

    # To do single session
    # session_to_do = ["RD045_20240227_183215", "RD045_20240228_171641",
    #                       "RD045_20240229_172110", "RD045_20240301_141157"]

    # To do sessions free licking & WF
    # session_to_do = ["RD040_20240208_172611", "RD040_20240211_170660", "RD040_20240212_160747"]

    # Group from cicada format
    # config_file = "//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Robin_Dard/group.yaml"
    # with open(config_file, 'r', encoding='utf8') as stream:
    #     config_dict = yaml.safe_load(stream)
    # sessions = config_dict['NWB_CI_LSENS']['Context_expert_sessions']
    # sessions = config_dict['NWB_CI_LSENS']['Context_good_params']
    # sessions = config_dict['NWB_CI_LSENS']['context_expert_widefield']
    # sessions = config_dict['NWB_CI_LSENS']['Context_contrast_expert']
    # sessions = config_dict['NWB_CI_LSENS']['context_contrast_widefield']
    # session_to_do = [session[0] for session in sessions]

    # Group from direct list of pah
    config_path = "//sv-nas1.rcp.epfl.ch/Petersen-Lab/z_LSENS/Pol_Bech/Sessions_list"
    yaml_file = "context_sessions_jrgeco_expert.yaml"
    config_file = os.path.join(config_path, yaml_file)
    with open(config_file, 'r', encoding='utf8') as stream:
        session_dict = yaml.safe_load(stream)
    session_to_do = session_dict['Sessions path']

    # Decide what to do :
    do_wf_movies_average = False
    do_wf_timecourses = False
    do_psths = True
    save_f0 = True
    do_wf_timecourses_mouse_average = True
    compare_context_baseline = True

    # Get list of mouse ID from list of session to do
    if os.path.exists(session_to_do[0]):
        subject_ids = list(np.unique([nwb_read.get_mouse_id(file) for file in session_to_do]))
    else:
        subject_ids = list(np.unique([session[0:5] for session in session_to_do]))

    experimenter_initials_1 = "RD"
    experimenter_initials_2 = "PB"
    root_path_1 = server_path.get_experimenter_nwb_folder(experimenter_initials_1)
    root_path_2 = server_path.get_experimenter_nwb_folder(experimenter_initials_2)
    output_path = os.path.join(f'{server_path.get_experimenter_saving_folder_root(experimenter_initials_1)}',
                               'Pop_results', 'Context_behaviour', 'WF_PSTHs_context_jrgeco_expert_20240614')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    all_nwb_names = os.listdir(root_path_1)

    session_dit = {'Sessions': session_to_do}
    with open(os.path.join(output_path, "session_to_do.yaml"), 'w') as stream:
        yaml.dump(session_dit, stream, default_flow_style=False, explicit_start=True)

    # ---------------------------------------------------------------------------------------------------------- #
    if save_f0:
        for subject_id in subject_ids:
            if os.path.exists(session_to_do[0]):
                nwb_files = [file for file in session_to_do if subject_id in file]
            else:
                nwb_names = [name for name in all_nwb_names if subject_id in name]
                nwb_files = []
                for session in session_to_do:
                    nwb_files += [os.path.join(root_path_1, name) for name in nwb_names if session in name]
            print(" ")
            print(f"nwb_files : {nwb_files}")

            save_f0_image(nwb_files)

    # ---------------------------------------------------------------------------------------------------------- #
    if do_wf_movies_average:
        for subject_id in subject_ids:
            if os.path.exists(session_to_do[0]):
                nwb_files = [file for file in session_to_do if subject_id in file]
            else:
                nwb_names = [name for name in all_nwb_names if subject_id in name]
                nwb_files = []
                for session in session_to_do:
                    nwb_files += [os.path.join(root_path_1, name) for name in nwb_names if session in name]
            print(" ")
            print(f"nwb_files : {nwb_files}")

            make_wf_movies(nwb_files, output_path)

    # ---------------------------------------------------------------------------------------------------------- #
    if do_wf_timecourses:
        trials_list = ['whisker_hit_trial', 'whisker_miss_trial']
        epochs_to_do = ['rewarded', 'non-rewarded']
        for subject_id in subject_ids:
            if os.path.exists(session_to_do[0]):
                nwb_files = [file for file in session_to_do if subject_id in file]
            else:
                nwb_names = [name for name in all_nwb_names if subject_id in name]
                nwb_files = []
                for session in session_to_do:
                    nwb_files += [os.path.join(root_path_1, name) for name in nwb_names if session in name]

            print(f"nwb_files : {nwb_files}")
            plot_wf_activity(nwb_files, trials_list, epochs_to_do, output_path)

    # ---------------------------------------------------------------------------------------------------------- #
    if do_wf_timecourses_mouse_average:
        trials_list = ['whisker_hit_trial', 'whisker_miss_trial']
        epochs_to_do = ['rewarded', 'non-rewarded']
        all_mice = []
        for subject_id in subject_ids:
            if os.path.exists(session_to_do[0]):
                nwb_files = [file for file in session_to_do if subject_id in file]
            else:
                nwb_names = [name for name in all_nwb_names if subject_id in name]
                nwb_files = []
                for session in session_to_do:
                    nwb_files += [os.path.join(root_path_1, name) for name in nwb_names if session in name]

            avg_mice_data = plot_wf_activity_mouse_average(nwb_files, subject_id, trials_list, epochs_to_do, output_path)
            all_mice.append(avg_mice_data)
        key_list = list(all_mice[0].keys())
        general_data_dict = dict()
        for my_key in key_list:
            for i in range(len(all_mice)):
                data = all_mice[i][my_key]
                if i == 0:
                    general_data_dict[my_key] = [data]
                else:
                    general_data_dict[my_key].append(data)

        for key, data in general_data_dict.items():
            print(' ')
            print('Do the plots')
            print(f"Key: {key}, Data shape : {len(data)} mice, with {data[0].shape} shape")
            data = np.stack(data)
            avg_data = np.nanmean(data, axis=0)

            path = os.path.join(output_path, f"{key}_single_frames")
            if not os.path.exists(path):
                os.makedirs(path)

            cutlet_idx = 0
            fig, axes = plt.subplots(1, 4, figsize=(7, 7))
            for start in range(200, 220, 5):
                plot_wf_single_frame(np.nanmean(avg_data[start:start + 5], axis=0),
                                     title=f'{key} {start}-{start + 5}',
                                     fig=fig,
                                     ax_to_plot=axes[cutlet_idx],
                                     colormap='hotcold',
                                     vmin=-0.005,
                                     vmax=0.02,
                                     cbar_shrink=0.75)
                cutlet_idx += 1
            fig.savefig(os.path.join(path, f'all_mice_avg.png'))
            fig.savefig(os.path.join(path, f'all_mice_avg.svg'))
            plt.close()
    # ---------------------------------------------------------------------------------------------------------- #
    if compare_context_baseline:
        for subject_id in subject_ids:
            print(' ')
            print(f'Mouse : {subject_id}')
            if os.path.exists(session_to_do[0]):
                nwb_files = [file for file in session_to_do if subject_id in file]
            else:
                nwb_names = [name for name in all_nwb_names if subject_id in name]
                nwb_files = []
                for session in session_to_do:
                    nwb_files += [os.path.join(root_path_1, name) for name in nwb_names if session in name]
            print(f"nwb_files : {nwb_files}")

            compare_quiet_windows_across_context(nwb_files, output_path, only_correct_trials=False)

    # ---------------------------------------------------------------------------------------------------------- #
    if do_psths:
        # Build one dataframe with all mice
        print("Build general data table")
        df = []
        for subject_id in subject_ids:
            print(f"Concatenate data from mouse {subject_id}")
            if os.path.exists(session_to_do[0]):
                nwb_files = [file for file in session_to_do if subject_id in file]
            else:
                nwb_names = [name for name in all_nwb_names if subject_id in name]
                nwb_files = []
                for session in session_to_do:
                    nwb_files += [os.path.join(root_path_1, name) for name in nwb_names if session in name]

            # Filter to keep only widefield sessions
            nwb_files = [nwb_file for nwb_file in nwb_files if 'wf' in nwb_read.get_session_type(nwb_file)]
            if not nwb_files:
                print(f"No widefield session for {subject_id}")
                continue
            print(f"{len(nwb_files)} NWBs : {nwb_files}")

            # Create dataframe of traces aligned to events
            # trials_dict = [{'whisker_stim': [1], 'lick_flag': [1]},
            #                {'whisker_stim': [1], 'lick_flag': [0]},
            #                {'auditory_stim': [1], 'lick_flag': [1]},
            #                {'auditory_stim': [1], 'lick_flag': [0]}]
            trials_dict = [{'whisker_stim': [1], 'lick_flag': [1]},
                           {'whisker_stim': [1], 'lick_flag': [0]}]
            # trials_dict = [{'no_stim': [1], 'lick_flag': [1]},
            #                {'no_stim': [1], 'lick_flag': [0]}]
            # trials_dict = [{'whisker_stim': [1], 'lick_flag': [1]},
            #                {'whisker_stim': [1], 'lick_flag': [0]},
            #                {'auditory_stim': [1], 'lick_flag': [1]},
            #                {'auditory_stim': [1], 'lick_flag': [0]},
            #                {'no_stim': [1], 'lick_flag': [1]},
            #                {'no_stim': [1], 'lick_flag': [0]}]

            # trial_names = ['whisker_hit',
            #                'whisker_miss',
            #                'auditory_hit',
            #                'auditory_miss']
            trial_names = ['whisker_hit',
                           'whisker_miss']
            # trial_names = ['false_alarm',
            #                'correct_rejection']
            # trial_names = ['whisker_hit',
            #                'whisker_miss',
            #                'auditory_hit',
            #                'auditory_miss',
            #                'false_alarm',
            #                'correct_rejection']

            epochs = ['rewarded', 'non-rewarded']

            t_range = (0.15, 0.25)

            subtract_baseline = True

            mouse_df = return_events_aligned_wf_table(nwb_files=nwb_files,
                                                      rrs_keys=['ophys', 'brain_area_fluorescence', 'dff0_traces'],
                                                      trials_dict=trials_dict,
                                                      trial_names=trial_names,
                                                      epochs=epochs,
                                                      time_range=t_range,
                                                      subtract_baseline=subtract_baseline)
            df.append(mouse_df)
        df = pd.concat(df, ignore_index=True)

        # ---------------------------------------------------------------------------------------------------------- #
        # Group data by sessions
        print(' ')
        print('Average data by session')
        session_avg_data = df.groupby(["mouse_id", "session_id", "trial_type", "epoch",
                                      "behavior_day", "behavior_type", "roi", "cell_type", "time"],
                                      as_index=False).agg(np.nanmean)
        # Group session data by mice
        print(' ')
        print('Average data by mouse')
        mice_avg_data = session_avg_data.drop(['session_id', 'behavior_day'], axis=1)
        mice_avg_data = mice_avg_data.groupby(["mouse_id", "trial_type", "epoch",
                                               "behavior_type", "roi", "cell_type", "time"],
                                              as_index=False).agg(np.nanmean)

        # --------------------------------------------------------------------------------------------------------- #
        print('Do some plots')
        # DO SOME PLOTS #
        figsize = (10, 10)
        y_lim = (-0.005, 0.025)
        # y_lim = (0.012, 0.06)
        save_formats = ['pdf', 'svg']
        # save_formats = ['pdf']

        # -------------------------------- Plot general average --------------------------------------------------- #
        # Plot all area to see successive activation
        print('Plot general average')
        # Whisker trials
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
        for ax in axs.flatten():
            ax.set_ylim(y_lim)
        plt.suptitle(f'Whisker trials average from {len(subject_ids)} mice ({subject_ids})')
        # plt.show()
        saving_folder = os.path.join(output_path)
        if not os.path.exists(saving_folder):
            os.makedirs(saving_folder)
        for save_format in save_formats:
            fig.savefig(os.path.join(saving_folder, f'whisker_trials_average.{save_format}'))
            plt.close()

        # Auditory trials
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=figsize)
        rwd_data_to_plot = mice_avg_data.loc[(mice_avg_data.trial_type.isin(['auditory_hit'])) &
                                             (mice_avg_data.cell_type.isin(
                                                 ['A1', 'tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                             (mice_avg_data.epoch == 'rewarded')]
        sns.lineplot(data=rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[0, 0])
        rwd_data_to_plot = mice_avg_data.loc[(mice_avg_data.trial_type.isin(['auditory_hit'])) &
                                             (mice_avg_data.cell_type.isin(
                                                 ['A1', 'tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                             (mice_avg_data.epoch == 'non-rewarded')]
        sns.lineplot(data=rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[1, 0])
        axs[0, 0].set_title('Auditory hit Rewarded context')
        axs[1, 0].set_title('Auditory hit Non rewarded context')

        nn_rwd_data_to_plot = mice_avg_data.loc[(mice_avg_data.trial_type.isin(['auditory_miss'])) &
                                                (mice_avg_data.cell_type.isin(
                                                    ['A1', 'tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                                (mice_avg_data.epoch == 'rewarded')]
        sns.lineplot(data=nn_rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[0, 1])
        nn_rwd_data_to_plot = mice_avg_data.loc[(mice_avg_data.trial_type.isin(['auditory_miss'])) &
                                                (mice_avg_data.cell_type.isin(
                                                    ['A1', 'tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                                (mice_avg_data.epoch == 'non-rewarded')]
        sns.lineplot(data=nn_rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[1, 1])
        axs[0, 1].set_title('Auditory miss Rewarded context')
        axs[1, 1].set_title('Auditory miss Non rewarded context')
        for ax in axs.flatten():
            ax.set_ylim(y_lim)
        plt.suptitle(f'Auditory trials average from {len(subject_ids)} mice ({subject_ids})')
        # plt.show()
        saving_folder = os.path.join(output_path)
        if not os.path.exists(saving_folder):
            os.makedirs(saving_folder)
        for save_format in save_formats:
            fig.savefig(os.path.join(saving_folder, f'auditory_trials_average.{save_format}'))
            plt.close()

        # # Catch trials
        # fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=figsize)
        # rwd_data_to_plot = mice_avg_data.loc[(mice_avg_data.trial_type.isin(['false_alarm'])) &
        #                                      (mice_avg_data.cell_type.isin(
        #                                          ['tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
        #                                      (mice_avg_data.epoch == 'rewarded')]
        # sns.lineplot(data=rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[0, 0])
        # rwd_data_to_plot = mice_avg_data.loc[(mice_avg_data.trial_type.isin(['false_alarm'])) &
        #                                      (mice_avg_data.cell_type.isin(
        #                                          ['tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
        #                                      (mice_avg_data.epoch == 'non-rewarded')]
        # sns.lineplot(data=rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[1, 0])
        # axs[0, 0].set_title('False alarm Rewarded context')
        # axs[1, 0].set_title('False alarm Non rewarded context')
        #
        # nn_rwd_data_to_plot = mice_avg_data.loc[(mice_avg_data.trial_type.isin(['correct_rejection'])) &
        #                                         (mice_avg_data.cell_type.isin(
        #                                             ['tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
        #                                         (mice_avg_data.epoch == 'rewarded')]
        # sns.lineplot(data=nn_rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[0, 1])
        # nn_rwd_data_to_plot = mice_avg_data.loc[(mice_avg_data.trial_type.isin(['correct_rejection'])) &
        #                                         (mice_avg_data.cell_type.isin(
        #                                             ['tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
        #                                         (mice_avg_data.epoch == 'non-rewarded')]
        # sns.lineplot(data=nn_rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[1, 1])
        # axs[0, 1].set_title('Correct rejection Rewarded context')
        # axs[1, 1].set_title('Correct rejection Non rewarded context')
        # for ax in axs.flatten():
        #     ax.set_ylim(y_lim)
        # plt.suptitle(f'Catch trials average from {len(subject_ids)} mice ({subject_ids})')
        # # plt.show()
        # saving_folder = os.path.join(output_path)
        # if not os.path.exists(saving_folder):
        #     os.makedirs(saving_folder)
        # fig.savefig(os.path.join(saving_folder, 'catch_trials_average.pdf'))
        # plt.close()

        # Plot by area
        areas = ['A1', 'wS1', 'wS2', 'wM1', 'wM2', 'ALM', 'tjM1', 'tjS1']
        for area in areas:
            # Whisker
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            data_to_plot = mice_avg_data.loc[(mice_avg_data.cell_type == area) &
                                             (mice_avg_data.trial_type.isin(['whisker_hit', 'whisker_miss']))]
            sns.lineplot(data=data_to_plot, x='time', y='activity', hue='epoch', style='trial_type', ax=ax)
            ax.set_ylim(y_lim)
            plt.suptitle(f"{area} response to whisker trials average from {len(subject_ids)} mice ({subject_ids})")
            # plt.show()
            saving_folder = os.path.join(output_path)
            if not os.path.exists(saving_folder):
                os.makedirs(saving_folder)
            for save_format in save_formats:
                fig.savefig(os.path.join(saving_folder, f'whisker_trials_average_{area}.{save_format}'))
                plt.close()

            # Auditory
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            data_to_plot = mice_avg_data.loc[(mice_avg_data.cell_type == area) &
                                             (mice_avg_data.trial_type.isin(['auditory_hit', 'auditory_miss']))]
            sns.lineplot(data=data_to_plot, x='time', y='activity', hue='epoch', style='trial_type', ax=ax)
            ax.set_ylim(y_lim)
            plt.suptitle(f"{area} response to auditory trials average from {len(subject_ids)} mice ({subject_ids})")
            # plt.show()
            saving_folder = os.path.join(output_path)
            if not os.path.exists(saving_folder):
                os.makedirs(saving_folder)
            for save_format in save_formats:
                fig.savefig(os.path.join(saving_folder, f'auditory_trials_average_{area}.{save_format}'))
                plt.close()

            # Catch
            # fig, ax = plt.subplots(1, 1, figsize=figsize)
            # data_to_plot = mice_avg_data.loc[(mice_avg_data.cell_type == area) &
            #                                  (mice_avg_data.trial_type.isin(['false_alarm', 'correct_rejection']))]
            # sns.lineplot(data=data_to_plot, x='time', y='activity', hue='epoch', style='trial_type', ax=ax)
            # ax.set_ylim(y_lim)
            # plt.suptitle(f"{area} response to catch trials average from {len(subject_ids)} mice ({subject_ids})")
            # # plt.show()
            # saving_folder = os.path.join(output_path)
            # if not os.path.exists(saving_folder):
            #     os.makedirs(saving_folder)
            # fig.savefig(os.path.join(saving_folder, f'catch_trials_average_{area}.pdf'))
            # plt.close()

        # ------------------------------------ Plot by mouse ----------------------------------------------------- #
        print(" ")
        print("Plot for each mouse")
        for subject_id in subject_ids:
            print(f"Mouse {subject_id}")
            # List subject sessions
            if os.path.exists(session_to_do[0]):
                subject_sessions = [nwb_read.get_session_id(session) for session in session_to_do
                                    if subject_id in session]
            else:
                subject_sessions = [session for session in session_to_do if subject_id in session]
            # Average per session : Plot with one point per time per session:
            session_avg_data_mouse = session_avg_data.loc[session_avg_data.mouse_id == subject_id]
            # Plot all area to see successive activation
            # Whisker
            fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=figsize)
            rwd_data_to_plot = session_avg_data_mouse.loc[(session_avg_data_mouse.trial_type.isin(['whisker_hit'])) &
                                                    (session_avg_data_mouse.cell_type.isin(
                                                        ['tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                                    (session_avg_data_mouse.epoch == 'rewarded')]
            sns.lineplot(data=rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[0, 0])
            rwd_data_to_plot = session_avg_data_mouse.loc[(session_avg_data_mouse.trial_type.isin(['whisker_hit'])) &
                                                    (session_avg_data_mouse.cell_type.isin(
                                                        ['tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                                    (session_avg_data_mouse.epoch == 'non-rewarded')]
            sns.lineplot(data=rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[1, 0])
            axs[0, 0].set_title('Whisker hit Rewarded context')
            axs[1, 0].set_title('Whisker hit Non rewarded context')

            nn_rwd_data_to_plot = session_avg_data_mouse.loc[(session_avg_data_mouse.trial_type.isin(['whisker_miss'])) &
                                                       (session_avg_data_mouse.cell_type.isin(
                                                           ['tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                                       (session_avg_data_mouse.epoch == 'rewarded')]
            sns.lineplot(data=nn_rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[0, 1])
            nn_rwd_data_to_plot = session_avg_data_mouse.loc[(session_avg_data_mouse.trial_type.isin(['whisker_miss'])) &
                                                       (session_avg_data_mouse.cell_type.isin(
                                                           ['tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                                       (session_avg_data_mouse.epoch == 'non-rewarded')]
            sns.lineplot(data=nn_rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[1, 1])
            axs[0, 1].set_title('Whisker miss Rewarded context')
            axs[1, 1].set_title('Whisker miss Non rewarded context')
            for ax in axs.flatten():
                ax.set_ylim(y_lim)
            plt.suptitle(f'{subject_id} : average from {len(subject_sessions)} sessions')
            # plt.show()
            saving_folder = os.path.join(output_path, f"{subject_id}")
            if not os.path.exists(saving_folder):
                os.makedirs(saving_folder)
            for save_format in save_formats:
                fig.savefig(os.path.join(saving_folder, f"{subject_id}_whisker_trials_average.{save_format}"))
                plt.close()

            # Auditory
            fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=figsize)
            rwd_data_to_plot = session_avg_data_mouse.loc[(session_avg_data_mouse.trial_type.isin(['auditory_hit'])) &
                                                    (session_avg_data_mouse.cell_type.isin(
                                                        ['A1', 'tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                                    (session_avg_data_mouse.epoch == 'rewarded')]
            sns.lineplot(data=rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[0, 0])
            rwd_data_to_plot = session_avg_data_mouse.loc[(session_avg_data_mouse.trial_type.isin(['auditory_hit'])) &
                                                    (session_avg_data_mouse.cell_type.isin(
                                                        ['A1', 'tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                                    (session_avg_data_mouse.epoch == 'non-rewarded')]
            sns.lineplot(data=rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[1, 0])
            axs[0, 0].set_title('Auditory hit Rewarded context')
            axs[1, 0].set_title('Auditory hit Non rewarded context')

            nn_rwd_data_to_plot = session_avg_data_mouse.loc[(session_avg_data_mouse.trial_type.isin(['auditory_miss'])) &
                                                       (session_avg_data_mouse.cell_type.isin(
                                                           ['A1', 'tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                                       (session_avg_data_mouse.epoch == 'rewarded')]
            sns.lineplot(data=nn_rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[0, 1])
            nn_rwd_data_to_plot = session_avg_data_mouse.loc[(session_avg_data_mouse.trial_type.isin(['auditory_miss'])) &
                                                       (session_avg_data_mouse.cell_type.isin(
                                                           ['A1', 'tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                                       (session_avg_data_mouse.epoch == 'non-rewarded')]
            sns.lineplot(data=nn_rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[1, 1])
            axs[0, 1].set_title('Auditory miss Rewarded context')
            axs[1, 1].set_title('Auditory miss Non rewarded context')
            for ax in axs.flatten():
                ax.set_ylim(y_lim)
            plt.suptitle(f'{subject_id} : average from {len(subject_sessions)} sessions')
            # plt.show()
            saving_folder = os.path.join(output_path, f"{subject_id}")
            if not os.path.exists(saving_folder):
                os.makedirs(saving_folder)
            for save_format in save_formats:
                fig.savefig(os.path.join(saving_folder, f"{subject_id}_auditory_trials_average.{save_format}"))
                plt.close()

            # Plot per area to compare the two contexts in each:
            areas = ['A1', 'wS1', 'wS2', 'wM1', 'wM2', 'tjM1']
            for area in areas:
                # Whiskers
                fig, ax = plt.subplots(1, 1, figsize=figsize)
                data_to_plot = session_avg_data_mouse.loc[(session_avg_data_mouse.cell_type == area) &
                                                    (session_avg_data_mouse.trial_type.isin(
                                                        ['whisker_hit', 'whisker_miss']))]
                sns.lineplot(data=data_to_plot, x='time', y='activity', hue='epoch', style='trial_type', ax=ax)
                ax.set_ylim(y_lim)
                plt.suptitle(f"{area} response to whisker trials : average from {len(subject_sessions)} sessions")
                # plt.show()
                saving_folder = os.path.join(output_path, f"{subject_id}")
                if not os.path.exists(saving_folder):
                    os.makedirs(saving_folder)
                fig.savefig(os.path.join(saving_folder, f'{subject_id}_whisker_trials_average_{area}.png'))
                plt.close()

                # Auditory
                fig, ax = plt.subplots(1, 1, figsize=figsize)
                data_to_plot = session_avg_data_mouse.loc[(session_avg_data_mouse.cell_type == area) &
                                                    (session_avg_data_mouse.trial_type.isin(
                                                        ['auditory_hit', 'auditory_miss']))]
                sns.lineplot(data=data_to_plot, x='time', y='activity', hue='epoch', style='trial_type', ax=ax)
                ax.set_ylim(y_lim)
                plt.suptitle(f"{area} response to auditory trials : average from {len(subject_sessions)} sessions")
                # plt.show()
                saving_folder = os.path.join(output_path, f"{subject_id}")
                if not os.path.exists(saving_folder):
                    os.makedirs(saving_folder)
                for save_format in save_formats:
                    fig.savefig(os.path.join(saving_folder, f'{subject_id}_auditory_trials_average_{area}.{save_format}'))
                    plt.close()

            # Plot with single session
            print('Plot single session')
            for session in subject_sessions:
                print(f"Session {session}")
                # Plot with all areas
                # Whisker
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
                # for ax in axs.flatten():
                #     ax.set_ylim(y_lim)
                plt.suptitle(f"{session}, whisker trials")
                # plt.show()
                saving_folder = os.path.join(output_path, f"{session[0:5]}", f"{session}")
                if not os.path.exists(saving_folder):
                    os.makedirs(saving_folder)
                for save_format in save_formats:
                    fig.savefig(os.path.join(saving_folder, f"{session}_whisker_trials.{save_format}"))
                    plt.close()

                # Auditory
                fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=figsize)
                rwd_data_to_plot = df.loc[(df.trial_type.isin(['auditory_hit'])) &
                                          (df.cell_type.isin(
                                              ['A1', 'tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                          (df.epoch == 'rewarded') &
                                          (df.session_id == session)]
                sns.lineplot(data=rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[0, 0])
                rwd_data_to_plot = df.loc[(df.trial_type.isin(['auditory_hit'])) &
                                          (df.cell_type.isin(
                                              ['A1', 'tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                          (df.epoch == 'non-rewarded') &
                                          (df.session_id == session)]
                sns.lineplot(data=rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[1, 0])
                axs[0, 0].set_title('Auditory hit Rewarded context')
                axs[1, 0].set_title('Auditory hit Non rewarded context')

                nn_rwd_data_to_plot = df.loc[(df.trial_type.isin(['auditory_miss'])) &
                                             (df.cell_type.isin(
                                                 ['A1', 'tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                             (df.epoch == 'rewarded') &
                                             (df.session_id == session)]
                sns.lineplot(data=nn_rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[0, 1])
                nn_rwd_data_to_plot = df.loc[(df.trial_type.isin(['auditory_miss'])) &
                                             (df.cell_type.isin(
                                                 ['A1', 'tjS1', 'wS1', 'wS2', 'tjM1', 'wM1', 'wM2'])) &
                                             (df.epoch == 'non-rewarded') &
                                             (df.session_id == session)]
                sns.lineplot(data=nn_rwd_data_to_plot, x='time', y='activity', hue='cell_type', ax=axs[1, 1])
                axs[0, 1].set_title('Auditory miss Rewarded context')
                axs[1, 1].set_title('Auditory miss Non rewarded context')
                # for ax in axs.flatten():
                #     ax.set_ylim(y_lim)
                plt.suptitle(f"{session}, auditory trials")
                # plt.show()
                saving_folder = os.path.join(output_path, f"{session[0:5]}", f"{session}")
                if not os.path.exists(saving_folder):
                    os.makedirs(saving_folder)
                for save_format in save_formats:
                    fig.savefig(os.path.join(saving_folder, f"{session}_auditory_trials.{save_format}"))
                    plt.close()

                # Plot by area
                areas = ['A1', 'wS1', 'wS2', 'wM1', 'wM2', 'tjM1']
                for area in areas:
                    # Whisker
                    sub_data_to_plot = df.loc[(df.cell_type == area) &
                                              (df.trial_type.isin(['whisker_hit', 'whisker_miss'])) &
                                              (df.session_id == session)]
                    fig, ax = plt.subplots(1, 1, figsize=figsize)
                    sns.lineplot(data=sub_data_to_plot, x='time', y='activity', hue='epoch', style='trial_type', ax=ax)
                    # ax.set_ylim(y_lim)
                    plt.suptitle(f"{session}, {area} response to whisker trials")
                    # plt.show()
                    saving_folder = os.path.join(output_path, f"{session[0:5]}", f"{session}")
                    if not os.path.exists(saving_folder):
                        os.makedirs(saving_folder)
                    fig.savefig(os.path.join(saving_folder, f"{session}_whisker_trials_{area}.pdf"))
                    plt.close()

                    # Auditory
                    sub_data_to_plot = df.loc[(df.cell_type == area) &
                                              (df.trial_type.isin(['auditory_hit', 'auditory_miss'])) &
                                              (df.session_id == session)]
                    fig, ax = plt.subplots(1, 1, figsize=figsize)
                    sns.lineplot(data=sub_data_to_plot, x='time', y='activity', hue='epoch', style='trial_type', ax=ax)
                    # ax.set_ylim(y_lim)
                    plt.suptitle(f"{session}, {area} response to auditory trials")
                    # plt.show()
                    saving_folder = os.path.join(output_path, f"{session[0:5]}", f"{session}")
                    if not os.path.exists(saving_folder):
                        os.makedirs(saving_folder)
                    for save_format in save_formats:
                        fig.savefig(os.path.join(saving_folder, f"{session}_auditory_trials_{area}.{save_format}"))
                        plt.close()

