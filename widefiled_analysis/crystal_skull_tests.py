import os
import sys
sys.path.append(os.getcwd())
import glob
import imageio as iio

import matplotlib.pyplot as plt
import yaml
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, rgb2hex, hex2color, TwoSlopeNorm
from matplotlib.lines import Line2D            
from itertools import product

import imageio as iio
import nwb_utils.utils_behavior as bhv_utils
import utils.behaviour_plot_utils as plot_utils
from utils.wf_plotting_utils import get_colormap
from utils.haas_utils import *
import nwb_wrappers.nwb_reader_functions as nwb_read
from nwb_utils import utils_misc
from nwb_utils.utils_plotting import lighten_color, remove_top_right_frame


def compute_dff0_and_anatomical(wf_data, save_path):
    wf_488 = np.rollaxis(wf_data[:, 0:-1:2, ...], axis=1)
    wf_405 = np.rollaxis(wf_data[:, 1:-1:2, ...], axis=1)

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(wf_488[0, 0, :, :])
    ax[1].imshow(wf_405[0, 0, :, :])
    ax[0].set_title('488')
    ax[1].set_title('405')
    fig.savefig(os.path.join(save_path, "anatomical.png"))

    dff0_488 = (wf_488-np.nanmean(wf_488[25:49], axis=0))/np.nanmean(wf_488[25:49], axis=0)
    dff0_405 = (wf_405-np.nanmean(wf_405[25:49], axis=0))/np.nanmean(wf_405[25:49], axis=0)

    return dff0_488, dff0_405


if __name__ == "__main__":

    mouseID = 'RD083'
    if mouseID == 'RD082':
        file = "/mnt/lsens-data/RD082/Training/RD082_20250519_132306/results.csv"
        wf_file = "/mnt/lsens-data/RD082/Recording/Imaging/RD082_20250519_132306/RD082_20250519_132306.mj2"
    elif mouseID == 'RD083':
        file = "/mnt/lsens-data/RD083/Training/RD083_20250519_114542/results.csv"
        wf_file = "/mnt/lsens-data/RD083/Recording/Imaging/RD083_20250519_114542/RD083_20250519_114542.mj2"

    save_path = f"/mnt/lsens-analysis/Pol_Bech/Pop_results/test/{mouseID}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(os.path.join(save_path, 'whisker'))
        os.makedirs(os.path.join(save_path, 'auditory'))
        os.makedirs(os.path.join(save_path, 'catch'))

    df = pd.read_csv(file)
    wf_data = iio.v3.imread(wf_file, plugin='pyav', format='gray16be')
    wf_data = wf_data.reshape(df.shape[0], -1, 250, 360, order='C')
    wf_data = wf_data.reshape(wf_data.shape[0], wf_data.shape[1], int(wf_data.shape[2] / 2), 2, int(wf_data.shape[3] / 2), 2).mean(axis=3).mean(axis=4)

    dff0_488, dff0_405 = compute_dff0_and_anatomical(wf_data, save_path)

    cmap = get_colormap('hotcold')
    norm = TwoSlopeNorm(vmin=-0.02, vcenter=0, vmax=0.02)

    wh_frame_488 = np.nanmean(dff0_488[:, df.is_whisker, :, :], axis=1)
    wh_frame_405 = np.nanmean(dff0_405[:, df.is_whisker, :, :], axis=1)
    wf_sub = wh_frame_488[:-1] - wh_frame_405

    for i in range(100):
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(wh_frame_488[i], cmap='viridis', norm=norm)
        ax[1].imshow(wh_frame_405[i], cmap='viridis', norm=norm)
        ax[2].imshow(wf_sub[i], cmap='viridis', norm=norm)
        fig.savefig(os.path.join(save_path, 'whisker', f'whisker_avg_frame_{i}.png'))
        fig.clear()

    aud_frame_488 = np.nanmean(dff0_488[:, df.is_auditory, :, :], axis=1)
    aud_frame_405 = np.nanmean(dff0_405[:, df.is_auditory, :, :], axis=1)
    aud_sub = aud_frame_488[:-1] - aud_frame_405

    for i in range(100):
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(aud_frame_488[i], cmap=cmap, norm=norm)
        ax[1].imshow(aud_frame_405[i], cmap=cmap, norm=norm)
        ax[2].imshow(aud_sub[i], cmap=cmap, norm=norm)
        fig.savefig(os.path.join(save_path, 'auditory', f'auditory_avg_frame_{i}.png'))
        fig.clear()

    catch_frame_488 = np.nanmean(dff0_488[:, ~df.is_stim, :, :], axis=1)
    catch_frame_405 = np.nanmean(dff0_405[:, ~df.is_stim, :, :], axis=1)
    catch_sub = catch_frame_488[:-1] - catch_frame_405

    for i in range(100):
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(catch_frame_488[i], cmap=cmap, norm=norm)
        ax[1].imshow(catch_frame_405[i], cmap=cmap, norm=norm)
        ax[2].imshow(catch_sub[i], cmap=cmap, norm=norm)
        fig.savefig(os.path.join(save_path, 'catch', f'catch_avg_frame_{i}.png'))
        fig.clear()