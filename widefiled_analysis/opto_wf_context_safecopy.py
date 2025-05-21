import itertools
import os
import sys
sys.path.append(os.getcwd())
import glob
import warnings

import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
import yaml
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, rgb2hex, hex2color
from matplotlib.lines import Line2D            
from itertools import product


import nwb_utils.utils_behavior as bhv_utils
import utils.behaviour_plot_utils as plot_utils
from utils.haas_utils import *
import nwb_wrappers.nwb_reader_functions as nwb_read
from nwb_utils import utils_misc
from nwb_utils.utils_plotting import lighten_color, remove_top_right_frame
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics.pairwise import paired_distances
from utils.wf_plotting_utils import reduce_im_dimensions, plot_grid_on_allen, generate_reduced_image_df


def get_frames_by_epoch(nwb_file, trials, wf_timestamps, start=0, stop=200):
    frames = []
    for tstamp in trials:
        frame = utils_misc.find_nearest(wf_timestamps, tstamp)
        data = nwb_read.get_widefield_dff0(nwb_file, ['ophys', 'dff0'], int(frame + start), int(frame + stop))
        if data.shape != (len(np.arange(start, stop)), 125, 160):
            data = np.ones([stop-start, 125, 160]) * np.nan
        else:
            data = data - np.nanmean(data[:48], axis=0)
        frames.append([data[49:100]])

    data_frames = np.array(frames)
    data_frames = np.stack(data_frames, axis=0)
    return data_frames


def get_dff0_traces_by_epoch(nwb_file, trials, wf_timestamps, start=0, stop=250):
    wf_data = pd.DataFrame(columns=['A1', 'ALM', 'tjM1', 'tjS1', 'RSC', 'wM1', 'wM2', 'wS1', 'wS2'])
    indices = nwb_read.get_cell_indices_by_cell_type(nwb_file, ['ophys', 'brain_area_fluorescence', 'dff0_traces'])
    for key in indices.keys():
        wf_data[key] = nwb_read.get_widefield_dff0_traces(nwb_file, ['ophys', 'brain_area_fluorescence', 'dff0_traces'])[indices[key][0]]

    data = []
    for tstamp in trials:
        frame = utils_misc.find_nearest(wf_timestamps, tstamp)
        wf = wf_data.loc[frame+start:frame+stop-1].to_numpy()
        if wf.shape != (len(np.arange(start, stop)), wf_data.shape[1]):
            wf = np.ones([len(np.arange(start, stop)), wf_data.shape[1]]) * np.nan
        else:
            wf = wf - np.nanmean(wf[:48], axis=0)
        data += [wf]

    data = np.array(data)
    data = np.stack(data, axis=0)

    wf_data = pd.DataFrame(columns=['A1', 'ALM', 'tjM1', 'tjS1', 'RSC', 'wM1', 'wM2', 'wS1', 'wS2'])
    for i, loc in enumerate(wf_data.keys()):
        wf_data[loc] = [data[j,:,i] for j in range(data.shape[0])]

    return wf_data


def get_reduced_im_by_epoch(nwb_file, trials, wf_timestamps, start=0, stop=200):
    frames = []
    rrs_keys = ['ophys', 'brain_grid_fluorescence', 'dff0_grid_traces']
    traces = nwb_read.get_roi_response_serie_data(nwb_file=nwb_file, keys=rrs_keys)
    area_dict = nwb_read.get_cell_indices_by_cell_type(nwb_file=nwb_file, keys=rrs_keys)

    for tstamp in trials:
        frame = utils_misc.find_nearest(wf_timestamps, tstamp)
        data = nwb_read.get_roi_response_serie_data(nwb_file=nwb_file, keys=rrs_keys)[:, int(frame + start):int(frame + stop)].T
        if data.shape != (len(np.arange(start, stop)), 42):
            data = np.ones([stop-start, 42]) * np.nan
        else:
            data = data - np.nanmean(data[:48], axis=0)

        frames.append([data])

    frames = np.stack(np.array(frames).squeeze(), axis=0)
    wf_data = pd.DataFrame(columns=area_dict.keys())
    for i, loc in enumerate(wf_data.keys()):
        wf_data[loc] = [frames[j,:,area_dict[loc].squeeze()] for j in range(frames.shape[0])]
    wf_data['time'] = [[np.linspace(-1,3.98,250)] for i in range(wf_data.shape[0])]
    return wf_data


def combine_data(nwb_files, output_path):
    # coords_list = {'wS1': [[-1.5, 3.5], [-1.5, 4.5], [-2.5, 3.5], [-2.5, 4.5]], 'wM1': [[1.5, 1.5]], 'wM2': [[2.5,1.5]], 'RSC': [[-0.5, 0.5], [-1.5,0.5]],
    #                'ALM': [[2.5, 2.5]], 'tjS1':[[0.5, 4.5]], 'tjM1':[[1.5, 3.5]], 'control': [[-5.0, 5.0]]} #AP, ML
    for nwb_file in nwb_files:
        session_df = []
        bhv_data = bhv_utils.build_standard_behavior_table([nwb_file])
        if bhv_data.trial_id.duplicated().sum()>0:
            bhv_data['trial_id'] = bhv_data.index.values

        bhv_data = bhv_data.loc[(bhv_data.early_lick==0) & (bhv_data.opto_grid_ap!=3.5)]
        bhv_data['opto_stim_coord'] = bhv_data.apply(lambda x: f"({x.opto_grid_ap}, {x.opto_grid_ml})",axis=1)
        wf_timestamps = nwb_read.get_widefield_timestamps(nwb_file, ['ophys', 'dff0'])
        session_id = nwb_read.get_session_id(nwb_file)
        mouse_id = nwb_read.get_mouse_id(nwb_file)
        print(f"--------{session_id}-------- ")

        for loc in bhv_data.opto_stim_coord.unique():
            opto_data = bhv_data.loc[bhv_data.opto_stim_coord==loc]

            if loc=='(-5.0, 5.0)':
                opto_data['opto_stim_loc'] = 'control'
            else:
                opto_data['opto_stim_loc'] = 'stim'

            trials = opto_data.start_time

            # wf_image = get_frames_by_epoch(nwb_file, trials, wf_timestamps, start=0, stop=250) #250 frames since frame rate in this sessions is 50Hz and trial duration is 5s
            # opto_data['wf_image_shape'] = [wf_image[i].shape for i in range(opto_data.shape[0])]           
            # opto_data['wf_images'] = [wf_image[i].flatten(order='C') for i in range(opto_data.shape[0])]

            wf_image = get_reduced_im_by_epoch(nwb_file, trials, wf_timestamps, start=0, stop=250)
            wf_image['trial_id'] = opto_data.trial_id.values
            opto_data = pd.merge(opto_data.reset_index(drop=True), wf_image, on='trial_id')

            roi_data = get_dff0_traces_by_epoch(nwb_file, trials, wf_timestamps, start=0, stop=250)
            roi_data['trial_id'] = opto_data.trial_id.values
            opto_data = pd.merge(opto_data.reset_index(drop=True), roi_data, on='trial_id')
            print(f"Final shape after merging: {opto_data.shape[0]}")
            session_df += [opto_data]

        session_df = pd.concat(session_df, ignore_index=True)
        if not os.path.exists(Path(output_path, session_id)):
            os.makedirs(Path(output_path, session_id))
        session_df.to_parquet(Path(output_path, session_id, 'results.parquet.gzip', compression='gzip'))


def plot_opto_effect_matrix(nwb_files, output_path):

    if not os.path.exists(Path(output_path, 'heatmaps')):
        os.makedirs(Path(output_path, 'heatmaps'))
    # Load opto effect
    opto_df = []
    opto_data_path = glob.glob(os.path.join(output_path, 'opto_results', "*", "opto_data.json"))
    opto_df += [pd.read_json(file) for file in opto_data_path]
    opto_df = pd.concat(opto_df)
    opto_df = opto_df.loc[opto_df.opto_grid_ap!=3.5]
    avg_opto_df = opto_df.groupby(by=['context', 'trial_type', 'opto_grid_ml', 'opto_grid_ap']).agg(
                                                                                 data=('data_mean', list),
                                                                                 data_sub=('data_mean_sub', list),
                                                                                 data_mean=('data_mean', 'mean'),
                                                                                 data_mean_sub=(
                                                                                 'data_mean_sub', 'mean'),
                                                                                 shuffle_dist=('shuffle_dist', 'sum'),
                                                                                 shuffle_dist_sub=(
                                                                                 'shuffle_dist_sub', 'sum'),
                                                                                 percentile_avg=('percentile', 'mean'),
                                                                                 percentile_avg_sub=(
                                                                                 'percentile_sub', 'mean'),
                                                                                 n_sigma_avg=('n_sigma', 'mean'),
                                                                                 n_sigma_avg_sub=(
                                                                                 'n_sigma_sub', 'mean')).reset_index()
    avg_opto_df['opto_stim_coord'] = avg_opto_df.apply(lambda x: f"({x.opto_grid_ap}, {x.opto_grid_ml})", axis=1)

    # Load and process opto_wf effect
    total_df = []
    for nwb_file in nwb_files:
        session_id = nwb_read.get_session_id(nwb_file)
        mouse_id = nwb_read.get_mouse_id(nwb_file)
        total_df += [pd.read_parquet(Path(output_path, session_id, 'results.parquet.gzip', compression='gzip'))]

    total_df = pd.concat(total_df, ignore_index=True)
    total_df.context = total_df.context.map({0:'non-rewarded', 1:'rewarded'})
    total_df['time'] = [[np.linspace(-1,3.98,250)] for i in range(total_df.shape[0])]
    total_df['legend']= total_df.apply(lambda x: f"{'control' if x.opto_stim_loc=='control' else 'stim'} - {'lick' if x.lick_flag==1 else 'no lick'}",axis=1)

    d = {c: lambda x: x.unique()[0] for c in ['opto_stim_loc', 'legend']}
    d['time'] = lambda x: list(x)[0][0]
    for c in ['(-0.5, 0.5)', '(-0.5, 1.5)', '(-0.5, 2.5)', '(-0.5, 3.5)', '(-0.5, 4.5)', '(-0.5, 5.5)', '(-1.5, 0.5)',
       '(-1.5, 1.5)', '(-1.5, 2.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(-1.5, 5.5)', '(-2.5, 0.5)', '(-2.5, 1.5)', '(-2.5, 2.5)',
       '(-2.5, 3.5)', '(-2.5, 4.5)', '(-2.5, 5.5)', '(-3.5, 0.5)', '(-3.5, 1.5)', '(-3.5, 2.5)', '(-3.5, 3.5)', '(-3.5, 4.5)',
       '(-3.5, 5.5)', '(0.5, 0.5)', '(0.5, 1.5)', '(0.5, 2.5)', '(0.5, 3.5)', '(0.5, 4.5)', '(0.5, 5.5)', '(1.5, 0.5)', 
       '(1.5, 1.5)', '(1.5, 2.5)', '(1.5, 3.5)', '(1.5, 4.5)', '(1.5, 5.5)', '(2.5, 0.5)', '(2.5, 1.5)', '(2.5, 2.5)', '(2.5, 3.5)', '(2.5, 4.5)', '(2.5, 5.5)', 
       'A1', 'ALM', 'tjM1', 'tjS1', 'RSC', 'wM1', 'wM2', 'wS1', 'wS2']:
        d[f"{c}"]= lambda x: np.nanmean(np.stack(x), axis=0)
          
    mouse_df = total_df.groupby(by=['mouse_id', 'context', 'trial_type', 'opto_stim_coord', 'lick_flag']).agg(d).reset_index()
    mouse_df = mouse_df.melt(id_vars=['mouse_id', 'context', 'trial_type', 'opto_stim_loc', 'opto_stim_coord', 'lick_flag', 'legend', 'time'],
                                 value_vars=['(-0.5, 0.5)', '(-0.5, 1.5)', '(-0.5, 2.5)', '(-0.5, 3.5)', '(-0.5, 4.5)', '(-0.5, 5.5)', '(-1.5, 0.5)',
       '(-1.5, 1.5)', '(-1.5, 2.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(-1.5, 5.5)', '(-2.5, 0.5)', '(-2.5, 1.5)', '(-2.5, 2.5)',
       '(-2.5, 3.5)', '(-2.5, 4.5)', '(-2.5, 5.5)', '(-3.5, 0.5)', '(-3.5, 1.5)', '(-3.5, 2.5)', '(-3.5, 3.5)', '(-3.5, 4.5)',
       '(-3.5, 5.5)', '(0.5, 0.5)', '(0.5, 1.5)', '(0.5, 2.5)', '(0.5, 3.5)', '(0.5, 4.5)', '(0.5, 5.5)', '(1.5, 0.5)', 
       '(1.5, 1.5)', '(1.5, 2.5)', '(1.5, 3.5)', '(1.5, 4.5)', '(1.5, 5.5)', '(2.5, 0.5)', '(2.5, 1.5)', '(2.5, 2.5)', '(2.5, 3.5)', '(2.5, 4.5)', '(2.5, 5.5)',],
                                 var_name='roi',
                                 value_name='dff0').explode(['time', 'dff0'])
    avg_df = mouse_df.groupby(by=['context', 'trial_type', 'lick_flag', 'opto_stim_coord', 'roi', 'time']).agg(lambda x: np.nanmean(x)).reset_index()

    # Plot timecourses per stim location trial type and lick over time
    for stim_coord in avg_df.opto_stim_coord.unique():
        for trial in avg_df.trial_type.unique():
            subset = avg_df[(avg_df.opto_stim_coord==stim_coord) & (avg_df.trial_type==trial)]
            subset.time = subset.time.round(2)
            fig,ax = plt.subplots(2,2, figsize=(8,12), sharex=True)
            fig.suptitle(f"{trial}, stim_loc: {stim_coord}")
            for name, group in subset.groupby(by = ['context', 'lick_flag']):
                row = 0 if name[0]=='rewarded' else 1
                col = 0 if name[1]==1 else 1
                df = group.pivot_table(values='dff0', index='roi', columns='time')
                g = sns.heatmap(df,vmin=-0.03, vmax=0.05, center=0, cmap='icefire', ax=ax[row,col], cbar=True, yticklabels=True)
                g.set_yticklabels(g.get_yticklabels(), size = 10)
                ymin, ymax = ax[row,col].get_ylim()
                ax[row, col].vlines(50, ymin, ymax, colors='white', linestyles='dashed')
                
            ax[0, 0].set_title('Lick')
            ax[0, 1].set_title('No Lick')
            ax[0, 0].set_ylabel('Rewarded')
            ax[1, 0].set_ylabel('Non-rewarded')
            fig.tight_layout()
            fig.savefig(os.path.join(output_path, 'heatmaps', f'{trial}_{stim_coord}_heatmap.png'))


    seismic_palette = sns.diverging_palette(265, 10, s=100, l=40, sep=30, n=200, center="light", as_cmap=True)

    mouse_df = total_df.groupby(by=['mouse_id', 'context', 'trial_type', 'opto_stim_coord']).agg(d).reset_index()
    mouse_df = mouse_df.melt(id_vars=['mouse_id', 'context', 'trial_type', 'opto_stim_loc', 'opto_stim_coord', 'time'],
                                 value_vars=['(-0.5, 0.5)', '(-0.5, 1.5)', '(-0.5, 2.5)', '(-0.5, 3.5)', '(-0.5, 4.5)', '(-0.5, 5.5)', '(-1.5, 0.5)',
       '(-1.5, 1.5)', '(-1.5, 2.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(-1.5, 5.5)', '(-2.5, 0.5)', '(-2.5, 1.5)', '(-2.5, 2.5)',
       '(-2.5, 3.5)', '(-2.5, 4.5)', '(-2.5, 5.5)', '(-3.5, 0.5)', '(-3.5, 1.5)', '(-3.5, 2.5)', '(-3.5, 3.5)', '(-3.5, 4.5)',
       '(-3.5, 5.5)', '(0.5, 0.5)', '(0.5, 1.5)', '(0.5, 2.5)', '(0.5, 3.5)', '(0.5, 4.5)', '(0.5, 5.5)', '(1.5, 0.5)', 
       '(1.5, 1.5)', '(1.5, 2.5)', '(1.5, 3.5)', '(1.5, 4.5)', '(1.5, 5.5)', '(2.5, 0.5)', '(2.5, 1.5)', '(2.5, 2.5)', '(2.5, 3.5)', '(2.5, 4.5)', '(2.5, 5.5)',],
                                 var_name='roi',
                                 value_name='dff0').explode(['time', 'dff0'])
    avg_df = mouse_df.groupby(by=['context', 'trial_type', 'opto_stim_coord', 'roi', 'time']).agg(lambda x: np.nanmean(x)).reset_index()

    time_df = avg_df[(avg_df.time>=0) & (avg_df.time<=0.5)].groupby(by=['context', 'trial_type', 'opto_stim_coord', 'roi']).agg(lambda x: np.nanmean(x)).reset_index()
    for name, group in time_df.groupby(by=['context', 'trial_type']):
        subset = group.pivot_table(index='opto_stim_coord', columns='roi', values='dff0', aggfunc='mean')
        subset = subset-subset.loc["(-5.0, 5.0)"]
        subset = subset.drop("(-5.0, 5.0)")
        mask = np.eye(subset.shape[0])      

        opto_subset = avg_opto_df.loc[(avg_opto_df.context==name[0]) & (avg_opto_df.trial_type==name[1]),['data_mean_sub', 'opto_stim_coord']].set_index('opto_stim_coord').sort_values(by='data_mean_sub')
        new_idx = opto_subset.index
        opto_subset.to_csv(os.path.join(output_path, 'heatmaps', f'all_stim_window_{name[1]}_{name[0]}_opto_matrix.csv'))
        subset = subset.reindex(new_idx)
        subset[new_idx].to_csv(os.path.join(output_path, 'heatmaps', f'all_stim_window_{name[1]}_{name[0]}_wf_matrix.csv'))

        fig,ax=plt.subplots(1,2,figsize=(7,4), width_ratios=[6,1])        
        g= sns.heatmap(subset[new_idx], mask=mask, vmin=-0.03, vmax=0.03, cmap='icefire', center=0, cbar=True, ax=ax[0], yticklabels=True, xticklabels=True, square=True)
        g.set_yticklabels(g.get_yticklabels(), size = 5)
        g.set_xticklabels(g.get_xticklabels(), size = 5)
        g.collections[0].colorbar.set_label("$\Delta F/F0_{stim} - \Delta F/F0_{control}$", rotation=-90, size=10)
        g.figure.axes[-1].tick_params(labelsize=5)

        g= sns.heatmap(opto_subset, vmin=-0.4, vmax=0.4, cmap=seismic_palette, cbar=True, ax=ax[1], yticklabels=True)
        g.set_yticklabels(g.get_yticklabels(), size = 5)
        g.collections[0].colorbar.set_label("$\Delta$ Lick", rotation=-90, size=10)
        g.figure.axes[-1].tick_params(labelsize=5)
        ax[0].set_ylabel("Inhibited")
        ax[0].set_xlabel("Measured")
        ax[1].set_ylabel("Inhibited")
        ax[1].set_xticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(output_path, 'heatmaps', f'all_stim_window_{name[1]}_{name[0]}_wf_opto_matrix.png'))

    time_df = avg_df[(avg_df.time>=0) & (avg_df.time<=0.12)].groupby(by=['context', 'trial_type', 'opto_stim_coord', 'roi', 'time']).agg(lambda x: np.nanmean(x)).reset_index()
    time_df = time_df.groupby(by=['context', 'trial_type', 'opto_stim_coord', 'roi']).agg(lambda x: np.nanmean(x)).reset_index()
    seismic_palette = sns.diverging_palette(265, 10, s=100, l=40, sep=30, n=200, center="light", as_cmap=True)

    for name, group in time_df.groupby(by=['context', 'trial_type']):
        subset = group.pivot_table(index='opto_stim_coord', columns='roi', values='dff0', aggfunc='mean')
        control = subset.loc["(-5.0, 5.0)"]
        subset = subset-control
        subset = subset.drop("(-5.0, 5.0)")
        mask = np.eye(subset.shape[0])        

        opto_subset = avg_opto_df.loc[(avg_opto_df.context==name[0]) & (avg_opto_df.trial_type==name[1]),['data_mean_sub', 'opto_stim_coord']].set_index('opto_stim_coord').sort_values(by='data_mean_sub')
        new_idx = opto_subset.index
        opto_subset.to_csv(os.path.join(output_path, 'heatmaps', f'100ms_stim_window_{name[1]}_{name[0]}_opto_matrix.csv'))
        subset = subset.reindex(new_idx)
        subset[new_idx].to_csv(os.path.join(output_path, 'heatmaps', f'100ms_stim_window_{name[1]}_{name[0]}_wf_matrix.csv'))

        fig,ax=plt.subplots(1,2,figsize=(7,4), width_ratios=[6,1])        
        g= sns.heatmap(subset[new_idx], mask=mask, vmin=-0.03, vmax=0.03, cmap='icefire', center=0, cbar=True, ax=ax[0], yticklabels=True, xticklabels=True, square=True)
        g.set_yticklabels(g.get_yticklabels(), size = 5)
        g.set_xticklabels(g.get_xticklabels(), size = 5)
        g.collections[0].colorbar.set_label("$\Delta F/F0_{stim} - \Delta F/F0_{control}$", rotation=-90, size=10)
        g.figure.axes[-1].tick_params(labelsize=5)

        g= sns.heatmap(opto_subset, vmin=-0.4, vmax=0.4, cmap=seismic_palette, cbar=True, ax=ax[1], yticklabels=True)
        g.set_yticklabels(g.get_yticklabels(), size = 5)
        g.collections[0].colorbar.set_label("$\Delta$ Lick", rotation=-90, size=10)
        g.figure.axes[-1].tick_params(labelsize=5)
        ax[0].set_ylabel("Inhibited")
        ax[0].set_xlabel("Measured")
        ax[1].set_ylabel("Inhibited")
        ax[1].set_xticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(output_path, 'heatmaps', f'100ms_{name[1]}_{name[0]}_wf_opto_matrix.png'))


def plot_opto_wf_psth(nwb_files, output_path):
    # coords_list = {'wS1': [[-1.5, 3.5]], 'wM1': [[1.5, 1.5]], 'wM2': [[2.5,1.5]], 'RSC': [[-0.5, 0.5], [-1.5,0.5]],
    #                'ALM': [[2.5, 2.5]], 'tjS1':[[0.5, 4.5]], 'tjM1':[[1.5, 3.5]], 'control': [[-5.0, 5.0]]}
    if not os.path.exists(Path(output_path, 'PSTHs')):
        os.makedirs(Path(output_path, 'PSTHs'))

    total_df = []
   
    coords_list = {'wS1': "(-1.5, 3.5)", 'wS2': "(-1.5, 4.5)", 'wM1': "(1.5, 1.5)", 'wM2': "(2.5, 1.5)", 'RSC': "(-0.5, 0.5)",
                'ALM': "(2.5, 2.5)", 'tjS1':"(0.5, 4.5)", 'tjM1':"(1.5, 3.5)", 'control': "(-5.0, 5.0)"}
    
    for nwb_file in nwb_files:
        session_id = nwb_read.get_session_id(nwb_file)
        mouse_id = nwb_read.get_mouse_id(nwb_file)
        total_df += [pd.read_parquet(Path(output_path, session_id, 'results.parquet.gzip', compression='gzip'))]

    total_df = pd.concat(total_df, ignore_index=True)
    total_df['time'] = [[np.linspace(-1,3.98,250)] for i in range(total_df.shape[0])]
    total_df['legend'] = total_df.apply(lambda x: f"{x.opto_stim_coord} - {'lick' if x.lick_flag==1 else 'no lick'}",axis=1)
    d = {c: lambda x: x.unique()[0] for c in ['opto_stim_loc', 'legend']}
    d['time'] = lambda x: list(x)[0][0]
    for c in ['(-0.5, 0.5)', '(-0.5, 1.5)', '(-0.5, 2.5)', '(-0.5, 3.5)', '(-0.5, 4.5)', '(-0.5, 5.5)', '(-1.5, 0.5)',
       '(-1.5, 1.5)', '(-1.5, 2.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(-1.5, 5.5)', '(-2.5, 0.5)', '(-2.5, 1.5)', '(-2.5, 2.5)',
       '(-2.5, 3.5)', '(-2.5, 4.5)', '(-2.5, 5.5)', '(-3.5, 0.5)', '(-3.5, 1.5)', '(-3.5, 2.5)', '(-3.5, 3.5)', '(-3.5, 4.5)',
       '(-3.5, 5.5)', '(0.5, 0.5)', '(0.5, 1.5)', '(0.5, 2.5)', '(0.5, 3.5)', '(0.5, 4.5)', '(0.5, 5.5)', '(1.5, 0.5)', 
       '(1.5, 1.5)', '(1.5, 2.5)', '(1.5, 3.5)', '(1.5, 4.5)', '(1.5, 5.5)', '(2.5, 0.5)', '(2.5, 1.5)', '(2.5, 2.5)', '(2.5, 3.5)', '(2.5, 4.5)', '(2.5, 5.5)', 
       'A1', 'ALM', 'tjM1', 'tjS1', 'RSC', 'wM1', 'wM2', 'wS1', 'wS2']:
        d[f"{c}"]= lambda x: np.nanmean(np.stack(x), axis=0)
          
    # mouse_df = total_df.groupby(by=['mouse_id', 'context', 'trial_type', 'opto_stim_coord', 'lick_flag']).agg(d).reset_index()
    mouse_df = total_df.groupby(by=['mouse_id', 'context', 'trial_type', 'opto_stim_coord']).agg(d).reset_index()

    for loc in coords_list.keys():
        if loc=='control':
            continue
        subset = mouse_df.loc[mouse_df.opto_stim_coord.isin([coords_list[loc], coords_list['control']])]
        for name, group in subset.groupby(by=['context', 'trial_type']):
            if 'auditory_trial' in [name[1]]:
                continue

            # group = group.melt(id_vars=['context', 'trial_type', 'opto_stim_loc', 'opto_stim_coord', 'lick_flag', 'legend', 'time'], 
            #                 value_vars=["(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 1.5)", "(-0.5, 0.5)", "(2.5, 2.5)", "(0.5, 4.5)", "(1.5, 3.5)",], 
            #                 var_name='roi', 
            #                 value_name='dff0').explode(['time', 'dff0'])

            group = group.melt(id_vars=['context', 'trial_type', 'opto_stim_loc', 'opto_stim_coord', 'time'], 
                            value_vars=["(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 1.5)", "(-0.5, 0.5)", "(2.5, 2.5)", "(0.5, 4.5)", "(1.5, 3.5)",], 
                            var_name='roi', 
                            value_name='dff0').explode(['time', 'dff0'])
            
            fig,ax =plt.subplots(3,3, figsize=(9,9), sharey=True, sharex=True)
            fig.suptitle(f"{coords_list[loc]}-stim_{'rewarded' if name[0]==1 else 'non_rewarded'}_{name[1]}")

            for i, roi in enumerate(["(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 1.5)", "(-0.5, 0.5)", "(2.5, 2.5)", "(0.5, 4.5)", "(1.5, 3.5)"]):
                
                # ax.flat[i] = sns.lineplot(data=group[group.roi.isin([roi])], x='time', y='dff0', hue='legend', 
                #                           hue_order=[f"{coords_list[loc]} - lick", f"{coords_list[loc]} - no lick", f"(-5.0, 5.0) - lick", f"(-5.0, 5.0) - no lick"], 
                #                           palette=['#005F60', '#FD5901', '#249EA0', '#FAAB36'], ax=ax.flat[i])
                
                ax.flat[i] = sns.lineplot(data=group[group.roi.isin([roi])], x='time', y='dff0', hue='opto_stim_coord', 
                            hue_order=[f"{coords_list[loc]}", f"(-5.0, 5.0)"], 
                            palette=['#005F60', '#FD5901'], ax=ax.flat[i])
                
                ax.flat[i].set_xlim([-0.1,0.3])
                ax.flat[i].set_ylim([-0.04, 0.07])
                ax.flat[i].set_title(roi)
                ax.flat[i].vlines(0, -0.04, 0.07, 'grey', 'dashed')
                ax.flat[i].spines[['top', 'right']].set_visible(False)
                ax.flat[i].get_legend().remove()

            handles, labels = ax.flat[i].get_legend_handles_labels()
            fig.legend(handles, labels, loc='outside upper right')
            fig.savefig(Path(output_path, 'PSTHs', f"{coords_list[loc]}-stim_{'rewarded' if name[0]==1 else 'non_rewarded'}_{name[1]}.png"))
            fig.savefig(Path(output_path, 'PSTHs', f"{coords_list[loc]}-stim_{'rewarded' if name[0]==1 else 'non_rewarded'}_{name[1]}.svg"))


def dimensionality_reduction(nwb_files, output_path):
    result_path = Path(output_path, 'PCA_150')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    total_df = []
    coords_list = {'wS1': "(-1.5, 3.5)", 'wS2': "(-1.5, 4.5)", 'wM1': "(1.5, 1.5)", 'wM2': "(2.5, 1.5)", 'RSC': "(-0.5, 0.5)",
                   'ALM': "(2.5, 2.5)", 'tjS1':"(0.5, 4.5)", 'tjM1':"(1.5, 3.5)", 'control': "(-5.0, 5.0)"}
    
    for nwb_file in nwb_files:
        session_id = nwb_read.get_session_id(nwb_file)
        mouse_id = nwb_read.get_mouse_id(nwb_file)
        df = [pd.read_parquet(Path(output_path, session_id, 'results.parquet.gzip', compression='gzip'))]
        # df['group'] = 'VGAT'
        total_df += df

    total_df = pd.concat(total_df, ignore_index=True)
    total_df.context = total_df.context.map({0:'non-rewarded', 1:'rewarded'})
    total_df['time'] = [[np.linspace(-1,3.98,250)] for i in range(total_df.shape[0])]
    total_df['legend']= total_df.apply(lambda x: f"{x.opto_stim_coord} - {'lick' if x.lick_flag==1 else 'no lick'}",axis=1)

    d = {c: lambda x: x.unique()[0] for c in ['opto_stim_loc', 'legend']}
    d['time'] = lambda x: list(x)[0][0]
    for c in ['(-0.5, 0.5)', '(-0.5, 1.5)', '(-0.5, 2.5)', '(-0.5, 3.5)', '(-0.5, 4.5)', '(-0.5, 5.5)', '(-1.5, 0.5)',
       '(-1.5, 1.5)', '(-1.5, 2.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(-1.5, 5.5)', '(-2.5, 0.5)', '(-2.5, 1.5)', '(-2.5, 2.5)',
       '(-2.5, 3.5)', '(-2.5, 4.5)', '(-2.5, 5.5)', '(-3.5, 0.5)', '(-3.5, 1.5)', '(-3.5, 2.5)', '(-3.5, 3.5)', '(-3.5, 4.5)',
       '(-3.5, 5.5)', '(0.5, 0.5)', '(0.5, 1.5)', '(0.5, 2.5)', '(0.5, 3.5)', '(0.5, 4.5)', '(0.5, 5.5)', '(1.5, 0.5)', 
       '(1.5, 1.5)', '(1.5, 2.5)', '(1.5, 3.5)', '(1.5, 4.5)', '(1.5, 5.5)', '(2.5, 0.5)', '(2.5, 1.5)', '(2.5, 2.5)', '(2.5, 3.5)', '(2.5, 4.5)', '(2.5, 5.5)', 
       'A1', 'ALM', 'tjM1', 'tjS1', 'RSC', 'wM1', 'wM2', 'wS1', 'wS2']:
        d[f"{c}"]= lambda x: np.nanmean(np.stack(x), axis=0)
          
    mouse_df = total_df.groupby(by=['mouse_id', 'context', 'trial_type', 'opto_stim_coord', 'lick_flag']).agg(d).reset_index()
    mouse_df = mouse_df.melt(id_vars=['mouse_id', 'context', 'trial_type', 'legend', 'opto_stim_coord', 'lick_flag', 'time'],
                                 value_vars=['(-0.5, 0.5)', '(-0.5, 1.5)', '(-0.5, 2.5)', '(-0.5, 3.5)', '(-0.5, 4.5)', '(-0.5, 5.5)', '(-1.5, 0.5)',
       '(-1.5, 1.5)', '(-1.5, 2.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(-1.5, 5.5)', '(-2.5, 0.5)', '(-2.5, 1.5)', '(-2.5, 2.5)',
       '(-2.5, 3.5)', '(-2.5, 4.5)', '(-2.5, 5.5)', '(-3.5, 0.5)', '(-3.5, 1.5)', '(-3.5, 2.5)', '(-3.5, 3.5)', '(-3.5, 4.5)',
       '(-3.5, 5.5)', '(0.5, 0.5)', '(0.5, 1.5)', '(0.5, 2.5)', '(0.5, 3.5)', '(0.5, 4.5)', '(0.5, 5.5)', '(1.5, 0.5)', 
       '(1.5, 1.5)', '(1.5, 2.5)', '(1.5, 3.5)', '(1.5, 4.5)', '(1.5, 5.5)', '(2.5, 0.5)', '(2.5, 1.5)', '(2.5, 2.5)', '(2.5, 3.5)', '(2.5, 4.5)', '(2.5, 5.5)',],
                                 var_name='roi',
                                 value_name='dff0').explode(['time', 'dff0'])
    avg_df = mouse_df.groupby(by=['context', 'trial_type', 'lick_flag', 'legend', 'opto_stim_coord', 'roi', 'time']).agg(lambda x: np.nanmean(x)).reset_index()
   
    avg_df.time = avg_df.time.round(2)
    avg_df = avg_df[(avg_df.time>=-0.15)&(avg_df.time<=0.15)]
    subset_df = avg_df[(avg_df.trial_type.isin(['whisker_trial', 'no_stim_trial'])) & (avg_df.opto_stim_coord=="(-5.0, 5.0)")].pivot(index=['context','trial_type', 'legend', 'time'], columns='roi', values='dff0')
    avg_data_for_pca = subset_df.to_numpy()
    labels = subset_df.keys()

    # Standardize average data for training: Based on trials with light on control location 
    scaler = StandardScaler()
    fit_scaler = scaler.fit(avg_data_for_pca)
    avg_data_for_pca = fit_scaler.transform(avg_data_for_pca)
    pca = PCA(n_components=15)
    results = pca.fit(np.nan_to_num(avg_data_for_pca))

    # Project whisker and catch trials 
    mouse_df = total_df.groupby(by=['mouse_id', 'context', 'trial_type', 'opto_stim_coord']).agg(d).reset_index()
    mouse_df = mouse_df.melt(id_vars=['mouse_id', 'context', 'trial_type', 'opto_stim_coord', 'time'],
                                 value_vars=['(-0.5, 0.5)', '(-0.5, 1.5)', '(-0.5, 2.5)', '(-0.5, 3.5)', '(-0.5, 4.5)', '(-0.5, 5.5)', '(-1.5, 0.5)',
       '(-1.5, 1.5)', '(-1.5, 2.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(-1.5, 5.5)', '(-2.5, 0.5)', '(-2.5, 1.5)', '(-2.5, 2.5)',
       '(-2.5, 3.5)', '(-2.5, 4.5)', '(-2.5, 5.5)', '(-3.5, 0.5)', '(-3.5, 1.5)', '(-3.5, 2.5)', '(-3.5, 3.5)', '(-3.5, 4.5)',
       '(-3.5, 5.5)', '(0.5, 0.5)', '(0.5, 1.5)', '(0.5, 2.5)', '(0.5, 3.5)', '(0.5, 4.5)', '(0.5, 5.5)', '(1.5, 0.5)', 
       '(1.5, 1.5)', '(1.5, 2.5)', '(1.5, 3.5)', '(1.5, 4.5)', '(1.5, 5.5)', '(2.5, 0.5)', '(2.5, 1.5)', '(2.5, 2.5)', '(2.5, 3.5)', '(2.5, 4.5)', '(2.5, 5.5)',],
                                 var_name='roi',
                                 value_name='dff0').explode(['time', 'dff0'])
    avg_df = mouse_df.groupby(by=['context', 'trial_type', 'opto_stim_coord', 'roi', 'time']).agg(lambda x: np.nanmean(x)).reset_index()
    avg_df = avg_df[(avg_df.time>=-0.15)&(avg_df.time<=0.15)]
    subset_df = avg_df[avg_df.trial_type.isin(['whisker_trial', 'no_stim_trial'])].pivot(index=['context','trial_type', 'opto_stim_coord', 'time'], columns='roi', values='dff0')
    avg_data_for_pca = subset_df.to_numpy()
    avg_data_for_pca = fit_scaler.transform(avg_data_for_pca)
    principal_components = pca.transform(np.nan_to_num(avg_data_for_pca))

    pc_df = pd.DataFrame(data=principal_components, index=subset_df.index)
    pc_df.columns = [f"PC {i+1}" for i in range(0, principal_components.shape[1])]
    subset_df = subset_df.join(pc_df).reset_index()

    # Explained variance ratio
    exp_var = [val * 100 for val in pca.explained_variance_ratio_]
    plot_y = [sum(exp_var[:i+1]) for i in range(len(exp_var))]
    plot_x = range(1, len(plot_y) + 1)
    fig,ax = plt.subplots(figsize=(7,4))
    ax.plot(plot_x, plot_y, marker="o", color="#9B1D20")
    for x, y in zip(plot_x, plot_y):
        plt.text(x, y + 3, f"{y:.1f}%", ha="center", va="bottom")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Cumulative percentage of variance explained")
    ax.set_ylim([50,100])
    ax.set_xticks(plot_x)
    ax.grid(axis="y")
    ax.spines[['top', 'right']].set_visible(False)
    for ext in ['.png', '.svg']:
        fig.savefig(Path(result_path, f'variance_explained{ext}'))

    # subset_df['lick_flag'] = subset_df.apply(lambda x: 0 if 'no lick' in x.legend else 1, axis=1)
    # subset_df['stim_loc'] = subset_df.apply(lambda x: x.legend.split(" -")[0], axis=1)

    ## Plot biplots to see which variables (i.e. rois) contain most of the explained variance (?)

    coeff = np.transpose(results.components_)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle("PCA1-2 Biplot")
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    fig1.suptitle("PCA2-3 Biplot")
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    fig2.suptitle("PCA1-3 Biplot")

    grid_list = ['(-0.5, 0.5)', '(-0.5, 1.5)',
       '(-0.5, 2.5)', '(-0.5, 3.5)', '(-0.5, 4.5)', '(-0.5, 5.5)',
       '(-1.5, 0.5)', '(-1.5, 1.5)', '(-1.5, 2.5)', '(-1.5, 3.5)',
       '(-1.5, 4.5)', '(-1.5, 5.5)', '(-2.5, 0.5)', '(-2.5, 1.5)',
       '(-2.5, 2.5)', '(-2.5, 3.5)', '(-2.5, 4.5)', '(-2.5, 5.5)',
       '(-3.5, 0.5)', '(-3.5, 1.5)', '(-3.5, 2.5)', '(-3.5, 3.5)',
       '(-3.5, 4.5)', '(-3.5, 5.5)', '(0.5, 0.5)', '(0.5, 1.5)', '(0.5, 2.5)',
       '(0.5, 3.5)', '(0.5, 4.5)', '(0.5, 5.5)', '(1.5, 0.5)', '(1.5, 1.5)',
       '(1.5, 2.5)', '(1.5, 3.5)', '(1.5, 4.5)', '(1.5, 5.5)', '(2.5, 0.5)',
       '(2.5, 1.5)', '(2.5, 2.5)', '(2.5, 3.5)', '(2.5, 4.5)', '(2.5, 5.5)']
    
    for j, roi in enumerate(['ALM', 'tjM1', 'tjS1', 'RSC', 'wM1', 'wM2','wS1', 'wS2']):
        pca_idx, coord = [(i, coord) for i, coord in enumerate(grid_list) if coord == coords_list[roi]][0]
        ax.arrow(x=0, y=0, dx=coeff[pca_idx, 0], dy=coeff[pca_idx, 1], color="#000000", width=0.003, head_width=0.03)
        ax.text(x=coeff[pca_idx, 0] * 1.15, y=coeff[pca_idx, 1] * 1.15, s=coord, size=13, color="#000000", ha="center", va="center")
        ax1.arrow(x=0, y=0, dx=coeff[pca_idx, 1], dy=coeff[pca_idx, 2], color="#000000", width=0.003, head_width=0.03)
        ax1.text(x=coeff[pca_idx, 1] * 1.15, y=coeff[pca_idx, 2] * 1.15, s=coord, size=13, color="#000000", ha="center", va="center")
        ax2.arrow(x=0, y=0, dx=coeff[pca_idx, 0], dy=coeff[pca_idx, 2], color="#000000", width=0.003, head_width=0.03)
        ax2.text(x=coeff[pca_idx, 0] * 1.15, y=coeff[pca_idx, 2] * 1.15, s=coord, size=13, color="#000000", ha="center", va="center")

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    ax1.set_xlabel("Principal Component 2")
    ax1.set_ylabel("Principal Component 3")

    ax2.set_xlabel("Principal Component 1")
    ax2.set_ylabel("Principal Component 3")

    for axes in [ax, ax1, ax2]:
        axes.set_aspect("equal", "box")

        axes.set_xlim([-1, 1])
        axes.set_ylim([-1, 1])
        axes.set_xticks(np.arange(-1, 1.1, 0.2))
        axes.set_yticks(np.arange(-1, 1.1, 0.2))

        axes.axhline(y=0, color="black", linestyle="--")
        axes.axvline(x=0, color="black", linestyle="--")
        axes.grid(True)

        circle = plt.Circle((0, 0), 0.99, color="gray", fill=False)
        axes.add_patch(circle)

    fig.savefig(Path(result_path, f"dimensionality_reduction_PC1vsPC2_biplot.png"))
    fig.savefig(Path(result_path, f"dimensionality_reduction_PC1vsPC2_biplot.svg"))

    fig1.savefig(Path(result_path, f"dimensionality_reduction_PC2vsPC3_biplot.png"))
    fig1.savefig(Path(result_path, f"dimensionality_reduction_PC2vsPC3_biplot.svg"))

    fig2.savefig(Path(result_path, f"dimensionality_reduction_PC1vsPC3_biplot.png"))
    fig2.savefig(Path(result_path, f"dimensionality_reduction_PC1vsPC3_biplot.svg"))

    ## Plot loadings in allen ccf
    fig, ax = plt.subplots(5, int(coeff.shape[1]/5), figsize=(8, 20))
    fig.suptitle("PC1-3 loadings")

    for i in range(coeff.shape[1]):
        im_pca = generate_reduced_image_df(coeff[np.newaxis, :, i], [eval(label) for label in labels])
        im_pca.drop(im_pca[(im_pca.x==5.5)&(im_pca.y==2.5)].index, inplace=True)        
        plot_grid_on_allen(im_pca, outcome='dff0', palette='seismic', result_path=None, dotsize=340, vmin=-im_pca.dff0.abs().max(), vmax=im_pca.dff0.abs().max(), norm=None, fig=fig, ax= ax.flat[i])
        ax.flat[i].set_axis_off()
        ax.flat[i].set_title(f"PC {i+1}")
    fig.savefig(os.path.join(result_path, f"grid_loadings_pc.png"))

    ## Plot PC timecourses

    long_df = subset_df.melt(id_vars=['context', 'trial_type', 'opto_stim_coord', 'time'], 
                            value_vars=['PC 1', 'PC 2', 'PC 3'], 
                            var_name='PC', 
                            value_name='data').explode(['time', 'data'])
    long_df = long_df[long_df.opto_stim_coord.isin(list(coords_list.values()))]
    # long_df['context_legend'] = long_df.apply(lambda x: f"{x.context} - {'lick' if x.lick_flag==1 else 'no lick'}", axis=1)
    fig, ax = plt.subplots(3, 4, figsize=(8,6))
    fig.suptitle("PC timecourses")
    g = sns.relplot(data=long_df[long_df.trial_type=='whisker_trial'], x='time', y='data', hue='context', 
                    hue_order=['rewarded', 'non-rewarded'], 
                    palette=['#348A18', '#6E188A'], 
                    col='PC', row='opto_stim_coord', kind='line', linewidth=2, facet_kws=dict(sharey=False))
    for k in range(g.axes.shape[0]):
        g.axes[k, 0].set_ylim(-30,30)
        g.axes[k, 1].set_ylim(-15,5)
        g.axes[k, 2].set_ylim(-15,5)

    g.figure.savefig(Path(result_path, 'whisker_PC_timecourses_by_region.png'))
    g.figure.savefig(Path(result_path, 'whisker_PC_timecourses_by_region.svg'))
    
    fig, ax = plt.subplots(3, 4, figsize=(8,6))
    fig.suptitle("PC timecourses")
    g = sns.relplot(data=long_df[long_df.trial_type=='no_stim_trial'], x='time', y='data', hue='context', 
                    hue_order=['rewarded', 'non-rewarded'], 
                    palette=['#348A18', '#6E188A'], 
                    col='PC', row='opto_stim_coord', kind='line', linewidth=2, facet_kws=dict(sharey=False))
    for k in range(g.axes.shape[0]):
        g.axes[k, 0].set_ylim(-30,30)
        g.axes[k, 1].set_ylim(-15,5)
        g.axes[k, 2].set_ylim(-15,5)

    g.figure.savefig(Path(result_path, 'catch_PC_timecourses_by_region.png'))
    g.figure.savefig(Path(result_path, 'catch_PC_timecourses_by_region.svg'))

    g = sns.relplot(data=long_df[long_df.trial_type=='whisker_trial'], x='time', y='data', hue='opto_stim_coord', 
                    hue_order=["(-5.0, 5.0)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)", "(1.5, 3.5)", "(-0.5, 0.5)"], 
                    palette=['#000000', '#011f4b', '#03396c', '#005b96', '#6497b1', '#b3cde0', '#ffccd5', '#c9184a', '#590d22'], 
                    col='context', row='PC', kind='line', linewidth=2, facet_kws=dict(sharey=False))
    for k in range(g.axes.shape[1]):
        g.axes[0, k].set_ylim(-30,30)
        g.axes[1, k].set_ylim(-15,5)
        g.axes[2, k].set_ylim(-15,5)
    g.figure.savefig(Path(result_path, 'whisker_PC_timecourses_by_trial_outcome.png'))
    g.figure.savefig(Path(result_path, 'whisker_PC_timecourses_by_trial_outcome.svg'))

    g = sns.relplot(data=long_df[long_df.trial_type=='no_stim_trial'], x='time', y='data', hue='opto_stim_coord', 
                    hue_order=["(-5.0, 5.0)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)", "(1.5, 3.5)", "(-0.5, 0.5)"], 
                    palette=['#000000', '#011f4b', '#03396c', '#005b96', '#6497b1', '#b3cde0', '#ffccd5', '#c9184a', '#590d22'], 
                    col='context', row='PC', kind='line', linewidth=2, facet_kws=dict(sharey=False))
    for k in range(g.axes.shape[1]):
        g.axes[0, k].set_ylim(-30,30)
        g.axes[1, k].set_ylim(-15,5)
        g.axes[2, k].set_ylim(-15,5)
    g.figure.savefig(Path(result_path, 'catch_PC_timecourses_by_trial_outcome.png'))
    g.figure.savefig(Path(result_path, 'catch_PC_timecourses_by_trial_outcome.svg'))

    ## Plot projected time courses onto PCx vs PCy
    color_dict = {"(-5.0, 5.0)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#000000'], N=avg_df.time.unique().shape[0]),
                  "(-1.5, 3.5)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#011f4b'], N=avg_df.time.unique().shape[0]),
                  "(-1.5, 4.5)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#03396c'], N=avg_df.time.unique().shape[0]),
                  "(1.5, 1.5)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#005b96'], N=avg_df.time.unique().shape[0]),
                  "(2.5, 1.5)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#6497b1'], N=avg_df.time.unique().shape[0]),
                  "(2.5, 2.5)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#b3cde0'], N=avg_df.time.unique().shape[0]),
                  "(0.5, 4.5)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#ffccd5'], N=avg_df.time.unique().shape[0]),
                  "(1.5, 3.5)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#c9184a'], N=avg_df.time.unique().shape[0]),
                  "(-0.5, 0.5)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#590d22'], N=avg_df.time.unique().shape[0])
}
    
    lines = ['#000000', '#011f4b', '#03396c', '#005b96', '#6497b1', '#b3cde0', '#ffccd5', '#c9184a', '#590d22']
    handles = [Line2D([0], [0], color=c, lw=4) for c in lines]

    for trial in subset_df.trial_type.unique():
        fig, ax= plt.subplots(1,2, figsize=(8,4), sharey=True, sharex=True)
        fig1, ax1 = plt.subplots(1,2, figsize=(8,4), sharey=True, sharex=True)
        fig2, ax2 = plt.subplots(1,2, figsize=(8,4), sharey=True, sharex=True)

        for i, (name, group) in enumerate(subset_df[subset_df.trial_type==trial].groupby(by=['context'])):
            for stim in color_dict.keys():
                ax.flat[i].scatter(group.loc[group.opto_stim_coord==stim, 'PC 1'], group.loc[group.opto_stim_coord==stim, 'PC 2'], c=group.loc[group.opto_stim_coord==stim, 'time'], cmap=color_dict[stim], label=stim, s=10)
                ax.flat[i].scatter(group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 1'], group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 2'], s=10, facecolors='none', edgecolors='r')
                ax1.flat[i].scatter(group.loc[group.opto_stim_coord==stim, 'PC 2'], group.loc[group.opto_stim_coord==stim, 'PC 3'], c=group.loc[group.opto_stim_coord==stim, 'time'], cmap=color_dict[stim], label=stim, s=10)
                ax1.flat[i].scatter(group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 2'], group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 3'], s=10, facecolors='none', edgecolors='r')
                ax2.flat[i].scatter(group.loc[group.opto_stim_coord==stim, 'PC 1'], group.loc[group.opto_stim_coord==stim, 'PC 3'], c=group.loc[group.opto_stim_coord==stim, 'time'], cmap=color_dict[stim], label=stim, s=10)
                ax2.flat[i].scatter(group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 1'], group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 3'], s=10, facecolors='none', edgecolors='r')

            ax.flat[i].set_xlim(-35,35)
            ax.flat[i].set_xlabel('PC 1')
            ax.flat[i].set_ylim(-15,5)
            ax.flat[i].set_ylabel('PC 2')
            ax.flat[i].set_title(f"{'Rewarded' if name=='rewarded' else 'Non-rewarded'} {trial}")

            ax1.flat[i].set_xlim(-15,5)
            ax1.flat[i].set_xlabel('PC 2')
            ax1.flat[i].set_ylim(-15,5)
            ax1.flat[i].set_ylabel('PC 3')
            ax1.flat[i].set_title(f"{'Rewarded' if name=='rewarded' else 'Non-rewarded'} {trial}")

            ax2.flat[i].set_xlim(-35,35)
            ax2.flat[i].set_xlabel('PC 1')
            ax2.flat[i].set_ylim(-15,5)
            ax2.flat[i].set_ylabel('PC 3')
            ax2.flat[i].set_title(f"{'Rewarded' if name=='rewarded' else 'Non-rewarded'} {trial}")

        fig.legend(handles, ["(-5.0, 5.0)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)", "(1.5, 3.5)", "(-0.5, 0.5)"], loc='outside upper right')
        fig.savefig(Path(result_path, f"{trial}_dimensionality_reduction_PC1vsPC2.png"))
        fig.savefig(Path(result_path, f"{trial}_dimensionality_reduction_PC1vsPC2.svg"))
        fig1.legend(handles, ["(-5.0, 5.0)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)", "(1.5, 3.5)", "(-0.5, 0.5)"], loc='outside upper right')
        fig1.savefig(Path(result_path, f"{trial}_dimensionality_reduction_PC2vsPC3.png"))
        fig1.savefig(Path(result_path, f"{trial}_dimensionality_reduction_PC2vsPC3.svg"))

        fig2.legend(handles, ["(-5.0, 5.0)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)", "(1.5, 3.5)", "(-0.5, 0.5)"], loc='outside upper right')
        fig2.savefig(Path(result_path, f"{trial}_dimensionality_reduction_PC1vsPC3.png"))
        fig2.savefig(Path(result_path, f"{trial}_dimensionality_reduction_PC1vsPC3.svg"))

    ## PCA projetions with arrowplots
    for trial in subset_df.trial_type.unique():
        fig, ax= plt.subplots(1,2, figsize=(8,4), sharey=True, sharex=True)
        fig1, ax1 = plt.subplots(1,2, figsize=(8,4), sharey=True, sharex=True)
        fig2, ax2 = plt.subplots(1,2, figsize=(8,4), sharey=True, sharex=True)

        for i, (name, group) in enumerate(subset_df[subset_df.trial_type==trial].groupby(by=['context'])):
            for j, stim in enumerate(color_dict.keys()):
                ax.flat[i].plot(group.loc[group.opto_stim_coord==stim, 'PC 1'], group.loc[group.opto_stim_coord==stim, 'PC 2'], lw=1, c=lines[j], alpha=1, label=stim)
                ax1.flat[i].plot(group.loc[group.opto_stim_coord==stim, 'PC 2'], group.loc[group.opto_stim_coord==stim, 'PC 3'], lw=1, c=lines[j], alpha=1, label=stim)
                ax2.flat[i].plot(group.loc[group.opto_stim_coord==stim, 'PC 1'], group.loc[group.opto_stim_coord==stim, 'PC 3'], lw=1, c=lines[j], alpha=1, label=stim)

                if  group.loc[group.opto_stim_coord==stim, 'PC 1'].shape[0]==0:
                    continue

                for t_step in [10, 20, 40, 60, 80]:
                    x= group.loc[(group.opto_stim_coord==stim) & (group.time.mul(100)>group.time.mul(100).values[t_step]-1) & (group.time.mul(100)<group.time.mul(100).values[t_step]+1), 'PC 1'].values[0]
                    y= group.loc[(group.opto_stim_coord==stim) & (group.time.mul(100)>group.time.mul(100).values[t_step]-1) & (group.time.mul(100)<group.time.mul(100).values[t_step]+1), 'PC 2'].values[0]
                    dx= group.loc[(group.opto_stim_coord==stim) & (group.time.mul(100)>group.time.mul(100).values[t_step+1]-1) & (group.time.mul(100)<group.time.mul(100).values[t_step+1]+1), 'PC 1'].values[0] - x
                    dy= group.loc[(group.opto_stim_coord==stim) & (group.time.mul(100)>group.time.mul(100).values[t_step+1]-1) & (group.time.mul(100)<group.time.mul(100).values[t_step+1]+1), 'PC 2'].values[0] - y

                    ax.flat[i].arrow(x, y, dx, dy, width=0.001, shape='full', color=lines[j], length_includes_head=True, head_width=.12)
                    
                    x= group.loc[(group.opto_stim_coord==stim) & (group.time.mul(100)>group.time.mul(100).values[t_step]-1) & (group.time.mul(100)<group.time.mul(100).values[t_step]+1), 'PC 2'].values[0]
                    y= group.loc[(group.opto_stim_coord==stim) & (group.time.mul(100)>group.time.mul(100).values[t_step]-1) & (group.time.mul(100)<group.time.mul(100).values[t_step]+1), 'PC 3'].values[0]
                    dx= group.loc[(group.opto_stim_coord==stim) & (group.time.mul(100)>group.time.mul(100).values[t_step+1]-1) & (group.time.mul(100)<group.time.mul(100).values[t_step+1]+1), 'PC 2'].values[0] - x
                    dy= group.loc[(group.opto_stim_coord==stim) & (group.time.mul(100)>group.time.mul(100).values[t_step+1]-1) & (group.time.mul(100)<group.time.mul(100).values[t_step+1]+1), 'PC 3'].values[0] - y

                    ax1.flat[i].arrow(x, y, dx, dy, width=0.001, shape='full', color=lines[j], length_includes_head=True, head_width=.12)

                    x= group.loc[(group.opto_stim_coord==stim) & (group.time.mul(100)>group.time.mul(100).values[t_step]-1) & (group.time.mul(100)<group.time.mul(100).values[t_step]+1), 'PC 1'].values[0]
                    y= group.loc[(group.opto_stim_coord==stim) & (group.time.mul(100)>group.time.mul(100).values[t_step]-1) & (group.time.mul(100)<group.time.mul(100).values[t_step]+1), 'PC 3'].values[0]
                    dx= group.loc[(group.opto_stim_coord==stim) & (group.time.mul(100)>group.time.mul(100).values[t_step+1]-1) & (group.time.mul(100)<group.time.mul(100).values[t_step+1]+1), 'PC 1'].values[0] - x
                    dy= group.loc[(group.opto_stim_coord==stim) & (group.time.mul(100)>group.time.mul(100).values[t_step+1]-1) & (group.time.mul(100)<group.time.mul(100).values[t_step+1]+1), 'PC 3'].values[0] - y

                    ax2.flat[i].arrow(x, y, dx, dy, width=0.001, shape='full', color=lines[j], length_includes_head=True, head_width=.12)
            
            ax.flat[i].set_xlabel('PC 1')
            ax.flat[i].set_ylabel('PC 2')
            ax.flat[i].set_title(f"{'Rewarded' if name=='rewarded' else 'Non-rewarded'} {trial}")

            ax1.flat[i].set_xlabel('PC 2')
            ax1.flat[i].set_ylabel('PC 3')
            ax1.flat[i].set_title(f"{'Rewarded' if name=='rewarded' else 'Non-rewarded'} {trial}")

            ax2.flat[i].set_xlabel('PC 1')
            ax2.flat[i].set_ylabel('PC 3')
            ax2.flat[i].set_title(f"{'Rewarded' if name=='rewarded' else 'Non-rewarded'} {trial}")

        fig.legend(handles, ["(-5.0, 5.0)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)", "(1.5, 3.5)", "(-0.5, 0.5)"], loc='outside upper right')
        fig.savefig(Path(result_path, f"{trial}_dimensionality_reduction_PC1vsPC2_lines.png"))
        fig.savefig(Path(result_path, f"{trial}_dimensionality_reduction_PC1vsPC2_lines.svg"))
        fig1.legend(handles, ["(-5.0, 5.0)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)", "(1.5, 3.5)", "(-0.5, 0.5)"], loc='outside upper right')
        fig1.savefig(Path(result_path, f"{trial}_dimensionality_reduction_PC2vsPC3_lines.png"))
        fig1.savefig(Path(result_path, f"{trial}_dimensionality_reduction_PC2vsPC3_lines.svg"))
        fig2.legend(handles, ["(-5.0, 5.0)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)", "(1.5, 3.5)", "(-0.5, 0.5)"], loc='outside upper right')
        fig2.savefig(Path(result_path, f"{trial}_dimensionality_reduction_PC1vsPC3_lines.png"))
        fig2.savefig(Path(result_path, f"{trial}_dimensionality_reduction_PC1vsPC3_lines.svg"))


    ## Control vs stim one by one projections
    for trial in subset_df.trial_type.unique():
        for stim in color_dict.keys():
            save_path = os.path.join(result_path, stim)
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)

            if stim == "(-5.0, 5.0)":
                continue

            fig, ax= plt.subplots(1,2, figsize=(8,4), sharey=True, sharex=True)
            fig.suptitle(stim)
            fig1, ax1 = plt.subplots(1,2, figsize=(8,4), sharey=True, sharex=True)
            fig1.suptitle(stim)
            fig2, ax2 = plt.subplots(1,2, figsize=(8,4), sharey=True, sharex=True)
            fig2.suptitle(stim)
            
            fig3 = plt.figure(figsize=(4,4))
            fig3.suptitle(f"{trial} - stim {stim}")
            ax3 = fig3.add_subplot(1, 1, 1, projection='3d')

            for i, (name, group) in enumerate(subset_df[subset_df.trial_type==trial].groupby(by=['context'])):
                if name=='rewarded':
                    color = LinearSegmentedColormap.from_list('', ['#FFFFFF', '#348A18'], N=avg_df.time.unique().shape[0])
                    control = LinearSegmentedColormap.from_list('', ['#FFFFFF', '#000000'], N=avg_df.time.unique().shape[0])
                else:
                    color = LinearSegmentedColormap.from_list('', ['#FFFFFF', '#6E188A'], N=avg_df.time.unique().shape[0])
                    control = LinearSegmentedColormap.from_list('', ['#FFFFFF', '#808080'], N=avg_df.time.unique().shape[0])

                ax.flat[i].scatter(group.loc[group.opto_stim_coord==stim, 'PC 1'], group.loc[group.opto_stim_coord==stim, 'PC 2'], c=group.loc[group.opto_stim_coord==stim, 'time'], cmap=color, label=name[0], s=10)
                ax.flat[i].scatter(group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 1'], group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 2'], s=10, facecolors='none', edgecolors='r')
                ax.flat[i].scatter(group.loc[group.opto_stim_coord=="(-5.0, 5.0)", 'PC 1'], group.loc[group.opto_stim_coord=="(-5.0, 5.0)", 'PC 2'], c=group.loc[group.opto_stim_coord=="(-5.0, 5.0)", 'time'], cmap=control, label=name[0], s=10)
                ax.flat[i].scatter(group.loc[(group.opto_stim_coord=="(-5.0, 5.0)") & (group.time==0), 'PC 1'], group.loc[(group.opto_stim_coord=="(-5.0, 5.0)") & (group.time==0), 'PC 2'], s=10, facecolors='none', edgecolors='r')
                
                ax1.flat[i].scatter(group.loc[group.opto_stim_coord==stim, 'PC 2'], group.loc[group.opto_stim_coord==stim, 'PC 3'], c=group.loc[group.opto_stim_coord==stim, 'time'], cmap=color, label=name[0], s=10)
                ax1.flat[i].scatter(group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 2'], group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 3'], s=10, facecolors='none', edgecolors='r')
                ax1.flat[i].scatter(group.loc[group.opto_stim_coord=="(-5.0, 5.0)", 'PC 2'], group.loc[group.opto_stim_coord=="(-5.0, 5.0)", 'PC 3'], c=group.loc[group.opto_stim_coord=="(-5.0, 5.0)", 'time'], cmap=control, label=name[0], s=10)
                ax1.flat[i].scatter(group.loc[(group.opto_stim_coord=="(-5.0, 5.0)") & (group.time==0), 'PC 2'], group.loc[(group.opto_stim_coord=="(-5.0, 5.0)") & (group.time==0), 'PC 3'], s=10, facecolors='none', edgecolors='r')
                
                ax2.flat[i].scatter(group.loc[group.opto_stim_coord==stim, 'PC 1'], group.loc[group.opto_stim_coord==stim, 'PC 3'], c=group.loc[group.opto_stim_coord==stim, 'time'], cmap=color, label=name[0], s=10)
                ax2.flat[i].scatter(group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 1'], group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 3'], s=10, facecolors='none', edgecolors='r')
                ax2.flat[i].scatter(group.loc[group.opto_stim_coord=="(-5.0, 5.0)", 'PC 1'], group.loc[group.opto_stim_coord=="(-5.0, 5.0)", 'PC 3'], c=group.loc[group.opto_stim_coord=="(-5.0, 5.0)", 'time'], cmap=control, label=name[0], s=10)
                ax2.flat[i].scatter(group.loc[(group.opto_stim_coord=="(-5.0, 5.0)") & (group.time==0), 'PC 1'], group.loc[(group.opto_stim_coord=="(-5.0, 5.0)") & (group.time==0), 'PC 3'], s=10, facecolors='none', edgecolors='r')

                y = group.loc[group.opto_stim_coord==stim, 'PC 1']
                x = group.loc[group.opto_stim_coord==stim, 'PC 2']
                z = group.loc[group.opto_stim_coord==stim, 'PC 3']

                ax3.plot(x, y, z, c=color(15))
                ax3.scatter(group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 2'],
                            group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 1'],
                            group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 3'],
                            c='r')
                y = group.loc[group.opto_stim_coord=="(-5.0, 5.0)", 'PC 1']
                x = group.loc[group.opto_stim_coord=="(-5.0, 5.0)", 'PC 2']
                z = group.loc[group.opto_stim_coord=="(-5.0, 5.0)", 'PC 3']

                ax3.plot(x, y, z, c=control(15))
                ax3.scatter(group.loc[(group.opto_stim_coord=="(-5.0, 5.0)") & (group.time==0), 'PC 2'],
                            group.loc[(group.opto_stim_coord=="(-5.0, 5.0)") & (group.time==0), 'PC 1'],
                            group.loc[(group.opto_stim_coord=="(-5.0, 5.0)") & (group.time==0), 'PC 3'],
                            c='r')
                ax.flat[i].set_xlim(-35, 35)
                ax.flat[i].set_ylim(-15, 5)
                ax.flat[i].set_xlabel('PC 1')
                ax.flat[i].set_ylabel('PC 2')
                ax.flat[i].set_title(f"{name}")

                ax1.flat[i].set_ylim(-15, 5)
                ax1.flat[i].set_ylim(-15, 5)
                ax1.flat[i].set_xlabel('PC 2')
                ax1.flat[i].set_ylabel('PC 3')
                ax1.flat[i].set_title(f"{name}")

                ax2.flat[i].set_xlim(-35,35)
                ax2.flat[i].set_ylim(-15, 5)
                ax2.flat[i].set_xlabel('PC 1')
                ax2.flat[i].set_ylabel('PC 3')
                ax2.flat[i].set_title(f"{name}")

                ax3.set_xlim(-15,5)
                ax3.set_ylim(-35,35)
                ax3.set_zlim(-15,5)
                ax3.set_xlabel('PC 2')
                ax3.set_ylabel('PC 1')
                ax3.set_zlabel('PC 3')

            fig.savefig(Path(save_path, f"{trial}_dimensionality_reduction_PC1vsPC2.png"))
            fig.savefig(Path(save_path, f"{trial}_dimensionality_reduction_PC1vsPC2.svg"))

            fig1.savefig(Path(save_path, f"{trial}_dimensionality_reduction_PC2vsPC3.png"))
            fig1.savefig(Path(save_path, f"{trial}_dimensionality_reduction_PC2vsPC3.svg"))

            fig2.savefig(Path(save_path, f"{trial}_dimensionality_reduction_PC1vsPC3.png"))
            fig2.savefig(Path(save_path, f"{trial}_dimensionality_reduction_PC1vsPC3.svg"))
            
            fig3.savefig(Path(save_path, f"{trial}_dimensionality_reduction_3d.png"))
            fig3.savefig(Path(save_path, f"{trial}_dimensionality_reduction_3d.svg"))


    ## compute mean squared error
    group = 'controls' if 'control' in str(output_path) else 'VGAT'
    opto_avg_df = load_opto_data(group)
    results_total = []
    for trial in subset_df.trial_type.unique():
        for i, (name, group) in enumerate(subset_df[subset_df.trial_type==trial].groupby(by=['context'])):
            for stim in subset_df.opto_stim_coord.unique():
                if stim == "(-5.0, 5.0)":
                    continue
                
                control = group[(group.trial_type==trial) & (group.opto_stim_coord=="(-5.0, 5.0)") & (group.time>=0)]
                inhibited = group[(group.trial_type==trial) & (group.opto_stim_coord==stim) & (group.time>=0)]
                euclidean_dist = np.sqrt(np.sum((inhibited[[f"PC 1", f"PC 2", f"PC 3"]].to_numpy() - control[[f"PC 1", f"PC 2", f"PC 3"]].to_numpy())**2, axis=1)).sum()

                diff = (inhibited[[f"PC 1", f"PC 2", f"PC 3"]].to_numpy() - control[[f"PC 1", f"PC 2", f"PC 3"]].to_numpy())[-1].sum() 

                results = {
                        'trial_type': trial,
                        'context': name,
                        'stim_loc': stim,
                        'opto_stim_coord': eval(stim),
                        'PC': 'all',
                        'error': diff,
                        'distance': euclidean_dist
                    }
                results_total += [results]

                for pc in range(1, 4):
                    if inhibited.empty == False:
                        diff = inhibited[f"PC {pc}"].values[-1] - control[f"PC {pc}"].values[-1]
                        euclidean_dist = paired_distances(control[f"PC {pc}"].values[:, np.newaxis], inhibited[f"PC {pc}"].values[:, np.newaxis], method='euclidean').sum()
                    else:
                        euclidean_dist = np.nan
                        diff = np.nan

                    results = {
                        'trial_type': trial,
                        'context': name,
                        'stim_loc': stim,
                        'opto_stim_coord': eval(stim),
                        'PC': pc,
                        'error': diff,
                        'distance': euclidean_dist
                    }
                    results_total += [results]

    results_total = pd.DataFrame(results_total)
    save_path = os.path.join(result_path, 'MSE')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    seismic_palette = sns.diverging_palette(265, 10, s=100, l=40, sep=30, n=200, center="light", as_cmap=True)

    for trial in results_total.trial_type.unique():
        for pc in range(1, 4):
            fig, ax= plt.subplots(1,2, figsize=(5,4))
            fig.suptitle(f"{trial} PC {pc} error per stim spot")

            fig1, ax1= plt.subplots(1,2, figsize=(5,4))
            fig1.suptitle(f"{trial} PC {pc} distance from control per stim spot")

            if pc==1:
                vmax = 30
                
            elif pc==2:
                vmax = 15
            else:
                vmax = 8

            for i, (name, group) in enumerate(results_total[(results_total.trial_type==trial) & (results_total.PC==pc)].groupby(by=['context'])):
                im_df = generate_reduced_image_df(group.error.values[np.newaxis,:], group.opto_stim_coord)
                im_df = im_df.rename(columns={'dff0': 'error'})
                plot_grid_on_allen(im_df, outcome='error', palette=seismic_palette, result_path=None, vmin=-vmax, vmax=vmax, fig=fig, ax=ax[i])

                im_df = generate_reduced_image_df(group.distance.values[np.newaxis,:], group.opto_stim_coord)
                im_df = im_df.rename(columns={'dff0': 'distance'})
                plot_grid_on_allen(im_df, outcome='distance', palette='viridis', result_path=None, vmin=0, vmax=im_df.distance.max(), fig=fig1, ax=ax1[i])

                ax[i].set_title(f"{name}")
                ax1[i].set_title(f"{name}")
            
            fig.tight_layout()
            fig.savefig(os.path.join(save_path, f"{trial}_PC{pc}_mse.png"))
            fig1.tight_layout()
            fig1.savefig(os.path.join(save_path, f"{trial}_PC{pc}_distance.png"))

        fig, ax= plt.subplots(1,2, figsize=(5,4))
        fig.suptitle(f"{trial} all PCs error per stim spot")

        fig1, ax1= plt.subplots(1,2, figsize=(5,4))
        fig1.suptitle(f"{trial} all PCs distance from control per stim spot")
        for i, (name, group) in enumerate(results_total[(results_total.trial_type==trial) & (results_total.PC=='all')].groupby(by=['context'])):
            im_df = generate_reduced_image_df(group.error.values[np.newaxis,:], group.opto_stim_coord)
            im_df = im_df.rename(columns={'dff0': 'error'})
            plot_grid_on_allen(im_df, outcome='error', palette=seismic_palette, result_path=None, vmin=-40, vmax=40, fig=fig, ax=ax[i])

            im_df = generate_reduced_image_df(group.distance.values[np.newaxis,:], group.opto_stim_coord)
            im_df = im_df.rename(columns={'dff0': 'distance'})
            plot_grid_on_allen(im_df, outcome='distance', palette='viridis', result_path=None, vmin=0, vmax=150, fig=fig1, ax=ax1[i])

            ax[i].set_title(f"{name}")
            ax1[i].set_title(f"{name}")
        
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, f"{trial}_all_PCs_error.png"))

        fig1.tight_layout()
        fig1.savefig(os.path.join(save_path, f"{trial}_all_PCs_distance.png"))

    ## Context difference
    results_total = []
    for trial in subset_df.trial_type.unique():
        control_rewarded = subset_df[(subset_df.trial_type==trial) & (subset_df.opto_stim_coord=="(-5.0, 5.0)") & (subset_df.context=="rewarded") & (subset_df.time>=0)]
        control_n_rewarded = subset_df[(subset_df.trial_type==trial) & (subset_df.opto_stim_coord=="(-5.0, 5.0)") & (subset_df.context=="non-rewarded") & (subset_df.time>=0)]
                            
        for i, (name, group) in enumerate(subset_df[subset_df.trial_type==trial].groupby(by=['opto_stim_coord'])):
            if name == "(-5.0, 5.0)":
                    continue

            rewarded = group[(group.context=="rewarded") & (group.time>=0)]
            n_rewarded = group[(group.context=="non-rewarded") & (group.time>=0)]

            euclidean_dist = np.sqrt(np.sum((rewarded[[f"PC 1", f"PC 2", f"PC 3"]].to_numpy() - n_rewarded[[f"PC 1", f"PC 2", f"PC 3"]].to_numpy())**2, axis=1)).sum() -\
                            np.sqrt(np.sum((control_rewarded[[f"PC 1", f"PC 2", f"PC 3"]].to_numpy() - control_n_rewarded[[f"PC 1", f"PC 2", f"PC 3"]].to_numpy())**2, axis=1)).sum()
            
            diff = (rewarded[[f"PC 1", f"PC 2", f"PC 3"]].to_numpy() - n_rewarded[[f"PC 1", f"PC 2", f"PC 3"]].to_numpy())[-1].sum() -\
                    (control_rewarded[[f"PC 1", f"PC 2", f"PC 3"]].to_numpy() - control_n_rewarded[[f"PC 1", f"PC 2", f"PC 3"]].to_numpy())[-1].sum()
            
            results = {
                    'trial_type': trial,
                    'stim_loc': name,
                    'opto_stim_coord': eval(name),
                    'PC': 'all',
                    'error': diff,
                    'distance': euclidean_dist
                }
            results_total += [results]

            for pc in range(1, 4):
                control_diff = control_rewarded[f"PC {pc}"].values[-1] - control_n_rewarded[f"PC {pc}"].values[-1]
                control_dist = paired_distances(control_rewarded[f"PC {pc}"].values[:, np.newaxis], control_n_rewarded[f"PC {pc}"].values[:, np.newaxis], method='euclidean').sum()

                diff = (rewarded[f"PC {pc}"].values[-1] - n_rewarded[f"PC {pc}"].values[-1])- control_diff
                euclidean_dist = paired_distances(rewarded[f"PC {pc}"].values[:, np.newaxis], n_rewarded[f"PC {pc}"].values[:, np.newaxis], method='euclidean').sum()- control_dist

                results = {
                    'trial_type': trial,
                    'stim_loc': name,
                    'opto_stim_coord': eval(name),
                    'PC': pc,
                    'error': diff,
                    'distance': euclidean_dist
                }
                results_total += [results]

    results_total = pd.DataFrame(results_total)
    save_path = os.path.join(result_path, 'MSE')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    seismic_palette = sns.diverging_palette(265, 10, s=100, l=40, sep=30, n=200, center="light", as_cmap=True)

    for trial in results_total.trial_type.unique():
        for pc in range(1, 4):
            fig, ax= plt.subplots(figsize=(4,4))
            fig.suptitle(f"{trial} PC {pc} error between contexts")

            fig1, ax1= plt.subplots(figsize=(4,4))
            fig1.suptitle(f"{trial} PC {pc} distance between contexts")

            if pc==1:
                vmax = 15
                
            elif pc==2:
                vmax = 10
            else:
                vmax = 5

            im_df = generate_reduced_image_df(results_total.loc[(results_total.trial_type==trial) & (results_total.PC==pc), 'error'].values[np.newaxis,:], results_total.loc[(results_total.trial_type==trial) & (results_total.PC==pc), 'opto_stim_coord'])
            im_df = im_df.rename(columns={'dff0': 'error'})
            plot_grid_on_allen(im_df, outcome='error', palette=seismic_palette, result_path=None, vmin=-vmax, vmax=vmax, fig=fig, ax=ax)

            im_df = generate_reduced_image_df(results_total.loc[(results_total.trial_type==trial) & (results_total.PC==pc), 'distance'].values[np.newaxis,:], results_total.loc[(results_total.trial_type==trial) & (results_total.PC==pc), 'opto_stim_coord'])
            im_df = im_df.rename(columns={'dff0': 'distance'})
            plot_grid_on_allen(im_df, outcome='distance', palette=seismic_palette, result_path=None, vmin=-im_df.distance.abs().max(), vmax=im_df.distance.abs().max(), fig=fig1, ax=ax1)
            
            fig.tight_layout()
            fig.savefig(os.path.join(save_path, f"context_{trial}_PC{pc}_mse.png"))
            fig1.tight_layout()
            fig1.savefig(os.path.join(save_path, f"context_{trial}_PC{pc}_distance.png"))


        fig, ax= plt.subplots(figsize=(4,4))
        fig.suptitle(f"{trial} all PCs error between contexts")

        fig1, ax1= plt.subplots(figsize=(4,4))
        fig1.suptitle(f"{trial} all PCs distance between contexts")

        im_df = generate_reduced_image_df(results_total.loc[(results_total.trial_type==trial) & (results_total.PC=='all'), 'error'].values[np.newaxis,:], results_total.loc[(results_total.trial_type==trial) & (results_total.PC=='all'), 'opto_stim_coord'])
        im_df = im_df.rename(columns={'dff0': 'error'})
        plot_grid_on_allen(im_df, outcome='error', palette=seismic_palette, result_path=None, vmin=-10, vmax=10, fig=fig, ax=ax)

        im_df = generate_reduced_image_df(results_total.loc[(results_total.trial_type==trial) & (results_total.PC=='all'), 'distance'].values[np.newaxis,:], results_total.loc[(results_total.trial_type==trial) & (results_total.PC=='all'), 'opto_stim_coord'])
        im_df = im_df.rename(columns={'dff0': 'distance'})
        plot_grid_on_allen(im_df, outcome='distance', palette=seismic_palette, result_path=None, vmin=-300, vmax=300, fig=fig1, ax=ax1)

        fig.tight_layout()
        fig.savefig(os.path.join(save_path, f"context_{trial}_all_PCs_mse.png"))
        fig1.tight_layout()
        fig1.savefig(os.path.join(save_path, f"context_{trial}_all_PCs_distance.png"))

    ## Clustering of trajectories
    save_path = os.path.join(result_path, 'tSNE')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    # tsne_df = subset_df.loc[(subset_df.context=='rewarded') & (subset_df.trial_type=='whisker_trial'), ['time', 'opto_stim_coord', 'PC 1', 'PC 2', 'PC 3']]
    # tsne_df = tsne_df.melt(id_vars=['opto_stim_coord', 'time'], value_vars=['PC 1', 'PC 2', 'PC 3'], var_name='PC', value_name='values')
    # tsne_df = tsne_df.pivot(index='opto_stim_coord', columns=['PC', 'time'], values='values')
    opto_subset = opto_avg_df[opto_avg_df.trial_type=='whisker_trial']
    opto_subset['opto_stim_coord'] = opto_subset.opto_stim_coord.apply(str)
    opto_subset = opto_subset[['context', 'opto_stim_coord', 'data_mean_sub']]
    opto_subset = opto_subset.pivot(index='opto_stim_coord', columns='context', values='data_mean_sub')

    tsne_df = subset_df.loc[subset_df.trial_type=='whisker_trial', ['context', 'time', 'opto_stim_coord', 'PC 1', 'PC 2', 'PC 3']]
    tsne_df = tsne_df.melt(id_vars=['opto_stim_coord', 'context', 'time'], value_vars=['PC 1', 'PC 2', 'PC 3'], var_name='PC', value_name='values')
    tsne_df = tsne_df.pivot(index='opto_stim_coord', columns=['context', 'PC', 'time'], values='values')
    tsne_mat = tsne_df.to_numpy()
    from sklearn.manifold import TSNE
    coord_embedded =TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(tsne_mat)
    fig, ax = plt.subplots()
    ax.scatter(coord_embedded[:, 0], coord_embedded[:, 1])
    fig.savefig(os.path.join(save_path, 'tsne_all_context_all_trials.png'))

    import matplotlib.cm as cm
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    n_cl_max = 10
    for n_cl in range(2, n_cl_max + 1):
        clusterer = KMeans(n_clusters=n_cl, random_state=100, n_init="auto", max_iter=500)
        cluster_id = clusterer.fit_predict(tsne_mat)
        silhouette_avg = silhouette_score(tsne_mat, cluster_id)
        print("For n_clusters =", n_cl, "The average silhouette_score is :", silhouette_avg)
        sample_silhouette_values = silhouette_samples(tsne_mat, cluster_id)

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(18, 7))
        ax0.set_xlim([-0.1, 1])
        ax0.set_ylim([0, len(tsne_mat) + (n_cl + 1) * 10])
        y_lower = 10
        for i in range(n_cl):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_id == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_cl)
            ax0.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax0.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax0.set_title("The silhouette plot for the various clusters.")
        ax0.set_xlabel("The silhouette coefficient values")
        ax0.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax0.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax0.set_yticks([])  # Clear the yaxis labels / ticks
        ax0.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_id.astype(float) / n_cl)
        ax1.scatter(
            coord_embedded[:, 0], coord_embedded[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax1.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax1.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax1.set_title("The visualization of the clustered data.")
        ax1.set_xlabel("Feature space for the 1st feature")
        ax1.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_cl,
            fontsize=14,
            fontweight="bold",
        )
        fig.savefig(os.path.join(save_path, f'rewarded_whisker_trial_{n_cl}_clusters_silhouette.png'))
        
    for i in range(2, 10):
        clusterer = KMeans(n_clusters=int(i), random_state=100, n_init="auto", max_iter=500) # 4 clusters is best according to silhouette score
        cluster_id = clusterer.fit_predict(tsne_mat)

        color_idx = np.array(['gray' for stim in tsne_df.index])
        color_idx[tsne_df.index.isin(['(-0.5, 0.5)'])] = 'r'
        color_idx[tsne_df.index.isin(['(0.5, 4.5)', '(1.5, 3.5)'])] = 'm'
        color_idx[tsne_df.index.isin(['(-1.5, 3.5)', '(-1.5, 4.5)'])] = 'c'
        color_idx[tsne_df.index.isin(['(1.5, 1.5)', '(2.5, 2.5)'])] = 'g'
        color_idx[tsne_df.index.isin(['(-5.0, 5.0)'])] = 'k'   

        fig, ax = plt.subplots()
        ax.scatter(coord_embedded[:, 0], coord_embedded[:, 1],c=cluster_id, cmap='brg')
        fig.savefig(os.path.join(save_path, f'{i}_tsne_clusters_id.png'))
        fig,ax = plt.subplots()
        ax.scatter(opto_subset.loc[tsne_df.drop('(-5.0, 5.0)').index, 'rewarded'], opto_subset.loc[tsne_df.drop('(-5.0, 5.0)').index, 'non-rewarded'], c=cluster_id[tsne_df.index!='(-5.0, 5.0)'], cmap='brg')
        ax.set_xlabel('Rewarded $\Delta$ PLick')
        ax.set_ylabel('Non-rewarded $\Delta$ PLick')
        fig.savefig(os.path.join(save_path, f'{i}_clusters_behaviour_correlations.png'))

        fig, ax = plt.subplots(figsize=(4,4))
        im_df = generate_reduced_image_df(cluster_id[np.newaxis, :], [eval(idx) for idx in tsne_df.index])
        im_df = im_df.rename(columns={'dff0': 'cluster_id'})
        plot_grid_on_allen(im_df, outcome='cluster_id', palette='tab10', dotsize=440, result_path=None, vmin=0, vmax=cluster_id.max(), fig=fig, ax=ax)
        ax.set_axis_off()
        fig.savefig(os.path.join(save_path, f'{i}_clusters_map.png'))

    ## correlate with opto behaviour results
    clusterer = KMeans(n_clusters=9, random_state=100, n_init="auto", max_iter=500) # 4 clusters is best according to silhouette score
    cluster_id = clusterer.fit_predict(tsne_mat)
    cluster_df = pd.DataFrame(cluster_id, index=tsne_df.index, columns=['cluster_id'])   
    opto_subset = opto_subset.join(cluster_df)
    sorted_df = opto_subset.groupby('cluster_id').agg('mean').sort_values(by=['rewarded', 'non-rewarded']).reset_index().reset_index().rename(columns={'index':'order'})
    opto_subset['sorted_idx'] = opto_subset['cluster_id'].map(dict(sorted_df[['cluster_id', 'order']].values))

    fig=plt.figure(figsize=(12,4))
    ax1=fig.add_subplot(1,3,1)
    ax1.scatter(opto_subset['rewarded'], opto_subset['sorted_idx'], c=opto_subset['cluster_id'], cmap='tab10')
    ax1.set_title('Rewarded opto_effect vs cluster_id')
    ax1.set_xlabel('$\Delta$ PLick')
    ax1.set_ylabel('Sorted cluster_id')
    ax1.set_xlim([-0.5,0.5])
    ax2=fig.add_subplot(1,3,2)
    ax2.scatter(opto_subset['non-rewarded'], opto_subset['sorted_idx'], c=opto_subset['cluster_id'], cmap='tab10')
    ax2.set_title('Non-rewarded opto_effect vs cluster_id')
    ax2.set_xlabel('$\Delta$ PLick')
    ax2.set_ylabel('Sorted cluster_id')
    ax2.set_xlim([-0.5,0.5])
    ax3=fig.add_subplot(1,3,3, projection='3d')
    ax3.scatter(opto_subset['rewarded'], opto_subset['non-rewarded'], opto_subset['sorted_idx'], c=opto_subset['cluster_id'], cmap='tab10')
    ax3.set_title('Rewarded and non-rewarded vs cluster_id')
    ax3.set_xlabel('Rewarded $\Delta$ PLick')
    ax3.set_ylabel('Non-rewarded $\Delta$ PLick')
    ax3.set_zlabel('Sorted cluster_id')
    ax3.set_xlim([-0.5,0.5])
    ax3.set_ylim([-0.5,0.5])
    fig.savefig(os.path.join(save_path, 'cluster_id_vs_behaviour_correlations.png'))
    fig.savefig(os.path.join(save_path, 'cluster_id_vs_behaviour_correlations.svg'))

    ## leave one out


def load_opto_data(group):
    opto_results = fr'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/optogenetic_results/{group}'
    opto_results = haas_pathfun(opto_results)
    single_mouse_result_files = glob.glob(os.path.join(opto_results, "*", "opto_data.json"))
    opto_df = []
    for file in single_mouse_result_files:
        d= pd.read_json(file)
        d['mouse_name'] = [file.split("/")[-2] for i in range(d.shape[0])]
        opto_df += [d]
    opto_df = pd.concat(opto_df)
    opto_df = opto_df.loc[opto_df.opto_grid_ap!=3.5]
    opto_avg_df = opto_df.groupby(by=['context', 'trial_type', 'opto_grid_ml', 'opto_grid_ap']).agg(
                                                                                 data=('data_mean', list),
                                                                                 data_sub=('data_mean_sub', list),
                                                                                 data_mean=('data_mean', 'mean'),
                                                                                 data_mean_sub=(
                                                                                 'data_mean_sub', 'mean'),
                                                                                 shuffle_dist=('shuffle_dist', 'sum'),
                                                                                 shuffle_dist_sub=(
                                                                                 'shuffle_dist_sub', 'sum'),
                                                                                 percentile_avg=('percentile', 'mean'),
                                                                                 percentile_avg_sub=(
                                                                                 'percentile_sub', 'mean'),
                                                                                 n_sigma_avg=('n_sigma', 'mean'),
                                                                                 n_sigma_avg_sub=(
                                                                                 'n_sigma_sub', 'mean'))
    opto_avg_df['shuffle_mean'] = opto_avg_df.apply(lambda x: np.mean(x.shuffle_dist), axis=1)
    opto_avg_df['shuffle_std'] = opto_avg_df.apply(lambda x: np.std(x.shuffle_dist), axis=1)
    opto_avg_df['shuffle_mean_sub'] = opto_avg_df.apply(lambda x: np.mean(x.shuffle_dist_sub), axis=1)
    opto_avg_df['shuffle_std_sub'] = opto_avg_df.apply(lambda x: np.std(x.shuffle_dist_sub), axis=1)

    opto_avg_df = opto_avg_df.reset_index()
    opto_avg_df['opto_stim_coord'] = opto_avg_df.apply(lambda x: tuple([x.opto_grid_ap, x.opto_grid_ml]), axis=1)
    return opto_avg_df


def leave_one_out_PCA(nwb_files, output_path):
    result_path = Path(output_path, 'PCA_leaveoneout')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    total_df = []
    coords_list = {'wS1': "(-1.5, 3.5)", 'wS2': "(-1.5, 4.5)", 'wM1': "(1.5, 1.5)", 'wM2': "(2.5, 1.5)", 'RSC': "(-0.5, 0.5)",
                   'ALM': "(2.5, 2.5)", 'tjS1':"(0.5, 4.5)", 'tjM1':"(1.5, 3.5)", 'control': "(-5.0, 5.0)"}
    
    for nwb_file in nwb_files:
        session_id = nwb_read.get_session_id(nwb_file)
        mouse_id = nwb_read.get_mouse_id(nwb_file)
        df = [pd.read_parquet(Path(output_path, session_id, 'results.parquet.gzip', compression='gzip'))]
        # df['group'] = 'VGAT'
        total_df += df

    total_df = pd.concat(total_df, ignore_index=True)
    total_df.context = total_df.context.map({0:'non-rewarded', 1:'rewarded'})
    total_df['time'] = [[np.linspace(-1,3.98,250)] for i in range(total_df.shape[0])]
    total_df['legend']= total_df.apply(lambda x: f"{x.opto_stim_coord} - {'lick' if x.lick_flag==1 else 'no lick'}",axis=1)

    d = {c: lambda x: x.unique()[0] for c in ['opto_stim_loc', 'legend']}
    d['time'] = lambda x: list(x)[0][0]
    for c in ['(-0.5, 0.5)', '(-0.5, 1.5)', '(-0.5, 2.5)', '(-0.5, 3.5)', '(-0.5, 4.5)', '(-0.5, 5.5)', '(-1.5, 0.5)',
       '(-1.5, 1.5)', '(-1.5, 2.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(-1.5, 5.5)', '(-2.5, 0.5)', '(-2.5, 1.5)', '(-2.5, 2.5)',
       '(-2.5, 3.5)', '(-2.5, 4.5)', '(-2.5, 5.5)', '(-3.5, 0.5)', '(-3.5, 1.5)', '(-3.5, 2.5)', '(-3.5, 3.5)', '(-3.5, 4.5)',
       '(-3.5, 5.5)', '(0.5, 0.5)', '(0.5, 1.5)', '(0.5, 2.5)', '(0.5, 3.5)', '(0.5, 4.5)', '(0.5, 5.5)', '(1.5, 0.5)', 
       '(1.5, 1.5)', '(1.5, 2.5)', '(1.5, 3.5)', '(1.5, 4.5)', '(1.5, 5.5)', '(2.5, 0.5)', '(2.5, 1.5)', '(2.5, 2.5)', '(2.5, 3.5)', '(2.5, 4.5)', '(2.5, 5.5)', 
       'A1', 'ALM', 'tjM1', 'tjS1', 'RSC', 'wM1', 'wM2', 'wS1', 'wS2']:
        d[f"{c}"]= lambda x: np.nanmean(np.stack(x), axis=0)
          
    mouse_df = total_df.groupby(by=['mouse_id', 'context', 'trial_type', 'opto_stim_coord', 'lick_flag']).agg(d).reset_index()
    mouse_df = mouse_df.melt(id_vars=['mouse_id', 'context', 'trial_type', 'legend', 'opto_stim_coord', 'lick_flag', 'time'],
                                 value_vars=['(-0.5, 0.5)', '(-0.5, 1.5)', '(-0.5, 2.5)', '(-0.5, 3.5)', '(-0.5, 4.5)', '(-0.5, 5.5)', '(-1.5, 0.5)',
       '(-1.5, 1.5)', '(-1.5, 2.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(-1.5, 5.5)', '(-2.5, 0.5)', '(-2.5, 1.5)', '(-2.5, 2.5)',
       '(-2.5, 3.5)', '(-2.5, 4.5)', '(-2.5, 5.5)', '(-3.5, 0.5)', '(-3.5, 1.5)', '(-3.5, 2.5)', '(-3.5, 3.5)', '(-3.5, 4.5)',
       '(-3.5, 5.5)', '(0.5, 0.5)', '(0.5, 1.5)', '(0.5, 2.5)', '(0.5, 3.5)', '(0.5, 4.5)', '(0.5, 5.5)', '(1.5, 0.5)', 
       '(1.5, 1.5)', '(1.5, 2.5)', '(1.5, 3.5)', '(1.5, 4.5)', '(1.5, 5.5)', '(2.5, 0.5)', '(2.5, 1.5)', '(2.5, 2.5)', '(2.5, 3.5)', '(2.5, 4.5)', '(2.5, 5.5)',],
                                 var_name='roi',
                                 value_name='dff0').explode(['time', 'dff0'])
    avg_df = mouse_df.groupby(by=['context', 'trial_type', 'lick_flag', 'legend', 'opto_stim_coord', 'roi', 'time']).agg(lambda x: np.nanmean(x)).reset_index()
   
    avg_df.time = avg_df.time.round(2)
    avg_df = avg_df[(avg_df.time>=-0.15)&(avg_df.time<=0.15)]
    subset_df = avg_df[(avg_df.trial_type.isin(['whisker_trial', 'no_stim_trial'])) & (avg_df.opto_stim_coord=="(-5.0, 5.0)")].pivot(index=['context','trial_type', 'legend', 'time'], columns='roi', values='dff0')
    avg_data_for_pca = subset_df.to_numpy()
    labels = subset_df.keys()

    # Standardize average data for training: Based on trials with light on control location 
    scaler = StandardScaler()
    fit_scaler = scaler.fit(avg_data_for_pca)
    avg_data_for_pca = fit_scaler.transform(avg_data_for_pca)
    pca = PCA(n_components=15)
    results = pca.fit(np.nan_to_num(avg_data_for_pca))

    # Project whisker and catch trials 
    mouse_df = total_df.groupby(by=['mouse_id', 'context', 'trial_type', 'opto_stim_coord']).agg(d).reset_index()
    mouse_df = mouse_df.melt(id_vars=['mouse_id', 'context', 'trial_type', 'opto_stim_coord', 'time'],
                                 value_vars=['(-0.5, 0.5)', '(-0.5, 1.5)', '(-0.5, 2.5)', '(-0.5, 3.5)', '(-0.5, 4.5)', '(-0.5, 5.5)', '(-1.5, 0.5)',
       '(-1.5, 1.5)', '(-1.5, 2.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(-1.5, 5.5)', '(-2.5, 0.5)', '(-2.5, 1.5)', '(-2.5, 2.5)',
       '(-2.5, 3.5)', '(-2.5, 4.5)', '(-2.5, 5.5)', '(-3.5, 0.5)', '(-3.5, 1.5)', '(-3.5, 2.5)', '(-3.5, 3.5)', '(-3.5, 4.5)',
       '(-3.5, 5.5)', '(0.5, 0.5)', '(0.5, 1.5)', '(0.5, 2.5)', '(0.5, 3.5)', '(0.5, 4.5)', '(0.5, 5.5)', '(1.5, 0.5)', 
       '(1.5, 1.5)', '(1.5, 2.5)', '(1.5, 3.5)', '(1.5, 4.5)', '(1.5, 5.5)', '(2.5, 0.5)', '(2.5, 1.5)', '(2.5, 2.5)', '(2.5, 3.5)', '(2.5, 4.5)', '(2.5, 5.5)',],
                                 var_name='roi',
                                 value_name='dff0').explode(['time', 'dff0'])
    avg_df = mouse_df.groupby(by=['context', 'trial_type', 'opto_stim_coord', 'roi', 'time']).agg(lambda x: np.nanmean(x)).reset_index()
    avg_df = avg_df[(avg_df.time>=-0.15)&(avg_df.time<=0.15)]
    subset_df = avg_df[avg_df.trial_type.isin(['whisker_trial', 'no_stim_trial'])].pivot(index=['context','trial_type', 'opto_stim_coord', 'time'], columns='roi', values='dff0')
    avg_data_for_pca = subset_df.to_numpy()
    avg_data_for_pca = fit_scaler.transform(avg_data_for_pca)
    principal_components = pca.transform(np.nan_to_num(avg_data_for_pca))
    loadings = pca.components_.T

    loo_df = []

    pc_df = pd.DataFrame(data=principal_components, index=subset_df.index)
    pc_df.columns = [f"PC {i+1}" for i in range(0, principal_components.shape[1])]
    pc_df['KO'] = 'full_loadings'
    loo_df += [pc_df]

    for i, ko in enumerate(subset_df.columns):
        ko_loadings = loadings.copy()
        ko_loadings[i] = 0

        principal_components = avg_data_for_pca @ ko_loadings

        pc_df = pd.DataFrame(data=principal_components, index=subset_df.index)
        pc_df.columns = [f"PC {i+1}" for i in range(0, principal_components.shape[1])]
        pc_df['KO'] = ko
        loo_df += [pc_df]

    loo_df = pd.concat(loo_df).reset_index()
    loo_df['legend'] = loo_df.apply(lambda x: f'stim_{x.opto_stim_coord}-del_{x.KO}', axis=1)

    lines = ['#005F60', '#FD5901', '#000000', '#808080']
    handles = [Line2D([0], [0], color=c, lw=4) for c in lines]

    ko_effect=[]
    for ko in loo_df.KO.unique():
        if ko=='full_loadings':
                continue
        for stim in loo_df.opto_stim_coord.unique():
            if stim=='(-5.0, 5.0)':
                continue

            if not os.path.exists(os.path.join(result_path, ko)):
                os.makedirs(os.path.join(result_path, ko))

            ko_df = loo_df.loc[(loo_df.KO.isin([ko, 'full_loadings'])) & (loo_df.opto_stim_coord.isin([stim, '(-5.0, 5.0)']))]  
            ko_df = ko_df.loc[ko_df.time>=0]      
            # fig,ax=plt.subplots(2,3, figsize=(12,8))
            # for i, group in enumerate(product(('rewarded', 'non-rewarded'), ('PC 1', 'PC 2', 'PC 3'))):
            #     g = sns.lineplot(ko_df.loc[(ko_df.context==group[0]) & (ko_df.trial_type=='whisker_trial')], x='time', y=group[1], hue='legend', 
            #                  hue_order=[f"stim_{stim}-del_{ko}", f"stim_{stim}-del_full_loadings", f"stim_(-5.0, 5.0)-del_{ko}", f"stim_(-5.0, 5.0)-del_full_loadings"], 
            #                  palette=['#005F60', '#FD5901', '#000000', '#808080'],
            #                  ax=ax.flat[i])
            #     g.get_legend().remove()
            #     ax.flat[i].set_title(group)

            # fig.legend(handles, [f"stim_{stim}-del_{ko}", f"stim_{stim}-del_full_loadings", f"control_stim-del_{ko}", f"control_stim-del_full_loadings"], loc='outside upper right')
            # fig.tight_layout()
            # fig.savefig(os.path.join(result_path, ko, f'{stim}-stim_whisker_trials.png'))

            for c in ['rewarded', 'non-rewarded']:
                euclidean_dist = np.sqrt(np.sum((ko_df.loc[(ko_df.context==c) & (ko_df.trial_type=='whisker_trial') & (ko_df.opto_stim_coord==stim) & (ko_df.KO==ko), [f"PC 1", f"PC 2", f"PC 3"]].to_numpy() -\
                                                    ko_df.loc[(ko_df.context==c) & (ko_df.trial_type=='whisker_trial') & (ko_df.opto_stim_coord==stim) & (ko_df.KO=='full_loadings'), [f"PC 1", f"PC 2", f"PC 3"]].to_numpy())**2, axis=1)).sum() 
                results ={
                    'KO': ko,
                    'stim': stim,
                    'context': c,
                    'trial_type': 'whisker_trial',
                    'distance': euclidean_dist
                }

                ko_effect += [results]


        for c in ['rewarded', 'non-rewarded']:
            euclidean_dist = np.sqrt(np.sum((ko_df.loc[(ko_df.context==c) & (ko_df.trial_type=='whisker_trial') & (ko_df.opto_stim_coord=='(-5.0, 5.0)') & (ko_df.KO==ko), [f"PC 1", f"PC 2", f"PC 3"]].to_numpy() -\
                                                ko_df.loc[(ko_df.context==c) & (ko_df.trial_type=='whisker_trial') & (ko_df.opto_stim_coord=='(-5.0, 5.0)') &  (ko_df.KO=='full_loadings'), [f"PC 1", f"PC 2", f"PC 3"]].to_numpy())**2, axis=1)).sum() 

            results ={
                    'KO': ko,
                    'stim': '(-5.0, 5.0)',
                    'context': c,
                    'trial_type': 'whisker_trial',
                    'distance': euclidean_dist
                }
                
            ko_effect += [results] 
    ko_effect = pd.DataFrame(ko_effect)
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    sns.heatmap(ko_effect.loc[ko_effect.context=='rewarded'].pivot(index='KO', columns='stim', values='distance'), ax=ax[0], vmin=0, vmax=15)
    ax[0].set_title('Rewarded whisker_trial')
    sns.heatmap(ko_effect.loc[ko_effect.context=='non-rewarded'].pivot(index='KO', columns='stim', values='distance'), ax=ax[1], vmin=0, vmax=15)
    ax[1].set_title('Non-rewarded whisker_trial')
    fig.tight_layout()
    fig.savefig(os.path.join(result_path, f'whisker_trials_ko_vs_stim.png'))

    for stim, group in ko_effect.groupby('stim'):
        fig,ax=plt.subplots(2,1, figsize=(12,8))
        fig.suptitle(f"{stim}-stim KO effect")
        ax[0].stem(group.loc[group.context=='rewarded', 'KO'], group.loc[group.context=='rewarded', 'distance'])
        ax[0].set_title('Rewarded whisker_trial')
        ax[1].stem(group.loc[group.context=='non-rewarded', 'KO'], group.loc[group.context=='non-rewarded', 'distance'])
        ax[1].set_title('Non-rewarded whisker_trial')

        for a in ax:
            a.set_ylabel('Euclidean distance')
            a.set_xlabel('ROI KO')
            a.set_xticks(range(group.loc[group.context=='rewarded', 'KO'].shape[0]), group.loc[group.context=='rewarded', 'KO'], rotation=90)

        fig.tight_layout()
        fig.savefig(os.path.join(result_path, f'{stim}_stim_stemplot.png'))

        fig,ax = plt.subplots(1,2, figsize=(8,4))
        fig.suptitle(f'{stim}-stim euclidean distance per KO')
        im_df = generate_reduced_image_df(group.loc[group.context=='rewarded', 'distance'].values[np.newaxis, :], group.loc[group.context=='rewarded', 'KO'].apply(eval))
        im_df = im_df.rename(columns={'dff0': 'distance'})
        plot_grid_on_allen(im_df, outcome='distance', palette='plasma', result_path=None, vmin=0, vmax=10, fig=fig, ax=ax[0])
        ax[0].set_title('Rewarded')

        im_df = generate_reduced_image_df(group.loc[group.context=='non-rewarded', 'distance'].values[np.newaxis, :], group.loc[group.context=='rewarded', 'KO'].apply(eval))
        im_df = im_df.rename(columns={'dff0': 'distance'})
        plot_grid_on_allen(im_df, outcome='distance', palette='plasma', result_path=None, vmin=0, vmax=10, fig=fig, ax=ax[1])
        ax[1].set_title('Non-rewarded')
        fig.savefig(os.path.join(result_path, f'{stim}_stim_effect_grids.png'))

    for ko, group in ko_effect.groupby('KO'):
        fig,ax = plt.subplots(1,2, figsize=(8,4))
        fig.suptitle(f'{ko}-KO euclidean distance per stim point')
        im_df = generate_reduced_image_df(group.loc[group.context=='rewarded', 'distance'].values[np.newaxis, :], group.loc[group.context=='rewarded', 'stim'].apply(eval))
        im_df = im_df.rename(columns={'dff0': 'distance'})
        plot_grid_on_allen(im_df, outcome='distance', palette='plasma', result_path=None, vmin=0, vmax=10, fig=fig, ax=ax[0])
        ax[0].set_title('Rewarded')

        im_df = generate_reduced_image_df(group.loc[group.context=='non-rewarded', 'distance'].values[np.newaxis, :], group.loc[group.context=='rewarded', 'stim'].apply(eval))
        im_df = im_df.rename(columns={'dff0': 'distance'})
        plot_grid_on_allen(im_df, outcome='distance', palette='plasma', result_path=None, vmin=0, vmax=10, fig=fig, ax=ax[1])
        ax[1].set_title('Non-rewarded')
        fig.savefig(os.path.join(result_path, f'{ko}_ko_effect_grids.png'))


def main(nwb_files, output_path):
    # combine_data(nwb_files, output_path)
    # plot_opto_effect_matrix(nwb_files, output_path)

    # output_path = os.path.join('//sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Pol_Bech', 'Pop_results',
    #                             'Context_behaviour', 'optogenetic_widefield_results')
    # output_path = haas_pathfun(output_path)
    dimensionality_reduction(nwb_files, output_path)


if __name__ == "__main__":

    for file in ['context_sessions_wf_opto_controls', 'context_sessions_wf_opto']:#'context_sessions_wf_opto', 
        config_file = f"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Session_list/{file}.yaml"
        config_file = haas_pathfun(config_file)

        with open(config_file, 'r', encoding='utf8') as stream:
            config_dict = yaml.safe_load(stream)

        nwb_files = [haas_pathfun(p.replace("\\", '/')) for p in config_dict['Session path']]
        
        output_path = os.path.join('//sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Pol_Bech', 'Pop_results',
                                    'Context_behaviour', 'optogenetic_widefield_results', 'controls' if 'controls' in str(config_file) else 'VGAT')
        output_path = haas_pathfun(output_path)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        main(nwb_files, output_path)
