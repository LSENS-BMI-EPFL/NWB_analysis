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

import nwb_utils.utils_behavior as bhv_utils
import utils.behaviour_plot_utils as plot_utils
from utils.haas_utils import *
from utils.wf_plotting_utils import reduce_im_dimensions
import nwb_wrappers.nwb_reader_functions as nwb_read
from nwb_utils import utils_misc
from nwb_utils.utils_plotting import lighten_color, remove_top_right_frame
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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
    result_path = Path(output_path, 'PCA_200')
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

    # avg_df = avg_df.melt(id_vars=['context', 'trial_type', 'stim_loc', 'coord', 'lick_flag', 'legend', 'time'],
    #                               value_vars=['A1', 'ALM', 'tjM1', 'tjS1', 'RSC', 'wM1', 'wM2','wS1', 'wS2'],
    #                               var_name='roi',
    #                               value_name='dff0').explode(['time', 'dff0']).reset_index()
    
    avg_df.time = avg_df.time.round(2)
    avg_df = avg_df[(avg_df.time>=-0.2)&(avg_df.time<=0.2)]
    avg_df = avg_df[avg_df.trial_type.isin(['whisker_trial', 'no_stim_trial'])].pivot(index=['context','trial_type', 'legend', 'time'], columns='roi', values='dff0')
    avg_data_for_pca = avg_df.to_numpy()
    scaler = StandardScaler()
    avg_data_for_pca = scaler.fit_transform(avg_data_for_pca)

    pca = PCA(n_components=15)
    results = pca.fit(np.nan_to_num(avg_data_for_pca))
    principal_components = pca.transform(np.nan_to_num(avg_data_for_pca))

    pc_df = pd.DataFrame(data=principal_components, index=avg_df.index)
    pc_df.columns = [f"PC {i+1}" for i in range(0, principal_components.shape[1])]

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
    fig.savefig(Path(result_path, 'variance_explained.png'))

    avg_df = avg_df.join(pc_df).reset_index()

    avg_df['lick_flag'] = avg_df.apply(lambda x: 0 if 'no lick' in x.legend else 1, axis=1)
    avg_df['stim_loc'] = avg_df.apply(lambda x: x.legend.split(" -")[0], axis=1)

    ## Plot biplots to see which variables (i.e. rois) contain most of the explained variance (?)
    labels = avg_df.stim_loc.unique()
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

    ## Plot PC timecourses

    long_df = avg_df.melt(id_vars=['context', 'trial_type', 'stim_loc', 'lick_flag', 'legend', 'time'], 
                            value_vars=['PC 1', 'PC 2', 'PC 3'], 
                            var_name='PC', 
                            value_name='data').explode(['time', 'data'])
    long_df = long_df[long_df.stim_loc.isin(list(coords_list.values()))]
    long_df['context_legend'] = long_df.apply(lambda x: f"{x.context} - {'lick' if x.lick_flag==1 else 'no lick'}", axis=1)
    fig, ax = plt.subplots(3, 4, figsize=(8,6))
    fig.suptitle("PC timecourses")
    g = sns.relplot(data=long_df[long_df.trial_type=='whisker_trial'], x='time', y='data', hue='context_legend', 
                    hue_order=['rewarded - lick', 'rewarded - no lick', 'non-rewarded - lick', 'non-rewarded - no lick'], 
                    palette=['#348A18', '#83E464', '#6E188A', '#C564E4'], 
                    col='stim_loc', row='PC', kind='line', linewidth=2)
    g.figure.savefig(Path(result_path, 'whisker_PC_timecourses_by_region.png'))
    g.figure.savefig(Path(result_path, 'whisker_PC_timecourses_by_region.svg'))
    
    fig, ax = plt.subplots(3, 4, figsize=(8,6))
    fig.suptitle("PC timecourses")
    g = sns.relplot(data=long_df[long_df.trial_type=='no_stim_trial'], x='time', y='data', hue='context_legend', 
                    hue_order=['rewarded - lick', 'rewarded - no lick', 'non-rewarded - lick', 'non-rewarded - no lick'], 
                    palette=['#348A18', '#83E464', '#6E188A', '#C564E4'], 
                    col='stim_loc', row='PC', kind='line', linewidth=2)
    g.figure.savefig(Path(result_path, 'catch_PC_timecourses_by_region.png'))
    g.figure.savefig(Path(result_path, 'catch_PC_timecourses_by_region.svg'))

    g = sns.relplot(data=long_df[long_df.trial_type=='whisker_trial'], x='time', y='data', hue='stim_loc', 
                    hue_order=["(-5.0, 5.0)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 1.5)", "(-0.5, 0.5)", "(2.5, 2.5)", "(0.5, 4.5)", "(1.5, 3.5)"], 
                    palette=['#000000', '#00ffff', '#24dbff', '#49b6ff', '#6d92ff', '#926dff', '#b649ff', '#db24ff', '#ff00ff'], 
                    col='context_legend', row='PC', kind='line', linewidth=2)
    g.figure.savefig(Path(result_path, 'whisker_PC_timecourses_by_trial_outcome.png'))
    g.figure.savefig(Path(result_path, 'whisker_PC_timecourses_by_trial_outcome.svg'))

    g = sns.relplot(data=long_df[long_df.trial_type=='no_stim_trial'], x='time', y='data', hue='stim_loc', 
                    hue_order=["(-5.0, 5.0)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 1.5)", "(-0.5, 0.5)", "(2.5, 2.5)", "(0.5, 4.5)", "(1.5, 3.5)"], 
                    palette=['#000000', '#00ffff', '#24dbff', '#49b6ff', '#6d92ff', '#926dff', '#b649ff', '#db24ff', '#ff00ff'], 
                    col='context_legend', row='PC', kind='line', linewidth=2)
    g.figure.savefig(Path(result_path, 'catch_PC_timecourses_by_trial_outcome.png'))
    g.figure.savefig(Path(result_path, 'catch_PC_timecourses_by_trial_outcome.svg'))

    ## Plot projected time courses onto PCx vs PCy
    color_dict = {"(-5.0, 5.0)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#000000'], N=avg_df.time.unique().shape[0]),
                  "(-1.5, 3.5)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#00ffff'], N=avg_df.time.unique().shape[0]),
                  "(-1.5, 4.5)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#24dbff'], N=avg_df.time.unique().shape[0]),
                  "(1.5, 1.5)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#49b6ff'], N=avg_df.time.unique().shape[0]),
                  "(2.5, 1.5)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#6d92ff'], N=avg_df.time.unique().shape[0]),
                  "(-0.5, 0.5)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#926dff'], N=avg_df.time.unique().shape[0]),
                  "(2.5, 2.5)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#b649ff'], N=avg_df.time.unique().shape[0]),
                  "(0.5, 4.5)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#db24ff'], N=avg_df.time.unique().shape[0]),
                  "(1.5, 3.5)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#ff00ff'], N=avg_df.time.unique().shape[0])}
    
    lines = ['#000000', '#00ffff', '#24dbff', '#49b6ff', '#6d92ff', '#926dff', '#b649ff', '#db24ff', '#ff00ff']
    handles = [Line2D([0], [0], color=c, lw=4) for c in lines]

    for trial in avg_df.trial_type.unique():
        fig, ax= plt.subplots(2,2, figsize=(8,8), sharey=True, sharex=True)
        fig1, ax1 = plt.subplots(2,2, figsize=(8,8), sharey=True, sharex=True)
        fig2, ax2 = plt.subplots(2,2, figsize=(8,8), sharey=True, sharex=True)

        for i, (name, group) in enumerate(avg_df[avg_df.trial_type==trial].groupby(by=['context', 'lick_flag'])):
            for stim in color_dict.keys():
                ax.flat[i].scatter(group.loc[group.stim_loc==stim, 'PC 1'], group.loc[group.stim_loc==stim, 'PC 2'], c=group.loc[group.stim_loc==stim, 'time'], cmap=color_dict[stim], label=stim, s=10)
                ax.flat[i].scatter(group.loc[(group.stim_loc==stim) & (group.time==0), 'PC 1'], group.loc[(group.stim_loc==stim) & (group.time==0), 'PC 2'], s=10, facecolors='none', edgecolors='r')
                ax1.flat[i].scatter(group.loc[group.stim_loc==stim, 'PC 2'], group.loc[group.stim_loc==stim, 'PC 3'], c=group.loc[group.stim_loc==stim, 'time'], cmap=color_dict[stim], label=stim, s=10)
                ax1.flat[i].scatter(group.loc[(group.stim_loc==stim) & (group.time==0), 'PC 2'], group.loc[(group.stim_loc==stim) & (group.time==0), 'PC 3'], s=10, facecolors='none', edgecolors='r')
                ax2.flat[i].scatter(group.loc[group.stim_loc==stim, 'PC 1'], group.loc[group.stim_loc==stim, 'PC 3'], c=group.loc[group.stim_loc==stim, 'time'], cmap=color_dict[stim], label=stim, s=10)
                ax2.flat[i].scatter(group.loc[(group.stim_loc==stim) & (group.time==0), 'PC 1'], group.loc[(group.stim_loc==stim) & (group.time==0), 'PC 3'], s=10, facecolors='none', edgecolors='r')

            ax.flat[i].set_xlabel('PC 1')
            ax.flat[i].set_ylabel('PC 2')
            ax.flat[i].set_title(f"{'Rewarded' if name[0]=='rewarded' else 'Non-rewarded'} {trial} {'lick' if name[1] ==1 else 'no-lick'}")

            ax1.flat[i].set_xlabel('PC 2')
            ax1.flat[i].set_ylabel('PC 3')
            ax1.flat[i].set_title(f"{'Rewarded' if name[0]=='rewarded' else 'Non-rewarded'} {trial} {'lick' if name[1] ==1 else 'no-lick'}")

            ax2.flat[i].set_xlabel('PC 1')
            ax2.flat[i].set_ylabel('PC 3')
            ax2.flat[i].set_title(f"{'Rewarded' if name[0]=='rewarded' else 'Non-rewarded'} {trial} {'lick' if name[1] ==1 else 'no-lick'}")

        fig.legend(handles, ["(-5.0, 5.0)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 1.5)", "(-0.5, 0.5)", "(2.5, 2.5)", "(0.5, 4.5)", "(1.5, 3.5)"], loc='outside upper right')
        fig.savefig(Path(result_path, f"{trial}_dimensionality_reduction_PC1vsPC2.png"))
        fig.savefig(Path(result_path, f"{trial}_dimensionality_reduction_PC1vsPC2.svg"))
        fig1.legend(handles, ["(-5.0, 5.0)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 1.5)", "(-0.5, 0.5)", "(2.5, 2.5)", "(0.5, 4.5)", "(1.5, 3.5)"], loc='outside upper right')
        fig1.savefig(Path(result_path, f"{trial}_dimensionality_reduction_PC2vsPC3.png"))
        fig1.savefig(Path(result_path, f"{trial}_dimensionality_reduction_PC2vsPC3.svg"))

        fig2.legend(handles, ["(-5.0, 5.0)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 1.5)", "(-0.5, 0.5)", "(2.5, 2.5)", "(0.5, 4.5)", "(1.5, 3.5)"], loc='outside upper right')
        fig2.savefig(Path(result_path, f"{trial}_dimensionality_reduction_PC1vsPC3.png"))
        fig2.savefig(Path(result_path, f"{trial}_dimensionality_reduction_PC1vsPC3.svg"))

    ## PCA projetions with arrowplots
    for trial in avg_df.trial_type.unique():
        fig, ax= plt.subplots(2,2, figsize=(8,8), sharey=True, sharex=True)
        fig1, ax1 = plt.subplots(2,2, figsize=(8,8), sharey=True, sharex=True)
        fig2, ax2 = plt.subplots(2,2, figsize=(8,8), sharey=True, sharex=True)

        for i, (name, group) in enumerate(avg_df[avg_df.trial_type==trial].groupby(by=['context', 'lick_flag'])):
            for j, stim in enumerate(color_dict.keys()):
                ax.flat[i].plot(group.loc[group.stim_loc==stim, 'PC 1'], group.loc[group.stim_loc==stim, 'PC 2'], lw=1, c=lines[j], alpha=1, label=stim)
                ax1.flat[i].plot(group.loc[group.stim_loc==stim, 'PC 2'], group.loc[group.stim_loc==stim, 'PC 3'], lw=1, c=lines[j], alpha=1, label=stim)
                ax2.flat[i].plot(group.loc[group.stim_loc==stim, 'PC 1'], group.loc[group.stim_loc==stim, 'PC 3'], lw=1, c=lines[j], alpha=1, label=stim)

                if  group.loc[group.stim_loc==stim, 'PC 1'].shape[0]==0:
                    continue

                for t_step in [10, 20, 40, 60, 80]:
                    x= group.loc[(group.stim_loc==stim) & (group.time.mul(100)>group.time.mul(100).values[t_step]-1) & (group.time.mul(100)<group.time.mul(100).values[t_step]+1), 'PC 1'].values[0]
                    y= group.loc[(group.stim_loc==stim) & (group.time.mul(100)>group.time.mul(100).values[t_step]-1) & (group.time.mul(100)<group.time.mul(100).values[t_step]+1), 'PC 2'].values[0]
                    dx= group.loc[(group.stim_loc==stim) & (group.time.mul(100)>group.time.mul(100).values[t_step+1]-1) & (group.time.mul(100)<group.time.mul(100).values[t_step+1]+1), 'PC 1'].values[0] - x
                    dy= group.loc[(group.stim_loc==stim) & (group.time.mul(100)>group.time.mul(100).values[t_step+1]-1) & (group.time.mul(100)<group.time.mul(100).values[t_step+1]+1), 'PC 2'].values[0] - y

                    ax.flat[i].arrow(x, y, dx, dy, width=0.001, shape='full', color=lines[j], length_includes_head=True, head_width=.12)
                    
                    x= group.loc[(group.stim_loc==stim) & (group.time.mul(100)>group.time.mul(100).values[t_step]-1) & (group.time.mul(100)<group.time.mul(100).values[t_step]+1), 'PC 2'].values[0]
                    y= group.loc[(group.stim_loc==stim) & (group.time.mul(100)>group.time.mul(100).values[t_step]-1) & (group.time.mul(100)<group.time.mul(100).values[t_step]+1), 'PC 3'].values[0]
                    dx= group.loc[(group.stim_loc==stim) & (group.time.mul(100)>group.time.mul(100).values[t_step+1]-1) & (group.time.mul(100)<group.time.mul(100).values[t_step+1]+1), 'PC 2'].values[0] - x
                    dy= group.loc[(group.stim_loc==stim) & (group.time.mul(100)>group.time.mul(100).values[t_step+1]-1) & (group.time.mul(100)<group.time.mul(100).values[t_step+1]+1), 'PC 3'].values[0] - y

                    ax1.flat[i].arrow(x, y, dx, dy, width=0.001, shape='full', color=lines[j], length_includes_head=True, head_width=.12)

                    x= group.loc[(group.stim_loc==stim) & (group.time.mul(100)>group.time.mul(100).values[t_step]-1) & (group.time.mul(100)<group.time.mul(100).values[t_step]+1), 'PC 1'].values[0]
                    y= group.loc[(group.stim_loc==stim) & (group.time.mul(100)>group.time.mul(100).values[t_step]-1) & (group.time.mul(100)<group.time.mul(100).values[t_step]+1), 'PC 3'].values[0]
                    dx= group.loc[(group.stim_loc==stim) & (group.time.mul(100)>group.time.mul(100).values[t_step+1]-1) & (group.time.mul(100)<group.time.mul(100).values[t_step+1]+1), 'PC 1'].values[0] - x
                    dy= group.loc[(group.stim_loc==stim) & (group.time.mul(100)>group.time.mul(100).values[t_step+1]-1) & (group.time.mul(100)<group.time.mul(100).values[t_step+1]+1), 'PC 3'].values[0] - y

                    ax2.flat[i].arrow(x, y, dx, dy, width=0.001, shape='full', color=lines[j], length_includes_head=True, head_width=.12)
            
            ax.flat[i].set_xlabel('PC 1')
            ax.flat[i].set_ylabel('PC 2')
            ax.flat[i].set_title(f"{'Rewarded' if name[0]=='rewarded' else 'Non-rewarded'} {trial} {'lick' if name[1] ==1 else 'no-lick'}")

            ax1.flat[i].set_xlabel('PC 2')
            ax1.flat[i].set_ylabel('PC 3')
            ax1.flat[i].set_title(f"{'Rewarded' if name[0]=='rewarded' else 'Non-rewarded'} {trial} {'lick' if name[1] ==1 else 'no-lick'}")

            ax2.flat[i].set_xlabel('PC 1')
            ax2.flat[i].set_ylabel('PC 3')
            ax2.flat[i].set_title(f"{'Rewarded' if name[0]=='rewarded' else 'Non-rewarded'} {trial} {'lick' if name[1] ==1 else 'no-lick'}")

        fig.legend(handles, ["(-5.0, 5.0)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 1.5)", "(-0.5, 0.5)", "(2.5, 2.5)", "(0.5, 4.5)", "(1.5, 3.5)"], loc='outside upper right')
        fig.savefig(Path(result_path, f"{trial}_dimensionality_reduction_PC1vsPC2_lines.png"))
        fig.savefig(Path(result_path, f"{trial}_dimensionality_reduction_PC1vsPC2_lines.svg"))
        fig1.legend(handles, ["(-5.0, 5.0)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 1.5)", "(-0.5, 0.5)", "(2.5, 2.5)", "(0.5, 4.5)", "(1.5, 3.5)"], loc='outside upper right')
        fig1.savefig(Path(result_path, f"{trial}_dimensionality_reduction_PC2vsPC3_lines.png"))
        fig1.savefig(Path(result_path, f"{trial}_dimensionality_reduction_PC2vsPC3_lines.svg"))
        fig2.legend(handles, ["(-5.0, 5.0)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 1.5)", "(-0.5, 0.5)", "(2.5, 2.5)", "(0.5, 4.5)", "(1.5, 3.5)"], loc='outside upper right')
        fig2.savefig(Path(result_path, f"{trial}_dimensionality_reduction_PC1vsPC3_lines.png"))
        fig2.savefig(Path(result_path, f"{trial}_dimensionality_reduction_PC1vsPC3_lines.svg"))


def main(nwb_files, output_path):
    # combine_data(nwb_files, output_path)
    # plot_opto_effect_matrix(nwb_files, output_path)

    # output_path = os.path.join('//sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Pol_Bech', 'Pop_results',
    #                             'Context_behaviour', 'optogenetic_widefield_results')
    # output_path = haas_pathfun(output_path)
    dimensionality_reduction(nwb_files, output_path)


if __name__ == "__main__":

    for file in ['context_sessions_wf_opto_controls']:#'context_sessions_wf_opto', 
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
