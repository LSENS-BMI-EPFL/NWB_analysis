import itertools
import os
import sys
sys.path.append(os.getcwd())
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
    for tstamp in trials:
        frame = utils_misc.find_nearest(wf_timestamps, tstamp)
        data = nwb_read.get_widefield_dff0_traces(nwb_file, ['ophys', 'dff0'], int(frame + start), int(frame + stop))
        if data.shape != (len(np.arange(start, stop)), 125, 160):
            continue
        frames.append(np.nanmean(data, axis=0))

    data_frames = np.array(frames)
    data_frames = np.stack(data_frames, axis=0)
    return data_frames


def combine_data(nwb_files, output_path):
    coords_list = {'wS1': [[-1.5, 3.5],[-1.5, 4.5],[-2.5, 3.5],[-2.5, 4.5]], 'wM1': [[1.5, 1.5]], 'wM2': [[2.5,1.5]], 'RSC': [[-0.5, 0.5], [-1.5,0.5]],
                   'ALM': [[2.5, 2.5]], 'tjS1':[[0.5, 4.5]], 'tjM1':[[1.5, 3.5]], 'control': [[-5.0, 5.0]]} #AP, ML
    for nwb_file in nwb_files:
        session_df = []
        bhv_data = bhv_utils.build_standard_behavior_table([nwb_file])
        bhv_data = bhv_data.loc[bhv_data.early_lick==0]
        wf_timestamps = nwb_read.get_widefield_timestamps(nwb_file, ['ophys', 'dff0'])
        session_id = nwb_read.get_session_id(nwb_file)
        mouse_id = nwb_read.get_mouse_id(nwb_file)
        print("---------------- ")

        for loc in coords_list.keys():
            for coord in coords_list[loc]:

                opto_data = bhv_data.loc[(bhv_data.opto_grid_ap.mul(10).astype(int)==int(coord[0]*10)) & (bhv_data.opto_grid_ml.mul(10).astype(int)==int(coord[1]*10))]
                opto_data['opto_stim_loc'] = loc
                opto_data['opto_stim_coord'] = f"{coord[0]} - {coord[1]}"
                trials = opto_data.start_time

                if len(trials) == 0:
                    print(f'No trials in session {session_id}, {loc} coords {coord}... continuing')
                else:
                    print(f'{opto_data.shape[0]} trials in session {session_id}, {loc} coords {coord}')
                    wf_image = get_frames_by_epoch(nwb_file, trials, wf_timestamps, start=0, stop=250) #250 frames since frame rate in this sessions is 50Hz and trial duration is 5s
                    opto_data['wf_image_shape'] = [wf_image[i].shape for i in range(opto_data.shape[0])]           
                    opto_data['wf_images'] = [wf_image[i].flatten(order='C') for i in range(opto_data.shape[0])]
                    roi_data = get_dff0_traces_by_epoch(nwb_file, trials, wf_timestamps, start=0, stop=250)
                    roi_data['trial_id'] = opto_data.trial_id.values
                    opto_data = pd.merge(opto_data.reset_index(drop=True), roi_data, on='trial_id')
                    print(f"Final shape after merging: {opto_data.shape[0]}")
                    session_df += [opto_data]

        session_df = pd.concat(session_df, ignore_index=True)
        if not os.path.exists(Path(output_path, session_id)):
            os.makedirs(Path(output_path, session_id))
        session_df.to_parquet(Path(output_path, session_id, 'results.parquet.gzip', compression='gzip'))


def plot_opto_wf_psth(nwb_files, output_path):
    # coords_list = {'wS1': [[-1.5, 3.5],[-1.5, 4.5],[-2.5, 3.5],[-2.5, 4.5]], 'wM1': [[1.5, 1.5]], 'wM2': [[2.5,1.5]], 'RSC': [[-0.5, 0.5], [-1.5,0.5]],
    #                'ALM': [[2.5, 2.5]], 'tjS1':[[0.5, 4.5]], 'tjM1':[[1.5, 3.5]], 'control': [[-5.0, 5.0]]}
    if not os.path.exists(Path(output_path, 'PSTHs')):
        os.makedirs(Path(output_path, 'PSTHs'))

    total_df = []
    coords_list = {'wS1': [[-1.5, 3.5]], 'wM1': [[1.5, 1.5]], 'wM2': [[2.5,1.5]], 'RSC': [[-0.5, 0.5], [-1.5,0.5]],
                   'ALM': [[2.5, 2.5]], 'tjS1':[[0.5, 4.5]], 'tjM1':[[1.5, 3.5]], 'control': [[-5.0, 5.0]]}
    
    for nwb_file in nwb_files:
        session_id = nwb_read.get_session_id(nwb_file)
        mouse_id = nwb_read.get_mouse_id(nwb_file)
        total_df += [pd.read_parquet(Path(output_path, session_id, 'results.parquet.gzip', compression='gzip'))]

    total_df = pd.concat(total_df, ignore_index=True)
    total_df['time'] = [[np.linspace(-1,3.98,250)] for i in range(total_df.shape[0])]
    total_df['legend'] = total_df.apply(lambda x: f"{x.opto_stim_loc} - {'lick' if x.lick_flag==1 else 'no lick'}",axis=1)
    total_df = total_df[~total_df.opto_stim_coord.isin(['-1.5 - 4.5', '-2.5 - 3.5', '-2.5 - 4.5'])].reset_index(drop=True)
    wf_shape = total_df.loc[0, 'wf_image_shape']
    mouse_df = total_df.groupby(by=['mouse_id', 'context', 'trial_type', 'opto_stim_loc', 'lick_flag']).agg(
        stim_loc=('opto_stim_loc', lambda x: x.unique()[0]),
        coord=('opto_stim_coord', lambda x: x.unique()[0]),
        legend=('legend', lambda x: x.unique()[0]),
        wf_images=('wf_images', lambda x: np.nanmean(np.stack(x), axis=0)),
        A1=('A1', lambda x: np.nanmean(np.stack(x), axis=0)),
        ALM=('ALM', lambda x: np.nanmean(np.stack(x), axis=0)),
        tjM1=('tjM1', lambda x: np.nanmean(np.stack(x), axis=0)),
        tjS1=('tjS1', lambda x: np.nanmean(np.stack(x), axis=0)),
        RSC=('RSC', lambda x: np.nanmean(np.stack(x), axis=0)),
        wM1=('wM1', lambda x: np.nanmean(np.stack(x), axis=0)),
        wM2=('wM2', lambda x: np.nanmean(np.stack(x), axis=0)),
        wS1=('wS1', lambda x: np.nanmean(np.stack(x), axis=0)),
        wS2=('wS2', lambda x: np.nanmean(np.stack(x), axis=0)),
        time=('time', lambda x: list(x)[0][0])
    ).reset_index()

    for loc in mouse_df.stim_loc.unique():
        if loc=='control':
            continue
        subset = mouse_df.loc[mouse_df.stim_loc.isin([loc, 'control'])]
        for name, group in subset.groupby(by=['context', 'trial_type']):

            group = group.melt(id_vars=['context', 'trial_type', 'stim_loc', 'coord', 'lick_flag', 'legend', 'time'], 
                            value_vars=['A1', 'ALM', 'tjM1', 'tjS1', 'RSC', 'wM1', 'wM2','wS1', 'wS2'], 
                            var_name='roi', 
                            value_name='dff0').explode(['time', 'dff0'])

            fig,ax =plt.subplots(3,3, figsize=(9,9), sharey=True, sharex=True)
            fig.suptitle(f"{loc}-stim_{'rewarded' if name[0]==1 else 'non_rewarded'}_{name[1]}")

            for i, roi in enumerate(['A1', 'ALM', 'tjM1', 'tjS1', 'RSC', 'wM1', 'wM2','wS1', 'wS2']):
                
                ax.flat[i] = sns.lineplot(data=group[group.roi.isin([roi])], x='time', y='dff0', hue='legend', 
                                          hue_order=[f"{loc} - lick", f"{loc} - no lick", f"control - lick", f"control - no lick"], 
                                          palette=['#005F60', '#FD5901', '#249EA0', '#FAAB36'], ax=ax.flat[i])
                ax.flat[i].set_xlim([-0.1,0.5])
                ax.flat[i].set_ylim([-0.04, 0.07])
                ax.flat[i].set_title(roi)
                ax.flat[i].vlines(0, -0.04, 0.07, 'grey', 'dashed')
                ax.flat[i].spines[['top', 'right']].set_visible(False)
                ax.flat[i].get_legend().remove()

            handles, labels = ax.flat[i].get_legend_handles_labels()
            fig.legend(handles, labels, loc='outside upper right')
            fig.savefig(Path(output_path, 'PSTHs', f"{loc}-stim_{'rewarded' if name[0]==1 else 'non_rewarded'}_{name[1]}.png"))
            fig.savefig(Path(output_path, 'PSTHs', f"{loc}-stim_{'rewarded' if name[0]==1 else 'non_rewarded'}_{name[1]}.svg"))


def dimensionality_reduction(nwb_files, output_path):
    result_path = Path(output_path, 'PCA')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    total_df = []
    coords_list = {'wS1': [[-1.5, 3.5]], 'wM1': [[1.5, 1.5]], 'wM2': [[2.5,1.5]], 'RSC': [[-0.5, 0.5], [-1.5,0.5]],
                   'ALM': [[2.5, 2.5]], 'tjS1':[[0.5, 4.5]], 'tjM1':[[1.5, 3.5]], 'control': [[-5.0, 5.0]]}
    
    for nwb_file in nwb_files:
        session_id = nwb_read.get_session_id(nwb_file)
        mouse_id = nwb_read.get_mouse_id(nwb_file)
        total_df += [pd.read_parquet(Path(output_path, session_id, 'results.parquet.gzip', compression='gzip'))]

    total_df = pd.concat(total_df, ignore_index=True)
    total_df['time'] = [[np.linspace(-1,3.98,250)] for i in range(total_df.shape[0])]
    total_df['legend'] = total_df.apply(lambda x: f"{x.opto_stim_loc} - {'lick' if x.lick_flag==1 else 'no lick'}",axis=1)
    total_df = total_df[~total_df.opto_stim_coord.isin(['-1.5 - 4.5', '-2.5 - 3.5', '-2.5 - 4.5'])].reset_index(drop=True)
    wf_shape = total_df.loc[0, 'wf_image_shape']
    mouse_df = total_df.groupby(by=['mouse_id', 'context', 'trial_type', 'opto_stim_loc', 'lick_flag']).agg(
        stim_loc=('opto_stim_loc', lambda x: x.unique()[0]),
        coord=('opto_stim_coord', lambda x: x.unique()[0]),
        legend=('legend', lambda x: x.unique()[0]),
        wf_images=('wf_images', lambda x: np.nanmean(np.stack(x), axis=0)),
        A1=('A1', lambda x: np.nanmean(np.stack(x), axis=0)),
        ALM=('ALM', lambda x: np.nanmean(np.stack(x), axis=0)),
        tjM1=('tjM1', lambda x: np.nanmean(np.stack(x), axis=0)),
        tjS1=('tjS1', lambda x: np.nanmean(np.stack(x), axis=0)),
        RSC=('RSC', lambda x: np.nanmean(np.stack(x), axis=0)),
        wM1=('wM1', lambda x: np.nanmean(np.stack(x), axis=0)),
        wM2=('wM2', lambda x: np.nanmean(np.stack(x), axis=0)),
        wS1=('wS1', lambda x: np.nanmean(np.stack(x), axis=0)),
        wS2=('wS2', lambda x: np.nanmean(np.stack(x), axis=0)),
        time=('time', lambda x: list(x)[0][0])
    ).reset_index()

    avg_df = mouse_df.groupby(by=['context', 'trial_type', 'stim_loc', 'lick_flag']).agg(
        coord=('coord', lambda x: x.unique()[0]),
        legend=('legend', lambda x: x.unique()[0]),
        wf_images=('wf_images', lambda x: np.nanmean(np.stack(x), axis=0)),
        A1=('A1', lambda x: np.nanmean(np.stack(x), axis=0)),
        ALM=('ALM', lambda x: np.nanmean(np.stack(x), axis=0)),
        tjM1=('tjM1', lambda x: np.nanmean(np.stack(x), axis=0)),
        tjS1=('tjS1', lambda x: np.nanmean(np.stack(x), axis=0)),
        RSC=('RSC', lambda x: np.nanmean(np.stack(x), axis=0)),
        wM1=('wM1', lambda x: np.nanmean(np.stack(x), axis=0)),
        wM2=('wM2', lambda x: np.nanmean(np.stack(x), axis=0)),
        wS1=('wS1', lambda x: np.nanmean(np.stack(x), axis=0)),
        wS2=('wS2', lambda x: np.nanmean(np.stack(x), axis=0)),
        time=('time', lambda x: list(x)[0])
    ).reset_index()

    avg_df = avg_df.melt(id_vars=['context', 'trial_type', 'stim_loc', 'coord', 'lick_flag', 'legend', 'time'],
                                  value_vars=['A1', 'ALM', 'tjM1', 'tjS1', 'RSC', 'wM1', 'wM2','wS1', 'wS2'],
                                  var_name='roi',
                                  value_name='dff0').explode(['time', 'dff0']).reset_index()
    avg_df.time = avg_df.time.astype('float64')
    avg_df = avg_df[avg_df.time.isin(avg_df.time.unique()[40:100])]
    avg_df = avg_df[avg_df.trial_type=='whisker_trial'].pivot(index=['context', 'legend', 'time'], columns='roi', values='dff0')
    avg_data_for_pca = avg_df.to_numpy()
    scaler = StandardScaler()
    avg_data_for_pca = scaler.fit_transform(avg_data_for_pca)

    pca = PCA(n_components=9)
    results = pca.fit(avg_data_for_pca)
    principal_components = pca.transform(avg_data_for_pca)

    pc_df = pd.DataFrame(data=principal_components, index=avg_df.index)
    pc_df.columns = [f"PC {i+1}" for i in range(0, principal_components.shape[1])]

    avg_df = avg_df.join(pc_df).reset_index()

    avg_df['lick_flag'] = avg_df.apply(lambda x: 0 if 'no lick' in x.legend else 1, axis=1)
    avg_df['stim_loc'] = avg_df.apply(lambda x: x.legend.split(" ")[0], axis=1)

    ## Plot biplots to see which variables (i.e. rois) contain most of the explained variance (?)
    labels = avg_df.stim_loc.unique()
    coeff = np.transpose(results.components_)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle("PCA1-2 Biplot")
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    fig1.suptitle("PCA2-3 Biplot")
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    fig2.suptitle("PCA1-3 Biplot")
    for j, roi in enumerate(['A1', 'ALM', 'tjM1', 'tjS1', 'RSC', 'wM1', 'wM2','wS1', 'wS2']):
        ax.arrow(x=0, y=0, dx=coeff[j, 0], dy=coeff[j, 1], color="#000000", width=0.003, head_width=0.03)
        ax.text(x=coeff[j, 0] * 1.15, y=coeff[j, 1] * 1.15, s=roi, size=13, color="#000000", ha="center", va="center")
        ax1.arrow(x=0, y=0, dx=coeff[j, 1], dy=coeff[j, 2], color="#000000", width=0.003, head_width=0.03)
        ax1.text(x=coeff[j, 1] * 1.15, y=coeff[j, 2] * 1.15, s=roi, size=13, color="#000000", ha="center", va="center")
        ax2.arrow(x=0, y=0, dx=coeff[j, 0], dy=coeff[j, 2], color="#000000", width=0.003, head_width=0.03)
        ax2.text(x=coeff[j, 0] * 1.15, y=coeff[j, 2] * 1.15, s=roi, size=13, color="#000000", ha="center", va="center")

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

    fig, ax = plt.subplots(3, 4, figsize=(8,6))
    fig.suptitle("PC timecourses")
    long_df = avg_df.melt(id_vars=['context', 'stim_loc', 'lick_flag', 'legend', 'time'], 
                            value_vars=['PC 1', 'PC 2', 'PC 3'], 
                            var_name='PC', 
                            value_name='data').explode(['time', 'data'])
    long_df['context_legend'] = long_df.apply(lambda x: f"{'reward' if x.context==1 else 'no-reward'} - {'lick' if x.lick_flag==1 else 'no lick'}", axis=1)
    g = sns.relplot(data=long_df, x='time', y='data', hue='context_legend', 
                    hue_order=['reward - lick', 'reward - no lick', 'no-reward - lick', 'no-reward - no lick'], 
                    palette=['#348A18', '#83E464', '#6E188A', '#C564E4'], 
                    col='stim_loc', row='PC', kind='line', linewidth=2)
    g.figure.savefig(Path(result_path, 'PC_timecourses_by_region.png'))
    g.figure.savefig(Path(result_path, 'PC_timecourses_by_region.svg'))

    g = sns.relplot(data=long_df, x='time', y='data', hue='stim_loc', 
                    hue_order=['control', 'wS1', 'wM1', 'wM2', 'RSC', 'ALM', 'tjM1', 'tjS1'], 
                    palette=['#000000', '#00ffff', '#2ad5ff', '#55aaff', '#807fff', '#aa55ff', '#d52aff', '#ff00ff'], 
                    col='context_legend', row='PC', kind='line', linewidth=2)
    g.figure.savefig(Path(result_path, 'PC_timecourses_by_trial_outcome.png'))
    g.figure.savefig(Path(result_path, 'PC_timecourses_by_trial_outcome.svg'))

    ## Plot projected time courses onto PCx vs PCy
    color_dict = {'control': LinearSegmentedColormap.from_list('', ['#FFFFFF', '#000000'], N=avg_df.time.unique().shape[0]),
                  'wS1': LinearSegmentedColormap.from_list('', ['#FFFFFF', '#00ffff'], N=avg_df.time.unique().shape[0]),
                  'wM1': LinearSegmentedColormap.from_list('', ['#FFFFFF', '#2ad5ff'], N=avg_df.time.unique().shape[0]),
                  'wM2': LinearSegmentedColormap.from_list('', ['#FFFFFF', '#55aaff'], N=avg_df.time.unique().shape[0]),
                  'RSC': LinearSegmentedColormap.from_list('', ['#FFFFFF', '#807fff'], N=avg_df.time.unique().shape[0]),
                  'ALM': LinearSegmentedColormap.from_list('', ['#FFFFFF', '#aa55ff'], N=avg_df.time.unique().shape[0]),
                  'tjM1': LinearSegmentedColormap.from_list('', ['#FFFFFF', '#d52aff'], N=avg_df.time.unique().shape[0]),
                  'tjS1': LinearSegmentedColormap.from_list('', ['#FFFFFF', '#ff00ff'], N=avg_df.time.unique().shape[0])}
    
    lines = ['#000000', '#00ffff', '#2ad5ff', '#55aaff', '#807fff', '#aa55ff', '#d52aff', '#ff00ff']
    handles = [Line2D([0], [0], color=c, lw=4) for c in lines]

    fig, ax= plt.subplots(2,2, figsize=(8,8), sharey=True, sharex=True)
    fig1, ax1 = plt.subplots(2,2, figsize=(8,8), sharey=True, sharex=True)
    fig2, ax2 = plt.subplots(2,2, figsize=(8,8), sharey=True, sharex=True)

    for i, (name, group) in enumerate(avg_df.groupby(by=['context', 'lick_flag'])):
        for stim in group.stim_loc.unique():
            ax.flat[i].scatter(group.loc[group.stim_loc==stim, 'PC 1'], group.loc[group.stim_loc==stim, 'PC 2'], c=group.loc[group.stim_loc==stim, 'time'], cmap=color_dict[stim], label=stim, s=10)
            ax.flat[i].scatter(group.loc[(group.stim_loc==stim) & (group.time==0), 'PC 1'], group.loc[(group.stim_loc==stim) & (group.time==0), 'PC 2'], s=10, facecolors='none', edgecolors='r')
            ax1.flat[i].scatter(group.loc[group.stim_loc==stim, 'PC 2'], group.loc[group.stim_loc==stim, 'PC 3'], c=group.loc[group.stim_loc==stim, 'time'], cmap=color_dict[stim], label=stim, s=10)
            ax1.flat[i].scatter(group.loc[(group.stim_loc==stim) & (group.time==0), 'PC 2'], group.loc[(group.stim_loc==stim) & (group.time==0), 'PC 3'], s=10, facecolors='none', edgecolors='r')
            ax2.flat[i].scatter(group.loc[group.stim_loc==stim, 'PC 1'], group.loc[group.stim_loc==stim, 'PC 3'], c=group.loc[group.stim_loc==stim, 'time'], cmap=color_dict[stim], label=stim, s=10)
            ax2.flat[i].scatter(group.loc[(group.stim_loc==stim) & (group.time==0), 'PC 1'], group.loc[(group.stim_loc==stim) & (group.time==0), 'PC 3'], s=10, facecolors='none', edgecolors='r')

        ax.flat[i].set_xlabel('PC 1')
        ax.flat[i].set_ylabel('PC 2')
        ax.flat[i].set_title(f"{'Rewarded' if name[0]==1 else 'Non-rewarded'} whisker trial {'lick' if name[1] ==1 else 'no-lick'}")

        ax1.flat[i].set_xlabel('PC 2')
        ax1.flat[i].set_ylabel('PC 3')
        ax1.flat[i].set_title(f"{'Rewarded' if name[0]==1 else 'Non-rewarded'} whisker trial {'lick' if name[1] ==1 else 'no-lick'}")

        ax2.flat[i].set_xlabel('PC 1')
        ax2.flat[i].set_ylabel('PC 3')
        ax2.flat[i].set_title(f"{'Rewarded' if name[0]==1 else 'Non-rewarded'} whisker trial {'lick' if name[1] ==1 else 'no-lick'}")

    fig.legend(handles, ['control', 'wS1', 'wM1', 'wM2', 'RSC', 'ALM', 'tjM1', 'tjS1'], loc='outside upper right')
    fig.savefig(Path(result_path, f"dimensionality_reduction_PC1vsPC2.png"))
    fig.savefig(Path(result_path, f"dimensionality_reduction_PC1vsPC2.svg"))
    fig1.legend(handles, ['control', 'wS1', 'wM1', 'wM2', 'RSC', 'ALM', 'tjM1', 'tjS1'], loc='outside upper right')
    fig1.savefig(Path(result_path, f"dimensionality_reduction_PC2vsPC3.png"))
    fig1.savefig(Path(result_path, f"dimensionality_reduction_PC2vsPC3.svg"))

    fig2.legend(handles, ['control', 'wS1', 'wM1', 'wM2', 'RSC', 'ALM', 'tjM1', 'tjS1'], loc='outside upper right')
    fig2.savefig(Path(result_path, f"dimensionality_reduction_PC1vsPC3.png"))
    fig2.savefig(Path(result_path, f"dimensionality_reduction_PC1vsPC3.svg"))

    ## PCA projetions with arrowplots
    fig, ax= plt.subplots(2,2, figsize=(8,8), sharey=True, sharex=True)
    fig1, ax1 = plt.subplots(2,2, figsize=(8,8), sharey=True, sharex=True)
    fig2, ax2 = plt.subplots(2,2, figsize=(8,8), sharey=True, sharex=True)

    for i, (name, group) in enumerate(avg_df.groupby(by=['context', 'lick_flag'])):
        for j, stim in enumerate(['control', 'wS1', 'wM1', 'wM2', 'RSC', 'ALM', 'tjM1', 'tjS1']):
            ax.flat[i].plot(group.loc[group.stim_loc==stim, 'PC 1'], group.loc[group.stim_loc==stim, 'PC 2'], lw=1, c=lines[j], alpha=1, label=stim)
            ax1.flat[i].plot(group.loc[group.stim_loc==stim, 'PC 2'], group.loc[group.stim_loc==stim, 'PC 3'], lw=1, c=lines[j], alpha=1, label=stim)
            ax2.flat[i].plot(group.loc[group.stim_loc==stim, 'PC 1'], group.loc[group.stim_loc==stim, 'PC 3'], lw=1, c=lines[j], alpha=1, label=stim)

            if name==(0,1) and stim=='ALM':
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
        ax.flat[i].set_title(f"{'Rewarded' if name[0]==1 else 'Non-rewarded'} whisker trial {'lick' if name[1] ==1 else 'no-lick'}")

        ax1.flat[i].set_xlabel('PC 2')
        ax1.flat[i].set_ylabel('PC 3')
        ax1.flat[i].set_title(f"{'Rewarded' if name[0]==1 else 'Non-rewarded'} whisker trial {'lick' if name[1] ==1 else 'no-lick'}")

        ax2.flat[i].set_xlabel('PC 1')
        ax2.flat[i].set_ylabel('PC 3')
        ax2.flat[i].set_title(f"{'Rewarded' if name[0]==1 else 'Non-rewarded'} whisker trial {'lick' if name[1] ==1 else 'no-lick'}")

    fig.legend(handles, ['control', 'wS1', 'wM1', 'wM2', 'RSC', 'ALM', 'tjM1', 'tjS1'], loc='outside upper right')
    fig.savefig(Path(result_path, f"dimensionality_reduction_PC1vsPC2_lines.png"))
    fig.savefig(Path(result_path, f"dimensionality_reduction_PC1vsPC2_lines.svg"))
    fig1.legend(handles, ['control', 'wS1', 'wM1', 'wM2', 'RSC', 'ALM', 'tjM1', 'tjS1'], loc='outside upper right')
    fig1.savefig(Path(result_path, f"dimensionality_reduction_PC2vsPC3_lines.png"))
    fig1.savefig(Path(result_path, f"dimensionality_reduction_PC2vsPC3_lines.svg"))
    fig2.legend(handles, ['control', 'wS1', 'wM1', 'wM2', 'RSC', 'ALM', 'tjM1', 'tjS1'], loc='outside upper right')
    fig2.savefig(Path(result_path, f"dimensionality_reduction_PC1vsPC3_lines.png"))
    fig2.savefig(Path(result_path, f"dimensionality_reduction_PC1vsPC3_lines.svg"))


def main(nwb_files, output_path):
    combine_data(nwb_files, output_path)
    plot_opto_wf_psth(nwb_files, output_path)
    dimensionality_reduction(nwb_files, output_path)


if __name__ == "__main__":

    for file in [ 'context_sessions_wf_opto', 'context_sessions_wf_opto_controls']:
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
