import itertools
import itertools
import os
import re
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

import nwb_wrappers.nwb_reader_functions as nwb_read
import nwb_utils.utils_behavior as bhv_utils
from nwb_utils import utils_misc
from nwb_utils.utils_plotting import lighten_color, remove_top_right_frame

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics.pairwise import paired_distances

from utils.wf_plotting_utils import reduce_im_dimensions, plot_grid_on_allen, generate_reduced_image_df
from utils.dlc_utils import *
from utils.widefield_utils import *
from utils.haas_utils import *
import utils.behaviour_plot_utils as plot_utils


# def load_opto_data(group):
#     opto_results = fr'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/optogenetic_results/{group}'
#     opto_results = haas_pathfun(opto_results)
#     single_mouse_result_files = glob.glob(os.path.join(opto_results, "*", "opto_data.json"))
#     opto_df = []
#     for file in single_mouse_result_files:
#         d= pd.read_json(file)
#         d['mouse_name'] = [file.split("/")[-2] for i in range(d.shape[0])]
#         opto_df += [d]
#     opto_df = pd.concat(opto_df)
#     opto_df = opto_df.loc[opto_df.opto_grid_ap!=3.5]
#     opto_avg_df = opto_df.groupby(by=['context', 'trial_type', 'opto_grid_ml', 'opto_grid_ap']).agg(
#                                                                                  data=('data_mean', list),
#                                                                                  data_sub=('data_mean_sub', list),
#                                                                                  data_mean=('data_mean', 'mean'),
#                                                                                  data_mean_sub=(
#                                                                                  'data_mean_sub', 'mean'),
#                                                                                  shuffle_dist=('shuffle_dist', 'sum'),
#                                                                                  shuffle_dist_sub=(
#                                                                                  'shuffle_dist_sub', 'sum'),
#                                                                                  percentile_avg=('percentile', 'mean'),
#                                                                                  percentile_avg_sub=(
#                                                                                  'percentile_sub', 'mean'),
#                                                                                  n_sigma_avg=('n_sigma', 'mean'),
#                                                                                  n_sigma_avg_sub=(
#                                                                                  'n_sigma_sub', 'mean'))
#     opto_avg_df['shuffle_mean'] = opto_avg_df.apply(lambda x: np.mean(x.shuffle_dist), axis=1)
#     opto_avg_df['shuffle_std'] = opto_avg_df.apply(lambda x: np.std(x.shuffle_dist), axis=1)
#     opto_avg_df['shuffle_mean_sub'] = opto_avg_df.apply(lambda x: np.mean(x.shuffle_dist_sub), axis=1)
#     opto_avg_df['shuffle_std_sub'] = opto_avg_df.apply(lambda x: np.std(x.shuffle_dist_sub), axis=1)

#     opto_avg_df = opto_avg_df.reset_index()
#     opto_avg_df['opto_stim_coord'] = opto_avg_df.apply(lambda x: tuple([x.opto_grid_ap, x.opto_grid_ml]), axis=1)
#     return opto_avg_df


def combine_data(nwb_files, output_path):

    for nwb_file in nwb_files:
        session_df = []
        bhv_data = bhv_utils.build_standard_behavior_table([nwb_file])
        if bhv_data.trial_id.duplicated().sum()>0:
            bhv_data['trial_id'] = bhv_data.index.values

        bhv_data = bhv_data.loc[(bhv_data.early_lick==0) & (bhv_data.opto_grid_ap!=3.5)]
        bhv_data['opto_stim_coord'] = bhv_data.apply(lambda x: f"({x.opto_grid_ap}, {x.opto_grid_ml})",axis=1)
        wf_timestamps = nwb_read.get_widefield_timestamps(nwb_file, ['ophys', 'dff0'])
        dlc_timestamps = nwb_read.get_dlc_timestamps(nwb_file, ['behavior', 'BehavioralTimeSeries'])
        if dlc_timestamps is None:
            dlc_timestamps = [[], []]

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

            side_dlc = get_dlc_data(nwb_file, trials, dlc_timestamps, view='side', start=0, stop=250)
            top_dlc = get_dlc_data(nwb_file, trials, dlc_timestamps, view='top', start=0, stop=250)
            side_dlc['trial_id'] = opto_data.trial_id.values
            top_dlc['trial_id'] = opto_data.trial_id.values

            wf_image = get_reduced_im_by_epoch(nwb_file, trials, wf_timestamps, start=0, stop=250)
            wf_image['trial_id'] = opto_data.trial_id.values

            opto_data = pd.merge(opto_data.reset_index(drop=True), side_dlc, on='trial_id')
            opto_data = pd.merge(opto_data.reset_index(drop=True), top_dlc, on='trial_id')
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


def coding_direction_sensory_vs_lick(df, result_path):
    # roi_list = ['(-0.5, 0.5)', '(-1.5, 0.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(0.5, 4.5)', '(1.5, 1.5)', '(1.5, 3.5)', '(2.5, 2.5)']

    roi_list = ['(-0.5, 0.5)', '(-0.5, 1.5)', '(-0.5, 2.5)', '(-0.5, 3.5)', '(-0.5, 4.5)', '(-0.5, 5.5)', '(-1.5, 0.5)',
       '(-1.5, 1.5)', '(-1.5, 2.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(-1.5, 5.5)', '(-2.5, 0.5)', '(-2.5, 1.5)', '(-2.5, 2.5)',
       '(-2.5, 3.5)', '(-2.5, 4.5)', '(-2.5, 5.5)', '(-3.5, 0.5)', '(-3.5, 1.5)', '(-3.5, 2.5)', '(-3.5, 3.5)', '(-3.5, 4.5)',
       '(-3.5, 5.5)', '(0.5, 0.5)', '(0.5, 1.5)', '(0.5, 2.5)', '(0.5, 3.5)', '(0.5, 4.5)', '(0.5, 5.5)', '(1.5, 0.5)', 
       '(1.5, 1.5)', '(1.5, 2.5)', '(1.5, 3.5)', '(1.5, 4.5)', '(1.5, 5.5)', '(2.5, 0.5)', '(2.5, 1.5)', '(2.5, 2.5)', '(2.5, 3.5)', '(2.5, 4.5)']
    
    time_array = np.asarray(df.loc[0, 'time'][0])
    lick_times =  df.lick_time.to_numpy() - (df.start_time+1).to_numpy()
    lick_start_frames = [utils_misc.find_nearest(time_array, lick) if ~np.isnan(lick) else np.nan for lick in lick_times]
    for roi in roi_list:
        df[f'{roi}_lick'] = df.apply(lambda x: np.roll(x[roi], 50-lick_start_frames[x.name]) if ~np.isnan(lick_start_frames[x.name]) else np.ones_like(x[roi])*np.nan, axis=1)
        # df[f'{roi}_lick'] = df.apply(lambda x: x[f"{roi}_lick"] - np.nanmean(x[f"{roi}_lick"][40:50]), axis=1)

    d = {c: lambda x: x.unique()[0] for c in ['legend']}
    d['time'] = lambda x: list(x)[0][0]
    for c in roi_list:
        d[f"{c}"]= lambda x: np.nanmean(np.stack(x), axis=0)
        d[f"{c}_lick"]= lambda x: np.nanmean(np.stack(x), axis=0)
    
    total_roi_list = roi_list
    total_roi_list += [f"{roi}_lick" for roi in roi_list]

    mouse_df = df.loc[df.opto_stim_coord=='(-5.0, 5.0)'].groupby(by=['mouse_id', 'context', 'trial_type', 'lick_flag']).agg(d).reset_index()
    mouse_df = mouse_df.melt(id_vars=['mouse_id', 'context', 'trial_type', 'legend', 'lick_flag', 'time'],
                                 value_vars=total_roi_list,
                                 var_name='roi',
                                 value_name='dff0').explode(['time', 'dff0'])
    control_df = mouse_df.groupby(by=['context', 'trial_type', 'lick_flag', 'legend', 'roi', 'time']).agg(lambda x: np.nanmean(x)).reset_index()
   
    control_df.time = control_df.time.round(2)
    control_df = control_df[(control_df.time>=-0.20)&(control_df.time<=0.20)]
    control_df = control_df.loc[control_df.trial_type!='auditory_trial']

    mouse_df = df.groupby(by=['mouse_id', 'context', 'trial_type', 'opto_stim_coord']).agg(d).reset_index()
    mouse_df = mouse_df.melt(id_vars=['mouse_id', 'context', 'trial_type', 'opto_stim_coord', 'time'],
                                 value_vars=roi_list,
                                 var_name='roi',
                                 value_name='dff0').explode(['time', 'dff0'])
    stim_df = mouse_df.groupby(by=['context', 'trial_type', 'opto_stim_coord', 'roi', 'time']).agg(lambda x: np.nanmean(x)).reset_index()
    stim_df = stim_df[(stim_df.time>=-0.20)&(stim_df.time<=0.20)]
    stim_df = stim_df.loc[stim_df.trial_type!='auditory_trial']
    stim_df = stim_df[~stim_df.roi.str.contains('lick')]

    whisker_lick = control_df.loc[(~control_df.roi.str.contains('lick')) &
                             (control_df.trial_type=='whisker_trial') & 
                             (control_df.lick_flag==1)].groupby(by=['roi', 'time'], as_index=False)['dff0'].mean().pivot(index='roi', columns='time').to_numpy()
    whisker_miss = control_df.loc[(~control_df.roi.str.contains('lick')) &
                             (control_df.trial_type=='whisker_trial') & 
                             (control_df.lick_flag==0)].groupby(by=['roi', 'time'], as_index=False)['dff0'].mean().pivot(index='roi', columns='time').to_numpy()
    catch_lick = control_df.loc[(control_df.roi.str.contains('lick')) &
                             (control_df.trial_type=='no_stim_trial') & 
                             (control_df.lick_flag==1)].groupby(by=['roi', 'time'], as_index=False)['dff0'].mean().pivot(index='roi', columns='time').to_numpy()
    catch_miss = control_df.loc[(~control_df.roi.str.contains('lick')) &
                            (control_df.trial_type=='no_stim_trial') & 
                            (control_df.lick_flag==0)].groupby(by=['roi', 'time'], as_index=False)['dff0'].mean().pivot(index='roi', columns='time').to_numpy()
    order = control_df.loc[(~control_df.roi.str.contains('lick')) &
                            (control_df.trial_type=='no_stim_trial') & 
                            (control_df.lick_flag==0)].groupby(by=['roi', 'time'], as_index=False)['dff0'].mean().pivot(index='roi', columns='time').index.to_list()
    
    cd_sensory = whisker_miss - catch_miss
    # cd_lick = catch_lick - catch_miss
    cd_lick = catch_lick #- catch_miss
    
    cd_sensory_im_df = generate_reduced_image_df(np.nanmean(cd_sensory[:, 11:15], axis=1)[np.newaxis, ...], [eval(roi) for roi in order])
    fig, ax=plt.subplots(1,2, figsize=(8,4))
    plot_grid_on_allen(cd_sensory_im_df, outcome='dff0', palette='icefire', dotsize=440, result_path=None, vmin=-0.01, vmax=0.01, norm='two-slope', fig=fig, ax=ax[0])
    ax[0].set_title('Sensory CD')
    ax[0].xaxis.set_visible(False)
    ax[0].yaxis.set_visible(False)    

    cd_lick_im_df = generate_reduced_image_df(np.nanmean(cd_lick[:, 11:15], axis=1)[np.newaxis, ...], [eval(roi) for roi in order])
    plot_grid_on_allen(cd_lick_im_df, outcome='dff0', palette='icefire', dotsize=440, result_path=None, vmin=-0.02, vmax=0.02, norm='two-slope', fig=fig, ax=ax[1])
    ax[1].set_title('Lick CD')
    ax[1].xaxis.set_visible(False)
    ax[1].yaxis.set_visible(False)
    fig.savefig(os.path.join(result_path, 'coding_direction_images.png'))

    scale = 0.007
    for stim in ['(-0.5, 0.5)', '(-1.5, 0.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(0.5, 4.5)', '(1.5, 1.5)', '(1.5, 3.5)', '(2.5, 2.5)']:
        fig, ax = plt.subplots(2,3, figsize=(8,4))
        for j, (name, group) in enumerate(control_df.groupby('context')):
            if name=='rewarded':
                wh_color = '#348A18'
            else:
                wh_color = '#6E188A'

            for lick in group.lick_flag.unique():
                whisker = group.loc[(~group.roi.str.contains('lick')) &
                                        (group.trial_type=='whisker_trial') & 
                                        (group.lick_flag==lick)].groupby(by=['roi', 'time'], as_index=False)['dff0'].mean().pivot(index='roi', columns='time').to_numpy()
                catch = group.loc[(~group.roi.str.contains('lick')) &
                                        (group.trial_type=='no_stim_trial') & 
                                        (group.lick_flag==lick)].groupby(by=['roi', 'time'], as_index=False)['dff0'].mean().pivot(index='roi', columns='time').to_numpy()

                projected_sensory=[]
                projected_lick=[]
                for i in range(whisker.shape[1]):
                    projected_sensory.append(np.dot(whisker[:, i], np.nanmean(cd_sensory[:, 11:15], axis=1)))
                    projected_lick.append(np.dot(whisker[:, i], np.nanmean(cd_lick[:, 11:15], axis=1)))

                ax[j, 0].plot(np.asarray(projected_sensory), c=wh_color, alpha=1 if lick else 0.3)
                ax[j, 1].plot(np.asarray(projected_lick), c=wh_color, alpha=1 if lick else 0.3)
                ax[j, 2].plot(np.asarray(projected_lick), np.asarray(projected_sensory), c=wh_color, alpha=1 if lick else 0.3)

                projected_sensory=[]
                projected_lick=[]
                for i in range(whisker.shape[1]):
                    projected_sensory.append(np.dot(catch[:, i], np.nanmean(cd_sensory[:, 11:15], axis=1)))
                    projected_lick.append(np.dot(catch[:, i], np.nanmean(cd_lick[:, 11:15], axis=1)))

                ax[j, 0].plot(np.asarray(projected_sensory), c='#000000', alpha=1 if lick else 0.3)
                ax[j, 1].plot(np.asarray(projected_lick), c='#000000', alpha=1 if lick else 0.3)
                ax[j, 2].plot(np.asarray(projected_lick), np.asarray(projected_sensory), c='#000000', alpha=1 if lick else 0.3)
        
        for j, (name, group) in enumerate(stim_df.groupby('context')):
        
            whisker = group.loc[(group.opto_stim_coord==stim) &
                                    (group.trial_type=='whisker_trial')].groupby(by=['roi', 'time'], as_index=False)['dff0'].mean().pivot(index='roi', columns='time').to_numpy()
            catch = group.loc[(group.opto_stim_coord==stim) &
                                    (group.trial_type=='no_stim_trial')].groupby(by=['roi', 'time'], as_index=False)['dff0'].mean().pivot(index='roi', columns='time').to_numpy()
            order = group.loc[(group.opto_stim_coord==stim) &
                                    (group.trial_type=='whisker_trial')].groupby(by=['roi', 'time'], as_index=False)['dff0'].mean().pivot(index='roi', columns='time').index.to_list()
            
            whisker[order.index(stim), :] = np.zeros_like(whisker[0,:])
            catch[order.index(stim), :] = np.zeros_like(catch[0,:])
            projected_sensory=[]
            projected_lick=[]
            for i in range(whisker.shape[1]):

                projected_sensory.append(np.dot(whisker[:, i], np.nanmean(cd_sensory[:, 11:15], axis=1)))
                projected_lick.append(np.dot(whisker[:, i], np.nanmean(cd_lick[:, 11:15], axis=1)))

            ax[j, 0].plot(np.asarray(projected_sensory), c='royalblue')
            ax[j, 1].plot(np.asarray(projected_lick), c='royalblue')
            ax[j, 2].plot(np.asarray(projected_lick), np.asarray(projected_sensory), c='royalblue')

            projected_sensory=[]
            projected_lick=[]
            for i in range(whisker.shape[1]):
                projected_sensory.append(np.dot(catch[:, i], np.nanmean(cd_sensory[:, 11:15], axis=1)))
                projected_lick.append(np.dot(catch[:, i], np.nanmean(cd_lick[:, 11:15], axis=1)))

            ax[j, 0].plot(np.asarray(projected_sensory), c='royalblue', alpha=0.3)
            ax[j, 1].plot(np.asarray(projected_lick), c='royalblue', alpha=0.3)
            ax[j, 2].plot(np.asarray(projected_lick), np.asarray(projected_sensory), c='royalblue', alpha=0.3)

            ax[j, 0].set_ylim([-scale, scale])
            ax[j, 0].set_title('sensory')
            # ax[j, 0].set_aspect('equal', 'box')

            ax[j, 1].set_ylim([-scale*2, scale*2])
            ax[j, 1].set_title('lick')
            # ax[j, 1].set_aspect('equal', 'box')

            ax[j, 2].set_ylim([-scale, scale])
            ax[j,2].set_ylabel('sensory')
            ax[j,2].set_xlabel('lick')
            ax[j, 2].set_xlim([-scale*2, scale*2])
            # ax[j, 2].set_aspect('equal', 'box')
        fig.tight_layout()
        fig.savefig(os.path.join(result_path, f'{stim}_coding_direction.png'))
        fig.savefig(os.path.join(result_path, f'{stim}_coding_direction.svg'))


def main(nwb_files, output_path):
    # combine_data(nwb_files, output_path)

    result_path = Path(output_path, 'coding_direction')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    coords_list = {'wS1': "(-1.5, 3.5)", 'wS2': "(-1.5, 4.5)", 'wM1': "(1.5, 1.5)", 'wM2': "(2.5, 1.5)", 'RSC': "(-0.5, 0.5)", "RSC_2": "(-1.5, 0.5)",
            'ALM': "(2.5, 2.5)", 'tjS1':"(0.5, 4.5)", 'tjM1':"(1.5, 3.5)", 'control': "(-5.0, 5.0)"}

    # opto_avg_df = load_opto_data(group)

    total_df = load_wf_opto_data(nwb_files, output_path)
    total_df.context = total_df.context.map({0:'non-rewarded', 1:'rewarded'})
    total_df['time'] = [[np.linspace(-1,3.98,250)] for i in range(total_df.shape[0])]
    total_df['legend']= total_df.apply(lambda x: f"{x.opto_stim_coord} - {'lick' if x.lick_flag==1 else 'no lick'}",axis=1)

    coding_direction_sensory_vs_lick(total_df, result_path)


if __name__ == "__main__":

    for file in ['context_sessions_wf_opto', 'context_sessions_wf_opto_controls']: #'context_sessions_wf_opto', 
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
