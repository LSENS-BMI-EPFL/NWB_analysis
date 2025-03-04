import itertools
import os
import sys
sys.path.append(os.getcwd())
import warnings

import matplotlib.pyplot as plt
import yaml
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

import nwb_utils.utils_behavior as bhv_utils
import utils.behaviour_plot_utils as plot_utils
from utils.haas_utils import *
import nwb_wrappers.nwb_reader_functions as nwb_read
from nwb_utils import utils_misc
from nwb_utils.utils_plotting import lighten_color, remove_top_right_frame


def get_frames_by_epoch(nwb_file, trials, wf_timestamps, start=0, stop=200):
    frames = []
    for tstamp in trials:
        frame = utils_misc.find_nearest(wf_timestamps, tstamp)
        data = nwb_read.get_widefield_dff0(nwb_file, ['ophys', 'dff0'], int(frame + start), int(frame + stop))
        if data.shape != (len(np.arange(start, stop)), 125, 160):
            continue
        frames.append(np.nanmean(data, axis=0))

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
            continue
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
    coords_list = {'wS1': [[-1.5, 3.5],[-1.5, 4.5],[-2.5, 3.5],[-2.5, 4.5]], 'wM1': [[1.5, 1.5]], 'wM2': [[2.5,1.5]], 
                   'RSC': [[-0.5, 0.5], [-1.5,0.5]], 'control': [[-5,5]]} #AP, ML
    for nwb_file in nwb_files:
        session_df = []
        bhv_data = bhv_utils.build_standard_behavior_table([nwb_file])
        bhv_data = bhv_data.loc[bhv_data.early_lick==0]
        wf_timestamps = nwb_read.get_widefield_timestamps(nwb_file, ['ophys', 'dff0'])
        session_id = nwb_read.get_session_id(nwb_file)
        mouse_id = nwb_read.get_mouse_id(nwb_file)
        for loc in coords_list.keys():
            for coord in coords_list[loc]:

                opto_data = bhv_data.loc[(bhv_data.opto_grid_ap==coord[0]) & (bhv_data.opto_grid_ml==coord[1])] 
                trials = opto_data.start_time
                wf_image = get_frames_by_epoch(nwb_file, trials, wf_timestamps, start=0, stop=250) #250 frames since frame rate in this sessions is 50Hz and trial duration is 5s
                opto_data['wf_images'] = [wf_image[i] for i in range(opto_data.shape[0])]

                roi_data = get_dff0_traces_by_epoch(nwb_file, trials, wf_timestamps, start=0, stop=250)
                roi_data['trial_id'] = opto_data.trial_id
                opto_data = pd.merge(opto_data.reset_index(drop=True), roi_data, on='trial_id')
                session_df += [opto_data]
        session_df = pd.concat(session_df, ignore_index=True)
        if not os.path.exists(Path(output_path, session_id)):
            os.makedirs(Path(output_path, session_id))
        session_df.to_parquet(Path(output_path, session_id, 'results.parquet.gzip', compression='gzip'))


def main(nwb_files, output_path):
    combine_data(nwb_files, output_path)



if __name__ == "__main__":

        output_path = os.path.join('//sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Pol_Bech', 'Pop_results',
                                    'Context_behaviour', 'optogenetic_widefield_results')
        output_path = haas_pathfun(output_path)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        config_file = "//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Session_list/context_sessions_wf_opto.yaml"
        config_file = haas_pathfun(config_file)

        with open(config_file, 'r', encoding='utf8') as stream:
            config_dict = yaml.safe_load(stream)

        nwb_files = [haas_pathfun(p.replace("\\", '/')) for p in config_dict['Session path']]

        main(nwb_files, output_path)
