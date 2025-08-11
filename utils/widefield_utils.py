import numpy as np
import pandas as pd
from pathlib import Path

import nwb_wrappers.nwb_reader_functions as nwb_read
from nwb_utils import utils_misc


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


def load_wf_opto_data(nwb_files, output_path):
    total_df = []
    coords_list = {'wS1': "(-1.5, 3.5)", 'wS2': "(-1.5, 4.5)", 'wM1': "(1.5, 1.5)", 'wM2': "(2.5, 1.5)", 'RSC': "(-0.5, 0.5)",
                   'ALM': "(2.5, 2.5)", 'tjS1':"(0.5, 4.5)", 'tjM1':"(1.5, 3.5)", 'control': "(-5.0, 5.0)"}
    
    for nwb_file in nwb_files:
        session_id = nwb_read.get_session_id(nwb_file)
        mouse_id = nwb_read.get_mouse_id(nwb_file)
        df = [pd.read_parquet(Path(output_path, session_id, 'results.parquet.gzip', compression='gzip'))]
        total_df += df

    return pd.concat(total_df, ignore_index=True)