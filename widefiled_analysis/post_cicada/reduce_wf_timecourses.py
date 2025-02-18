import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from utils.wf_plotting_utils import *


def loop_over_dict(data_dict):
    total_df = []
    for mouse in data_dict.keys():
        for trial_type in data_dict[mouse].keys():
            for session_data in data_dict[mouse][trial_type]:
                session = session_data[0]
                data = session_data[1]
                data, coords = reduce_im_dimensions(data) 
                df = generate_reduced_image_df(data, coords=coords)
                df['mouse_name'] = mouse
                df['session'] = session
                df['trial_type'] = trial_type
                total_df += [df]
    return pd.concat(total_df)


def aggregate_frames(df, n_frames_to_avg):
     curr_idx = df.frame.unique()
     new_idx = np.array([curr_idx for i in range(n_frames_to_avg)]).T.flatten()[:len(curr_idx)]
     df['to_agg'] = df['frame'].map({frame: new_idx[frame] for frame in total_avg.frame.unique()})
     return df.groupby(by=['trial_type', 'x', 'y', 'to_agg']).agg(dff0=('dff0', np.nanmean), frame_avg=('frame', lambda x: '_'.join(str(i) for i in x.values))).reset_index()


if __name__ == "__main__":
    cwd = os.getcwd()
    # cwd = r"M:\z_LSENS\Share\Pol_Bech\Bech_Dard et al 2025\figure4_data\Widefield Time Courses_2025_02_04.10-34-44"
    data_path = Path(cwd, 'general_data_dict.npy')
    result_path = Path(cwd)
    data_dict = np.load(data_path, allow_pickle=True).item()

    avg_df = loop_over_dict(data_dict=data_dict)
    mouse_avg = avg_df.groupby(by=['mouse_name', 'trial_type', 'frame', 'x', 'y']).agg(np.nanmean).reset_index()
    total_avg = mouse_avg.groupby(by=['trial_type', 'frame', 'x', 'y']).agg(np.nanmean).reset_index()
    total_avg = aggregate_frames(total_avg, 2)

    for name, group in total_avg.groupby(by=['trial_type', 'frame_avg']):
        fig,ax = plt.subplots(figsize=(4,4))
        fig.suptitle(name[0])
        fig, ax = plot_grid_on_allen(group, outcome='dff0', palette=get_colormap('hotcold'), dotsize=500, norm='two_slope', vmin=-0.005, vmax=0.02, result_path=None, fig=fig, ax=ax)
        fig.savefig(Path(result_path, name[0], f"reduced_{name[0]}_{name[1]}.png"))
        plt.close()
