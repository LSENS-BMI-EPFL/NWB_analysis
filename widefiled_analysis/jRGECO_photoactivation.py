import os
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.colors
import matplotlib.pyplot as plt
import nwb_wrappers.nwb_reader_functions as nwb_read

import warnings

warnings.filterwarnings("ignore")
from itertools import product
from utils.wf_plotting_utils import *
from nwb_utils import server_path, utils_misc, utils_behavior

power_map = {1.45: 4, 2.9: 8, 4.37: 12}
part_map = {"tjS1": [0.5, 3.5], "tjM1": [1.5, 2.5], "tjM2": [2.5, 2.5]}
bregma = (87.5, 120)


def compute_loc_in_wf(AP, ML, step):
    return bregma[0] + AP * step, bregma[
        1] - ML * step  # AP is substracted because more posterior from bregma is -x mm AP


def mean_loc(subgroup, step):
    results = []

    for index, row in subgroup.iterrows():

        center = compute_loc_in_wf(row.opto_grid_ap, row.opto_grid_ml, step)
        for stim, measure in product(np.arange(3), np.arange(3)):

            if stim == 0:
                if row.opto_grid_ap > 2:
                    continue
                if row.opto_grid_ml > 5 or row.opto_grid_ml < 2:
                    continue

                offset = np.array([[0, 0], [1.4 * step, -0.86 * step], [2.06 * step, -1.66 * step]])
            elif stim == 1:
                if row.opto_grid_ap > 2 or row.opto_grid_ap < -3:
                    continue
                if row.opto_grid_ml > 5 or row.opto_grid_ml < 2:
                    continue

                offset = np.array([[-1.4 * step, 0.86 * step], [0, 0], [0.66 * step, -0.8 * step]])
            elif stim == 2:
                if row.opto_grid_ap < -2:
                    continue
                if row.opto_grid_ml > 5 or row.opto_grid_ml < 2:
                    continue

                offset = np.array([[-2.06 * step, 1.66 * step], [-0.66 * step, 0.8 * step], [0, 0]])

            data = {}
            data['mouse_id'] = row.mouse_id
            data['opto_stim_amplitude'] = row.opto_stim_amplitude
            data['opto_grid_ap'] = row.opto_grid_ap
            data['opto_grid_ml'] = row.opto_grid_ml
            data['stim'] = stim
            data['measure'] = measure

            x = range(int(round(center[0]) + offset[measure, 0] - np.floor(step / 2)),
                      int(round(center[0]) + offset[measure, 0] + np.ceil(step / 2)))
            y = range(int(round(center[1]) + offset[measure, 1] + - np.floor(step / 2)),
                      int(round(center[1]) + offset[measure, 1] + np.ceil(step / 2)))

            data['dff0'] = np.nanmean(row['wf_mean'][y, x])
            results += [data]

    return pd.DataFrame(results)


def get_frames_by_epoch(nwb_file, trials, wf_timestamps, start=-200, stop=200):
    frames = []
    missing_trials = []
    for i, tstamp in enumerate(trials):
        frame = utils_misc.find_nearest(wf_timestamps, tstamp)
        data = nwb_read.get_widefield_dff0(nwb_file, ['ophys', 'dff0'], int(frame + start), int(frame + stop))
        if data.shape != (len(np.arange(start, stop)), 125, 160):
            missing_trials += [i]
            continue
        frames.append(data)

    data_frames = np.array(frames)
    data_frames = np.stack(data_frames, axis=0)
    return missing_trials, data_frames


def preprocess_data(nwb_file):
    mouse_id = nwb_read.get_mouse_id(nwb_file)
    session_id = nwb_read.get_session_id(nwb_file)

    trial_table = nwb_read.get_trial_table(nwb_file)
    wf_timestamps = nwb_read.get_widefield_timestamps(nwb_file, ['ophys', 'dff0'])
    missing_trials, image = get_frames_by_epoch(nwb_file, trial_table.start_time, wf_timestamps, start=0, stop=200)
    image = image - np.nanmean(image[:, 50:100, :, :], axis=1)[:, np.newaxis, :, :]
    image = [image[i] for i in range(image.shape[0])]
    trial_table = trial_table.drop(missing_trials)
    trial_table['widefield'] = image
    trial_table['wf_mean'] = trial_table.apply(lambda x: np.nanmean(x['widefield'][100:150, :, :], axis=0), axis=1)
    trial_table["mouse_id"] = mouse_id
    trial_table["session_id"] = session_id
    trial_table['opto_stim_amplitude'] = trial_table['opto_stim_amplitude'].map(power_map)

    return trial_table


def plot_mouse_data(results, output_path):
    step = get_wf_scalebar(1)
    for mouse_id, group in results.groupby('mouse_id'):
        save_path = os.path.join(output_path, mouse_id)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        pop_data = {'mouse_id': mouse_id, 'roi_data': []}

        wf_mean = group.groupby(['opto_stim_amplitude', 'opto_grid_ap', 'opto_grid_ml']).apply(
            lambda x: np.nanmean(np.stack(x['wf_mean']), axis=0)).reset_index()

        for amp, subgroup in wf_mean.groupby('opto_stim_amplitude'):
            pop_data['opto_stim_amp'] = amp
            plot_single_frame(np.nanmax(np.stack(subgroup[0]), axis=0), 'max projection', norm=True, colormap='hotcold',
                              colorbar_label='dFF0', save_path=os.path.join(save_path, f"{amp}_max_projection"),
                              vmin=-0.05, vmax=0.2, show=False)

            for i, part in enumerate(part_map.keys()):
                pop_data['roi_name'] = part
                pop_data['roi_mean'] = subgroup.loc[(subgroup['opto_grid_ap'] == part_map[part][0]) & (
                            subgroup['opto_grid_ml'] == part_map[part][1]), 0].values[0]

                pop_data['time'] = [np.arange(-2, 2, 0.02)]
                pop_data['roi_data'] = group.groupby(['opto_stim_amplitude', 'opto_grid_ap', 'opto_grid_ml']).get_group(
                    (amp, part_map[part][0], part_map[part][1]))['widefield']

                plot_single_frame(subgroup.loc[(subgroup['opto_grid_ap'] == part_map[part][0]) & (
                            subgroup['opto_grid_ml'] == part_map[part][1]), 0].values[0],
                                  f'Avg {part}', norm=True, colormap='hotcold',
                                  colorbar_label='dFF0', save_path=os.path.join(save_path, f"{amp}_{part}_average"),
                                  vmin=-0.05, vmax=0.2, show=False)


def plot_pop_data(results, output_path):
    scale = get_wf_scalebar(1)
    wf_mean = results.groupby(['mouse_id', 'opto_stim_amplitude', 'opto_grid_ap', 'opto_grid_ml']).apply(
        lambda x: np.nanmean(np.stack(x['wf_mean']), axis=0)).reset_index()

    # total_mean = wf_mean.groupby(['opto_stim_amplitude', 'opto_grid_ap', 'opto_grid_ml']).reset_index()

    avg_data = pd.DataFrame(columns=['mouse_id', "opto_stim_amplitude", "roi_name", "time", "roi_data"])

    for amp, subgroup in wf_mean.groupby('opto_stim_amplitude'):
        plot_single_frame(np.nanmax(np.stack(subgroup[0]), axis=0), 'max projection', norm=True, colormap='hotcold',
                          colorbar_label='dFF0', save_path=os.path.join(output_path, f"{amp}_max_projection"),
                          vmin=-0.05, vmax=0.2, show=False)

        for i, part in enumerate(part_map.keys()):
            plot_single_frame(subgroup.loc[(subgroup['opto_grid_ap'] == part_map[part][0]) & (
                    subgroup['opto_grid_ml'] == part_map[part][1]), 0].values[0],
                              f'Avg {part}', norm=True, colormap='hotcold',
                              colorbar_label='dFF0', save_path=os.path.join(output_path, f"{amp}_{part}_average"),
                              vmin=-0.05, vmax=0.2, show=False)


def plot_dff0_matrix(results, output_path):
    dset = results.groupby(['mouse_id', 'opto_stim_amplitude', 'stim', 'measure']).agg('mean').reset_index().groupby(
        ['opto_stim_amplitude', 'stim', 'measure']).agg(
        'mean').reset_index()

    dset.to_csv(os.path.join(output_path, "jRGECO_mat.csv"))

    for amp in dset.opto_stim_amplitude.unique():
        fig, ax = plt.subplots()
        fig.suptitle(f'{amp} mW')
        sns.heatmap(
            dset.loc[dset['opto_stim_amplitude'] == amp, :].pivot(index='stim', columns='measure', values='dff0'),
            cmap=get_colormap('hotcold'),
            norm=TwoSlopeNorm(vmin=-0.01, vcenter=0, vmax=0.2),
            ax=ax)
        # fig.show()
        fig.savefig(os.path.join(output_path, f'{amp}_power_matrix.png'))
        fig.savefig(os.path.join(output_path, f'{amp}_power_matrix.svg'))


def save_fig(fig, loc, name):
    if not os.path.exists(loc):
        os.makedirs(loc)
    os.path.join(loc, name)
    fig.savefig(os.path.join(loc, name + '.png'), dpi=100)
    fig.savefig(os.path.join(loc, name + '.svg'), dpi=100)


def main(nwb_files, output_path):
    results = []
    for nwb_file in nwb_files:
        trial_table = preprocess_data(nwb_file)
        results += [trial_table]

    results = pd.concat(results)
    dff0_mat = mean_loc(results, step=get_wf_scalebar(1))
    # plot_dff0_matrix(dff0_mat, output_path)
    # plot_mouse_data(results, output_path)
    plot_pop_data(results, output_path)


if __name__ == "__main__":
    config_file = r"M:\analysis\Pol_Bech\Sessions_list\jRGECO_photoactivation.yaml"
    with open(config_file, 'r', encoding='utf8') as stream:
        config_dict = yaml.safe_load(stream)
    sessions = config_dict['Session path']

    output_path = os.path.join(f'{server_path.get_experimenter_saving_folder_root("PB")}',
                               'Pop_results', 'jRGECO_photoactivation')

    main(sessions, output_path)
