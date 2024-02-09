import os
import yaml
import math
import numpy as np
import dask.array as da
import NWB_reader_functions as NWB_read
import tifffile as tiff

def find_nearest(array, value, is_sorted=True):
    """
    Return the index of the nearest content in array of value.
    from https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    return -1 or len(array) if the value is out of range for sorted array
    Args:
        array:
        value:
        is_sorted:

    Returns:

    """
    if len(array) == 0:
        return -1

    if is_sorted:
        if value < array[0]:
            return -1
        elif value > array[-1]:
            return len(array)
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
            return idx - 1
        else:
            return idx
    else:
        array = np.asarray(array)
        idx = (np.abs(array - value)).idxmin()
        return idx


def plot_wf_activity(nwb_files):

    for nwb_file in nwb_files:
        mouse_id = NWB_read.get_mouse_id(nwb_file)
        session_id = NWB_read.get_session_id(nwb_file)
        wf_timestamps = NWB_read.get_widefield_timestamps(nwb_file, ['ophys', 'dff0'])

        for trial_type in NWB_read.get_behavioral_events_names(nwb_file):
            trials = NWB_read.get_behavioral_events_times(nwb_file, trial_type)
            # if trials == None:
            #     continue
            frames = []
            for tstamp in trials:
                frame = find_nearest(wf_timestamps, tstamp)
                data = NWB_read.get_widefield_dff0(nwb_file, ['ophys', 'dff0'], frame-200, frame+200)
                frames.append(data)

            data_frames = np.array(frames)
            data_frames = np.stack(data_frames, axis=0)
            avg_data = np.nanmean(data_frames, axis=0)
            save_path = fr'M:\analysis\Pol_Bech\Pop_results\Context_behaviour\{mouse_id}\{session_id}'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            tiff.imwrite(os.path.join(save_path, f'{trial_type}.tiff'), avg_data)



if __name__ == "__main__":
    experimenter = 'Pol_Bech'

    root_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'NWB')
    output_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'Pop_results', 'Context_behaviour')
    all_nwb_names = os.listdir(root_path)

    subject_ids = ['RD039', 'PB000']
    # plots_to_do = ['single_session']
    session_to_do = ["RD039_20240205_150044", "PB000_20240205_181158"]

    for subject_id in subject_ids:
        nwb_names = [name for name in all_nwb_names if subject_id in name]
        nwb_files = []
        for session in session_to_do:
            nwb_files += [os.path.join(root_path, name) for name in nwb_names if session in name]

        plot_wf_activity(nwb_files)
