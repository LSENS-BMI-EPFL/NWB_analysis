import numpy as np
import pandas as pd
import re

from scipy.signal import firwin, filtfilt
import nwb_wrappers.nwb_reader_functions as nwb_read
import nwb_utils.utils_misc


def compute_jaw_opening_epoch(df):
    nfilt = 100  # Number of taps to use in FIR filter
    fw_base = 5  # Cut-off frequency for lowpass filter, in Hz
    nyq_rate = 200 / 2.0
    cutoff = min(1.0, fw_base / nyq_rate)
    b = firwin(nfilt, cutoff=cutoff, window='hamming')
    padlen = 3 * nfilt
    filtered_jaw = filtfilt(b, [1.0], df['jaw_angle'], axis=0,
                            padlen=padlen)
    jaw_opening = np.where(filtered_jaw < filtered_jaw.std(), np.zeros_like(filtered_jaw), 1)

    return jaw_opening


def compute_whisker_movement_epoch(df): ## TODO: revisit combination of parameters
    nfilt = 100  # Number of taps to use in FIR filter
    fw_base = 10  # Cut-off frequency for lowpass filter, in Hz
    nyq_rate = 200 / 2.0
    cutoff = min(1.0, fw_base / nyq_rate)
    b = firwin(nfilt, cutoff=cutoff, window='hamming')
    padlen = 3 * nfilt
    filtered_wh = filtfilt(b, [1.0], df['whisker_velocity'].abs(), axis=0,
                            padlen=padlen)
    movement = np.where(filtered_wh < 1*filtered_wh.std(), np.zeros_like(filtered_wh), 1)
    quiet = np.where(filtered_wh < 1, np.ones_like(filtered_wh), 0)
    return movement, quiet


def compute_movement_and_quiet(df, fw_base, mov_thr, quiet_thr):
    """
    Provided a vector of px values from dlc, extract periods of movement and quiet as boxcars of same dimensions as df
    Arguments:
    :param df: column from dlc dataframe to extract movement and quiet periods
    :param fw_base: frequency to lowpass filter
    :param mov_thr: threshold for movement periods, will multiply by std
    :param quiet_thr: threshold for quiet periods. I.e. for whisker < 1 pixel/50ms
    :return: movement, quiet = boxcars of 1 and 0
    """
    nfilt = 100  # Number of taps to use in FIR filter
    nyq_rate = 200 / 2.0
    cutoff = min(1.0, fw_base / nyq_rate)
    b = firwin(nfilt, cutoff=cutoff, window='hamming')
    padlen = 3 * nfilt
    filtered = filtfilt(b, [1.0], df.abs(), axis=0,
                            padlen=padlen)
    movement = np.where(filtered < mov_thr*filtered.std(), np.zeros_like(filtered), 1)
    quiet = np.where(filtered < quiet_thr, np.ones_like(filtered), 0)

    return movement, quiet


def get_start_stop_epochs(boxcar):
    """
    Merge epochs that are very close (0.5s) together
    :param boxcar:
    :return:
    """
    starts = np.where(np.diff(boxcar) > 0)[0]
    stops = np.where(np.diff(boxcar) < 0)[0]
    if len(stops) < len(starts):
        stops = stops[:-1]
    onset = np.vstack((starts, stops)).T.flatten()
    ends = [item for item in np.where(np.diff(onset) < 100)[0] if item % 2 == 1] # Merge when separation between licks is less than 0.5 s
    ends += [item + 1 for item in ends]

    return [item for i, item in enumerate(onset) if i not in ends]


def filter_min_len(onset, min_len=100):
    """
    Eliminate periods that don't have the required duration
    Arguments:
    :param onset: np.array of shape (#events, 2)
    :param min_len: int
    :return: onset -events that don't have enough duration
    """
    short_periods = []
    for i, item in enumerate(onset):
        if item[1] - item[0] < min_len:
            short_periods += [i]
    return np.delete(onset, short_periods, axis=0)


def get_isquiet(dlc_timestamps, quiet, frames, nframes_before):

    data_quiet = []
    for frame in frames:
        dlc_frame = nwb_utils.utils_misc.find_nearest(dlc_timestamps, frame)
        data_quiet += True if quiet[dlc_frame-nframes_before:dlc_frame].sum() == 0 else False

    return data_quiet


def filter_part_by_camview(view):
    if view == 'side':
        return ['jaw_y', 'jaw_angle', 'jaw_distance', 'jaw_velocity',
                'nose_angle', 'nose_distance',
                'particle_x', 'particle_y',
                'pupil_area', 'spout_y',
                'tongue_angle', 'tongue_distance', 'tongue_velocity']

    elif view == 'top':
        return ['top_nose_angle', 'top_nose_distance', 'top_nose_velocity',
                'top_particle_x', 'top_particle_y',
                'whisker_angle', 'whisker_velocity']

    else:
        print('Wrong view name')
        return 0


def get_likelihood_filtered_bodypart(nwb_file, keys, part, threshold=0.8):

    kinematic = part.split("_")[-1]
    root = re.sub(kinematic, '', part)
    suffix = 'tip_likelihood' if 'whisker' in part or 'top_nose' in part else 'likelihood'
    data = nwb_read.get_dlc_data(nwb_file, keys, part)
    likelihood = nwb_read.get_dlc_data(nwb_file, keys, root+suffix)

    return np.where(likelihood >= threshold, data, 0 if 'tongue' in part else np.nan)


def get_dlc_data(nwb_file, trials, timestamps, view, parts='all', start=0, stop=250):

    keys = ['behavior', 'BehavioralTimeSeries']

    if parts == 'all':
        dlc_parts = filter_part_by_camview(view)
    else:
        dlc_parts = [part for part in filter_part_by_camview(view) if part in parts]

    dlc_data = pd.DataFrame(columns=dlc_parts)

    view_timestamps = timestamps[0 if view == 'side' else 1]
    trial_data = []
    if len(view_timestamps) == 0:
        for i, tstamp in enumerate(trials):
            trace = pd.DataFrame(np.ones([abs(start-stop), len(dlc_parts)])*np.nan, columns=dlc_parts)
            trace['trl_type_idx'] = i
            trace['time'] = np.arange(start/100, stop/100, 0.01)-1
            trial_data += [trace.groupby('trl_type_idx').agg(lambda x: x.tolist())]
        return pd.concat(trial_data)
    
    for part in dlc_parts:
        # print(f"Getting data for {part}")
        dlc_data[part] = get_likelihood_filtered_bodypart(nwb_file, keys, part, threshold=0.5)
    
    view_timestamps = view_timestamps[:len(dlc_data)]

    for i, tstamp in enumerate(trials):
        frame = nwb_utils.utils_misc.find_nearest(view_timestamps, tstamp)

        trace = dlc_data.loc[frame+(start+1):frame+stop]
        if trace.shape == (len(np.arange(start, stop)), len(dlc_parts)):
            trace = trace.apply(lambda x: x - np.nanmean(x.iloc[0:50]))
        elif trace.shape == (len(np.arange(start, stop))-1, len(dlc_parts)):
            print(f"{view} has one frame less than requested")
            trace = dlc_data.loc[frame+(start+1):frame+stop+1]
            print(f"New shape {trace.shape[0]}")
        elif trace.shape == (len(np.arange(start, stop))+1, len(dlc_parts)):
            print(f"{view} has one frame more than requested")
            trace = trace[:-1, :]
            print(f"New shape {trace.shape[0]}")

        else:
            print(f'{view} has less data for this trial than requested: {trace.__len__()} frames')
            trace = pd.DataFrame(np.ones([abs(start-stop), len(dlc_parts)])*np.nan, columns=dlc_parts)

        trace['trl_type_idx'] = i
        trace['time'] = np.arange(start/100, stop/100, 0.01)-1
        trial_data += [trace.groupby('trl_type_idx').agg(lambda x: x.tolist())]
    return pd.concat(trial_data)