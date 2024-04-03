import numpy as np
import pandas as pd

from nwb_utils import utils_misc


def center_rrs_on_events(traces, traces_ts, event_ts, time_range, sampling_rate):
    time_range = (int(np.ceil(sampling_rate * time_range[0])),
                  int(np.floor(sampling_rate * time_range[1])))

    event_frames = []
    for event in event_ts:
        event_frames.append(utils_misc.find_nearest(traces_ts, event))

    n_cells = traces.shape[0]
    n_tp = int(time_range[0] + time_range[1] + 1)
    activity_aligned = np.zeros((n_cells, len(event_frames), n_tp)) * np.nan
    for idx, frame in enumerate(event_frames):
        if frame+time_range[1]+1 > traces.shape[1]:   # Remove event if to close to the end
            continue
        activity_aligned[:, idx] = traces[:, frame-time_range[0]:frame+time_range[1]+1]

    return activity_aligned


def select_activity_around_events_pd(activity, activity_ts, rois, events, time_range, sampling_rate, **metadata):

    dfs = []
    activity_aligned = center_rrs_on_events(activity, activity_ts,
                                            events, time_range,
                                            sampling_rate)
    n_cells, n_events, n_t = activity_aligned.shape
    time_stamps_vect = np.linspace(-time_range[0], time_range[1], n_t)
    time_stamps_vect = np.tile(time_stamps_vect, n_cells * n_events)
    event_vect = np.tile(np.repeat(np.arange(n_events), n_t), n_cells)
    rois_vect = np.repeat(rois, n_events * n_t)
    activity_reshaped = activity_aligned.flatten()

    df = dict({'activity': activity_reshaped, 'time': time_stamps_vect, 'event': event_vect, 'roi': rois_vect})
    df = pd.DataFrame.from_dict(df)
    
    # Add session metadata.
    for key, val in metadata.items():
        df[key] = val

    dfs.append(df)

    return pd.concat(dfs, ignore_index=True)
