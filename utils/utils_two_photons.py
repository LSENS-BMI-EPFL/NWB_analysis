import numpy as np

from utils import utils_misc


def center_rrs_on_events(traces, traces_ts, event_ts, time_range, sampling_rate):
    time_range = (int(np.ceil(sampling_rate * 2)), int(np.floor(sampling_rate * 4)))
    
    event_frames = []
    for event in event_ts:
        event_frames.append(utils_misc.find_nearest(traces_ts, event))

    n_cells = traces.shape[0]
    n_tp = int(time_range[0] + time_range[1] + 1)
    activity_aligned = np.zeros((n_cells, len(event_frames), n_tp))
    for idx, frame in enumerate(event_frames):
        activity_aligned[:, idx] = traces[:, frame-time_range[0]:frame+time_range[1]+1]

    return activity_aligned