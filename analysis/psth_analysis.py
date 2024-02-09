import os
import numpy as np

import utils.server_path
import nwb_wrappers.nwb_reader_functions as nwb_read
import utils.utils_misc

nwb_list =  ['AR103_20230827_180738.nwb',
             'AR103_20230826_173720.nwb',
             'AR103_20230825_190303.nwb',
             'AR103_20230824_100910.nwb',
             'AR103_20230823_102029.nwb']
nwb_path = utils.server_path.get_experimenter_nwb_folder('AR')
nwb_list = [os.path.join(nwb_path, nwb) for nwb in nwb_list]
nwb_file = nwb_list[0]

event_names = nwb_read.get_behavioral_events_names(nwb_file)
event_times = nwb_read.get_behavioral_events_times(nwb_file, 'whisker_miss_trial')[0]

traces = nwb_read.get_roi_response_serie_data(nwb_file, 'dff')
traces_ts = nwb_read.get_roi_response_serie_timestamps(nwb_file, 'dff')
sampling_rate = nwb_read.get_rrs_sampling_rate(nwb_file, 'dff')




event_frames = []
for event in event_times:
    event_frames.append(utils.utils_misc.find_nearest(traces_ts, event))

frame_range = (int(np.ceil(sampling_rate * 2)), int(np.floor(sampling_rate * 4)))

n_cells = traces.shape[0]
n_tp = int(frame_range[0] + frame_range[1] + 1)
activity = np.zeros((n_cells, len(event_frames), n_tp))
for idx, frame in enumerate(event_frames):
    activity[:, idx] = traces[:, frame-frame_range[0]:frame+frame_range[1]+1]

import matplotlib.pyplot as plt
plt.plot(activity.mean(axis=(0,1)))
