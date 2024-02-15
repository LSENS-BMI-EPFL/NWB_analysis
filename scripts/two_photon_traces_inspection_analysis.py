import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nwb_wrappers.nwb_reader_functions as nwb_read
from nwb_utils import server_path, utils_two_photons, utils_behavior


nwb_list = [
            # 'AR103_20230823_102029.nwb',
            #  'AR103_20230826_173720.nwb',
            #  'AR103_20230825_190303.nwb',
            #  'AR103_20230824_100910.nwb',
            #  'AR103_20230827_180738.nwb',
             'GF333_21012021_125450.nwb',
             'GF333_26012021_142304.nwb'
             ]

nwb_path = server_path.get_experimenter_nwb_folder('AR')
nwb_list = [os.path.join(nwb_path, nwb) for nwb in nwb_list]

F = []
time_stamp = []
Fneu = []
F0 = []
dff = []
dcnv = []
F_fissa = []
dff_fissa = []
events = []

for nwb_file in nwb_list:
    data = nwb_read.get_roi_response_serie_data(nwb_file, 'F')
    F.append(data)
    ts = nwb_read.get_roi_response_serie_timestamps(nwb_file, 'F')
    time_stamp.append(ts)
    data = nwb_read.get_roi_response_serie_data(nwb_file, 'Fneu')
    Fneu.append(data)
    # data = nwb_read.get_roi_response_serie_data(nwb_file, 'F0')
    # F0.append(data)
    data = nwb_read.get_roi_response_serie_data(nwb_file, 'dff')
    dff.append(data)
    data = nwb_read.get_roi_response_serie_data(nwb_file, 'dcnv')
    dcnv.append(data)
    data = nwb_read.get_roi_response_serie_data(nwb_file, 'F_fissa')
    F_fissa.append(data)
    data = nwb_read.get_roi_response_serie_data(nwb_file, 'dff_fissa')
    dff_fissa.append(data)
    ev = nwb_read.get_trial_timestamps_from_table(nwb_file, {'whisker_stim': [1], 'lick_flag':[0]})[0]
    events.append(ev)

F = np.concatenate(F, axis=1)
time_stamp = np.concatenate(time_stamp)
Fneu = np.concatenate(Fneu, axis=1)
# F0 = np.concatenate(F0, axis=1)
dff = np.concatenate(dff, axis=1)
dcnv = np.concatenate(dcnv, axis=1)
F_fissa = np.concatenate(F_fissa, axis=1)
dff_fissa = np.concatenate(dff_fissa, axis=1)
events = [e for e in ev for ev in events]








10, 44, 38

icell = 44

f, axes = plt.subplots(4,1, sharex=True)

axes[0].plot(F[icell])
axes[0].plot(Fneu[icell])
# axes[0].plot(F0[icell])
axes[0].axhline(1, linestyle='--', color='k')

axes[1].plot(F[icell]-0.7*Fneu[icell])
axes[1].plot(Fneu[icell])
# axes[1].plot(F0[icell])
axes[1].axhline(1, linestyle='--', color='k')

axes[2].plot(dff[icell])

axes[3].plot(dcnv[icell])

plt.suptitle(icell)



icell = 1

f, axes = plt.subplots(2,1, sharex=True)

axes[0].plot(F[icell])
axes[0].plot(Fneu[icell])
axes[1].plot(F_fissa[icell])

