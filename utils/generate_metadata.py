import os
import yaml
import numpy as np
import pandas as pd
import nwb_wrappers.nwb_reader_functions as nwb_read
import warnings
warnings.filterwarnings('ignore')


def main(nwb_files, output_path, overwrite=False):
    if os.path.exists(os.path.join(output_path, "context_mice_metadata.xlsx")) and not overwrite:
        metadata = pd.read_excel(os.path.join(output_path, "context_mice_metadata.xlsx"))
        sessions_done = metadata.session_id.to_list()
    else:
        metadata = pd.DataFrame(columns=["mouse_id", "session_id", "session_type", "video_tstamps", "top_frames",
                                         "side_frames", "top_difference", "side_difference", "wf_tstamps", "wf_frames",
                                         "wf_difference"])
        sessions_done = []

    for nwb_file in nwb_files:
        if len(sessions_done) > 0 and nwb_read.get_session_id(nwb_file) in sessions_done:
            continue

        results = {
            "mouse_id": nwb_read.get_mouse_id(nwb_file),
            "session_id": nwb_read.get_session_id(nwb_file),
            "session_type": nwb_read.get_session_type(nwb_file),

        }

        if nwb_read.get_dlc_timestamps(nwb_file, ['behavior', 'BehavioralTimeSeries']) is not None:
            results["video_tstamps"] = len(
                nwb_read.get_dlc_timestamps(nwb_file, ['behavior', 'BehavioralTimeSeries'])[0])
            if len(nwb_read.get_dlc_data(nwb_file, ['behavior', 'BehavioralTimeSeries'], 'whisker_tip_x')) is not None:
                results["top_frames"] = len(
                    nwb_read.get_dlc_data(nwb_file, ['behavior', 'BehavioralTimeSeries'], 'whisker_tip_x'))
                results["side_frames"] = len(
                    nwb_read.get_dlc_data(nwb_file, ['behavior', 'BehavioralTimeSeries'], 'jaw_x'))
            else:
                results["top_frames"] = 0
                results["side_frames"] = 0
        else:
            results["video_tstamps"] = 0
            results["top_frames"] = 0
            results["side_frames"] = 0

        if nwb_read.get_widefield_timestamps(nwb_file, ['ophys', 'dff0']) is not None:
            results["wf_tstamps"] = len(nwb_read.get_widefield_timestamps(nwb_file, ['ophys', 'dff0']))

            if nwb_read.get_widefield_dff0_traces(nwb_file, ['ophys', 'brain_area_fluorescence', 'dff0_traces']) is not None:
                results["wf_frames"] = len(nwb_read.get_widefield_dff0_traces(nwb_file, ['ophys', 'brain_area_fluorescence', 'dff0_traces'])[0])
            else:
                results["wf_frames"] = 0
        else:
            results["wf_tstamps"] = 0
            results["wf_frames"] = 0

        results["top_difference"] = (np.asarray(results["video_tstamps"]) - np.asarray(results["top_frames"])).tolist()
        results["side_difference"] = (
                    np.asarray(results["video_tstamps"]) - np.asarray(results["side_frames"])).tolist()
        results["wf_difference"] = (np.asarray(results["wf_tstamps"])) - np.asarray(results["wf_frames"]).tolist()

        metadata = pd.concat([metadata, pd.DataFrame(results, index=[0])], ignore_index=True)

    metadata.to_excel(os.path.join(output_path, "context_mice_metadata.xlsx"))


if __name__ == '__main__':
    config_file = r"M:\analysis\Pol_Bech\Sessions_list\context_sessions_jrgeco.yaml"
    output_path = r"M:\z_LSENS\Share\Pol_Bech\Session_list"
    with open(config_file, 'r', encoding='utf8') as stream:
        config_dict = yaml.safe_load(stream)

    # Choose session from dict wit keys
    nwb_files = config_dict['Session path']
    main(nwb_files, output_path, overwrite=False)
