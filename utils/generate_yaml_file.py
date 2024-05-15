import os
import yaml
import nwb_wrappers.nwb_reader_functions as nwb_read


if __name__ == "__main__":

    yaml_path = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Pol_Bech\Sessions_list"
    nwb_folder = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Pol_Bech\NWB"

    nwb_files = [file for file in os.listdir(nwb_folder) if file.split("_")[0] not in['PB000', 'PB157']]

    for file in nwb_files:
        nwb_fullpath = os.path.join(nwb_folder, file)
        behaviour, _ = nwb_read.get_bhv_type_and_training_day_index(nwb_fullpath)
        session_type = nwb_read.get_session_type(nwb_fullpath)

        if 'context' in behaviour:
            if 'wf' in session_type:
                with open(os.path.join(yaml_path, "context_widefield_sessions_id.yaml"), 'r', encoding='utf8') as stream:
                    session_data = yaml.safe_load(stream)
                    session_data['Sessions id'].append(nwb_read.get_session_id(nwb_fullpath))
                with open(os.path.join(yaml_path, "context_widefield_sessions_id.yaml"), 'w', encoding='utf8') as stream:
                    yaml.safe_dump(session_data, stream)

                with open(os.path.join(yaml_path, "context_widefield_sessions_path.yaml"), 'r', encoding='utf8') as stream:
                    session_data = yaml.safe_load(stream)
                    session_data['Sessions path'].append(os.path.join(nwb_folder, file))
                with open(os.path.join(yaml_path, "context_widefield_sessions_path.yaml"), 'w', encoding='utf8') as stream:
                    yaml.safe_dump(session_data, stream)

