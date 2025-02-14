import os
import yaml
import numpy as np
import pandas as pd
import nwb_wrappers.nwb_reader_functions as nwb_read
import warnings

warnings.filterwarnings('ignore')


def filter_files_by_keyword(directory, mouse_id, keywords, expert_table):
    """Filter files in a directory by a keyword."""
    filtered_files = [f for f in os.listdir(directory) if mouse_id in f]
    if keywords['behaviour'] is not None:
        filtered_files = [f for f in filtered_files if
                          nwb_read.get_bhv_type_and_training_day_index(os.path.join(directory, f))[0] in keywords[
                              'behaviour']]

    if keywords['imaging'] is not None:
        filtered_files = [f for f in filtered_files if
                          keywords['imaging'] in nwb_read.get_session_type(os.path.join(directory, f))]

    if keywords['opto']:
        filtered_files = [f for f in filtered_files if
                         'opto_session' in nwb_read.get_session_type(os.path.join(directory, f))]

    if keywords['expert'] and expert_table:
        expert_table = pd.read_excel(expert_table)
        expert_sessions = expert_table.loc[expert_table['w_context_expert'] == True, 'session_id'].to_list()
        filtered_files = [f for f in filtered_files if
                         'opto_session' not in nwb_read.get_session_type(os.path.join(directory, f))]
        filtered_files = [f for f in filtered_files if
                         'pharma_session' not in nwb_read.get_session_type(os.path.join(directory, f))]
        filtered_files = [f for f in filtered_files if f.split(".")[0] in expert_sessions]

    return filtered_files


def read_yaml(file_path):
    """Read YAML file and return its content."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file) or {}
    return {}


def write_yaml(file_path, content):
    """Write content to a YAML file."""
    with open(file_path, 'w') as file:
        yaml.dump(content, file)


def append_to_yaml(file_path, keys, items, directory):
    """Append items to a specific key in a YAML file."""
    data = read_yaml(file_path)
    for key in keys:
        if 'path' in key:
            to_append = [os.path.join(directory, file) for file in items]
        else:
            to_append = [item.split(".")[0] for item in items]

        if key in data:
            for item in to_append:
                if item not in data[key]:
                    data[key].extend(to_append)
                else:
                    continue
            data[key] = np.unique(data[key]).tolist()
            data[key] = sorted(data[key])
        else:
            data[key] = sorted(to_append)

    write_yaml(file_path, data)


def main(directory, mouse_id, keywords, yaml_file, yaml_key, expert_table=False):
    """Main function to filter files and append to YAML."""

    for mouse in mouse_id:
        filtered_files = filter_files_by_keyword(directory, mouse, keywords, expert_table)
        append_to_yaml(yaml_file, yaml_key, filtered_files, directory)
        print(f"Filtered items appended to '{yaml_key}' in {yaml_file}")


if __name__ == "__main__":
    # Specify the directory to search, keyword, YAML file, and key
    # mouse_id = ['PB191', 'PB192', 'PB193', 'PB194', 'PB195', 'PB196', 'PB197', 'PB198', 'PB199', 'PB200', 'PB201']
    # mouse_id = ['PB191', 'PB192', 'PB193', 'PB194', 'PB200', 'PB201']
    ids = np.linspace(170, 204).tolist()
    mouse_id = [f'PB{int(id)}' for id in ids]

    directory = fr"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\{'Robin_Dard' if 'RD' in mouse_id[0] else 'Pol_Bech'}\NWB"

    keywords = {'behaviour': ['whisker_context', 'context'],
                'imaging': 'wf',  # wf or None
                'opto': True,
                'expert': False}  # or True

    # for session_file in ['context_sessions', 'context_sessions_controls']:
    #     yaml_file = fr"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\z_LSENS\Share\Pol_Bech\Session_list\{session_file}.yaml"
    # # yaml_file = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\z_LSENS\Share\Pol_Bech\Session_list\context_sessions_controls.yaml"
    #
    #     yaml_key = ['Session id', 'Session path']
    #
    #     expert_table = '//sv-nas1.rcp.epfl.ch/Petersen-Lab/z_LSENS/Share/Pol_Bech/Session_list/context_perf_table.xlsx'
    #     exclude = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\z_LSENS\Share\Pol_Bech\Session_list\exclude_sessions.yaml"
    #
    #     main(directory, mouse_id, keywords, yaml_file, yaml_key, expert_table)

    yaml_file = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\z_LSENS\Share\Pol_Bech\Session_list\context_sessions_wf_opto.yaml"
    yaml_key = ['Session id', 'Session path']

    expert_table = r"M:\analysis\Pol_Bech\Pop_results\Context_behaviour\Single mouse behavior_2024_12_15.13-59-34\context_expert_sessions.xlsx"
    exclude = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\z_LSENS\Share\Pol_Bech\Session_list\exclude_sessions.yaml"

    main(directory, mouse_id, keywords, yaml_file, yaml_key, expert_table)
