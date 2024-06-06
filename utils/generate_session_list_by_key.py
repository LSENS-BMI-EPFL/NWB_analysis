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
    directory = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Pol_Bech\NWB"
    # mouse_id = ["PB182", "PB184"]
    mouse_id = ["PB176", "PB177", "PB178", "PB179", "PB180",
                "PB181"]
    keywords = {'behaviour': ['whisker_context', 'context'],
                'imaging': None,  # wf or None
                'opto': True,
                'expert': False}  # or True

    yaml_file = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Pol_Bech\Sessions_list\context_sessions_opto.yaml"
    yaml_key = ['Session id', 'Session path']

    expert_table = r"M:\analysis\Pol_Bech\Pop_results\Context_behaviour\Context_behaviour_analysis_may2024\context_expert_sessions.xlsx"

    main(directory, mouse_id, keywords, yaml_file, yaml_key, expert_table)
