import os


EXPERIMENTER_MAP = {
    'AR': 'Anthony_Renard',
    'RD': 'Robin_Dard',
    'AB': 'Axel_Bisi',
    'MP': 'Mauro_Pulin',
    'PB': 'Pol_Bech',
    'MM': 'Meriam_Malekzadeh',
    'LS': 'Lana_Smith',
    'GF': 'Anthony_Renard',
    'MI': 'Anthony_Renard',
}


def get_experimenter_nwb_folder(experimenter_initials):
    experimenter = EXPERIMENTER_MAP[experimenter_initials]
    nwb_folder = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'NWB')

    return nwb_folder


def get_experimenter_saving_folder_root(experimenter_initials):
    experimenter = EXPERIMENTER_MAP[experimenter_initials]
    saving_folder = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter)

    return saving_folder


def from_session_list_to_path_list(sessions_list):
    sessions_path = []
    for session_id in sessions_list:
        experimenter = session_id[0:2]
        nwb_folder = get_experimenter_nwb_folder(experimenter)
        files = os.listdir(nwb_folder)
        nwb_files = [os.path.join(nwb_folder, name) for name in files if 'nwb' in name]
        for nwb_path in nwb_files:
            if session_id in nwb_path:
                sessions_path.append(nwb_path)
                break
            else:
                continue

    return sessions_path

