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
    nwb_folder = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis',
                                   experimenter, 'NWB')

    return nwb_folder