from pynwb import NWBHDF5IO
from pynwb.base import TimeSeries


def get_mouse_id(nwb_file):
    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()
    mouse_id = nwb_data.subject.subject_id

    return mouse_id


def get_session_id(nwb_file):
    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()
    session_id = nwb_data.session_id

    return session_id


def get_bhv_type_and_training_day_index(nwb_file):
    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()
    description = nwb_data.session_description.split('_')
    if description[0] == 'free':
        behavior_type = description[0] + '_' + description[1]
        day = int(description[2])
    else:
        behavior_type = description[0]
        day = int(description[1])

    return behavior_type, day


def get_trial_table(nwb_file):
    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()
    nwb_objects = nwb_data.objects
    objects_list = [data for key, data in nwb_objects.items()]
    data_to_take = None
    for ind, obj in enumerate(objects_list):
        if 'trial' in obj.name:
            data = obj
            if isinstance(data, TimeSeries):
                continue
            else:
                data_to_take = data
                break
        else:
            continue
    trial_data_frame = data_to_take.to_dataframe()

    return trial_data_frame