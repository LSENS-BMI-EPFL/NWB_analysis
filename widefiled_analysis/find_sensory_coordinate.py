import os
import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass


from nwb_wrappers import nwb_reader_functions as nwb_read


mouse_line = 'jrgeco'  # controls_gfp, gcamp, jrgeco, controls_tdtomato

config_file = f'//sv-nas1.rcp.epfl.ch/Petersen-lab/z_LSENS/Share/Pol_Bech/Session_list/context_sessions_{mouse_line}_expert.yaml'

with open(config_file, 'r', encoding='utf8') as stream:
    config_dict = yaml.safe_load(stream)
nwb_files = config_dict['Session path']

root_folder = r'\\sv-nas1.rcp.epfl.ch\Petersen-lab\analysis\Robin_Dard\Pop_results\Context_behaviour\roi_coordinates'

rrs_keys = ['ophys', 'brain_area_fluorescence', 'dff0_traces']
segmentation_list = ['ophys', 'brain_areas', 'brain_area_segmentation']
# rrs_keys = ['ophys', 'brain_grid_fluorescence', 'dff0_grid_traces']
segmentation_grid_list = ['ophys', 'grid_areas', 'brain_grid_area_segmentation']

scale = 1
x = [62 * scale, 167 * scale]
y = [162 * scale, 152 * scale]
c = np.round(np.sqrt((x[1] - x[0]) ** 2 + (y[0] - y[1]) ** 2) / 6)
bregma = (88, 120)
output_folder = os.path.join(root_folder, '20250317')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

coordinates_df = []
for file in nwb_files:
    session = nwb_read.get_session_id(file)
    mouse = session[0:5]
    print(' ')
    print(f"Mouse: {mouse}, Session: {session}")
    saving_folder = os.path.join(output_folder)
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    area_dict = nwb_read.get_cell_indices_by_cell_type(nwb_file=file, keys=rrs_keys)
    masks_list = nwb_read.get_image_mask(nwb_file=file, segmentation_info=segmentation_list)
    sensory_areas = ['wS1', 'wS2', 'A1']
    coor_df = pd.DataFrame()
    ap_list = []
    ml_list = []
    area_list = []
    origin_x = []
    origin_y = []
    for sensory_area in sensory_areas:
        print(f"Area: {sensory_area}")
        area_list.append(sensory_area)
        mask = masks_list[int(area_dict.get(sensory_area)[0])]
        mass_center = center_of_mass(mask)
        origin_x.append(mass_center[1])
        origin_y.append(mass_center[0])
        corrected_mass_center = np.array([mass_center[1] - bregma[0], bregma[1] - mass_center[0]])
        mass_center_coordinates = corrected_mass_center / c
        ap_list.append(mass_center_coordinates[0])
        ml_list.append(mass_center_coordinates[1])
        print(f"AP: {mass_center_coordinates[0]}, ML: {mass_center_coordinates[1]}")
    coor_df['AP'] = ap_list
    coor_df['ML'] = ml_list
    coor_df['Area'] = area_list
    coor_df['Mouse'] = mouse
    coor_df['Session'] = session
    coordinates_df.append(coor_df)

coordinates_df = pd.concat(coordinates_df)
coordinates_df.to_csv(os.path.join(output_folder, 'GECO_coordinates_table.csv'))
coordinates_df.to_excel(os.path.join(output_folder, 'GECO_coordinates_table.xlsx'))
grid_masks = nwb_read.get_image_mask(nwb_file=nwb_files[0], segmentation_info=segmentation_grid_list)


all_grid_spots = np.zeros(grid_masks[0].shape)
for grid_spot in grid_masks:
    all_grid_spots += grid_spot.astype(int)

fig, ax = plt.subplots()
ax.imshow(all_grid_spots, cmap='Greys', alpha=0.3)
markers = ['+', 'x', '*']
for idx, roi in enumerate(coordinates_df.Area.unique()):
    for sess in coordinates_df.Session.unique():
        x_loc = coordinates_df.loc[(coordinates_df.Area == roi) & (coordinates_df.Session == sess)]['AP'].values[:] * c + bregma[0]
        y_loc = bregma[1] - coordinates_df.loc[(coordinates_df.Area == roi) & (coordinates_df.Session == sess)]['ML'].values[:] * c
        ax.plot(x_loc[0], y_loc[0], marker=markers[idx], markersize=6)
ax.set_xlim(0, 160)
ax.set_ylim(0, 125)
ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False)
ax.plot(88, 120, marker='p', c='r', markersize=6)

