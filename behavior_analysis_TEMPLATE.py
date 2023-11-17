#! /usr/bin/env/python3
"""
@project: NWB_analysis
@file: behavior_analysis_TEMPLATE.py
@time: 11/17/2023 4:15 PM
@description: Template script for making figures. Copy this file and rename it to make_figures_<INITIALS>.py and edit to your needs.
"""


# Imports
import os
import pandas as pd
import behavior_analysis_utils as bhv_utils

# Imports plotting functions
from behavior_analysis_ab import plot_single_mouse_weight_across_days, plot_single_session, plot_single_mouse_across_days

########################################################################################################################
# EDIT plot_behaviour and plot_group_behavior with desired functions.
########################################################################################################################

def plot_behavior(nwb_list, output_folder):
    bhv_data = bhv_utils.build_standard_behavior_table(nwb_list)

    # Plot all single session figures
    colors = ['#225ea8', '#00FFFF', '#238443', '#d51a1c', '#cccccc']

    plot_single_mouse_weight_across_days(combine_bhv_data=bhv_data, color_palette=colors, saving_path=output_folder)
    plot_single_session(combine_bhv_data=bhv_data, color_palette=colors, saving_path=output_folder)
    plot_single_mouse_across_days(combine_bhv_data=bhv_data, color_palette=colors, saving_path=output_folder)

    return

def plot_group_behavior(nwb_list):


    return

########################################################################################################################
# Main
# EDIT with your name and subject IDs to plot
########################################################################################################################

if __name__ == '__main__':

    # Use the functions to do the plots
    experimenter = 'Axel_Bisi'

    # Paths
    root_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, r'NWB')
    output_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'results')
    all_nwb_names = os.listdir(root_path)

    # Choose list of subject IDs to make plots for
    subject_ids = [11, 111]
    subject_ids = ['AB0{}'.format(i) if i<100 else 'AB{}'.format(i) for i in subject_ids]


    # Single-mouse plots
    # ------------------
    for subject_id in subject_ids:
        print(" ")
        print(f"Subject ID : {subject_id}")

        # Get NWB files for each mouse
        nwb_names = [name for name in all_nwb_names if subject_id in name]
        nwb_files = [os.path.join(root_path, name) for name in nwb_names]

        # Make output folder, per mouse
        results_path = os.path.join(output_path, subject_id)
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        # Run plotting functions for each mouse
        plot_behavior(nwb_list=nwb_files, output_folder=results_path)
