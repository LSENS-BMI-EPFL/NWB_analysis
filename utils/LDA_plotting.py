import os
from random import shuffle
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.append('/Users/nigro/Desktop/NWB_analysis')

from utils.LDA_tables import *
from utils.LDA_loader import *
import pathlib
import subprocess
from pathlib import Path
import scipy as sci
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import spikeinterface as si
import spikeinterface.preprocessing as sip
from sklearn.manifold import TSNE
from nwb_utils.utils_misc import find_nearest
from utils.lfp_utils import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


def plot_lda_results(data_folder,save_path, brain_regions, window_ripple=0.05, window_sensory=0.05, classes_labels=None, 
                     index_order=False, shuffle_tot=None):
    '''
    Make the plots 2x3x3 figures for the LDA results. The function will iterate through every mice data and fit LDA models on them 
    using the make_lda_table_for_plot.
    
    '''

    #read files and loop through them
    save_path.mkdir(parents=True, exist_ok=True)
    names = os.listdir(data_folder)
    files = [os.path.join(data_folder, name) for name in names] 

    for file_id, file in enumerate(files):
        print(' ')
        print(f'Mouse: {names[file_id][0:5]}')
        file_path = os.path.join(data_folder, file)
        df = pd.read_pickle(file_path)
        dico_colors = {
            # auditory
            "auditory_trial_R+_1": "darkblue",   # lick
            "auditory_trial_R+_0": "lightblue",  # no lick

            "auditory_trial_R-_1": "darkblue",   # lick
            "auditory_trial_R-_0": "lightblue",  # no lick 

            # whisker rewarded
            "whisker_trial_R+_1": "darkgreen",
            "whisker_trial_R+_0": "lightgreen",

            # whisker non rewarded
            "whisker_trial_R-_1": "darkred",
            "whisker_trial_R-_0": "lightcoral",
            }
        lda_table = make_lda_table_for_one_mouse(df, brain_regions=brain_regions, window_sensory=window_sensory, window_ripple=window_ripple, classes_labels=classes_labels, shuffle_tot=shuffle_tot)
    

        # add a new attibute to the new_trial that cobine the trial type, the rewarded group and the lick flag
        lda_table['lick_flag'] = lda_table['lick_flag'].apply(lambda x: str(x))
        lda_table['trial_combination_type']= lda_table['trial_type'] + "_" + lda_table['rewarded_group']+ '_' + lda_table['lick_flag']
        lda_plot = lda_table.dropna(subset=["LD1", "LD2"])

        palette={}
        for i in lda_table['trial_combination_type'].unique():
            palette[i] = dico_colors.get(i, "lightgrey")

        # plotting using relplot to create a grid of scatter plots
        g = sns.relplot(
            data=lda_plot,
            x="LD1",
            y="LD2",
            hue="trial_combination_type",
            style= 'trial_order_group' if index_order else None,  # add a different marker for the first half and the second half of the trials to see if there is an effect of trial order on the LDA projection 
            col="lda_type",
            row="baseline_substracted",
            kind="scatter",
            alpha=0.7,
            height=4,
            aspect=1,
            facet_kws={"margin_titles": True},
            palette=palette
        )

        # add the centroids in the plot 
        centroid_palette_R_pos={
            "no_stim_trial": "lightgrey",
            "auditory_trial": "blue",
            "whisker_trial": "green"
            }
        
        centroid_palette_R_neg={
            "no_stim_trial": "lightgrey",
            "auditory_trial": "blue",
            "whisker_trial": "red"
            }
        
        centroids=compute_centroids(lda_plot)
        for (baseline,lda_type),subdf in centroids.groupby(['baseline_substracted','lda_type']):
            
            ax = g.axes_dict[(baseline, lda_type)]

            sns.scatterplot(
                data=subdf,
                x=f"centroid_LD1",
                y=f"centroid_LD2",
                hue="trial_type",
                palette=centroid_palette_R_pos if subdf["rewarded_group"].iloc[0] == "R+" else centroid_palette_R_neg,
                marker="X",
                zorder=10,
                s=120,
                linewidth=0.3,
                edgecolor="black",
                legend=False ,
                ax=ax
            )

        # plot the centroids inside the plot 

        g.figure.suptitle(f"{names[file_id][0:5]} LDA results", y=1.02)
        out_file = save_path / f"{names[file_id][0:5]}_LDA_plot.png"
        g.savefig(out_file, dpi=200, bbox_inches="tight")

        # Close the figure to avoid memory issues when processing many files
        plt.close(g.figure) 

        print(f"Saved: {out_file}")

def plot_lda_binary_results(data_folder,save_path, brain_regions, window_ripple=0.05, window_sensory=0.05, classes_labels=None):
    '''
    Make the same LDA but with only two classes (e.g. whisker vs acoustic ) to see if we can better separate them.
    '''
    save_path.mkdir(parents=True, exist_ok=True)
    names = os.listdir(data_folder)
    files = [os.path.join(data_folder, name) for name in names] 

    for file_id, file in enumerate(files):
        print(' ')
        print(f'Mouse: {names[file_id][0:5]}')
        file_path = os.path.join(data_folder, file)
        df = pd.read_pickle(file_path)
        palette = {
            # auditory
            "auditory_trial_R+_1": "darkblue",   # lick
            "auditory_trial_R+_0": "lightblue",  # no lick

            "auditory_trial_R-_1": "darkblue",   # lick
            "auditory_trial_R-_0": "lightblue",  # no lick 

            # whisker rewarded
            "whisker_trial_R+_1": "darkgreen",
            "whisker_trial_R+_0": "lightgreen",

            # whisker non rewarded
            "whisker_trial_R-_1": "darkred",
            "whisker_trial_R-_0": "lightcoral",
            }
        lda_table = make_lda_table_for_one_mouse(df, brain_regions=brain_regions, window_sensory=window_sensory, window_ripple=window_ripple, classes_labels=classes_labels)

        lda_table['lick_flag'] = lda_table['lick_flag'].apply(lambda x: str(x))
        lda_table['trial_combination_type']= lda_table['trial_type'] + "_" + lda_table['rewarded_group']+ '_' + lda_table['lick_flag']
        lda_plot = lda_table.dropna(subset=["LD1"])
        
        g= sns.FacetGrid(
                lda_plot,
                col="lda_type",
                row="baseline_substracted",
                hue="trial_combination_type",
                margin_titles=True,
                height=3.5,
                aspect=1.4,
                palette=palette
            )
        g.map_dataframe(sns.histplot, x="LD1", bins=30,stat="density",common_norm=False, alpha=0.7)

        g.set_axis_labels("LD1 (projection LDA)", "")
        g.add_legend()
        g.figure.subplots_adjust(hspace=0.35, wspace=0.25)
        
        g.figure.suptitle(f"{names[file_id][0:5]} LDA results", y=1.02)
        out_file = save_path / f"{names[file_id][0:5]}_LDA_plot.png"
        g.savefig(out_file, dpi=200, bbox_inches="tight")

        # Close the figure to avoid memory issues when processing many files
        plt.close(g.figure) 

        print(f"Saved: {out_file}")

def plot_lda_results_with_index_order(data_folder,save_path, brain_regions, window_ripple=0.05, window_sensory=0.05, classes_labels=None, shuffle_tot=None):
    '''
    Make the plots 2x3x3 figures for the LDA results. The function will iterate through every mice data and fit LDA models on them 
    using the make_lda_table_for_plot.
    '''

    #read files and loop through them
    save_path.mkdir(parents=True, exist_ok=True)
    names = os.listdir(data_folder)
    files = [os.path.join(data_folder, name) for name in names] 

    for file_id, file in enumerate(files):
        print(' ')
        print(f'Mouse: {names[file_id][0:5]}')
        file_path = os.path.join(data_folder, file)
        df = pd.read_pickle(file_path)
        dico_colors = {
            # auditory
            "auditory_trial_R+_first_half": "lightblue",   # lick
            "auditory_trial_R+_second_half": "darkblue",  # no lick

            "auditory_trial_R-_first_half": "lightblue",   # lick
            "auditory_trial_R-_second_half": "darkblue",  # no lick 

            # whisker rewarded
            "whisker_trial_R+_first_half": "lightgreen",
            "whisker_trial_R+_second_half": "darkgreen",

            # whisker non rewarded
            "whisker_trial_R-_first_half": "lightcoral",
            "whisker_trial_R-_second_half": "darkred",
            }
        # define the classses labels if not defined 

        if classes_labels is None:
            classes_labels = ["no_stim_trial", "whisker_trial", "auditory_trial"]

        lda_table = make_lda_table_for_one_mouse(df, brain_regions=brain_regions, window_sensory=window_sensory, window_ripple=window_ripple, classes_labels=classes_labels, shuffle_tot=shuffle_tot)
    
        # add a new attibute to the new_trial that cobine the trial type, the rewarded group and the lick flag
        lda_table['lick_flag'] = lda_table['lick_flag'].apply(lambda x: str(x))
        lda_table['trial_combination_type']= lda_table['trial_type'] + "_" + lda_table['rewarded_group']+ '_' + lda_table['trial_order_group']
        lda_plot = lda_table.dropna(subset=["LD1", "LD2"])
        
        palette={}
        for i in lda_table['trial_combination_type'].unique():
            palette[i] = dico_colors.get(i, "lightgrey")

        # plotting using relplot to create a grid of scatter plots
        g = sns.relplot(
            data=lda_plot,
            x="LD1",
            y="LD2",
            hue="trial_combination_type",
            col="lda_type",
            row="baseline_substracted",
            kind="scatter",
            alpha=0.7,
            height=4,
            aspect=1,
            facet_kws={"margin_titles": True},
            palette=palette
        )

        # add the centroids in the plot 
        centroid_palette_R_pos={
            "no_stim_trial": "lightgrey",
            "auditory_trial": "blue",
            "whisker_trial": "green"
            }
        
        centroid_palette_R_neg={
            "no_stim_trial": "lightgrey",
            "auditory_trial": "blue",
            "whisker_trial": "red"
            }
        centroids=compute_centroids(lda_plot)
        for (baseline,lda_type),subdf in centroids.groupby(['baseline_substracted','lda_type']):
            
            ax = g.axes_dict[(baseline, lda_type)]

            sns.scatterplot(
                data=subdf,
                x=f"centroid_LD1",
                y=f"centroid_LD2",
                hue="trial_type",
                palette=centroid_palette_R_pos if subdf["rewarded_group"].iloc[0] == "R+" else centroid_palette_R_neg,
                marker="X",
                zorder=10,
                s=120,
                linewidth=0.3,
                edgecolor="black",
                legend=False ,
                ax=ax
            )

        # plot the centroids inside the plot 

        g.figure.suptitle(f"{names[file_id][0:5]} LDA results", y=1.02)
        out_file = save_path / f"{names[file_id][0:5]}_LDA_plot.png"
        g.savefig(out_file, dpi=200, bbox_inches="tight")

        # Close the figure to avoid memory issues when processing many files
        plt.close(g.figure) 

        print(f"Saved: {out_file}")

def plot_lda_results_from_table(data_folder,save_path,in_filename="lda_big_table_all_mice.pkl",index_order=False, shuffle_tot=None):
    '''
    Make the plots 2x3x3 figures for the LDA results. The function will iterate through every mice data and fit LDA models on them 
    using the make_lda_table_for_plot.
    
    '''

    data_folder = Path(data_folder)
    in_file = data_folder / in_filename
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file}")

    big_table = pd.read_pickle(in_file)
    big_table=big_table[big_table['shuffle_index']==-1].copy()  # keep only real data for the plot, but we can easily change that if we want to plot the shuffle distribution too

    for file_id, mouse in enumerate(big_table['mouse'].unique()):
        print(' ')
        print(f'Mouse: {mouse}')
        lda_table = big_table[big_table['mouse'] == mouse].copy()

        dico_colors = {
            # auditory
            "auditory_trial_R+_1": "darkblue",   # lick
            "auditory_trial_R+_0": "lightblue",  # no lick

            "auditory_trial_R-_1": "darkblue",   # lick
            "auditory_trial_R-_0": "lightblue",  # no lick 

            # whisker rewarded
            "whisker_trial_R+_1": "darkgreen",
            "whisker_trial_R+_0": "lightgreen",

            # whisker non rewarded
            "whisker_trial_R-_1": "darkred",
            "whisker_trial_R-_0": "lightcoral",
            }
    

        # add a new attibute to the new_trial that cobine the trial type, the rewarded group and the lick flag
        lda_table['lick_flag'] = lda_table['lick_flag'].apply(lambda x: str(x))
        lda_table['trial_combination_type']= lda_table['trial_type'] + "_" + lda_table['rewarded_group']+ '_' + lda_table['lick_flag']
        lda_plot = lda_table.dropna(subset=["LD1", "LD2"])

        palette={}
        for i in lda_table['trial_combination_type'].unique():
            palette[i] = dico_colors.get(i, "lightgrey")

        # plotting using relplot to create a grid of scatter plots
        g = sns.relplot(
            data=lda_plot,
            x="LD1",
            y="LD2",
            hue="trial_combination_type",
            style= 'trial_order_group' if index_order else None,  # add a different marker for the first half and the second half of the trials to see if there is an effect of trial order on the LDA projection 
            col="lda_type",
            row="baseline_substracted",
            kind="scatter",
            alpha=0.7,
            height=4,
            aspect=1,
            facet_kws={"margin_titles": True},
            palette=palette
        )

        # add the centroids in the plot 
        centroid_palette_R_pos={
            "no_stim_trial": "lightgrey",
            "auditory_trial": "blue",
            "whisker_trial": "green"
            }
        
        centroid_palette_R_neg={
            "no_stim_trial": "lightgrey",
            "auditory_trial": "blue",
            "whisker_trial": "red"
            }
        centroids=compute_centroids(lda_plot)
        for (baseline,lda_type),subdf in centroids.groupby(['baseline_substracted','lda_type']):
            
            ax = g.axes_dict[(baseline, lda_type)]

            sns.scatterplot(
                data=subdf,
                x=f"centroid_LD1",
                y=f"centroid_LD2",
                hue="trial_type",
                palette=centroid_palette_R_pos if subdf["rewarded_group"].iloc[0] == "R+" else centroid_palette_R_neg,
                marker="X",
                zorder=10,
                s=120,
                linewidth=0.3,
                edgecolor="black",
                legend=False ,
                ax=ax
            )

        # plot the centroids inside the plot 

        g.figure.suptitle(f"{mouse} LDA results", y=1.02)
        out_file = save_path / f"{mouse}_LDA_plot.png"
        g.savefig(out_file, dpi=200, bbox_inches="tight")

        # Close the figure to avoid memory issues when processing many files
        plt.close(g.figure) 

        print(f"Saved: {out_file}")
