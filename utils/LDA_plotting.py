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
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import spikeinterface as si
import spikeinterface.preprocessing as sip
from sklearn.manifold import TSNE
from nwb_utils.utils_misc import find_nearest
from utils.lfp_utils import *
from utils.spindle_association import filter_ripples_with_spindle_coupling
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

#######################################
# PLOTS FOR THE LDA RESULTS IN THE LDA SPACE (centroids, scatter)
#######################################


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

def plot_lda_binary_results(data_folder, save_path, pair=None,
                            in_filename="lda_big_table_all_mice_pairwise.pkl"):
    '''
    Plot binary LDA results (LD1 histogram) from the pre-computed pairwise table
    produced by make_lda_big_table_all_mice(pairwise=True).

    Parameters
    ----------
    data_folder : str or Path
        Folder containing the pairwise LDA table (in_filename).
    save_path : str or Path
        Folder where plots will be saved.
    pair : str or None
        Which pair to display, e.g. "no_stim_trial-whisker_trial".
        If None, uses the first pair found in the table.
        Available pairs follow the pattern "<class_a>-<class_b>".
    in_filename : str
        Name of the pickle file to load (default: "lda_big_table_all_mice_pairwise.pkl").
    '''
    data_folder = Path(data_folder)
    region = Path(in_filename).stem.split("_")[-1]
    save_path = Path(save_path) / region / "lda_binary"
    save_path.mkdir(parents=True, exist_ok=True)

    in_file = data_folder / in_filename
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file}")

    big_table = pd.read_pickle(in_file)
    big_table = big_table[big_table["shuffle_index"] == -1].copy()  # real data only

    available_pairs = big_table["pair"].unique().tolist()
    if pair is None:
        pair = available_pairs[0]
        print(f"No pair specified. Using: '{pair}'")
        print(f"Available pairs: {available_pairs}")
    elif pair not in available_pairs:
        raise ValueError(f"Pair '{pair}' not found. Available pairs: {available_pairs}")

    big_table = big_table[big_table["pair"] == pair].copy()
    '''
    palette = {
        # auditory
        "auditory_trial_R+_1": "darkblue",
        "auditory_trial_R+_0": "lightblue",
        "auditory_trial_R-_1": "darkblue",
        "auditory_trial_R-_0": "lightblue",
        # whisker rewarded
        "whisker_trial_R+_1": "darkgreen",
        "whisker_trial_R+_0": "lightgreen",
        # whisker non rewarded
        "whisker_trial_R-_1": "darkred",
        "whisker_trial_R-_0": "lightcoral",
        # no stim
        "no_stim_trial_R+_1": "dimgray",
        "no_stim_trial_R+_0": "lightgray",
        "no_stim_trial_R-_1": "dimgray",
        "no_stim_trial_R-_0": "lightgray",
    }
    '''
    palette = {
        "auditory_trial_R+": "blue",
        "auditory_trial_R-": "lightblue",
        "whisker_trial_R+": "green",
        "whisker_trial_R-": "red",
        "no_stim_trial_R+": "gray",
        "no_stim_trial_R-": "lightgray"
    }

    for mouse in big_table["mouse"].unique():
        print(' ')
        print(f'Mouse: {mouse}')
        lda_table = big_table[big_table["mouse"] == mouse].copy()

        lda_table["lick_flag"] = lda_table["lick_flag"].apply(lambda x: str(x))
        '''''
        lda_table["trial_combination_type"] = (
            lda_table["trial_type"] + "_" + lda_table["rewarded_group"] + "_" + lda_table["lick_flag"]
        )
        '''
        lda_table["trial_combination_type"] = (
            lda_table["trial_type"] + "_" + lda_table["rewarded_group"]
        )
        lda_plot = lda_table.dropna(subset=["LD1"])

        g = sns.FacetGrid(
            lda_plot,
            col="lda_type",
            row="baseline_substracted",
            hue="trial_combination_type",
            margin_titles=True,
            height=3.5,
            aspect=1.4,
            palette=palette,
            sharey=False,
        )
        g.map_dataframe(sns.histplot, x="LD1", bins=30, stat="density", common_norm=False, alpha=0.7)

        g.set_axis_labels("LD1 (projection LDA)", "")
        g.add_legend()
        g.figure.subplots_adjust(hspace=0.35, wspace=0.25)

        pair_label = pair.replace("_trial", "").replace("_", " ")
        g.figure.suptitle(f"{mouse}  |  {pair_label}", y=1.02)

        pair_slug = pair.replace(" ", "_")
        out_file = save_path / f"{mouse}_{pair_slug}_LDA_binary.png"
        g.savefig(out_file, dpi=200, bbox_inches="tight")
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
            "no_stim_trial_R+_first_half": "lightgrey",
            "no_stim_trial_R+_second_half": "darkgrey",

            # whisker non rewarded
            "whisker_trial_R-_first_half": "lightcoral",
            "whisker_trial_R-_second_half": "darkred",
            "no_stim_trial_R-_first_half": "lightgrey",
            "no_stim_trial_R-_second_half": "darkgrey"
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

def plot_lda_results_from_table(data_folder,save_path,in_filename,index_order=False, shuffle_tot=None):
    '''
    Make the plots 2x3x3 figures for the LDA results. The function will iterate through every mice data and fit LDA models on them 
    using the make_lda_table_for_plot.
    
    '''

    data_folder = Path(data_folder)
    in_file = data_folder / in_filename
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file}")
    # take the targeted secondary region 
    region = Path(in_filename).stem.split("_")[-1]
    save_path = Path(save_path) / region / "lda_scatter"
    save_path.mkdir(parents=True, exist_ok=True)

    big_table = pd.read_pickle(in_file)
    big_table=big_table[big_table['shuffle_index']==-1].copy()  # keep only real data for the plot, but we can easily change that if we want to plot the shuffle distribution too
    if Path(in_filename).stem.split('_')[-2] == "multiclass":
        big_table = big_table[~big_table['lda_type'].str.contains("all_ripples_to_sensory_lda")].copy()

    for mouse in big_table['mouse'].unique():
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

            #no stim
            "no_stim_trial_R+_1": "dimgray",
            "no_stim_trial_R+_0": "lightgray",
            "no_stim_trial_R-_1": "dimgray",
            "no_stim_trial_R-_0": "lightgray",
            }


        if not {"LD1", "LD2", "baseline_substracted"}.issubset(lda_table.columns):
            print(f"  Skipping {mouse}: missing LDA columns (no data for this region)")
            continue

        # add a new attibute to the new_trial that cobine the trial type, the rewarded group and the lick flag
        lda_table['lick_flag'] = lda_table['lick_flag'].apply(lambda x: str(x))
        lda_table['trial_combination_type']= lda_table['trial_type'] + "_" + lda_table['rewarded_group']+ '_' + lda_table['lick_flag']
        lda_plot = lda_table.dropna(subset=["LD1", "LD2"])

        if lda_plot.empty:
            print(f"  Skipping {mouse}: no valid LD1/LD2 data")
            continue

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

        # annotate each subplot with the number of units used in the LDA
        if 'n_units' in lda_plot.columns:
            for (baseline, lda_type), ax in g.axes_dict.items():
                subset = lda_plot[(lda_plot['baseline_substracted'] == baseline) & (lda_plot['lda_type'] == lda_type)]
                if not subset.empty:
                    n_units = int(subset['n_units'].iloc[0])
                    ax.text(0.02, 0.98, f"n={n_units} units", transform=ax.transAxes,
                            fontsize=8, va='top', ha='left', color='black')

        g.figure.suptitle(f"{mouse} LDA results", y=1.02)
        out_file = save_path / f"{mouse}_LDA_plot.png"
        g.savefig(out_file, dpi=200, bbox_inches="tight")

        # Close the figure to avoid memory issues when processing many files
        plt.close(g.figure) 

        print(f"Saved: {out_file}")

def plot_lda_whisker_proba_in_time_with_bhv(
    data_folder_lda_table,
    data_folder_ripples,
    save_path,
    in_filename="lda_big_table_all_mice_pairwise.pkl",
    pair="no_stim_trial-whisker_trial",
    lda_projection="all_ripples_to_sensory_lda",
    target_class="whisker_trial",
    lowess_frac=0.3,
    block_size=20,
):
    """
    For each mouse: 3 stacked subplots sharing the x-axis (time).
      ax0 — CA1: P(target_class) for each ripple (LOWESS overlay)
      ax1 — SSp: same
      ax2 — behavioural hit-rate per block

    Uses the pairwise LDA table (default: no_stim_trial vs whisker_trial).
    Only real data (shuffle_index == -1), baseline-subtracted, ripples projected
    into the sensory LDA space (lda_projection).

    Ripples are coloured by the trial type during which they occurred
    (no_stim vs whisker, with whisker split by rewarded_group).
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

    file = Path(data_folder_lda_table) / in_filename
    big_table = pd.read_pickle(file)
    target_region = Path(in_filename).stem.split("_")[-1]
    save_path = Path(save_path) / target_region / "all_ripples_proba_plots"
    save_path.mkdir(parents=True, exist_ok=True)

    names_ripples_table = os.listdir(data_folder_ripples)

    prob_col    = f"prob_{target_class}"
    target_base = target_class.replace("_trial", "")

    # ── filter ──────────────────────────────────────────────────────────────
    big_table = big_table[big_table["shuffle_index"] == -1].copy()
    big_table = big_table[big_table["pair"] == pair].copy()
    big_table = big_table[big_table["lda_type"].str.contains(lda_projection)].copy()
    big_table = big_table[big_table["baseline_substracted"] == True].copy()
    big_table = big_table[big_table[prob_col].notna()].copy()

    if big_table.empty:
        print(f"No data found for pair='{pair}', projection='{lda_projection}'.")
        return

    big_table["brain_region"] = big_table["lda_type"].apply(lambda x: x.split("_")[0])

    # lick_flag_str pour groupby, lick_colored pour la palette seaborn
    big_table["lick_flag_str"] = big_table["lick_flag"].apply(
        lambda x: str(int(x)) if pd.notna(x) else "nan"
    )
    big_table["lick_colored"] = big_table["lick_flag_str"] + "_" + big_table["rewarded_group"]

    # ── LOWESS (computed once, before the per-mouse loop) ───────────────────
    big_table = big_table.sort_values("ripple_times").reset_index(drop=True)
    smoothed_vals = np.full(len(big_table), np.nan)
    for _, grp in big_table.groupby(["mouse", "brain_region", "lick_flag_str"]):
        if len(grp) < 4:
            continue
        idx = grp.index
        smoothed = sm_lowess(
            grp[prob_col].values,
            grp["ripple_times"].values,
            frac=lowess_frac,
            return_sorted=False,
        )
        smoothed_vals[big_table.index.get_indexer(idx)] = smoothed
    big_table["prob_lowess"] = smoothed_vals

    colors_rplus  = {"1_R+": "darkgreen", "0_R+": "lightgreen"}
    colors_rminus = {"1_R-": "darkred",   "0_R-": "lightcoral"}

    # ── per-mouse figures ────────────────────────────────────────────────────
    for mouse in big_table["mouse"].unique():

        rewarded_group = big_table.loc[big_table["mouse"] == mouse, "rewarded_group"].iloc[0]
        is_rplus       = rewarded_group == "R+"
        mouse_colors   = colors_rplus  if is_rplus else colors_rminus
        mouse_classes  = [f"1_{rewarded_group}", f"0_{rewarded_group}"]

        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        fig.suptitle(
            f"Mouse: {mouse}  —  Ripple {pair} - P({target_base}) + behaviour", fontsize=12
        )

        # ── probability panels ───────────────────────────────────────────────
        df_mouse = big_table[big_table["mouse"] == mouse]

        regions = df_mouse["brain_region"].unique()
        for ax, region in zip([ax0, ax1], regions):
            sub = df_mouse[df_mouse["brain_region"] == region]
            sns.scatterplot(
                data=sub, x="ripple_times", y=prob_col,
                hue="lick_colored", hue_order=mouse_classes,
                palette=mouse_colors, s=10, alpha=0.3,
                legend=False, ax=ax,
            )
            sns.lineplot(
                data=sub, x="ripple_times", y="prob_lowess",
                hue="lick_colored", hue_order=mouse_classes,
                palette=mouse_colors, linewidth=2,
                legend=False, ax=ax,
            )
            ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.set_title(region, fontsize=9)
            ax.set_ylabel(f"P({target_base})", fontsize=8)
            ax.set_xlabel("")

        # shared legend on ax0
        legend_handles = [
            plt.Line2D([0], [0], color=mouse_colors[cls], linewidth=2,
                       label=("lick" if cls.startswith("1") else "no lick"))
            for cls in mouse_classes
        ]
        ax0.legend(handles=legend_handles, title="Lick", fontsize=8,
                   title_fontsize=8, loc="upper left",
                   bbox_to_anchor=(1.01, 1), frameon=False)

        # ── behaviour panel ──────────────────────────────────────────────────
        df_ripple = None
        for name in names_ripples_table:
            if mouse == name[:5] and target_region in name: # we look for the ripple table that corresponds to the current mouse and region (if it exists)
                df_ripple = pd.read_pickle(os.path.join(data_folder_ripples, name))
                break

        if df_ripple is not None:
            block_m_df, hr_w_col = make_bhv_block_table(df_ripple, is_rplus, block_size)

            bhv_items  = {hr_w_col: ("green" if is_rplus else "red"), "hr_n": "black", "hr_a": "royalblue"}
            bhv_labels = {hr_w_col: f"whisker {rewarded_group}", "hr_n": "no stim", "hr_a": "auditory"}
            for hr, c in bhv_items.items():
                sns.lineplot(data=block_m_df, x="start_time", y=hr,
                             color=c, label=bhv_labels[hr], ax=ax2)

            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel("Hit rate", fontsize=8)
            ax2.set_xlabel("Time (s)", fontsize=8)
            ax2.legend(title="Trial type", fontsize=8, title_fontsize=8,
                       loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
        else:
            ax2.text(0.5, 0.5, "no behavioural data", ha="center", va="center",
                     transform=ax2.transAxes, color="grey")
            ax2.set_ylabel("Hit rate", fontsize=8)
            ax2.set_xlabel("Time (s)", fontsize=8)

        plt.tight_layout()

        out_file = save_path / f"{mouse}_ripple_{target_base}_proba_time_bhv.png"
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_file}")


def accuracy_plot(data_folder, save_path):
    '''
    This function will read the big table with all the LDA results for all the mice and make a plot of the accuracy for each pair of trial types, separated by brain region
        and projection type and rewarded group.
    '''
    file_path = os.path.join(data_folder, "lda_big_table_all_mice_pairwise.pkl")
    df = pd.read_pickle(file_path)
    df = df[df["shuffle_index"] == -1].copy()  # keep only real data
    df["brain_region"] = df["lda_type"].apply(lambda x: x.split("_")[0])
    df["projection_type"] = df["lda_type"].apply(lambda x: "_".join(x.split("_")[1:]))
    df["brain_region"] = df["brain_region"].replace("second", "SSp")

    pair_order_all = [
        "no_stim_trial-auditory_trial",
        "no_stim_trial-whisker_trial",
        "auditory_trial-whisker_trial"
    ]
    df = df[df["pair"].isin(pair_order_all)].copy()
    pair_order = [p for p in pair_order_all if p in df["pair"].unique()]

    # aggregate accuracy per mouse (mean across folds/trials if needed)
    df_acc = (
        df[df["baseline_substracted"] == True]
        .groupby(["mouse", "rewarded_group", "brain_region", "projection_type", "pair"], as_index=False)["accuracy"]
        .mean()
    )

    palette_reward = {
        "R+": "#2ca25f",
        "R-": "#de2d26"
    }

    g = sns.FacetGrid(
        df_acc,
        row="projection_type",
        col="brain_region",
        margin_titles=True,
        height=4,
        aspect=1.3,
        sharey=True
    )

    g.map_dataframe(
        sns.boxplot,
        x="pair",
        y="accuracy",
        order=pair_order,
        hue="rewarded_group",
        palette=palette_reward,
        hue_order=["R+", "R-"],
        boxprops={"facecolor": "none"}
    )

    g.map_dataframe(
        sns.stripplot,
        x="pair",
        y="accuracy",
        hue="rewarded_group",
        palette=palette_reward,
        hue_order=["R+", "R-"],
        order=pair_order,
        alpha=0.8,
        dodge=True
    )

    # add chance level line at 0.5 for binary classification
    for ax in g.axes.flat:
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance level")
        ax.tick_params(axis="x", rotation=30)

    g.set_axis_labels("Pair of trial types", "Accuracy")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.add_legend(title="Rewarded group")
    g.figure.suptitle("LDA classification accuracy", y=1.02)
    g.figure.subplots_adjust(hspace=0.3, wspace=0.2)

    region = Path("lda_big_table_all_mice_pairwise.pkl").stem.split("_")[-1]
    save_path = Path(save_path) / region / "accuracy"
    save_path.mkdir(parents=True, exist_ok=True)
    out_file = save_path / "accuracy_plot.png"
    g.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(g.figure)
    print(f"Saved: {out_file}")
    
##################################################
# PLOTS FOR THE PAIRWISE PROBABILITY DISTRIBUTIONS 
##################################################

def plot_mean_whisker_proba_distibution_per_region(data_folder, save_path, pair="no_stim_trial-whisker_trial",
                                                    ax=None, spindle_coupled_only=False, cooccurrence_window=0.05):
    """
    For each brain region: 2 boxplots (R+ / R-) of the mean P(whisker) across all ripples
    projected onto the whisker-auditory pairwise LDA (all_ripples_to_sensory_lda).
    Individual mouse points are overlaid on the boxplots.
    If ax is provided, draws on it without creating a figure or saving.

    spindle_coupled_only : if True, keep only ripples with spindle_coupling == True
                           (uses filter_ripples_with_spindle_coupling with cooccurrence_window).
    cooccurrence_window  : half-window in seconds for spindle coupling (default 0.05 s).
    """
    data_folder = Path(data_folder)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    files = sorted(data_folder.glob("lda_big_table_all_mice_pairwise_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No pairwise files found in {data_folder}")

    all_tables = []
    for f in files:
        df = pd.read_pickle(f)
        df = df[df["shuffle_index"] == -1].copy()
        df = df[
            (df["pair"] == pair) &
            (df["lda_type"].str.endswith("all_ripples_to_sensory_lda")) &
            (df["baseline_substracted"] == True)
        ].copy()
        if df.empty:
            continue
        df["brain_region"] = df["lda_type"].str.replace("_all_ripples_to_sensory_lda", "", regex=False)
        all_tables.append(df)

    if not all_tables:
        raise ValueError(f"No data found for {pair} pair with all_ripples projection, the available pairs are {[f['pair'].unique() for f in all_tables]}")

    big_table = pd.concat(all_tables, axis=0, ignore_index=True)

    # annotate spindle coupling then keep only coupled ripples if requested
    if spindle_coupled_only:
        big_table = filter_ripples_with_spindle_coupling(big_table, cooccurrence_window)
        big_table = big_table[big_table["spindle_coupling"]].copy()

    # mean P(whisker) per mouse × brain_region
    df_mean = (
        big_table
        .groupby(["mouse", "rewarded_group", "brain_region"], as_index=False)["prob_whisker_trial"]
        .mean()
    )

    region_order_all = ["ca1", "SSp", "DMS", "MO-ALM", "MO-wM1", "MO-wM2", "DLS", "mPFC", "PPC"]
    region_order = [r for r in region_order_all if r in df_mean["brain_region"].unique()]

    palette_reward = {"R+": "#2ca25f", "R-": "#de2d26"}

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(max(8, len(region_order) * 1.6), 5))

    sns.boxplot(
        data=df_mean,
        x="brain_region",
        y="prob_whisker_trial",
        hue="rewarded_group",
        order=region_order,
        hue_order=["R+", "R-"],
        palette=palette_reward,
        boxprops={"facecolor": "none"},
        flierprops={"marker": ""},
        ax=ax,
    )
    sns.stripplot(
        data=df_mean,
        x="brain_region",
        y="prob_whisker_trial",
        hue="rewarded_group",
        order=region_order,
        hue_order=["R+", "R-"],
        palette=palette_reward,
        dodge=True,
        alpha=0.8,
        jitter=True,
        ax=ax,
    )

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    ax.set_xlabel("Brain region")
    ax.set_ylabel("Mean P(whisker) per mouse")
    coupled_tag = f" — spindle-coupled only (±{cooccurrence_window}s)" if spindle_coupled_only else ""
    ax.set_title(f"All ripples projected on {pair} LDA{coupled_tag}")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title="Group")

    if standalone:
        plt.tight_layout()
        suffix = "_spindle_coupled" if spindle_coupled_only else ""
        out_file = save_path / f"mean_whisker_proba_per_region_{pair.replace('-', '_')}{suffix}.png"
        fig.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_file}")


def plot_delta_whisker_proba_per_region(data_folder, save_path, pair="no_stim_trial-whisker_trial",
                                        ax=None, spindle_coupled_only=False, cooccurrence_window=0.05):
    """
    For each brain region: delta P(whisker) = mean(second_half) - mean(first_half) per mouse,
    shown as 2 boxplots (R+ / R-) with individual mouse points overlaid.
    Uses all ripples projected onto the whisker-auditory pairwise LDA.
    If ax is provided, draws on it without creating a figure or saving.

    spindle_coupled_only : if True, keep only ripples with spindle_coupling == True
                           (uses filter_ripples_with_spindle_coupling with cooccurrence_window).
    cooccurrence_window  : half-window in seconds for spindle coupling (default 0.05 s).
    """
    data_folder = Path(data_folder)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    files = sorted(data_folder.glob("lda_big_table_all_mice_pairwise_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No pairwise files found in {data_folder}")

    all_tables = []
    for f in files:
        df = pd.read_pickle(f)
        df = df[df["shuffle_index"] == -1].copy()
        df = df[
            (df["pair"] == pair) &
            (df["lda_type"].str.endswith("all_ripples_to_sensory_lda")) &
            (df["baseline_substracted"] == True) &
            (df["trial_order_group"].isin(["first_half", "second_half"]))
        ].copy()
        if df.empty:
            continue
        df["brain_region"] = df["lda_type"].str.replace("_all_ripples_to_sensory_lda", "", regex=False)
        all_tables.append(df)

    if not all_tables:
        raise ValueError(f"No data found for {pair} pair with trial_order_group")

    big_table = pd.concat(all_tables, axis=0, ignore_index=True)

    # annotate spindle coupling then keep only coupled ripples if requested
    if spindle_coupled_only:
        big_table = filter_ripples_with_spindle_coupling(big_table, cooccurrence_window)
        big_table = big_table[big_table["spindle_coupling"]].copy()

    # mean P(whisker) per mouse × brain_region × half
    df_half = (
        big_table
        .groupby(["mouse", "rewarded_group", "brain_region", "trial_order_group"], as_index=False)
        ["prob_whisker_trial"].mean()
    )

    # pivot to get first_half and second_half as columns, then compute delta
    df_pivot = df_half.pivot_table(
        index=["mouse", "rewarded_group", "brain_region"],
        columns="trial_order_group",
        values="prob_whisker_trial"
    ).reset_index()

    if "first_half" not in df_pivot.columns or "second_half" not in df_pivot.columns:
        raise ValueError("trial_order_group must contain 'first_half' and 'second_half'")

    df_pivot["delta"] = df_pivot["second_half"] - df_pivot["first_half"]
    df_delta = df_pivot.dropna(subset=["delta"])

    region_order_all = ["ca1", "SSp", "DMS", "MO-ALM", "MO-wM1", "MO-wM2", "DLS", "mPFC", "PPC"]
    region_order = [r for r in region_order_all if r in df_delta["brain_region"].unique()]

    palette_reward = {"R+": "#2ca25f", "R-": "#de2d26"}

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(max(8, len(region_order) * 1.6), 5)) # create an independent figure if no ax is provided, with width adapted to the number of regions

    sns.boxplot(
        data=df_delta,
        x="brain_region",
        y="delta",
        hue="rewarded_group",
        order=region_order,
        hue_order=["R+", "R-"],
        palette=palette_reward,
        boxprops={"facecolor": "none"},
        flierprops={"marker": ""},
        ax=ax,
    )
    sns.stripplot(
        data=df_delta,
        x="brain_region",
        y="delta",
        hue="rewarded_group",
        order=region_order,
        hue_order=["R+", "R-"],
        palette=palette_reward,
        dodge=True,
        alpha=0.8,
        jitter=True,
        ax=ax,
    )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Brain region")
    ax.set_ylabel("ΔP(whisker)  [2nd half − 1st half]")
    coupled_tag = f" — spindle-coupled only (±{cooccurrence_window}s)" if spindle_coupled_only else ""
    ax.set_title(f"Learning effect on ripple whisker probability — {pair}{coupled_tag}")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title="Group")

    if standalone:
        plt.tight_layout()
        suffix = "_spindle_coupled" if spindle_coupled_only else ""
        out_file = save_path / f"delta_whisker_proba_per_region_{pair.replace('-', '_')}{suffix}.png"
        fig.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_file}")

def plot_accuracy_per_region(data_folder, save_path, pair="no_stim_trial-whisker_trial", ax=None):
    """
    For each brain region: sensory LDA accuracy (5-fold CV) per mouse, split by rewarded group.

    Reads all pairwise LDA tables in data_folder, filters by `pair` and lda_type=="sensory_lda".
    The chance level line is drawn at 0.5 (binary classification).
    If ax is provided, draws on it without creating a figure or saving.

    Saved under <save_path>/accuracy_per_region_<pair>.png
    """
    data_folder = Path(data_folder)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    files = sorted(data_folder.glob("lda_big_table_all_mice_pairwise_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No pairwise LDA files found in {data_folder}")

    all_tables = []
    for f in files:
        df = pd.read_pickle(f)
        df = df[df["shuffle_index"] == -1].copy()
        df = df[
            (df["pair"] == pair) &
            (df["lda_type"].apply(lambda x: "_".join(x.split("_")[1:])) == "sensory_lda") &
            (df["baseline_substracted"] == True)
        ].copy()
        if df.empty:
            continue
        df["brain_region"] = df["lda_type"].apply(lambda x: x.split("_")[0])

        all_tables.append(df)

    if not all_tables:
        raise ValueError(f"No sensory_lda data found for pair '{pair}' in {data_folder}")

    big_table = pd.concat(all_tables, axis=0, ignore_index=True)

    # One accuracy value per mouse × region (same for all rows in the group)
    df_accuracy = (
        big_table
        .groupby(["mouse", "rewarded_group", "brain_region"], as_index=False)["accuracy"]
        .first()
    )

    region_order_all = ["ca1", "SSp", "DMS", "MO-ALM", "MO-wM1", "MO-wM2", "DLS", "mPFC", "PPC"]
    region_order = [r for r in region_order_all if r in df_accuracy["brain_region"].unique()]

    palette_reward = {"R+": "#2ca25f", "R-": "#de2d26"}

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(max(8, len(region_order) * 1.6), 5))

    sns.boxplot(
        data=df_accuracy,
        x="brain_region",
        y="accuracy",
        hue="rewarded_group",
        order=region_order,
        hue_order=["R+", "R-"],
        palette=palette_reward,
        boxprops={"facecolor": "none"},
        flierprops={"marker": ""},
        ax=ax,
    )
    sns.stripplot(
        data=df_accuracy,
        x="brain_region",
        y="accuracy",
        hue="rewarded_group",
        order=region_order,
        hue_order=["R+", "R-"],
        palette=palette_reward,
        dodge=True,
        alpha=0.8,
        jitter=True,
        ax=ax,
    )

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Brain region")
    ax.set_ylabel("Accuracy (5-fold CV)")
    ax.set_title(f"Sensory LDA accuracy per region — {pair}")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title="Group")

    if standalone:
        plt.tight_layout()
        out_file = save_path / f"accuracy_per_region_{pair.replace('-', '_')}.png"
        fig.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_file}")

def combine_plots(data_folder, save_path, pair="no_stim_trial-whisker_trial",
                  spindle_coupled_only=False, cooccurrence_window=0.05):
    """
    Combined 3-row figure:
      Row 0 — Mean P(whisker) per mouse × region
      Row 1 — ΔP(whisker) [2nd half − 1st half] per mouse × region
      Row 2 — Sensory LDA accuracy (5-fold CV) per mouse × region

    spindle_coupled_only : if True, rows 0 and 1 use only spindle-coupled ripples.
    cooccurrence_window  : half-window in seconds for spindle coupling (default 0.05 s).

    Saved under <save_path>/combined_summary_<pair>[_spindle_coupled].png
    """
    data_folder = Path(data_folder)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 13))
    coupled_tag = f" — spindle-coupled only (±{cooccurrence_window}s)" if spindle_coupled_only else ""
    fig.suptitle(f"Summary — {pair}{coupled_tag}", fontsize=13)

    plot_mean_whisker_proba_distibution_per_region(data_folder, save_path, pair=pair, ax=axes[0],
                                                   spindle_coupled_only=spindle_coupled_only,
                                                   cooccurrence_window=cooccurrence_window)
    plot_delta_whisker_proba_per_region(data_folder, save_path, pair=pair, ax=axes[1],
                                        spindle_coupled_only=spindle_coupled_only,
                                        cooccurrence_window=cooccurrence_window)
    plot_accuracy_per_region(data_folder, save_path, pair=pair, ax=axes[2])

    plt.tight_layout()
    suffix = "_spindle_coupled" if spindle_coupled_only else ""
    out_file = save_path / f"combined_summary_{pair.replace('-', '_')}{suffix}.png"
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")

def correlation_lda_res_mouse_perf(data_folder, data_folder_ripples, save_path,
                                   pair='no_stim_trial-whisker_trial',
                                   spindle_coupled_only=False, cooccurrence_window=0.05,
                                   perf_table_path=None):
    """
    Relplot: col = brain region, row = LDA result type (mean ripple / delta ripple),
    x = LDA value per mouse, y = whisker hit rate.
    Regression line + r/p annotation per rewarded_group per facet.

    spindle_coupled_only : if True, keep only ripples with spindle_coupling == True
                           (uses filter_ripples_with_spindle_coupling with cooccurrence_window).
    cooccurrence_window  : half-window in seconds for spindle coupling (default 0.05 s).
    perf_table_path      : path to a pre-computed performance table (.pkl) produced by
                           make_mouse_performance_table().  If provided, data_folder_ripples
                           is ignored and no ripple tables are reloaded.

    Saved under <save_path>/correlation_lda_perf_<pair>[_spindle_coupled].png
    """
    from scipy.stats import linregress

    data_folder = Path(data_folder)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # --- Load LDA tables ---
    files = sorted(data_folder.glob("lda_big_table_all_mice_pairwise_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No pairwise files found in {data_folder}")

    all_tables = []
    for f in tqdm(files, desc="Loading LDA tables"):
        df = pd.read_pickle(f)
        df = df[df["shuffle_index"] == -1].copy()
        df = df[
            (df["pair"] == pair) &
            (df["lda_type"].str.endswith("all_ripples_to_sensory_lda")) &
            (df["baseline_substracted"] == True)
        ].copy()
        if df.empty:
            continue
        df["brain_region"] = df["lda_type"].str.replace("_all_ripples_to_sensory_lda", "", regex=False)
        all_tables.append(df)

    if not all_tables:
        raise ValueError(f"No data found for pair '{pair}' with all_ripples_to_sensory_lda projection")

    big_table = pd.concat(all_tables, axis=0, ignore_index=True)

    # annotate spindle coupling then keep only coupled ripples if requested
    if spindle_coupled_only:
        big_table = filter_ripples_with_spindle_coupling(big_table, cooccurrence_window)
        big_table = big_table[big_table["spindle_coupling"]].copy()

    # mean P(whisker) per mouse × rewarded_group × brain_region
    df_mean = (
        big_table
        .groupby(["mouse", "rewarded_group", "brain_region"], as_index=False)["prob_whisker_trial"]
        .mean()
        .rename(columns={"prob_whisker_trial": "mean_ripple"})
    )

    # delta P(whisker) = second_half − first_half, per mouse × rewarded_group × brain_region
    df_half = (
        big_table
        .groupby(["mouse", "rewarded_group", "brain_region", "trial_order_group"], as_index=False)
        ["prob_whisker_trial"].mean()
    )
    df_pivot = df_half.pivot_table(
        index=["mouse", "rewarded_group", "brain_region"],
        columns="trial_order_group",
        values="prob_whisker_trial"
    ).reset_index()

    if "first_half" not in df_pivot.columns or "second_half" not in df_pivot.columns:
        raise ValueError("trial_order_group must contain 'first_half' and 'second_half'")

    df_pivot["delta_ripple"] = df_pivot["second_half"] - df_pivot["first_half"]
    df_delta = df_pivot[["mouse", "rewarded_group", "brain_region", "delta_ripple"]].dropna(subset=["delta_ripple"])

    # --- Mouse performance: global hit rate + delta hit rate (2nd half − 1st half) ---
    if perf_table_path is not None:
        perf_df = pd.read_pickle(perf_table_path)
    else:
        data_folder_ripples = Path(data_folder_ripples)
        wh_perf_list = []
        for file in tqdm(os.listdir(data_folder_ripples), desc='Loading ripple tables'):
            df_ripple = pd.read_pickle(data_folder_ripples / file)
            mouse = df_ripple["mouse"].iloc[0]
            mask_w = df_ripple["trial_type"] == "whisker_trial"
            if "context" in df_ripple.columns:
                mask_w &= df_ripple["context"] == "active"

            wh_perf = df_ripple.loc[mask_w, "lick_flag"].mean()

            delta_perf = np.nan
            if "trial_order_group" in df_ripple.columns:
                first  = df_ripple.loc[mask_w & (df_ripple["trial_order_group"] == "first_half"),  "lick_flag"].mean()
                second = df_ripple.loc[mask_w & (df_ripple["trial_order_group"] == "second_half"), "lick_flag"].mean()
                delta_perf = second - first

            wh_perf_list.append({"mouse": mouse, "whisker_hit_rate": wh_perf, "delta_perf": delta_perf})

        perf_df = pd.DataFrame(wh_perf_list).drop_duplicates("mouse")

    # --- Merge ---
    merged = pd.merge(df_mean, df_delta, on=["mouse", "rewarded_group", "brain_region"], how="inner")
    perf_merge_keys = ["mouse", "rewarded_group"] if "rewarded_group" in perf_df.columns else ["mouse"]
    merged = pd.merge(merged, perf_df, on=perf_merge_keys, how="inner")

    # --- Reshape to long format ---
    long = pd.melt(
        merged,
        id_vars=["mouse", "rewarded_group", "brain_region", "whisker_hit_rate", "delta_perf"],
        value_vars=["mean_ripple", "delta_ripple"],
        var_name="lda_result_type",
        value_name="lda_value",
    )

    row_labels = {"mean_ripple": "Mean P(whisker)", "delta_ripple": "ΔP(whisker) [2nd−1st]"}
    long["lda_result_type"] = long["lda_result_type"].map(row_labels)
    row_order = [row_labels["mean_ripple"], row_labels["delta_ripple"]]

    # mean row → global hit rate ; delta row → delta hit rate
    long["perf_value"] = np.where(
        long["lda_result_type"] == row_labels["delta_ripple"],
        long["delta_perf"],
        long["whisker_hit_rate"],
    )

    region_order_all = ["ca1", "SSp", "DMS", "MO-ALM", "MO-wM1", "MO-wM2", "DLS", "mPFC", "PPC"]
    region_order = [r for r in region_order_all if r in long["brain_region"].unique()]

    palette_reward = {"R+": "#2ca25f", "R-": "#de2d26"}

    # --- Relplot ---
    g = sns.relplot(
        data=long,
        x="lda_value",
        y="perf_value",
        col="brain_region",
        row="lda_result_type",
        hue="rewarded_group",
        hue_order=["R+", "R-"],
        col_order=region_order,
        row_order=row_order,
        palette=palette_reward,
        kind="scatter",
        height=3.5,
        aspect=1.0,
        facet_kws={"margin_titles": True},
        alpha=0.85,
        s=60,
    )

    # regression line + r/p annotation per rewarded_group per facet
    for (row_val, col_val), ax in g.axes_dict.items():
        sub = long[(long["lda_result_type"] == row_val) & (long["brain_region"] == col_val)]
        y_text = {"R+": 0.95, "R-": 0.80}
        for rg, color in palette_reward.items():
            sub_rg = sub[sub["rewarded_group"] == rg].dropna(subset=["lda_value", "perf_value"])
            if len(sub_rg) < 3:
                continue
            slope, intercept, r, p, _ = linregress(sub_rg["lda_value"], sub_rg["perf_value"])
            x_range = np.linspace(sub_rg["lda_value"].min(), sub_rg["lda_value"].max(), 100)
            ax.plot(x_range, slope * x_range + intercept, color=color, linewidth=1.5, alpha=0.75)
            ax.text(0.05, y_text[rg], f"r={r:.2f}, p={p:.2f}",
                    transform=ax.transAxes, fontsize=7, color=color, va="top")

    # y-axis label differs per row
    y_labels = {row_labels["mean_ripple"]: "Whisker hit rate", row_labels["delta_ripple"]: "Δ hit rate [2nd−1st]"}
    for (row_val, _), ax in g.axes_dict.items():
        ax.set_ylabel(y_labels.get(row_val, "Performance"))

    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    coupled_tag = f" — spindle-coupled only (±{cooccurrence_window}s)" if spindle_coupled_only else ""
    g.figure.suptitle(f"LDA result vs mouse performance — {pair}{coupled_tag}", y=1.02)
    g.figure.subplots_adjust(hspace=0.35, wspace=0.25)

    suffix = "_spindle_coupled" if spindle_coupled_only else ""
    out_file = save_path / f"correlation_lda_perf_{pair.replace('-', '_')}{suffix}.png"
    g.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(g.figure)
    print(f"Saved: {out_file}")
