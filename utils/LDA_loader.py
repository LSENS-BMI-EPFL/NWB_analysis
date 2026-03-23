import os
from random import shuffle
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.append('/Users/nigro/Desktop/NWB_analysis')
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
from utils.LDA_tables import *
from utils.LDA_plotting import *




def prepare_vector_LDA(
    df,
    brain_region: str,
    window_sensory=0.05,
    window_ripple=0.05,
    substract_baseline=True,
    context_value="active",
    classes_labels=None,
    shuffle_i= None
    
    ):
    """
    Compute pseudobulk raw counts per patient × cell type.

    Parameters
    ----------
    df : DataFrame
        The ripples table
    brain_region : str
        The brain region to process
    window_sensory : float
        The window size for sensory data (default 0.05)
    window_ripple : float
        The window size for ripple data (default 0.05)
    substract_baseline : bool
        Whether to subtract baseline from sensory data (default True)
    context_value : str
        The context value to filter the dataframe (default "active")
    classes_labels : tuple of str
        The class labels to keep for LDA (default ("no_stim_trial", "whisker_trial", "auditory_trial"))

    Returns
    -------
    X_sensory: (n_trials, n_features)
    y_sensory: (n_trials,)
    meta_trials: DataFrame (n_trials, meta cols)
    X_ripples: (n_ripples, n_features)
    y_ripples: (n_ripples,)
    meta_ripples: DataFrame (n_ripples, meta cols)
    """

    # Create population vectors
    new_df = build_table_population_vectors(
        df,
        window_sensory=window_sensory,
        window_ripple=window_ripple,
        substract_baseline=substract_baseline,
    )

    df_ctx = new_df[new_df.context == context_value].copy()
        
    # Filter by classes if specified
    if classes_labels is not None:
        df_ctx = df_ctx[df_ctx["trial_type"].isin(classes_labels)].copy()
    
    if df_ctx.empty:
        raise ValueError(f"No trials found for classes_labels={classes_labels}")

    # create a new column that order when the trial happen per trial type, we use trial half function to split the groups

    df_ctx = df_ctx.sort_values(["trial_type", "start_time"]).copy()

    df_ctx["trial_rank"] = df_ctx.groupby("trial_type").cumcount()
    df_ctx["n_trials_type"] = df_ctx.groupby("trial_type")["trial_type"].transform("size")

    df_ctx["trial_order_group"] = np.where(
        df_ctx["trial_rank"] < df_ctx["n_trials_type"] / 2,
        "first_half",
        "second_half")


    
    # apply a shuffle to create a null distribution of LDA, if no index it will be None

    df_ctx["shuffle_index"] = -1 if shuffle_i is None else shuffle_i  # add a column with the shuffle index for reproducibility and to keep track of it in the meta data, -1 means no shuffle
    if shuffle_i is not None: 
        rng=np.random.default_rng(seed=shuffle_i)  # create a random generator with the shuffle index as seed for reproducibility
        df_ctx['trial_type']= rng.permutation(df_ctx['trial_type'].to_numpy())
        df_ctx['shuffle_index']=shuffle_i
    
    # Define column names for sensory and ripple data based on the brain region
    sensory_col = f"{brain_region}_sensory"
    ripple_col = f"{brain_region}_ripple_content"

    # ------ Neural response during sensory stimulation ------
    X_sensory = np.stack(df_ctx[sensory_col].values)
    y_sensory = df_ctx["trial_type"].to_numpy()

    # Keep some metadata for save results later 
    meta_trials = df_ctx[["mouse","session", "start_time", "trial_type", "lick_flag", "context", "ripples_per_trial", "rewarded_group","shuffle_index","trial_order_group"]].copy()
    meta_trials["baseline_substracted"] = substract_baseline
    meta_trials['trial_index'] = df_ctx.index  # keep original index if meaningful


    # --- Ripples ---
    ripple_vectors = []
    ripple_labels = []
    ripple_meta_rows = []

    for trial_idx, row in df_ctx.iterrows():
        # row[ripple_col] is expected to be a list of vectors
        for r_i, ripple_vec in enumerate(row[ripple_col]):
            ripple_vectors.append(ripple_vec)
            ripple_labels.append(row["trial_type"])
            ripple_meta_rows.append(
                {   
                    "shuffle_index": row["shuffle_index"],
                    "trial_index": trial_idx,
                    "mouse": row["mouse"], 
                    "session": row["session"],
                    "start_time": row["start_time"],    
                    "trial_type": row["trial_type"],
                    "lick_flag": row["lick_flag"],
                    "context": row["context"],
                    "rewarded_group": row["rewarded_group"],
                    "trial_order_group": row["trial_order_group"],
                    "ripple_in_trial": r_i,
                }
            )

    if len(ripple_vectors) == 0:
        raise ValueError(f"No ripples found in {ripple_col} (region={brain_region}).")

    X_ripples = np.stack(ripple_vectors)
    y_ripples = np.array(ripple_labels)
    meta_ripples = pd.DataFrame(ripple_meta_rows)
    meta_ripples['baseline_substracted'] = substract_baseline

    return X_sensory, y_sensory, meta_trials, X_ripples, y_ripples, meta_ripples


def fit_lda(X,y):
    """
    fit LDA on the data and return the transformed data and explained variance.

    Parameters
    ----------
    X: (n_samples, n_features)
    y: (n_samples,)
    classes_labels: tuple of str
        The class labels to keep for LDA (default ("no_stim_trial", "whisker_trial", "auditory_trial"))


    Returns
    -------
    lda: fitted LDA model
    X_lda: (n_samples, n_components)
    explained_variance: (n_components,)
  
    """
    # Check if there are features to fit
    if X.shape[1]==0:
        return None, np.zeros((X.shape[0], 0)), np.array([])
    # perform the LDA 
    lda = LinearDiscriminantAnalysis()

    X_lda = lda.fit_transform(X, y)

    # Explained variance ratio
    explained_variance = lda.explained_variance_ratio_

    return lda, X_lda, explained_variance

def project_lda(lda, X):
    """
    Project new data onto the LDA components.

    Parameters
    ----------
    lda: fitted LDA model
    X: (n_samples, n_features) new data to project
    y: (n_samples,) labels for the new data (used to filter classes if needed)

    Returns
    -------
    X_lda: (n_samples, n_components) projected data
    """
    # check if lda is None (which can happen if there were no features to fit)
    if lda is None:
        return np.zeros((X.shape[0], 0))
    return lda.transform(X)

def make_lda_subtables(X_sensory_lda, meta_trials, X_ripples_to_sensory_lda, meta_ripples, X_ripples_lda, brain_region, classes_labels=None):
    """
    Build three DataFrames:
      - df_sensory_lda: trials in LDA space
      - df_ripples_lda: ripples in LDA space 
      - df_ripples_to_sensory_lda: ripples projected in the LDA space fitted only on sensory data (to check if they carry the same info)
     it will associate the coordinates form the the LDA models and the metadata of associate to the trial or the ripple
    """

    # Define column names for LDA components
    n_comp = X_sensory_lda.shape[1]
    lda_cols = [f"LD{i+1}" for i in range(n_comp)]

    # build_sensory_lda table
    df_sensory_lda = pd.DataFrame(X_sensory_lda, columns=lda_cols)
    df_sensory_lda = pd.concat([df_sensory_lda.reset_index(drop=True),
                                meta_trials.reset_index(drop=True)], axis=1)
    df_sensory_lda["classes_used"] = "|".join(classes_labels) if classes_labels is not None else None
    df_sensory_lda.index = pd.Index(meta_trials["trial_index"].values, name="trial")
    df_sensory_lda["lda_type"] = f"{brain_region}_sensory_lda" # add a column that describe the lda type 


    # build_ripples_to_sensory_lda table
    df_ripples_to_sensory_lda = pd.DataFrame(X_ripples_to_sensory_lda, columns=lda_cols)
    df_ripples_to_sensory_lda = pd.concat([df_ripples_to_sensory_lda.reset_index(drop=True),
                                meta_ripples.reset_index(drop=True)], axis=1)
    df_ripples_to_sensory_lda["classes_used"] = "|".join(classes_labels) if classes_labels is not None else None
    ripple_index = (
        meta_ripples["trial_index"].astype(str)
        + "_"
        + meta_ripples["ripple_in_trial"].astype(str))
    df_ripples_to_sensory_lda.index = pd.Index(ripple_index, name="ripple")
    df_ripples_to_sensory_lda["lda_type"] = f"{brain_region}_ripples_to_sensory_lda"


    # build_ripples_lda table
    df_ripples_lda = pd.DataFrame(X_ripples_lda, columns=lda_cols)
    df_ripples_lda = pd.concat([df_ripples_lda.reset_index(drop=True),
                                meta_ripples.reset_index(drop=True)], axis=1)
    df_ripples_lda["classes_used"] = "|".join(classes_labels) if classes_labels is not None else None
    df_ripples_lda.index = pd.Index(ripple_index, name="ripple")
    df_ripples_lda["lda_type"] = f"{brain_region}_ripples_lda"


    return df_sensory_lda, df_ripples_to_sensory_lda, df_ripples_lda


def run_lda_analysis(df, brain_region, window_sensory=0.05, window_ripple=0.05, substract_baseline=True, context_value="active", classes_labels=None, scale_data=True, shuffle_i=None):
    """
    Run the whole LDA analysis pipeline for a given brain region and baseline substraction and return the resulting DataFrames.
    Before fitting the model the popuplation vector are standardize with a a simple z-score. 
    """
    X_sensory, y_sensory, meta_trials, X_ripples, y_ripples, meta_ripples = prepare_vector_LDA(
        df,
        brain_region=brain_region,
        window_sensory=window_sensory,
        window_ripple=window_ripple,
        substract_baseline=substract_baseline,
        context_value=context_value,
        classes_labels=classes_labels,
        shuffle_i=shuffle_i
    )
    if scale_data and X_sensory.shape[1]>0:  # only scale if there are features to scale
        scaler = StandardScaler()
        # scale wiht the scaler of population vectors from sensory input 
        X_sensory = scaler.fit_transform(X_sensory)
        X_ripples = scaler.transform(X_ripples)
    model_lda, X_sensory_lda, expl_variance = fit_lda(X_sensory, y_sensory)
    X_ripples_to_sensory_lda = project_lda(model_lda, X_ripples)

    if scale_data and X_ripples.shape[1]>0:  # only scale if there are features to scale
        scaler_ripples = StandardScaler() 

        # for the fitting lda on popluation vector of ripples input, we apply a new scaller
        X_ripples = scaler_ripples.fit_transform(X_ripples)
    model_lda_ripples, X_ripples_lda, expl_variance_ripples = fit_lda(X_ripples, y_ripples)

    df_sensory_lda, df_ripples_to_sensory_lda, df_ripples_lda = make_lda_subtables(
        X_sensory_lda, meta_trials, X_ripples_to_sensory_lda, meta_ripples, X_ripples_lda, brain_region, classes_labels=classes_labels
    )

    return df_sensory_lda, df_ripples_to_sensory_lda, df_ripples_lda


def compute_centroids(
    df,
    lda_cols=["LD1", "LD2"],
    group_col="trial_type",
    meta_cols=["mouse", "baseline_substracted", "lda_type", "rewarded_group","shuffle_index"],
    classes_labels=None,
):
    """
    Compute centroids of the LDA projections for each class, separately for each
    combination of meta_cols.

    Returns
    -------
    centroids: DataFrame with columns:
      meta_cols + [group_col] + ['centroid_LD1', 'centroid_LD2']
    """

    # garder seulement les meta_cols qui existent vraiment dans df
    meta_cols = [c for c in meta_cols if c in df.columns]

    # enlever les lignes sans coordonnées LDA
    use_cols = meta_cols + [group_col] + lda_cols
    tmp = df.dropna(subset=[group_col] + lda_cols)[use_cols].copy()

     # keep only selected trial types when requested
    if classes_labels is not None:
        tmp = tmp[tmp[group_col].isin(classes_labels)].copy()
        
    centroids = (
        tmp.groupby(meta_cols + [group_col])[lda_cols]
        .mean()
        .reset_index()
        .rename(columns={c: f"centroid_{c}" for c in lda_cols})
    )

    return centroids

def compute_distances_between_centroids(centroids, lda_cols=["LD1", "LD2"], group_col="trial_type"):
    """
    Compute pairwise distances between centroids of a give mousee and a given condition (e.g. baseline substracted or not, LDA type, shuffle index) in the LDA space. 
    The distance is computed between the centroids of each trial type (e.g. no stim, auditory, whisker).

    Parameters
    ----------
    centroids: DataFrame with columns [group_col, 'centroid_LD1', 'centroid_LD2']
    lda_cols: list of str, column names for the LDA components (default ["LD1", "LD2"])
    group_col: str, column name for the class labels (default "trial_type")

    Returns
    -------
    distances: DataFrame with columns [group_col_1, group_col_2, distance]
    """
    from scipy.spatial.distance import pdist, squareform

    centroid_coords = centroids[[f"centroid_{col}" for col in lda_cols]].values
    distance_matrix = squareform(pdist(centroid_coords))
    
    # Create a DataFrame for distances that is readable
    groups = centroids[group_col].values
    distance_df = pd.DataFrame(distance_matrix, index=groups, columns=groups)
    
    # Convert to long format
    distances = distance_df.where(np.triu(np.ones(distance_df.shape), k=1).astype(bool)).stack().reset_index()
    distances.columns = [f"{group_col}_1", f"{group_col}_2", "distance"]

    # force a stable order of pair names; keep known trial order first,
    # then append any custom labels to avoid KeyError.
    trial_order = ["no_stim_trial", "auditory_trial", "whisker_trial"]
    dynamic_order = [x for x in groups if x not in trial_order]
    full_order = trial_order + dynamic_order
    order_map = {k: i for i, k in enumerate(full_order)}

    #add meta of mouse name and reward group
    distances['mouse'] = centroids['mouse'].iloc[0]
    distances['rewarded_group'] = centroids['rewarded_group'].iloc[0]
    distances['shuffle_index'] = centroids['shuffle_index'].iloc[0]

    # build a pair column with the two groups sorted by the order in trial_order
    distances["pair"] = distances.apply(
        lambda r: "-".join(
            sorted(
                [r[f"{group_col}_1"], r[f"{group_col}_2"]],
                key=lambda x: order_map[x]
            )
        ),
        axis=1
    )
    
    return distances



def centroid_distance_table_for_each_condition(centroids, lda_cols=["LD1", "LD2"],group_col='trial_type', condition_cols=['lda_type','baseline_substracted','shuffle_index']):
    """
    Compute the distances between centroids for each combination of conditions specified in condition_cols. By default it will compute the distances separately for 
    each LDA type and for baseline substracted or not and for each shuffle index.
    """
     # keep only the columns that exist 
    condition_cols = [c for c in condition_cols if c in centroids.columns]

    out = []
    for keys, sub in centroids.groupby(condition_cols, dropna=False):
        # key become a tuple if several columns
        if not isinstance(keys, tuple):
            keys = (keys,)

        # compute the centroids for this condition
        d = compute_distances_between_centroids(sub, lda_cols=lda_cols, group_col=group_col)

        # add the meta-cols 
        for col, val in zip(condition_cols, keys):
            d[col] = val

        out.append(d)

    if len(out) == 0:
        return pd.DataFrame(columns=condition_cols + [f"{group_col}_1", f"{group_col}_2", "distance", "pair"])

    return pd.concat(out, ignore_index=True)


def make_centroid_distance_table_for_all_mice(
    data_folder,
    save_path,
    lda_cols=["LD1", "LD2"],
    condition_cols=["mouse", "lda_type", "baseline_substracted", "shuffle_index"],
    in_filename="lda_big_table_all_mice.pkl",
    classes_labels=None,
    group_col="trial_type",
    out_filename="centroid_distances_all_mice_with_shuffle.pkl",
):
    """
    Compute centroid distances for all mice from the big LDA table produced by
    make_lda_big_table_all_mice.

    Parameters
    ----------
    data_folder : str or Path
        Folder containing the big LDA table generated by
        make_lda_big_table_all_mice.
    save_path : str or Path
        Folder where the output table will be saved.
    in_filename : str
        Name of the input pickle file to read from data_folder.
    lda_cols : list of str
        LDA component columns to use (default ["LD1", "LD2"]).
    condition_cols : list of str
        Columns to group by. Default groups by mouse × lda_type ×
        baseline_substracted × shuffle_index, so each shuffle iteration and
        each mouse gets its own set of pairwise distances.
    group_col : str
        Column containing the class labels (default "trial_type").
    out_filename : str
        Name of the output pickle file.

    Returns
    -------
    final_table : DataFrame
        Pairwise centroid distances for all mice and conditions.
        shuffle_index == -1  →  real data
        shuffle_index >= 0   →  null distribution
    """
    data_folder = Path(data_folder)
    in_file = data_folder / in_filename
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file}")

    big_table = pd.read_pickle(in_file)

    centroids = compute_centroids(
        big_table,
        lda_cols=lda_cols,
        group_col=group_col,
        classes_labels=classes_labels,
    )

    final_table = centroid_distance_table_for_each_condition(
        centroids,
        lda_cols=lda_cols,
        group_col=group_col,
        condition_cols=condition_cols,
    )

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    out_file = save_path / out_filename
    final_table.to_pickle(out_file)
    print(f"Saved: {out_file}")

    return final_table

def centroids_distance_plot(data_folder,save_path):
    # change name for binary plotting
    file_path = os.path.join(data_folder, "centroid_distances_all_mice_with_shuffle.pkl")
    df = pd.read_pickle(file_path)

    df= df[df["shuffle_index"]==0].copy()  # keep only real data for the plot, but we can easily change that if we want to plot the shuffle distribution too

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

    g = sns.FacetGrid(
        df.loc[df.baseline_substracted==True], # I keep only the non baseline substracted for the plot because I find it more interpretable, but I can easily change that if needed
        row="projection_type",
        col="brain_region",
        margin_titles=True,
        height=4,
        aspect=1.3,
        sharey=True
        )
    palette_reward = {
    "R+": "#2ca25f",   # vert
    "R-": "#de2d26"    # rouge
    }
    g.map_dataframe(
        sns.boxplot,
        x="pair",
        y="distance",
        order=pair_order,
        hue='rewarded_group',
        palette=palette_reward,
        hue_order=['R+','R-'],
        color='lightgrey',
        boxprops={'facecolor': 'none'}
        )

    
    g.map_dataframe(
        sns.stripplot,
        x="pair",
        y="distance",
        hue='rewarded_group',
        palette=palette_reward,
        hue_order=['R+','R-'],
        order=pair_order,
        alpha=0.8,
        dodge=True
        )
    g.set_axis_labels("Pair of trial types", "Centroid distance")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.add_legend(title="Rewarded group")

    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=30)

    g.figure.suptitle("Distances between LDA class centroids", y=1.02)
    g.figure.subplots_adjust(hspace=0.3, wspace=0.2)

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # change the name for whisker/auditory plot 
    out_file = save_path / "centroid_distances_plot44.png"

    g.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(g.figure)

    print(f"Saved: {out_file}")



def plot_centroid_results_with_shuffle(data_folder, save_path):
    """
    Plot LDA centroid distances with real data (stripplot) and shuffle mean (pointplot).
    Similar to centroids_distance_plot but adds pointplot of shuffle means beside stripplot.
    """
    file_path = os.path.join(data_folder, "centroid_distances_all_mice_with_shuffle.pkl")
    df = pd.read_pickle(file_path)

    #drop the mouse AB142
    df = df[df["mouse"] != "AB142"].copy()  # drop mouse AB142 which is an outlier with very high distances for the shuffle distribution, 
    # probably due to a problem in the data or the LDA fitting for this mouse, we will investigate that later but for now we drop it to avoid that it 
    # dominates the plot and make it unreadable. We can easily change that if we want to include it in the plot.

    # extract brain region and projection type from lda_type column
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

    # Keep only baseline_substracted == True
    df = df.loc[df["baseline_substracted"] == True].copy()

    df_real = df[df["shuffle_index"] == -1].copy()
    df_shuffle = df[df["shuffle_index"] != -1].copy()

    # Mean of shuffles per mouse / condition / pair (keep brain_region, projection_type via lda_type)
    df_shuffle_mean = (
        df_shuffle
        .groupby(
            ["mouse", "lda_type", "baseline_substracted", "rewarded_group", "pair"],
            as_index=False
        )["distance"]
        .mean()
    )
    
    # Recalculate brain_region and projection_type for shuffles (since they were lost in groupby)
    df_shuffle_mean["brain_region"] = df_shuffle_mean["lda_type"].apply(lambda x: x.split("_")[0])
    df_shuffle_mean["projection_type"] = df_shuffle_mean["lda_type"].apply(lambda x: "_".join(x.split("_")[1:]))
    df_shuffle_mean["brain_region"] = df_shuffle_mean["brain_region"].replace("second", "SSp")

    palette_reward = {
        "R+": "#2ca25f",
        "R-": "#de2d26"
    }

    g = sns.FacetGrid(
        df_real,
        row="projection_type",
        col="brain_region",
        margin_titles=True,
        height=4,
        aspect=1.3,
        sharey=False
    )

    # Stripplot for real data
    g.map_dataframe(
        sns.pointplot,
        x="pair",
        y="distance",
        hue="rewarded_group",
        palette=palette_reward,
        hue_order=['R+', 'R-'],
        order=pair_order,
        dodge=0.4,          # décale les points pour les 2 groupes
        errorbar="se",      # ou "ci", "sd"
        capsize=0.2,        # petites barres horizontales
        errwidth=1.5,
        join=False       # épaisseur des barres d'erreur
        )

    # Pointplot for shuffle means, added for each facet
    for (row_val, col_val), ax in g.axes_dict.items():
        facet_shuffle = df_shuffle_mean[
            (df_shuffle_mean["projection_type"] == row_val) &
            (df_shuffle_mean["brain_region"] == col_val)
        ].copy()
        
        if not facet_shuffle.empty:
            #I droped that line because there should not be any NaN
            #facet_shuffle = facet_shuffle.dropna(subset=["pair", "distance", "rewarded_group"]) 

            if not facet_shuffle.empty:
                sns.pointplot(
                    data=facet_shuffle,
                    x="pair",
                    y="distance",
                    hue="rewarded_group",
                    order=pair_order,
                    hue_order=["R+", "R-"],
                    palette=palette_reward,
                    dodge=0.3,
                    errorbar='se',
                    capsize=0.2,
                    errwidth=1.5,
                    markers="D",
                    scale=1.2,
                    join=False,
                    ax=ax,
                    legend=False
                )

    g.set_axis_labels("Pair of trial types", "Centroid distance")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")

    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=30)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    # Clean global legend
    legend_handles = [
        mpatches.Patch(color=palette_reward["R+"], label="R+ (Real)"),
        mpatches.Patch(color=palette_reward["R-"], label="R- (Real)"),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='black', markersize=8, label='Shuffle mean'),
    ]
    g.figure.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.03, 1.0))

    g.figure.suptitle("Distances between LDA class centroids (Real data + Shuffle mean)", y=1.02)
    g.figure.subplots_adjust(hspace=0.3, wspace=0.2)

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    out_file = save_path / "centroid_distances_plot_with_shuffle4.png"

    g.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(g.figure)

    print(f"Saved: {out_file}")

    ''''
    def shuffle_centroids_distance_distribution(save_path,data_folder):
    file_path = os.path.join(data_folder, "centroid_distances_all_mice_with_shuffle.pkl")
    df = pd.read_pickle(file_path)

    for mouse in df

    # keep only shuffle data
    df_shuffle = df[df["shuffle_index"] != -1].copy()

    # compute mean and std of distances for each pair across all shuffles and mice
    shuffle_stats = (
        df_shuffle.groupby("pair")["distance"]
        .agg(["mean", "std"])
        .reset_index()
    )

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    out_file = save_path / "shuffle_centroid_distance_distribution.pkl"
    shuffle_stats.to_pickle(out_file)
    print(f"Saved: {out_file}")
'''














