import os
import sys
sys.path.append('/Users/nigro/Desktop/NWB_analysis')
import pathlib
import subprocess
from pathlib import Path
import scipy as sci
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface as si
import spikeinterface.preprocessing as sip
from sklearn.manifold import TSNE
from nwb_utils.utils_misc import find_nearest
from utils.lfp_utils import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler




def prepare_vector_LDA(
    df,
    brain_region: str,
    window_sensory=0.05,
    window_ripple=0.05,
    substract_baseline=True,
    context_value="active",
    classes_labels=None
    
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
    
    # Define column names for sensory and ripple data based on the brain region
    sensory_col = f"{brain_region}_sensory"
    ripple_col = f"{brain_region}_ripple_content"

    # Neural response during sensory stimulation
    X_sensory = np.stack(df_ctx[sensory_col].values)
    y_sensory = df_ctx["trial_type"].to_numpy()

    # Keep some metadata for save results later 
    meta_trials = df_ctx[["mouse","session", "start_time", "trial_type", "lick_flag", "context", "ripples_per_trial", "rewarded_group"]].copy()
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
                    "trial_index": trial_idx,
                    "mouse": row["mouse"], 
                    "session": row["session"],
                    "start_time": row["start_time"],    
                    "trial_type": row["trial_type"],
                    "lick_flag": row["lick_flag"],
                    "context": row["context"],
                    "rewarded_group": row["rewarded_group"],
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
    df_sensory_lda["lda_type"] = f"{brain_region}_sensory_lda"

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


def run_lda_analysis(df, brain_region, window_sensory=0.05, window_ripple=0.05, substract_baseline=True, context_value="active", classes_labels=None, scale_data=True):
    """
    Run the whole LDA analysis pipeline for a given brain region and baseline substraction and return the resulting DataFrames.
    """
    X_sensory, y_sensory, meta_trials, X_ripples, y_ripples, meta_ripples = prepare_vector_LDA(
        df,
        brain_region=brain_region,
        window_sensory=window_sensory,
        window_ripple=window_ripple,
        substract_baseline=substract_baseline,
        context_value=context_value,
        classes_labels=classes_labels
    )
    if scale_data and X_sensory.shape[1]>0:  # only scale if there are features to scale
        scaler = StandardScaler()
        X_sensory = scaler.fit_transform(X_sensory)
        X_ripples = scaler.transform(X_ripples)
    model_lda, X_sensory_lda, expl_variance = fit_lda(X_sensory, y_sensory)
    X_ripples_to_sensory_lda = project_lda(model_lda, X_ripples)
    if scale_data and X_ripples.shape[1]>0:  # only scale if there are features to scale
        scaler_ripples = StandardScaler()
        X_ripples = scaler_ripples.fit_transform(X_ripples)
    model_lda_ripples, X_ripples_lda, expl_variance_ripples = fit_lda(X_ripples, y_ripples)

    df_sensory_lda, df_ripples_to_sensory_lda, df_ripples_lda = make_lda_subtables(
        X_sensory_lda, meta_trials, X_ripples_to_sensory_lda, meta_ripples, X_ripples_lda, brain_region, classes_labels=classes_labels
    )

    return df_sensory_lda, df_ripples_to_sensory_lda, df_ripples_lda

def make_lda_table_for_plot (df,brain_regions,window_sensory=0.05,window_ripple=0.05,classes_labels=["no_stim_trial", "whisker_trial", "auditory_trial"]):
    '''
    Run the whole LDA analysis for all brain regions and baseline substraction and return a big table with all the results to plot for one mouse .
    '''

    baseline_subraction = [False, True]
    tables=[]

    for brain_region in brain_regions:
        for substract_baseline in baseline_subraction:
            df_sensory_lda, df_ripples_to_sensory_lda, df_ripples_lda = run_lda_analysis(
                df,
                brain_region=brain_region,
                window_sensory=window_sensory,
                window_ripple=window_ripple,
                substract_baseline=substract_baseline,
                classes_labels=classes_labels
            )
            # extend the tables list with the three resulting tables for this brain region and baseline substraction
            tables.extend([df_sensory_lda, df_ripples_to_sensory_lda, df_ripples_lda])
    big_table = pd.concat(tables, axis=0)
    return big_table       

def plot_lda_results(data_folder,save_path, brain_regions, window_ripple=0.05, window_sensory=0.05, classes_labels=None):
    '''
    Make the plots 2x3x3 figures for the LDA results.
    
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
        lda_table = make_lda_table_for_plot(df, brain_regions=brain_regions, window_sensory=window_sensory, window_ripple=window_ripple, classes_labels=classes_labels)

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
            col="lda_type",
            row="baseline_substracted",
            kind="scatter",
            alpha=0.7,
            height=4,
            aspect=1,
            facet_kws={"margin_titles": True},
            palette=palette
        )

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
        lda_table = make_lda_table_for_plot(df, brain_regions=brain_regions, window_sensory=window_sensory, window_ripple=window_ripple, classes_labels=classes_labels)

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

def compute_centroids(
    df,
    lda_cols=["LD1", "LD2"],
    group_col="trial_type",
    meta_cols=["mouse", "baseline_substracted", "lda_type", "rewarded_group", "lick_flag"],
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

    centroids = (
        tmp.groupby(meta_cols + [group_col])[lda_cols]
        .mean()
        .reset_index()
        .rename(columns={c: f"centroid_{c}" for c in lda_cols})
    )

    return centroids

def compute_distances_between_centroids(centroids, lda_cols=["LD1", "LD2"], group_col="trial_type"):
    """
    Compute pairwise distances between centroids.

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

    # force the order of the distances names 
    trial_order = ["no_stim_trial", "auditory_trial", "whisker_trial"]
    order_map = {k: i for i, k in enumerate(trial_order)}

    #add meta of mouse name and reward group
    distances['mouse'] = centroids['mouse'].iloc[0]
    distances['rewarded_group'] = centroids['rewarded_group'].iloc[0]

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



def centroid_distance_table_for_each_condition(centroids, lda_cols=["LD1", "LD2"],group_col='trial_type', condition_cols=['lda_type','baseline_substracted']):
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


def make_centroid_distance_table_for_all_mice(data_folder, brain_regions=['ca1','second'], lda_cols=["LD1", "LD2"], categories_conditons=['lda_type','baseline_substracted'], group_col='trial_type'):
      #read files and loop through them
    names = os.listdir(data_folder)
    files = [os.path.join(data_folder, name) for name in names] 
    all_centroids_distances=[]
    for file_id, file in enumerate(files):
        print(' ')
        print(f'Mouse: {names[file_id][0:5]}')
        file_path = os.path.join(data_folder, file)
        df = pd.read_pickle(file_path)
        lda_table = make_lda_table_for_plot(df, brain_regions=brain_regions)
        centroids = compute_centroids(lda_table, lda_cols=lda_cols, group_col=group_col)
        centroids_distances=centroid_distance_table_for_each_condition(centroids, lda_cols=lda_cols, group_col=group_col, condition_cols=categories_conditons)
        all_centroids_distances.append(centroids_distances)

    return pd.concat(all_centroids_distances, axis=0, ignore_index=True)




