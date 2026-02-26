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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis




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
    meta_trials = df_ctx[["session", "start_time", "trial_type", "lick_flag", "context", "ripples_per_trial", "rewarded_group"]].copy()
    meta_trials.index = df_ctx.index  # keep original index if meaningful

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
    return lda.transform(X)

def make_lda_tables(X_sensory_lda, meta_trials, X_ripples_lda, meta_ripples, classes_labels=None):
    """
    Build two DataFrames:
      - df_sensory_lda: trials in LDA space
      - df_ripples_lda: ripples projected in the SAME LDA space
    """
    n_comp = X_sensory_lda.shape[1]
    lda_cols = [f"LD{i+1}" for i in range(n_comp)]

    df_sensory_lda = pd.DataFrame(X_sensory_lda, columns=lda_cols)
    df_sensory_lda = pd.concat([df_sensory_lda.reset_index(drop=True),
                                meta_trials.reset_index(drop=True)], axis=1)
    df_sensory_lda["classes_used"] = "|".join(classes_labels) if classes_labels is not None else None
    df_sensory_lda.index = pd.Index(range(len(df_sensory_lda)), name="trial")

    df_ripples_lda = pd.DataFrame(X_ripples_lda, columns=lda_cols)
    df_ripples_lda = pd.concat([df_ripples_lda.reset_index(drop=True),
                                meta_ripples.reset_index(drop=True)], axis=1)
    df_ripples_lda["classes_used"] = "|".join(classes_labels) if classes_labels is not None else None
    df_ripples_lda.index = pd.Index(range(len(df_ripples_lda)), name="ripple")

    return df_sensory_lda, df_ripples_lda