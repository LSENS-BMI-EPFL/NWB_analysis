import os
from random import shuffle
import sys
sys.path.append('/Users/nigro/Desktop/NWB_analysis')
from sklearn.cluster import DBSCAN
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
from utils.LDA_loader import prepare_vector_LDA
import seaborn as sns



def fit_DBSCAN(X, eps=15, min_samples=5):
    '''
    Fit a DBSCAN model to the population vectors and return the cluster labels.
    Parameters:
        X : array-like, shape (n_samples, n_features)
            The input data.
        eps : float, optional, default=3
            The maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples : int, optional, default=5
            The number of samples in a neighborhood for a point to be considered as a core point.
    Returns:
        labels : array, shape (n_samples,)
            Cluster labels for each point. Noisy samples are given the label -1.
    '''
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    return labels

def run_DBSCAN_analysis(df,brain_region, window_sensory, window_ripple, substract_baseline, context_value, classes_labels, shuffle_i, eps=0.5, min_samples=5):
    '''
    Run DBSCAN analysis on the input data and print the number of clusters found.
    Parameters:
        X : array-like, shape (n_samples, n_features)
            The input data.
        eps : float, optional, default=0.5
            The maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples : int, optional, default=5
            The number of samples in a neighborhood for a point to be considered as a core point.
    '''
    # prepare the data for DBSCAN
    X_sensory, y_sensory, meta_trials, X_ripples, y_ripples, meta_ripples = prepare_vector_LDA(
        df,
        brain_region=brain_region,
        window_sensory=window_sensory,
        window_ripple=window_ripple,
        substract_baseline=substract_baseline,
        classes_labels=classes_labels,
        shuffle_i=shuffle_i
    )
    # if there are no ripples in this brain region, return None
    if X_ripples.shape[1]==0: 
        return None, None 

    # fit the model and get the labels
    X_ripples= StandardScaler().fit_transform(X_ripples)
    labels = fit_DBSCAN(X_ripples, eps=eps, min_samples=min_samples)

    # concat the labels to the meta_ripples dataframe 
    ripple_index = (
    meta_ripples["trial_index"].astype(str)
    + "_"
    + meta_ripples["ripple_in_trial"].astype(str))
    
    df_ripples_meta = meta_ripples.reset_index(drop=True).copy()
    df_ripples_meta.index = pd.Index(ripple_index, name="ripple")
    df_ripples_meta["brain_region"] = brain_region
    df_ripples_meta["classes_used"] = "|".join(classes_labels) if classes_labels is not None else None
    df_ripples_meta["cluster_labels"] = labels

    return df_ripples_meta, X_ripples

def add_tsne_coordinates(
    df_ripples_meta,
    X_ripples,
    perplexity=30,
    random_state=42,
    scale_before_tsne=True
    ):
    """
    Compute 2D t-SNE coordinates from ripple population vectors and add them
    to the ripple metadata dataframe.

    Parameters
    ----------
    df_ripples_meta : pd.DataFrame
        Metadata dataframe returned by run_DBSCAN_analysis.
    X_ripples : np.ndarray
        Ripple population vectors, shape (n_ripples, n_neurons).
    perplexity : float
        t-SNE perplexity.
    random_state : int
        Random seed for reproducibility.
    scale_before_tsne : bool
        Whether to z-score the ripple vectors before t-SNE.

    Returns
    -------
    df_tsne : pd.DataFrame
        Copy of df_ripples_meta with added columns 'tsne_1' and 'tsne_2'.
    """
    # if there are no ripples or the features were only one, return None
    if X_ripples is None or X_ripples.shape[0] < 2 or X_ripples.shape[1] < 2:
        return None

    if scale_before_tsne and X_ripples.shape[1]>0:
        X_input = StandardScaler().fit_transform(X_ripples)
    else:
        X_input = X_ripples.copy()

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto"
    )

    X_tsne = tsne.fit_transform(X_input)

    df_tsne = df_ripples_meta.copy()
    df_tsne["tsne_1"] = X_tsne[:, 0]
    df_tsne["tsne_2"] = X_tsne[:, 1]

    return df_tsne

def make_tsne_table_for_one_mouse(
    df,
    brain_regions,
    window_sensory=0.05,
    window_ripple=0.05,
    substract_baseline=True,
    classes_labels=None,
    shuffle_i=None,
    perplexity=30,
    random_state=42,
    scale_before_tsne=True,
    eps=15,
    min_samples=5
):
    """
    Compute t-SNE coordinates for one mouse and return a dataframe with the results.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the mouse data.
    brain_region : str
        Brain region to analyze.
    window_sensory : float
        Sensory window size.
    window_ripple : float
        Ripple window size.
    substract_baseline : bool
        Whether to substract baseline from the vectors.
    classes_labels : list of str or None
        Trial types to keep. If None, keep all trial types.
    shuffle_i : int or None
        Shuffle index for data shuffling. If None, use real data without shuffling.
    perplexity : float
        t-SNE perplexity.
    random_state : int
        Random seed for reproducibility.
    scale_before_tsne : bool
        Whether to z-score the ripple vectors before t-SNE.

    Returns
    -------
    df_tsne : pd.DataFrame
        DataFrame containing t-SNE coordinates and metadata.
    """

    if classes_labels is None:
        classes_labels = ["no_stim_trial", "whisker_trial", "auditory_trial"]

    baseline_subraction = [False, True]
    tables=[]

    for brain_region in brain_regions:
        for substract_baseline in baseline_subraction:
            df_ripples_meta, X_ripples = run_DBSCAN_analysis(
                df=df,
                brain_region=brain_region,
                window_sensory=window_sensory,
                window_ripple=window_ripple,
                substract_baseline=substract_baseline,
                classes_labels=classes_labels,
                shuffle_i=shuffle_i,
                context_value=None,
                eps=eps,
                min_samples=min_samples
            )
            # if there are no ripples in this brain region, or if the features were only one, skip to the next iteration
            if df_ripples_meta is None:
                continue

            df_tsne = add_tsne_coordinates(
                df_ripples_meta,
                X_ripples,
                perplexity=perplexity,
                random_state=random_state,
                scale_before_tsne=scale_before_tsne
            )
            # if there are no ripples or the features were only one, df_tsne will be None, so we skip it in that case
            if df_tsne is None:
                continue

            # extend the tables list with the three resulting tables for this brain region and baseline substraction
            tables.extend([df_tsne])
    return pd.concat(tables, axis=0)


def plot_tsne(data_folder,save_path, eps=15, min_samples=5):
    """
    Plot a 2D t-SNE embedding colored by a metadata column.

    Parameters
    ----------
    df_tsne : pd.DataFrame
        DataFrame containing 'tsne_1', 'tsne_2', and metadata columns.
    color_col : str
        Column used for coloring points.
    title : str or None
        Figure title.
    figsize : tuple
        Figure size.
    noise_label : int
        Label used for DBSCAN noise points.
    save_path : str or Path or None
        Path to save the figure.
    """

    save_path.mkdir(parents=True, exist_ok=True)
    names = os.listdir(data_folder)
    files = [os.path.join(data_folder, name) for name in names] 
    
    for file_id, file in enumerate(files):
        print(' ')
        print(f'Mouse: {names[file_id][0:5]}')
        file_path = os.path.join(data_folder, file)
        df = pd.read_pickle(file_path)

        df_tsne = make_tsne_table_for_one_mouse(
            df,
            brain_regions=['ca1','second'],
            window_sensory=0.05,
            window_ripple=0.05,
            substract_baseline=True,
            eps=eps,
            min_samples=min_samples,
            classes_labels=["no_stim_trial", "whisker_trial", "auditory_trial"],
            shuffle_i=None,
            perplexity=30,
            random_state=42,
            scale_before_tsne=True)

        g= sns.relplot(
            data=df_tsne,
            x="tsne_1",
            y="tsne_2",
            col="brain_region",
            hue="cluster_labels",
            row="baseline_substracted",
            kind="scatter",
            alpha=0.7,
            height=4,
            aspect=1,
            facet_kws={"margin_titles": True},
            palette="tab10",
        )

        g.figure.suptitle(
            f"Mouse {names[file_id][0:5]} - t-SNE colored by DBSCAN clusters\n"
            f"DBSCAN eps={eps}, min_samples={min_samples}",
            y=1.02
        )
        g.figure.subplots_adjust(top=0.88)

        save_file = Path(save_path) / f"{names[file_id][0:5]}_tsne_dbscan.png"
        g.figure.savefig(save_file, bbox_inches="tight")
        plt.close(g.figure)
        print(f"Saved: {save_file}")


