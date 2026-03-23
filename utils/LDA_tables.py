import os
from random import shuffle
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.append('/Users/nigro/Desktop/NWB_analysis')

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


def make_lda_table_for_one_mouse (df,brain_regions,window_sensory=0.05,window_ripple=0.05,classes_labels=None,shuffle_tot=None):
    '''
    Run the whole LDA analysis for all brain regions and baseline substraction and return a big table with all the results to plot for one mouse.
    If shuffle_tot is not None, it will also run the analysis with shuffled labels for a number of times equal to shuffle_tot to create a null distribution 
    of LDA results to compare with the real one. The real data table will have shuffle_index = -1, the shuffled tables will have shuffle_index 
    from 0 to shuffle_tot-1.

    '''
    # ligne ajoutée par github copilot pour éviter une erreur de variable non définie,
    #  à vérifier si c'est pertinent ou pas
    
    if classes_labels is None:
        classes_labels = ["no_stim_trial", "whisker_trial", "auditory_trial"]

    baseline_subraction = [False, True]
    tables=[]
    # we create a shuffle index but also the non shuffled version (with shuffle index None) to have a null distribution of LDA results to compare with the real one.
    #  If shuffle_tot is None, we will only have the non shuffled version, if shuffle_tot is an integer n, we will have n shuffled version with shuffle index from 0 to n-1.

    if shuffle_tot is None:
        shuffle_list = [None]
    else:
        shuffle_list = [None] + list(range(shuffle_tot))

    for sh_i in shuffle_list:
        for brain_region in brain_regions:
            for substract_baseline in baseline_subraction:
                df_sensory_lda, df_ripples_to_sensory_lda, df_ripples_lda = run_lda_analysis(
                    df,
                    brain_region=brain_region,
                    window_sensory=window_sensory,
                    window_ripple=window_ripple,
                    substract_baseline=substract_baseline,
                    classes_labels=classes_labels,
                    shuffle_i=sh_i
                )
                # extend the tables list with the three resulting tables for this brain region and baseline substraction
                tables.extend([df_sensory_lda, df_ripples_to_sensory_lda, df_ripples_lda])
    big_table = pd.concat(tables, axis=0)
    return big_table    

def _compute_lda_for_mouse_file(
    file_path,
    brain_regions,
    window_sensory,
    window_ripple,
    classes_labels,
    shuffle_tot,
):
    """Worker used by multiprocessing: load one mouse file and compute its LDA table.
    """
    df = pd.read_pickle(file_path)
    return make_lda_table_for_one_mouse(
        df=df,
        brain_regions=brain_regions,
        window_sensory=window_sensory,
        window_ripple=window_ripple,
        classes_labels=classes_labels,
        shuffle_tot=shuffle_tot,
    ) 

def make_lda_big_table_all_mice(
    data_folder,
    save_path,
    brain_regions,
    window_ripple=0.05,
    window_sensory=0.05,
    classes_labels=None,
    shuffle_tot=None,
    n_jobs=None,
    use_multiprocessing=False,
):
    """
    Build and save one big LDA table for all mice, including the LDA projections for sensory and ripple data, the metadata for each trial and ripple, 
    and the shuffle index if specified.

    Parameters
    ----------
    data_folder : str or Path
        Folder containing one pickle file per mouse.
    save_path : str or Path
        Folder where output tables will be saved.
    brain_regions : list of str
        Brain regions to include.
    window_ripple : float
        Ripple window size.
    window_sensory : float
        Sensory window size.
    classes_labels : list of str
        Trial types to keep.
    shuffle_tot : int or None
        If None: only real data.
        If int: include real data + shuffles from 0 to shuffle_tot-1.
    n_jobs : int or None
        Number of worker processes.
        If None, uses cpu_count - 1.
    use_multiprocessing : bool
        If True, parallelize by mouse file using multiprocessing.

    Returns
    -------
    final_table : DataFrame
        Concatenated LDA table across all mice.
    """
    
    if classes_labels is None:
        classes_labels = ["no_stim_trial", "whisker_trial", "auditory_trial"]

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    data_folder = Path(data_folder)
    files = sorted(data_folder.glob("*.pkl"))
    if len(files) == 0:
        raise ValueError(f"No .pkl file found in {data_folder}")

    if n_jobs is None:
        n_jobs = max(1, (mp.cpu_count() or 2) - 2)  # leave some CPUs free

    all_tables = []

    if use_multiprocessing and n_jobs > 1 and len(files) > 1:
        max_workers = min(n_jobs, len(files))
        print(f"Running LDA in parallel on {max_workers} workers...")

        results_by_file = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(
                    _compute_lda_for_mouse_file,
                    str(file_path),
                    brain_regions,
                    window_sensory,
                    window_ripple,
                    classes_labels,
                    shuffle_tot,
                ): file_path
                for file_path in files
            }

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                mouse_name = file_path.stem[:5]
                try:
                    results_by_file[file_path] = future.result()
                    print(f"Done: {mouse_name}")
                except Exception as exc:
                    raise RuntimeError(f"LDA failed for {file_path.name}: {exc}") from exc

        # Keep deterministic ordering in the final concatenation.
        all_tables = [results_by_file[file_path] for file_path in files]
    else:
        for file_path in files:
            print(" ")
            print(f"Mouse: {file_path.stem[:5]}")
            lda_table_mouse = _compute_lda_for_mouse_file(
                str(file_path),
                brain_regions,
                window_sensory,
                window_ripple,
                classes_labels,
                shuffle_tot,
            )
            all_tables.append(lda_table_mouse)

    final_table = pd.concat(all_tables, axis=0, ignore_index=False)

    out_file_all = save_path / "lda_big_table_all_mice.pkl"
    final_table.to_pickle(out_file_all)
    print(f"Saved: {out_file_all}")

    return final_table  