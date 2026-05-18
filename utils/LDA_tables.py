import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.append('/Users/nigro/Desktop/NWB_analysis')

from utils.LDA_loader import *
from pathlib import Path
import numpy as np
import pandas as pd
import itertools


def make_lda_table_for_one_mouse(df, brain_regions, window_sensory=0.05, window_ripple=0.05, classes_labels=None, shuffle_tot=None, scale_data=True, project_all_ripples=False, uniform_priors=False):
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
                try:
                    df_sensory_lda, df_ripples_to_sensory_lda, df_ripples_lda, df_all_ripples = run_lda_analysis(
                        df,
                        brain_region=brain_region,
                        window_sensory=window_sensory,
                        window_ripple=window_ripple,
                        substract_baseline=substract_baseline,
                        classes_labels=classes_labels,
                        shuffle_i=sh_i,
                        scale_data=scale_data,
                        project_all_ripples=project_all_ripples,
                        uniform_priors=uniform_priors,
                    )
                except ValueError as e:
                    print(f"  Skipping {brain_region} baseline={substract_baseline} sh={sh_i}: {e}")
                    continue
                sub = [df_sensory_lda, df_ripples_to_sensory_lda, df_ripples_lda]
                if df_all_ripples is not None:
                    sub.append(df_all_ripples)
                tables.extend(sub)
    if not tables:
        raise ValueError("No LDA results produced for any brain region / baseline combination.")
    big_table = pd.concat(tables, axis=0)
    return big_table

def make_lda_table_for_one_mouse_pairwise(df, brain_regions, window_sensory=0.05, window_ripple=0.05,
                                           classes_labels=None, shuffle_tot=None, scale_data=True,
                                           project_all_ripples=False, uniform_priors=False):
    '''
    Run the whole LDA analysis for all brain regions and all pairwise combinations of trial types,
    then return a big table with all the results.
    
    For each pair of trial types, calls make_lda_table_for_one_mouse with only the two classes of the pair.
    A column 'pair' is added to identify which pair was used (e.g. "no_stim_trial_vs_whisker_trial").
    
    If shuffle_tot is not None, shuffled versions are also computed (see make_lda_table_for_one_mouse).
    '''
    if classes_labels is None:
        classes_labels = ["no_stim_trial", "whisker_trial", "auditory_trial"]
    
    trial_order = ["no_stim_trial", "auditory_trial", "whisker_trial"]  # Define the desired order of trial types  
    classes_labels = sorted(classes_labels, key=lambda x: trial_order.index(x) if x in trial_order else len(trial_order))
    
    pairs = list(itertools.combinations(classes_labels, 2))  # 3 pairs for 3 classes
    
    tables = []
    for class_a, class_b in pairs:
        pair_label = f"{class_a}-{class_b}"
        try:
            df_pair = make_lda_table_for_one_mouse(
                df=df,
                brain_regions=brain_regions,
                window_sensory=window_sensory,
                window_ripple=window_ripple,
                classes_labels=[class_a, class_b],
                shuffle_tot=shuffle_tot,
                scale_data=scale_data,
                project_all_ripples=project_all_ripples,
                uniform_priors=uniform_priors,
            )
        except ValueError as e:
            print(f"  Skipping pair {pair_label}: {e}")
            continue
        df_pair["pair"] = pair_label
        tables.append(df_pair)

    if not tables:
        raise ValueError("No LDA results produced for any pair.")
    big_table = pd.concat(tables, axis=0, ignore_index=True)
    return big_table
    

def _compute_lda_for_mouse_file(
    file_path,
    brain_regions,
    window_sensory,
    window_ripple,
    classes_labels,
    shuffle_tot,
    pairwise=False,
    scale_data=True,
    project_all_ripples=False,
    uniform_priors=False,
):
    """Worker used by multiprocessing: load one mouse file and compute its LDA table.

    If pairwise=True, runs binary pairwise LDA (one LDA per pair of trial types)
    via make_lda_table_for_one_mouse_pairwise. Otherwise runs the standard
    multiclass LDA via make_lda_table_for_one_mouse.
    """
    df = pd.read_pickle(file_path)
    # we drop the LFP columns to save memory and speed up the computations, as they are not used for the LDA analysis. We keep only the columns with metadata and population vectors.
    lfp_cols = [c for c in df.columns if any(k in c for k in ['_lfp', '_band_lfp', '_ripple_power', 'lfp_ts', 'whisker_trace', 'whisker_speed', 'tongue_trace', 'dlc_trial_ts'])]
    df = df.drop(columns=lfp_cols)
    if pairwise:
        return make_lda_table_for_one_mouse_pairwise(
            df=df,
            brain_regions=brain_regions,
            window_sensory=window_sensory,
            window_ripple=window_ripple,
            classes_labels=classes_labels,
            shuffle_tot=shuffle_tot,
            scale_data=scale_data,
            project_all_ripples=project_all_ripples,
            uniform_priors=uniform_priors,
        )
    return make_lda_table_for_one_mouse(
        df=df,
        brain_regions=brain_regions,
        window_sensory=window_sensory,
        window_ripple=window_ripple,
        classes_labels=classes_labels,
        shuffle_tot=shuffle_tot,
        scale_data=scale_data,
        project_all_ripples=project_all_ripples,
        uniform_priors=uniform_priors,
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
    pairwise=False,
    scale_data=True,
    project_all_ripples=False,
    uniform_priors=False,
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
    pairwise : bool
        If True, run pairwise binary LDA (one LDA per pair of trial types). Otherwise, run standard multiclass LDA.

    Returns
    -------
    final_table : DataFrame
        Concatenated LDA table across all mice.
    """
    
    if classes_labels is None:
        classes_labels = ["no_stim_trial", "whisker_trial", "auditory_trial"]

    second_target=brain_regions[1]

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    data_folder = Path(data_folder)
    files = sorted(data_folder.glob("*.pkl"))
    if len(files) == 0:
        raise ValueError(f"No .pkl file found in {data_folder}")
    
    keep_files= []
    for file_path in files:
        brain_regions_in_file = file_path.stem.split("_")[3] 
        if brain_regions_in_file == second_target:
            keep_files.append(file_path)

    if len(keep_files) == 0:
        raise ValueError(f"No file found for brain region {second_target} in {data_folder}")
    if n_jobs is None:
        n_jobs = max(1, (mp.cpu_count() or 2) - 2)  # leave some CPUs free

    all_tables = []

    if use_multiprocessing and n_jobs > 1 and len(keep_files) > 1:
        max_workers = min(n_jobs, len(keep_files))
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
                    pairwise,
                    scale_data,
                    project_all_ripples,
                    uniform_priors,
                ): file_path
                for file_path in keep_files
            }

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                mouse_name = file_path.stem[:5]
                try:
                    results_by_file[file_path] = future.result()
                    print(f"Done: {mouse_name}")
                except ValueError as exc:
                    print(f"  Skipping {mouse_name}: {exc}")
                except Exception as exc:
                    raise RuntimeError(f"LDA failed for {file_path.name}: {exc}") from exc

        # Keep deterministic ordering in the final concatenation (skip files that failed).
        all_tables = [results_by_file[file_path] for file_path in keep_files if file_path in results_by_file]
    else:
        for file_path in keep_files:
            print(" ")
            print(f"Mouse: {file_path.stem[:5]}")
            try:
                lda_table_mouse = _compute_lda_for_mouse_file(
                    str(file_path),
                    brain_regions,
                    window_sensory,
                    window_ripple,
                    classes_labels,
                    shuffle_tot,
                    pairwise,
                    scale_data,
                    project_all_ripples,
                    uniform_priors,
                )
            except ValueError as exc:
                print(f"  Skipping {file_path.stem[:5]}: {exc}")
                continue
            all_tables.append(lda_table_mouse)

    if not all_tables:
        raise ValueError(f"No LDA results produced for any mouse in {data_folder}")
    final_table = pd.concat(all_tables, axis=0, ignore_index=False)

    suffix = "pairwise" if pairwise else "multiclass"
    out_file_all = save_path / f"lda_big_table_all_mice_{suffix}_{second_target}.pkl"
    final_table.to_pickle(out_file_all)
    print(f"Saved: {out_file_all}")

    return final_table  