
import sys
sys.path.append('/Users/nigro/Desktop/NWB_analysis')
import numpy as np
import os
import pandas as pd
from pathlib import Path
from utils.LDA_loader import prepare_vector_LDA
from utils.lfp_utils import make_bhv_block_table
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

def correlate_with_class_means(X, X_ref, y_ref, classes_labels):
    """
    Compute Pearson correlation between each row of X and the mean population
    vector of each class computed from X_ref.

    Parameters
    ----------
    X : (n_samples, n_features)   — vectors to correlate (e.g. ripples)
    X_ref : (n_ref, n_features)   — reference data used to compute class means (e.g. sensory trials)
    y_ref : (n_ref,)              — class labels for X_ref
    classes_labels : list of str  — ordered list of class names

    Returns
    -------
    corr : (n_samples, n_classes) — Pearson r for each sample × class mean
    """
    # Mean vector per class  →  (n_classes, n_features)
    mean_vectors = np.stack([X_ref[y_ref == cls].mean(axis=0) for cls in classes_labels])

    # Vectorised Pearson correlation
    X_c = X - X.mean(axis=1, keepdims=True) # we center each sample of X 
    M_c = mean_vectors - mean_vectors.mean(axis=1, keepdims=True) # we center each class mean vector

    num = X_c @ M_c.T                                                    # (n_samples, n_classes)
    denom = (np.linalg.norm(X_c, axis=1, keepdims=True)
             * np.linalg.norm(M_c, axis=1, keepdims=True).T)            # (n_samples, n_classes)

    return np.where(denom > 0, num / denom, 0.0)


def compute_ripple_sensory_correlations(df, brain_region, window_sensory=0.05, window_ripple=0.05,
                                        substract_baseline=True, context_value="active",
                                        classes_labels=None, scale_data=True):
    """
    For each ripple, compute Pearson correlation with the mean sensory population
    vector of each class.

    If the brain region has no neurons (0 features), returns a NaN matrix so the
    ripple rows are still present in the final table.

    Returns
    -------
    corr_matrix : (n_ripples, n_classes) — NaN when no neural data for this region
    meta_ripples : DataFrame with ripple metadata
    classes_labels : list of str (the classes used)
    """
    if classes_labels is None:
        classes_labels = ["no_stim_trial", "whisker_trial", "auditory_trial"]

    X_sensory, y_sensory, _, X_ripples, _, meta_ripples = prepare_vector_LDA(
        df,
        brain_region=brain_region,
        window_sensory=window_sensory,
        window_ripple=window_ripple,
        substract_baseline=substract_baseline,
        context_value=context_value,
        classes_labels=classes_labels,
    )

    # No neurons recorded in this region → return NaN correlations
    if X_sensory.shape[1] == 0:
        corr_matrix = np.full((len(meta_ripples), len(classes_labels)), np.nan)
        return corr_matrix, meta_ripples, classes_labels

    if scale_data:
        scaler = StandardScaler()
        X_sensory = scaler.fit_transform(X_sensory)
        X_ripples = scaler.transform(X_ripples)

    corr_matrix = correlate_with_class_means(X_ripples, X_sensory, y_sensory, classes_labels)

    return corr_matrix, meta_ripples, classes_labels

def compute_ripple_spindle_coupling_proba(meta_ripples, block_size=300, cooccurrence_window=0.75):

    meta_ripples['block_index'] = meta_ripples['ripple_times'] // block_size

    # transform on the specific column: each group element x is a Series of lags for one block
    meta_ripples['cooccurrence_prob'] = meta_ripples.groupby('block_index')['spindle_coupling_lags'].transform(
        lambda x: np.nansum(np.abs(x) <= cooccurrence_window) / len(x)
    )

    return meta_ripples

def make_ripple_correlation_table_for_one_mouse(df, brain_regions, window_sensory=0.05, window_ripple=0.05,
                                                  context_value="active",
                                                 classes_labels=None, scale_data=True):
    """
    Run compute_ripple_sensory_correlations for all brain regions for one mouse
    and return a single concatenated DataFrame.
    """
    if classes_labels is None:
        classes_labels = ["no_stim_trial", "whisker_trial", "auditory_trial"]
    substract_baseline =[True,False]  # we compute correlations with and without baseline substraction

    tables = []
    for brain_region in brain_regions:
        for substract in substract_baseline:
            corr_matrix, meta_ripples, cls = compute_ripple_sensory_correlations(
                df,
                brain_region=brain_region,
                window_sensory=window_sensory,
                window_ripple=window_ripple,
                substract_baseline=substract,
                context_value=context_value,
                classes_labels=classes_labels,
                scale_data=scale_data,
            )

            # compute also the ripple-spindle coupling probability for each ripple (based on its block)
            meta_ripples = compute_ripple_spindle_coupling_proba(meta_ripples, block_size=300, cooccurrence_window=0.75)
            
            # Build correlation columns
            corr_df = pd.DataFrame(corr_matrix, columns=[f"corr_{c}" for c in cls])
            table = pd.concat([meta_ripples.reset_index(drop=True), corr_df], axis=1)
            table["brain_region"] = brain_region
            table["baseline_substracted"] = substract
            tables.append(table)


    return pd.concat(tables, axis=0, ignore_index=True)


def make_ripple_correlation_table_all_mice(data_folder, save_path, brain_regions,
                                            window_sensory=0.05, window_ripple=0.05,
                                            context_value="active",
                                            classes_labels=None, scale_data=True,
                                            out_filename="ripple_correlation_table_all_mice.pkl"):
    """
    Build and save a table of ripple–sensory correlations for all mice.

    For each mouse pickle file in data_folder, computes the Pearson correlation
    between each ripple population vector and the mean sensory population vector
    of each class, for every brain region.

    Parameters
    ----------
    data_folder : str or Path
    save_path : str or Path
    brain_regions : list of str
    out_filename : str

    Returns
    -------
    final_table : DataFrame with columns:
        ripple metadata | corr_<class> (one per class) | brain_region | baseline_substracted | mouse
    """
    if classes_labels is None:
        classes_labels = ["no_stim_trial", "whisker_trial", "auditory_trial"]

    data_folder = Path(data_folder)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    files = sorted(data_folder.glob("*.pkl"))
    if len(files) == 0:
        raise ValueError(f"No .pkl file found in {data_folder}")

    all_tables = []
    for file_path in files:
        mouse_name = file_path.stem[:5]
        print(f"Mouse: {mouse_name}")
        df = pd.read_pickle(file_path)
        try:
            table = make_ripple_correlation_table_for_one_mouse(
                df,
                brain_regions=brain_regions,
                window_sensory=window_sensory,
                window_ripple=window_ripple,
                context_value=context_value,
                classes_labels=classes_labels,
                scale_data=scale_data,
            )
            all_tables.append(table)
        except Exception as exc:
            print(f"  Skipped {mouse_name}: {exc}")

    final_table = pd.concat(all_tables, axis=0, ignore_index=True)

    out_file = save_path / out_filename
    final_table.to_pickle(out_file)
    print(f"Saved: {out_file}")

    return final_table

def plot_correlation_table(data_folder, save_path,
                           in_filename="ripple_correlation_table_all_mice.pkl"):
    """
    For each mouse in the correlation table, make one figure with 4 3-D scatter
    subplots (2 brain regions × with/without baseline subtraction).

    Axes:
        X  corr_no_stim_trial
        Y  corr_whisker_trial
        Z  corr_auditory_trial

    The figure is saved as  <save_path>/<mouse>_ripple_correlations_3D.png
    """
    file = Path(data_folder) / in_filename
    big_table = pd.read_pickle(file)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    classes_labels = ["no_stim_trial", "whisker_trial", "auditory_trial"]
    corr_cols      = [f"corr_{c}" for c in classes_labels]
    axis_labels    = ["No-stim corr.", "Whisker corr.", "Auditory corr."]

    # colour maps for the 3 classes
    class_colors = {
        "no_stim_trial":  "grey",
        "whisker_trial":  "gold",
        "auditory_trial": "steelblue",
    }

    mouse_list    = big_table["mouse"].unique()
    brain_regions = big_table["brain_region"].unique()

    for mouse in mouse_list:
        df_mouse = big_table[big_table["mouse"] == mouse]

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f"Mouse: {mouse}  —  Ripple–sensory correlations (3-D)", fontsize=14)

        subplot_idx = 1
        for region in brain_regions:
            for baseline in [True, False]:
                ax = fig.add_subplot(2, 2, subplot_idx, projection="3d")
                subtitle = (f"{region}  |  "
                            f"{'baseline subtracted' if baseline else 'no baseline subtraction'}")
                ax.set_title(subtitle, fontsize=9, pad=6)

                mask = (df_mouse["brain_region"] == region) & \
                       (df_mouse["baseline_substracted"] == baseline)
                sub = df_mouse[mask]

                if sub.empty or sub[corr_cols].isna().all().all():
                    ax.text(0, 0, 0, "no data", ha="center", va="center", color="gray")
                    ax.set_xlabel(axis_labels[0], fontsize=7)
                    ax.set_ylabel(axis_labels[1], fontsize=7)
                    ax.set_zlabel(axis_labels[2], fontsize=7)
                    subplot_idx += 1
                    continue

                x = sub[corr_cols[0]].values
                y = sub[corr_cols[1]].values
                z = sub[corr_cols[2]].values

                # colour each point by its most-correlated class
                best_class = sub[corr_cols].idxmax(axis=1).map(
                    lambda col: col.replace("corr_", "")
                )
                colors = best_class.map(class_colors).values

                ax.scatter(x, y, z, c=colors, s=15, alpha=0.55, edgecolors="none")

                # diagonal line (equal-correlation reference)
                lim = max(np.nanmax(np.abs([x, y, z])), 0.01)
                ax.plot([-lim, lim], [-lim, lim], [-lim, lim],
                        color="grey", linewidth=0.8, linestyle="--", alpha=0.5)

                ax.set_xlabel(axis_labels[0], fontsize=7)
                ax.set_ylabel(axis_labels[1], fontsize=7)
                ax.set_zlabel(axis_labels[2], fontsize=7)
                ax.tick_params(labelsize=6)

                subplot_idx += 1

        # shared legend
        legend_patches = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=col, markersize=8, label=cls.replace("_", " "))
            for cls, col in class_colors.items()
        ]
        fig.legend(handles=legend_patches, title="Dominant class",
                   loc="lower center", ncol=3, fontsize=9, title_fontsize=9,
                   frameon=False, bbox_to_anchor=(0.5, 0.01))

        plt.tight_layout(rect=[0, 0.06, 1, 0.97])

        out_file = save_path / f"{mouse}_ripple_correlations_3D.png"
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_file}")

def plot_correlation_table_in_time(data_folder, save_path, in_filename="ripple_correlation_table_all_mice.pkl",
                                   lowess_frac=0.3):
    """
    For each mouse, scatter raw correlation values over ripple time with a LOWESS curve overlay,
    one subplot per brain_region × baseline_substracted, one hue per class.
    """

    file = Path(data_folder) / in_filename
    big_table = pd.read_pickle(file)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    classes_labels = ["no_stim_trial", "whisker_trial", "auditory_trial"]
    corr_cols      = [f"corr_{c}" for c in classes_labels]

    class_colors = {
        "no_stim_trial":  "grey",
        "whisker_trial":  "gold",
        "auditory_trial": "steelblue",
    }

    id_cols = [c for c in big_table.columns if c not in corr_cols]
    long_table = big_table.melt(
        id_vars=id_cols,
        value_vars=corr_cols,
        var_name="class",
        value_name="correlation",
    )
    long_table["class"] = long_table["class"].str.replace("corr_", "", regex=False)

    # Compute LOWESS per (mouse, brain_region, baseline_substracted, class)
    long_table = long_table.sort_values("ripple_times")
    smoothed_vals = np.full(len(long_table), np.nan)
    group_keys = ["mouse", "brain_region", "baseline_substracted", "class"]
    for _, grp in long_table.groupby(group_keys):
        if len(grp) < 4:
            continue
        idx = grp.index
        smoothed = sm_lowess(grp["correlation"].values, grp["ripple_times"].values,
                             frac=lowess_frac, return_sorted=False)
        smoothed_vals[long_table.index.get_indexer(idx)] = smoothed
    long_table["corr_lowess"] = smoothed_vals

    for mouse in big_table["mouse"].unique():
        df_mouse = long_table[long_table["mouse"] == mouse]

        g = sns.FacetGrid(
            df_mouse,
            col="brain_region",
            col_order= ["ca1", "second"],
            row="baseline_substracted",
            sharey=False, sharex=False,
            height=3.5, aspect=1.6,
        )
        g.map_dataframe(
            sns.scatterplot,
            x="ripple_times", y="correlation",
            hue="class", hue_order=classes_labels,
            palette=class_colors, s=10, alpha=0.3, legend=False,
        )
        g.map_dataframe(
            sns.lineplot,
            x="ripple_times", y="corr_lowess",
            hue="class", hue_order=classes_labels,
            palette=class_colors, linewidth=2,
        )
        g.add_legend(title="Class")
        g.set_axis_labels("Ripple time (s)", "Pearson r")
        g.figure.suptitle(f"Mouse: {mouse}  —  Ripple–sensory correlations over time", y=1.02)

        out_file = save_path / f"{mouse}_ripple_correlations_time.png"
        g.figure.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(g.figure)
        print(f"  Saved: {out_file}")

def plot_correlation_table_in_time_with_bhv(data_folder_corr_table, data_folder_ripples, save_path,
                                             in_filename="ripple_correlation_table_all_mice.pkl",
                                             lowess_frac=0.3, block_size=20):
    """
    For each mouse: 3 stacked subplots sharing the x-axis (time).
      ax0 — CA1 ripple–sensory correlations (baseline subtracted, LOWESS overlay)
      ax1 — SSp ripple–sensory correlations (baseline subtracted, LOWESS overlay)
      ax2 — behavioural hit-rate per block
    """

    file = Path(data_folder_corr_table) / in_filename
    big_table = pd.read_pickle(file)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    names_ripples_table = os.listdir(data_folder_ripples)

    classes_labels = ["no_stim_trial", "whisker_trial", "auditory_trial"]
    corr_cols      = [f"corr_{c}" for c in classes_labels]

    id_cols = [c for c in big_table.columns if c not in corr_cols]
    long_table = big_table.melt(
        id_vars=id_cols,
        value_vars=corr_cols,
        var_name="class",
        value_name="correlation",
    )
    long_table["class"] = long_table["class"].str.replace("corr_", "", regex=False)

    long_table = long_table.sort_values("ripple_times")
    smoothed_vals = np.full(len(long_table), np.nan)
    group_keys = ["mouse", "brain_region", "baseline_substracted", "class"]
    for _, grp in long_table.groupby(group_keys):
        if len(grp) < 4:
            continue
        idx = grp.index
        smoothed = sm_lowess(grp["correlation"].values, grp["ripple_times"].values,
                             frac=lowess_frac, return_sorted=False)
        smoothed_vals[long_table.index.get_indexer(idx)] = smoothed
    long_table["corr_lowess"] = smoothed_vals

    # Tag whisker trials with rewarded_group before the per-mouse loop
    mask_whisker = long_table["class"] == "whisker_trial"
    long_table.loc[mask_whisker, "class"] = (
        "whisker_trial_" + long_table.loc[mask_whisker, "rewarded_group"]
    )

    # Updated labels/colors to reflect the split whisker classes
    corr_classes_rplus  = ["no_stim_trial", "whisker_trial_R+", "auditory_trial"]
    corr_classes_rminus = ["no_stim_trial", "whisker_trial_R-", "auditory_trial"]
    corr_colors_rplus  = {"no_stim_trial": "black", "whisker_trial_R+": "green",  "auditory_trial": "royalblue"}
    corr_colors_rminus = {"no_stim_trial": "black", "whisker_trial_R-": "red",    "auditory_trial": "royalblue"}

    for mouse in big_table["mouse"].unique():

        # Determine rewarded_group for this mouse (scalar)
        rewarded_group = big_table.loc[big_table["mouse"] == mouse, "rewarded_group"].iloc[0]
        is_rplus = rewarded_group == "R+"
        mouse_classes = corr_classes_rplus  if is_rplus else corr_classes_rminus
        mouse_colors  = corr_colors_rplus   if is_rplus else corr_colors_rminus

        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        fig.suptitle(f"Mouse: {mouse}  —  Ripple–sensory correlations + behaviour", fontsize=12)

        # ── correlation panels (baseline subtracted only) ──────────────────
        df_mouse = long_table[(long_table["mouse"] == mouse) & (long_table["baseline_substracted"] == True)]

        regions = df_mouse["brain_region"].unique()
        for ax, region in zip([ax0, ax1], regions):
            sub = df_mouse[df_mouse["brain_region"] == region]
            sns.scatterplot(data=sub, x="ripple_times", y="correlation",
                            hue="class", hue_order=mouse_classes,
                            palette=mouse_colors, s=10, alpha=0.3,
                            legend=False, ax=ax)
            sns.lineplot(data=sub, x="ripple_times", y="corr_lowess",
                         hue="class", hue_order=mouse_classes,
                         palette=mouse_colors, linewidth=2,
                         legend=False, ax=ax)
            ax.set_title(region, fontsize=9)
            ax.set_ylabel("Pearson r", fontsize=8)
            ax.set_xlabel("")

        # shared legend for the correlation classes
        legend_handles = [
            plt.Line2D([0], [0], color=mouse_colors[cls], linewidth=2,
                       label=cls.replace("_", " "))
            for cls in mouse_classes
        ]
        ax0.legend(handles=legend_handles, title="Class", fontsize=8,
                   title_fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1),
                   frameon=False)

        # ── behaviour panel ─────────────────────────────────────────────────
        df_ripple = None
        for name in names_ripples_table:
            if mouse == name[:5]:
                df_ripple = pd.read_pickle(os.path.join(data_folder_ripples, name))
                break

        if df_ripple is not None:
            block_m_df, hr_w_col = make_bhv_block_table(df_ripple, is_rplus, block_size)

            bhv_items = {hr_w_col: ("green" if is_rplus else "red"), "hr_n": "black", "hr_a": "royalblue"}
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

        out_file = save_path / f"{mouse}_ripple_correlations_time_bhv.png"
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_file}")

    












