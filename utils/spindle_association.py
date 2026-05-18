
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
from tqdm import tqdm


def compute_ripple_spindle_coupling_proba(meta_ripples, block_size=300, cooccurrence_window=0.05):
    '''

    '''

    meta_ripples['block_index'] = meta_ripples['ripple_times'] // block_size

    # transform on the specific column: each group element x is a Series of lags for one block
    meta_ripples['cooccurrence_prob'] = meta_ripples.groupby('block_index')['spindle_coupling_lags'].transform(
        lambda x: np.nansum(np.abs(x) <= cooccurrence_window) / len(x)
    )

    return meta_ripples

def filter_ripples_with_spindle_coupling(ripple_table, cooccurrence_window=0.05):
    df = ripple_table.copy()
    df['spindle_coupling'] = (
        df['spindle_coupling_lags'].notnull() &
        (df['spindle_coupling_lags'].abs() <= cooccurrence_window)
    )
    return df

def plot_mean_spindle_coupling_per_region(data_folder, save_path, second_target):
    """
    For each brain region in the correlation table, plot the distribution of
    mean ripple–spindle coupling probability across mice (one point per mouse).

    cooccurrence_prob is independent of baseline subtraction, so we keep only
    baseline_substracted=True to avoid counting each ripple twice.

    Saved under <save_path>/<second_target>/spindle_coupling/
    """
    in_filename = f"ripple_correlation_table_all_mice_{second_target}.pkl"
    file = Path(data_folder) / in_filename
    big_table = pd.read_pickle(file)

    save_path = Path(save_path) / second_target / "spindle_coupling"
    save_path.mkdir(parents=True, exist_ok=True)

    # one baseline version is enough — cooccurrence_prob doesn't depend on it
    df = big_table[big_table["baseline_substracted"] == True].copy()

    mean_coupling = (
        df.groupby(["mouse", "brain_region"])["cooccurrence_prob"]
        .mean()
        .reset_index()
        .rename(columns={"cooccurrence_prob": "mean_coupling_prob"})
    )

    brain_regions = mean_coupling["brain_region"].unique()

    fig, ax = plt.subplots(figsize=(max(5, len(brain_regions) * 1.5), 5))

    sns.stripplot(
        data=mean_coupling, x="brain_region", y="mean_coupling_prob",
        order=sorted(brain_regions),
        ax=ax, size=8, jitter=True, color="steelblue", alpha=0.8,
    )
    sns.pointplot(
        data=mean_coupling, x="brain_region", y="mean_coupling_prob",
        order=sorted(brain_regions),
        ax=ax, color="black", markers="D", linestyles="none",
        errorbar="se", capsize=0.1,
    )

    ax.set_xlabel("Brain region")
    ax.set_ylabel("Mean coupling probability")
    ax.set_title(f"Ripple–spindle coupling per region  (target: {second_target})")
    plt.tight_layout()

    out_file = save_path / f"mean_spindle_coupling_per_region_{second_target}.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")

def plot_distibution_spindle_lags_per_region(data_folder, save_path, second_target):
    """
    For a given brain_region, plot the KDE distribution of ripple–spindle coupling lags,
    one curve per mouse.

    spindle_coupling_lags is the delay (s) between each ripple and its nearest spindle
    in the secondary region. NaN = no spindle detected for that trial.

    Saved under <save_path>/<second_target>/spindle_coupling/spindle_lags_<brain_region>.png
    """
    in_filename = f"ripple_correlation_table_all_mice_{second_target}.pkl"
    file = Path(data_folder) / in_filename
    big_table = pd.read_pickle(file)

    save_path = Path(save_path) / second_target / "spindle_coupling"
    save_path.mkdir(parents=True, exist_ok=True)

    # one baseline version avoids counting each ripple twice
    df = big_table[
        (big_table["baseline_substracted"] == True) &
        (big_table["brain_region"] == second_target)
    ].copy()

    # coerce None → NaN, then drop
    df["spindle_coupling_lags"] = pd.to_numeric(df["spindle_coupling_lags"], errors="coerce")
    df = df.dropna(subset=["spindle_coupling_lags"])

    if df.empty:
        print(f"No spindle lag data in {second_target}")
        return

    mice = df["mouse"].unique()
    palette = sns.color_palette("tab10", n_colors=len(mice))

    fig, ax = plt.subplots(figsize=(8, 5))

    for mouse, color in zip(mice, palette):
        lags = df.loc[df["mouse"] == mouse, "spindle_coupling_lags"].values
        sns.kdeplot(lags, ax=ax, label=mouse, color=color, linewidth=1.8)

    all_lags = df["spindle_coupling_lags"].values
    uniform_density = 1.0 / (all_lags.max() - all_lags.min())
    ax.axhline(uniform_density, color="red", linestyle="--", linewidth=1.2, label="α = 5 % (uniform)")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(axis="x", which="minor", length=3, labelbottom=False)
    ax.set_xlabel("Ripple–spindle lag (s)")
    ax.set_ylabel("Density")
    ax.set_title(f"Spindle coupling lag distribution — {second_target}")
    ax.legend(title="Mouse", fontsize=8, title_fontsize=8, frameon=False)

    plt.tight_layout()
    out_file = save_path / f"spindle_lags_{second_target}.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_mean_spindle_lag_distribution_all_regions(data_folder, save_path,
                                                    bw=0.15, n_bootstrap=1000, n_grid=500):
    """
    One curve per mouse = mean of per-region KDEs (gaussian_kde, bw=0.15).
    Global 95% CI band from bootstrap: resample mice with replacement 1000×,
    compute mean KDE across sampled mice, CI = percentile 2.5 / 97.5.

    Loads all ripple_correlation_table_all_mice_*.pkl files in data_folder.
    Saved under <save_path>/spindle_lags_by_mouse.png
    """
    from scipy.stats import gaussian_kde as _gkde

    data_folder = Path(data_folder)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    files = sorted(data_folder.glob("ripple_correlation_table_all_mice_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No correlation tables found in {data_folder}")

    tables = []
    for f in files:
        df = pd.read_pickle(f)
        df = df[df["baseline_substracted"] == True].copy()
        df["spindle_coupling_lags"] = pd.to_numeric(df["spindle_coupling_lags"], errors="coerce")
        tables.append(df)

    big = pd.concat(tables, axis=0, ignore_index=True).dropna(subset=["spindle_coupling_lags"])
    if big.empty:
        print("No spindle lag data found.")
        return

    lag_min = big["spindle_coupling_lags"].min()
    lag_max = big["spindle_coupling_lags"].max()
    grid = np.linspace(lag_min, lag_max, n_grid)

    def kde_on_grid(lags):
        if len(lags) < 2:
            return np.zeros(n_grid)
        vals = _gkde(lags, bw_method=bw)(grid)
        vals /= np.trapezoid(vals, grid)
        return vals

    mice = sorted(big["mouse"].unique())
    palette = sns.color_palette("tab10", n_colors=len(mice))

    # per-mouse mean KDE across regions
    mouse_kdes = {}
    for mouse in mice:
        df_m = big[big["mouse"] == mouse]
        region_kdes = [
            kde_on_grid(df_m.loc[df_m["brain_region"] == r, "spindle_coupling_lags"].values)
            for r in df_m["brain_region"].unique()
        ]
        region_kdes = [k for k in region_kdes if k.sum() > 0]
        if region_kdes:
            mouse_kdes[mouse] = np.stack(region_kdes).mean(axis=0)

    if not mouse_kdes:
        print("No valid KDEs computed.")
        return

    # bootstrap CI: resample mice with replacement
    rng = np.random.default_rng(42)
    kde_matrix = np.stack(list(mouse_kdes.values()))   # (n_mice, n_grid)
    n_mice = len(kde_matrix)
    boot_means = np.stack([
        kde_matrix[rng.integers(0, n_mice, size=n_mice)].mean(axis=0)
        for _ in range(n_bootstrap)
    ])
    ci_low  = np.percentile(boot_means, 2.5,  axis=0)
    ci_high = np.percentile(boot_means, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))

    # individual mouse curves
    for mouse, color in zip(mice, palette):
        if mouse not in mouse_kdes:
            continue
        n_regions = big.loc[big["mouse"] == mouse, "brain_region"].nunique()
        ax.plot(grid, mouse_kdes[mouse], color=color, linewidth=1.6,
                label=f"{mouse} (n={n_regions} regions)")

    # global mean + bootstrap CI
    global_mean = kde_matrix.mean(axis=0)
    ax.plot(grid, global_mean, color="black", linewidth=2.2, label="Mean")
    ax.fill_between(grid, ci_low, ci_high, color="black", alpha=0.15, label="95% CI (bootstrap)")

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(axis="x", which="minor", length=3, labelbottom=False)
    ax.set_xlabel("Ripple–spindle lag (s)")
    ax.set_ylabel("Density")
    ax.set_title("Spindle–SWR coupling by mouse (bootstrap 95% CI across mice)")
    ax.legend(title="Mouse", fontsize=8, title_fontsize=8, frameon=False)

    plt.tight_layout()
    out_file = save_path / "spindle_lags_by_mouse.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_spindle_lag_distribution_per_region(data_folder, save_path,
                                              bw=0.15, n_bootstrap=1000, n_grid=500):
    """
    One curve per region = mean of per-mouse KDEs (gaussian_kde, bw=0.15).
    95% CI band from bootstrap: resample mice with replacement 1000×,
    compute mean KDE across sampled mice, CI = percentile 2.5 / 97.5.

    Loads all ripple_correlation_table_all_mice_*.pkl files in data_folder.
    Saved under <save_path>/spindle_lags_by_region.png
    """
    from scipy.stats import gaussian_kde as _gkde

    data_folder = Path(data_folder)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    files = sorted(data_folder.glob("ripple_correlation_table_all_mice_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No correlation tables found in {data_folder}")

    tables = []
    for f in files:
        df = pd.read_pickle(f)
        df = df[df["baseline_substracted"] == True].copy()
        df["spindle_coupling_lags"] = pd.to_numeric(df["spindle_coupling_lags"], errors="coerce")
        tables.append(df)

    big = pd.concat(tables, axis=0, ignore_index=True).dropna(subset=["spindle_coupling_lags"])
    if big.empty:
        print("No spindle lag data found.")
        return

    lag_min = big["spindle_coupling_lags"].min()
    lag_max = big["spindle_coupling_lags"].max()
    grid = np.linspace(lag_min, lag_max, n_grid)

    def kde_on_grid(lags):
        if len(lags) < 2:
            return np.zeros(n_grid)
        vals = _gkde(lags, bw_method=bw)(grid)
        vals /= np.trapezoid(vals, grid)
        return vals

    regions = sorted(big["brain_region"].unique())
    palette = sns.color_palette("tab10", n_colors=len(regions))

    fig, ax = plt.subplots(figsize=(9, 5))

    for region, color in zip(regions, palette):
        df_r = big[big["brain_region"] == region]
        mice = df_r["mouse"].unique()

        mouse_kdes = []
        for mouse in mice:
            lags = df_r.loc[df_r["mouse"] == mouse, "spindle_coupling_lags"].values
            k = kde_on_grid(lags)
            if k.sum() > 0:
                mouse_kdes.append(k)

        if not mouse_kdes:
            continue

        kde_matrix = np.stack(mouse_kdes)   # (n_mice, n_grid)
        n_mice = len(kde_matrix)

        region_mean = kde_matrix.mean(axis=0)

        rng = np.random.default_rng(42)
        boot_means = np.stack([
            kde_matrix[rng.integers(0, n_mice, size=n_mice)].mean(axis=0)
            for _ in range(n_bootstrap)
        ])
        ci_low  = np.percentile(boot_means, 2.5,  axis=0)
        ci_high = np.percentile(boot_means, 97.5, axis=0)

        ax.plot(grid, region_mean, color=color, linewidth=1.8,
                label=f"{region} (n={n_mice} mice)")
        ax.fill_between(grid, ci_low, ci_high, color=color, alpha=0.15)

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(axis="x", which="minor", length=3, labelbottom=False)
    ax.set_xlabel("Ripple–spindle lag (s)")
    ax.set_ylabel("Density")
    ax.set_title("Spindle–SWR coupling by region (bootstrap 95% CI across mice)")
    ax.legend(title="Region", fontsize=8, title_fontsize=8, frameon=False)

    plt.tight_layout()
    out_file = save_path / "spindle_lags_by_region.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_boxplot_spindle_coupling_per_region(data_folder, save_path,
                                              region_order=None):
    """
    Boxplot of mean spindle–SWR coupling probability per brain region.

    Each box = mean across mice; error bars = CI 95.
    Individual mouse points are overlaid as a stripplot.

    Loads all ripple_correlation_table_all_mice_*.pkl files in data_folder.
    Saved under <save_path>/spindle_coupling_boxplot_per_region.png
    """
    data_folder = Path(data_folder)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    files = sorted(data_folder.glob("ripple_correlation_table_all_mice_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No correlation tables found in {data_folder}")

    tables = []
    for f in files:
        df = pd.read_pickle(f)
        df = df[df["baseline_substracted"] == True].copy()
        tables.append(df)

    big = pd.concat(tables, axis=0, ignore_index=True)

    mean_coupling = (
        big.groupby(["mouse","rewarded_group", "brain_region"])["cooccurrence_prob"]
        .mean()
        .reset_index()
        .rename(columns={"cooccurrence_prob": "mean_coupling_prob"})
    )

    if region_order is None:
        region_order_all = [ "SSp", "DMS", "MO-ALM", "MO-wM1", "MO-wM2", "DLS", "mPFC", "PPC"]
        region_order = [r for r in region_order_all if r in mean_coupling["brain_region"].unique()]
        #remaining = sorted(set(mean_coupling["brain_region"].unique()) - set(region_order))
        #region_order = region_order + remaining

    palette_reward = {"R+": "#2ca25f", "R-": "#de2d26"}
    fig, ax = plt.subplots(figsize=(max(6, len(region_order) * 1.5), 5))

    sns.boxplot(
        data=mean_coupling,
        x="brain_region",
        y="mean_coupling_prob",
        hue="rewarded_group",
        order=region_order,
        hue_order=["R+", "R-"],
        palette=palette_reward,
        boxprops={"facecolor": "none"},
        flierprops={"marker": ""},
        ax=ax,
    )
    sns.stripplot(
        data=mean_coupling,
        x="brain_region",
        y="mean_coupling_prob",
        hue="rewarded_group",
        order=region_order,
        hue_order=["R+", "R-"],
        palette=palette_reward,
        dodge=True,
        alpha=0.8,
        jitter=True,
        ax=ax,
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title="Group")
    ax.set_xlabel("Brain region")
    ax.set_ylabel("Mean spindle–SWR coupling probability")
    ax.set_title("Spindle–SWR coupling probability per region")
    plt.tight_layout()

    out_file = save_path / "spindle_coupling_boxplot_per_region.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")

def plot_boxplot_delta_spindle_coupling_per_region(data_folder, save_path,
                                              region_order=None):
    """
    Boxplot of delta mean spindle–SWR coupling probability per brain region between beginning and end of session.

    Each box = delta across mice; error bars = CI 95 across mice.
    Individual mouse points are overlaid as a stripplot.

    Loads all ripple_correlation_table_all_mice_*.pkl files in data_folder.
    Saved under <save_path>/delta_spindle_coupling_boxplot_per_region.png
    """
    data_folder = Path(data_folder)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    files = sorted(data_folder.glob("ripple_correlation_table_all_mice_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No correlation tables found in {data_folder}")

    tables = []
    for f in files:
        df = pd.read_pickle(f)
        df = df[df["baseline_substracted"] == True].copy()
        tables.append(df)

    big_table = pd.concat(tables, axis=0, ignore_index=True)

    # mean P(whisker) per mouse × brain_region × half
    df_half = (
        big_table
        .groupby(["mouse", "rewarded_group", "brain_region", "trial_order_group"], as_index=False)
        ["cooccurrence_prob"].mean().rename(columns={"cooccurrence_prob": "mean_coupling_prob"})
    )

    # pivot to get first_half and second_half as columns, then compute delta
    df_pivot = df_half.pivot_table(
        index=["mouse", "rewarded_group", "brain_region"],
        columns="trial_order_group",
        values="mean_coupling_prob"
    ).reset_index()

    if "first_half" not in df_pivot.columns or "second_half" not in df_pivot.columns:
        raise ValueError("trial_order_group must contain 'first_half' and 'second_half'")

    df_pivot["delta"] = df_pivot["second_half"] - df_pivot["first_half"]
    df_delta = df_pivot.dropna(subset=["delta"])

    if region_order is None:
        region_order_all = [ "SSp", "DMS", "MO-ALM", "MO-wM1", "MO-wM2", "DLS", "mPFC", "PPC"]
        region_order = [r for r in region_order_all if r in df_delta["brain_region"].unique()]
        #remaining = sorted(set(df_delta["brain_region"].unique()) - set(region_order))
        #region_order = region_order + remaining

    palette_reward = {"R+": "#2ca25f", "R-": "#de2d26"}
    fig, ax = plt.subplots(figsize=(max(6, len(region_order) * 1.5), 5))

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
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title="Group")
    ax.set_xlabel("Brain region")
    ax.set_ylabel("Δ spindle–SWR coupling probability  [2nd half − 1st half]")
    ax.set_title("Delta spindle–SWR coupling probability per region")
    plt.tight_layout()

    out_file = save_path / "delta_spindle_coupling_boxplot_per_region.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")

def correlation_coupling_res_mouse_perf(data_folder, data_folder_ripples, save_path):
    """
    Relplot: col = brain region, row = coupling result type (mean coupling / delta coupling),
    x = coupling probability per mouse, y = whisker hit rate.
    Regression line + r/p annotation per rewarded_group per facet.

    Loads all ripple_correlation_table_all_mice_*.pkl files in data_folder.
    Saved under <save_path>/correlation_coupling_perf.png
    """
    from scipy.stats import linregress

    data_folder = Path(data_folder)
    data_folder_ripples = Path(data_folder_ripples)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # --- Load coupling tables ---
    files = sorted(data_folder.glob("ripple_correlation_table_all_mice_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No correlation tables found in {data_folder}")

    tables = []
    for f in tqdm(files, desc="Loading coupling tables"):
        df = pd.read_pickle(f)
        # cooccurrence_prob is independent of baseline subtraction — keep one version
        df = df[df["baseline_substracted"] == True].copy()
        tables.append(df)

    big = pd.concat(tables, axis=0, ignore_index=True)

    # mean coupling prob per mouse × rewarded_group × brain_region
    df_mean = (
        big.groupby(["mouse", "rewarded_group", "brain_region"], as_index=False)["cooccurrence_prob"]
        .mean()
        .rename(columns={"cooccurrence_prob": "mean_coupling"})
    )

    # delta coupling = second_half − first_half, per mouse × rewarded_group × brain_region
    df_half = (
        big.groupby(["mouse", "rewarded_group", "brain_region", "trial_order_group"], as_index=False)
        ["cooccurrence_prob"].mean()
    )
    df_pivot = df_half.pivot_table(
        index=["mouse", "rewarded_group", "brain_region"],
        columns="trial_order_group",
        values="cooccurrence_prob"
    ).reset_index()

    if "first_half" not in df_pivot.columns or "second_half" not in df_pivot.columns:
        raise ValueError("trial_order_group must contain 'first_half' and 'second_half'")

    df_pivot["delta_coupling"] = df_pivot["second_half"] - df_pivot["first_half"]
    df_delta = df_pivot[["mouse", "rewarded_group", "brain_region", "delta_coupling"]].dropna(subset=["delta_coupling"])

    # --- Mouse performance: global hit rate + delta hit rate (2nd half − 1st half) ---
    wh_perf_list = []
    for file in tqdm(os.listdir(data_folder_ripples), desc="Loading ripple tables"):
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
    merged = pd.merge(merged, perf_df, on="mouse", how="inner")

    # --- Reshape to long format ---
    long = pd.melt(
        merged,
        id_vars=["mouse", "rewarded_group", "brain_region", "whisker_hit_rate", "delta_perf"],
        value_vars=["mean_coupling", "delta_coupling"],
        var_name="coupling_result_type",
        value_name="coupling_value",
    )

    row_labels = {"mean_coupling": "Mean coupling prob", "delta_coupling": "Δ coupling [2nd−1st]"}
    long["coupling_result_type"] = long["coupling_result_type"].map(row_labels)
    row_order = [row_labels["mean_coupling"], row_labels["delta_coupling"]]

    # mean row → global hit rate ; delta row → delta hit rate
    long["perf_value"] = np.where(
        long["coupling_result_type"] == row_labels["delta_coupling"],
        long["delta_perf"],
        long["whisker_hit_rate"],
    )

    region_order_all = ["SSp", "DMS", "MO-ALM", "MO-wM1", "MO-wM2", "DLS", "mPFC", "PPC"]
    region_order = [r for r in region_order_all if r in long["brain_region"].unique()]

    palette_reward = {"R+": "#2ca25f", "R-": "#de2d26"}

    # --- Relplot ---
    g = sns.relplot(
        data=long,
        x="coupling_value",
        y="perf_value",
        col="brain_region",
        row="coupling_result_type",
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
        sub = long[(long["coupling_result_type"] == row_val) & (long["brain_region"] == col_val)]
        y_text = {"R+": 0.95, "R-": 0.80}
        for rg, color in palette_reward.items():
            sub_rg = sub[sub["rewarded_group"] == rg].dropna(subset=["coupling_value", "perf_value"])
            if len(sub_rg) < 3:
                continue
            slope, intercept, r, p, _ = linregress(sub_rg["coupling_value"], sub_rg["perf_value"])
            x_range = np.linspace(sub_rg["coupling_value"].min(), sub_rg["coupling_value"].max(), 100)
            ax.plot(x_range, slope * x_range + intercept, color=color, linewidth=1.5, alpha=0.75)
            ax.text(0.05, y_text[rg], f"r={r:.2f}, p={p:.2f}",
                    transform=ax.transAxes, fontsize=7, color=color, va="top")

    y_labels = {row_labels["mean_coupling"]: "Whisker hit rate", row_labels["delta_coupling"]: "Δ hit rate [2nd−1st]"}
    for (row_val, _), ax in g.axes_dict.items():
        ax.set_ylabel(y_labels.get(row_val, "Performance"))

    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.figure.suptitle("Spindle–SWR coupling vs mouse performance", y=1.02)
    g.figure.subplots_adjust(hspace=0.35, wspace=0.25)

    out_file = save_path / "correlation_coupling_perf.png"
    g.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(g.figure)
    print(f"Saved: {out_file}")