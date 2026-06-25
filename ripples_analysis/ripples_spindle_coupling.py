import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.spindle_association import *
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

if __name__ == "__main__":

    # ── PIPELINE FLAGS ────────────────────────────────────────────────────────
    RUN_PLOTS_PER_REGION   = False  # coupling + lag distributions per region
    RUN_SUMMARY_PLOTS      = True   # group-level summary across all regions
    RUN_CORRELATIONS       = False  # spindle coupling vs behavioural performance
    # ─────────────────────────────────────────────────────────────────────────

    # ── CONFIG ────────────────────────────────────────────────────────────────
    # Reads from correlation tables produced by ripples_correlations.py (STAGE 1)
    regions = ['SSp', 'DMS', 'MO-ALM', 'MO-wM1', 'MO-wM2', 'DLS', 'mPFC', 'PPC']

    BASE = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro")

    data_folder_corr    = BASE / "corr_tables"
    data_folder_ripples = BASE / "ripple_tables_bis"
    save_path           = BASE / "spindle_swr_coupling_new"
    perf_table_path     = BASE / "performance_tables" / "mouse_performance_table.pkl"
    # ─────────────────────────────────────────────────────────────────────────

    # ── STAGE 1 — PER-REGION PLOTS ───────────────────────────────────────────
    if RUN_PLOTS_PER_REGION:

        for region in regions:
            plot_mean_spindle_coupling_per_region(data_folder_corr, save_path, second_target=region)
            plot_distibution_spindle_lags_per_region(data_folder_corr, save_path, second_target=region)

    # ── STAGE 2 — SUMMARY PLOTS (all regions) ────────────────────────────────
    if RUN_SUMMARY_PLOTS:

        plot_mean_spindle_lag_distribution_all_regions(data_folder_corr, save_path)
        plot_spindle_lag_distribution_per_region(data_folder_corr, save_path)
        plot_boxplot_spindle_coupling_per_region(data_folder_corr, save_path, region_order=None)
        plot_boxplot_delta_spindle_coupling_per_region(data_folder_corr, save_path, region_order=None)

    # ── STAGE 3 — CORRELATIONS COUPLING vs PERFORMANCE ───────────────────────
    if RUN_CORRELATIONS:

        correlation_coupling_res_mouse_perf(
            data_folder_corr, data_folder_ripples, save_path,
            perf_table_path=perf_table_path,
        )
