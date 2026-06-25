import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.ripples_cor import *
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

if __name__ == "__main__":

    # ── PIPELINE FLAGS ────────────────────────────────────────────────────────
    RUN_BUILD_TABLES       = False    # slow — re-run only if data changes
    RUN_PLOTS_MOUSE        = True   # 3D scatter + correlations over time per mouse
    RUN_PLOTS_MOUSE_BHV    = True   # correlations over time + behaviour panel per mouse
    RUN_SUMMARY_PLOTS      = True   # mean and delta whisker correlation per region
    # ─────────────────────────────────────────────────────────────────────────

    # ── CONFIG ────────────────────────────────────────────────────────────────
    trial_types         = ['no_stim_trial', 'whisker_trial', 'auditory_trial']
    regions             = ['SSp', 'DMS', 'MO-ALM', 'MO-wM1', 'MO-wM2', 'DLS', 'mPFC', 'PPC']
    window_ripple       = 0.05
    window_sensory      = 0.05

    BASE = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro")

    data_folder_ripples = BASE / "ripple_tables_bis"
    data_folder_corr    = BASE / "corr_tables"
    save_path           = BASE / "corr_plots"
    # ─────────────────────────────────────────────────────────────────────────

    # ── STAGE 1 — BUILD TABLES ───────────────────────────────────────────────
    if RUN_BUILD_TABLES:

        for region in regions:
            make_ripple_correlation_table_all_mice(
                data_folder_ripples, data_folder_corr,
                brain_regions=['ca1', region],
                window_sensory=window_sensory,
                window_ripple=window_ripple,
                context_value="active",
                classes_labels=trial_types,
                scale_data=True,
            )

    # ── STAGE 2 — PER-MOUSE PLOTS ────────────────────────────────────────────
    if RUN_PLOTS_MOUSE:

        for region in regions:
            plot_correlation_table(data_folder_corr, save_path, second_target=region)
            plot_correlation_table_in_time(data_folder_corr, save_path, second_target=region)

    # ── STAGE 3 — PER-MOUSE PLOTS WITH BEHAVIOUR ─────────────────────────────
    if RUN_PLOTS_MOUSE_BHV:

        for region in regions:
            plot_correlation_table_in_time_with_bhv(
                data_folder_corr_table=data_folder_corr,
                data_folder_ripples=data_folder_ripples,
                save_path=save_path,
                second_target=region,
            )

    # ── STAGE 4 — SUMMARY PLOTS (all regions) ────────────────────────────────
    if RUN_SUMMARY_PLOTS:

        for baseline_substracted in [True, False]:
            combine_plots_corr(
                data_folder_corr, save_path,
                baseline_substracted=baseline_substracted,
            )
