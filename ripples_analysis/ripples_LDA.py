import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.LDA_loader import *
from utils.LDA_tables import *
from utils.LDA_plotting import *
from utils.ripples_cor import *
from utils.spindle_association import *
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

if __name__ == "__main__":

    # ── PIPELINE FLAGS ────────────────────────────────────────────────────────
    RUN_BUILD_TABLES  = False   # long — relancer si les données changent
    RUN_PLOTS_MOUSE   = False   # scatter / histos par souris
    RUN_SUMMARY_PLOTS = False   # figures de groupe par région
    RUN_CORRELATIONS  = True    # LDA vs performance comportementale
    # ─────────────────────────────────────────────────────────────────────────

    # ── CONFIG ────────────────────────────────────────────────────────────────
    task        = 'fast-learning'   # 'context' or 'fast-learning'
    trial_types = ['no_stim_trial', 'whisker_trial', 'auditory_trial']
    pair        = 'no_stim_trial-whisker_trial'
    regions     = ['SSp', 'DMS', 'MO-ALM', 'MO-wM1', 'MO-wM2', 'DLS', 'mPFC', 'PPC']

    spindle_coupled_only = False
    cooccurrence_window  = 0.05
    window_ripple        = 0.05
    window_sensory       = 0.05

    BASE = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro")

    data_folder         = BASE / "lda_tables"
    data_folder_ripples = BASE / "ripple_tables_bis"
    save_path           = BASE / "lda_plots"
    perf_table_path     = BASE / "performance_tables" / "mouse_performance_table.pkl"
    # ─────────────────────────────────────────────────────────────────────────

    # ── STAGE 1 — BUILD TABLES ───────────────────────────────────────────────
    if RUN_BUILD_TABLES:

        # Performance table (une seule fois, nécessaire pour les corrélations)
        make_mouse_performance_table(data_folder_ripples, BASE / "performance_tables")

        # LDA tables pairwise + multiclass pour chaque région secondaire
        for region in regions:
            for pairwise in [True, False]:
                make_lda_big_table_all_mice(
                    data_folder, save_path,
                    brain_regions=['ca1', region],
                    window_ripple=window_ripple,
                    window_sensory=window_sensory,
                    classes_labels=trial_types,
                    use_multiprocessing=False,
                    project_all_ripples=True,
                    pairwise=pairwise,
                    uniform_priors=False,
                )

        # make_centroid_distance_table_for_all_mice(data_folder, save_path, lda_cols=["LD1","LD2"], group_col='trial_type', classes_labels=trial_types)
        # for region in regions:
        #     make_ripple_correlation_table_all_mice(data_folder, save_path, brain_regions=['ca1', region], window_sensory=window_sensory, window_ripple=window_ripple, context_value="active", classes_labels=trial_types, scale_data=True)

    # ── STAGE 2 — PLOTS PAR SOURIS ───────────────────────────────────────────
    if RUN_PLOTS_MOUSE:

        for region in regions:
            plot_lda_results_from_table(
                data_folder, save_path,
                in_filename=f"lda_big_table_all_mice_multiclass_{region}.pkl",
            )
            plot_lda_whisker_proba_in_time_with_bhv(
                data_folder_lda_table=data_folder,
                data_folder_ripples=data_folder_ripples,
                save_path=save_path,
                in_filename=f"lda_big_table_all_mice_pairwise_{region}.pkl",
                pair=pair,
            )

        # plot_lda_results(data_folder, save_path, brain_regions=['ca1', 'second'], window_ripple=window_ripple, window_sensory=window_sensory, classes_labels=trial_types, index_order=True)
        # plot_lda_results_with_index_order(data_folder, save_path, brain_regions=['ca1', 'second'], window_ripple=window_ripple, window_sensory=window_sensory, classes_labels=trial_types)
        # plot_lda_binary_results(data_folder, save_path, pair=pair)
        # plot_tsne(data_folder, save_path, eps=5, min_samples=5)  # à vérifier : peut aussi être un summary plot

    # ── STAGE 3 — SUMMARY PLOTS ──────────────────────────────────────────────
    if RUN_SUMMARY_PLOTS:

        combine_plots(
            data_folder, save_path,
            pair=pair,
            spindle_coupled_only=spindle_coupled_only,
            cooccurrence_window=cooccurrence_window,
        )
        # plot_mean_whisker_proba_distibution_per_region(data_folder, save_path, pair=pair, spindle_coupled_only=spindle_coupled_only)
        # plot_delta_whisker_proba_per_region(data_folder, save_path, pair=pair, spindle_coupled_only=spindle_coupled_only)
        # plot_accuracy_per_region(data_folder, save_path, pair=pair)
        # accuracy_plot(data_folder, save_path)

        # Spindle-SWR coupling
        # plot_mean_spindle_lag_distribution_all_regions(data_folder, save_path)
        # plot_spindle_lag_distribution_per_region(data_folder, save_path)
        # plot_boxplot_spindle_coupling_per_region(data_folder, save_path, region_order=None)
        # plot_boxplot_delta_spindle_coupling_per_region(data_folder, save_path, region_order=None)
        # for region in regions:
        #     plot_distibution_spindle_lags_per_region(data_folder, save_path, second_target=region)

        # Centroid distance
        # centroids_distance_plot(data_folder, save_path)
        # plot_centroid_results_with_shuffle(data_folder, save_path)

        # Correlation tables (ripple-ripple)  — à vérifier : peut aussi aller en STAGE 4
        # plot_correlation_table(data_folder, save_path)
        # plot_correlation_table_in_time(data_folder, save_path)
        # for region in regions:
        #     plot_correlation_table_in_time_with_bhv(data_folder_corr_table=data_folder, data_folder_ripples=data_folder_ripples, save_path=save_path, second_target=region)

    # ── STAGE 4 — CORRELATIONS LDA vs PERFORMANCE ───────────────────────────
    if RUN_CORRELATIONS:

        correlation_lda_res_mouse_perf(
            data_folder, data_folder_ripples, save_path,
            pair=pair,
            spindle_coupled_only=spindle_coupled_only,
            cooccurrence_window=cooccurrence_window,
            perf_table_path=perf_table_path,
        )
        # correlation_coupling_res_mouse_perf(data_folder, data_folder_ripples, save_path)
