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
    RUN_BUILD_TABLES  = True   # slow — re-run only if data changes
    RUN_PLOTS_MOUSE   = False   # scatter / histograms per mouse
    RUN_SUMMARY_PLOTS = False   # group-level figures per region
    RUN_CORRELATIONS  = False  # LDA vs behavioural performance
    # ─────────────────────────────────────────────────────────────────────────

    # ── CONFIG ────────────────────────────────────────────────────────────────
    task        = 'fast-learning'   # 'context' or 'fast-learning'
    trial_types = ['no_stim_trial', 'whisker_trial', 'auditory_trial']
    pair        = 'no_stim_trial-whisker_trial'
    regions     = ['ORB']

    spindle_coupled_only = True
    cooccurrence_window  = 0.05
    window_ripple        = 0.05
    window_sensory       = 0.05
    uniform_priors       = False   # switch to True to use uniform-prior LDA tables

    BASE = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro")

    data_folder         = BASE / ("lda_tables_prior_uniform_new" if uniform_priors else "lda_tables_new")
    data_folder_ripples = BASE / "ripple_tables_bis"
    save_path_mouse     = BASE / "lda_plots_new" / ("per_mouse_uniform_prior" if uniform_priors else "per_mouse")
    save_path           = BASE / "lda_plots_new" / ("summary_proba_whisker_uniform_prior" if uniform_priors else "summary_proba_whisker")
    perf_table_path     = BASE / "performance_tables" / "mouse_performance_table.pkl"
    # ─────────────────────────────────────────────────────────────────────────

    # ── STAGE 1 — BUILD TABLES ───────────────────────────────────────────────
    if RUN_BUILD_TABLES:

        # Performance table
        #make_mouse_performance_table(data_folder_ripples, BASE / "performance_tables")

        # LDA tables pairwise + multiclass × uniform / non-uniform priors for each secondary region
        
        for region in regions:
            for pairwise in [True, False]:
                for use_uniform in [True, False]:
                    table_save_path = BASE / ("lda_tables_prior_uniform_new" if use_uniform else "lda_tables_new")
                    make_lda_big_table_all_mice(
                        data_folder_ripples, table_save_path,
                        brain_regions=['ca1', region],
                        window_ripple=window_ripple,
                        window_sensory=window_sensory,
                        classes_labels=trial_types,
                        use_multiprocessing=False,
                        project_all_ripples=True,
                        pairwise=pairwise,
                        uniform_priors=use_uniform,
                    )
        

        # OLD anaysis
        # make_centroid_distance_table_for_all_mice(data_folder, save_path, lda_cols=["LD1","LD2"], group_col='trial_type', classes_labels=trial_types)

    # ── STAGE 2 — PER-MOUSE PLOTS ───────────────────────────────────────────
    if RUN_PLOTS_MOUSE:

        for region in regions:
            plot_lda_results_from_table(
                data_folder, save_path_mouse,
                in_filename=f"lda_big_table_all_mice_multiclass_{region}.pkl",
            )
            plot_lda_whisker_proba_in_time_with_bhv(
                data_folder_lda_table=data_folder,
                data_folder_ripples=data_folder_ripples,
                save_path=save_path_mouse,
                in_filename=f"lda_big_table_all_mice_pairwise_{region}.pkl",
                pair=pair,
            )

        # OLD anaysis
        # plot_lda_results(data_folder, save_path, brain_regions=['ca1', 'second'], window_ripple=window_ripple, window_sensory=window_sensory, classes_labels=trial_types, index_order=True)
        # plot_lda_results_with_index_order(data_folder, save_path, brain_regions=['ca1', 'second'], window_ripple=window_ripple, window_sensory=window_sensory, classes_labels=trial_types)
        # plot_lda_binary_results(data_folder, save_path, pair=pair)
        # plot_tsne(data_folder, save_path, eps=5, min_samples=5)  # check: could also belong in summary plots

    # ── STAGE 3 — SUMMARY PLOTS ──────────────────────────────────────────────
    if RUN_SUMMARY_PLOTS:

        combine_plots(
            data_folder, save_path,
            pair=pair,
            spindle_coupled_only=spindle_coupled_only,
            cooccurrence_window=cooccurrence_window
            )
        plot_mean_whisker_proba_distibution_per_region(
            data_folder, save_path, 
            pair=pair, 
            spindle_coupled_only=spindle_coupled_only,
            cooccurrence_window=cooccurrence_window,
        )
        plot_delta_whisker_proba_per_region(
            data_folder, save_path, 
            pair=pair, 
            spindle_coupled_only=spindle_coupled_only,
            cooccurrence_window=cooccurrence_window)
        
        plot_coupling_effect_on_whisker_proba(
            data_folder, save_path,
            pair=pair,
            cooccurrence_window=cooccurrence_window,
        )
        # plot_accuracy_per_region(data_folder, save_path, pair=pair)
        # accuracy_plot(data_folder, save_path)

        # OLD analysis
        # Centroid distance
        # centroids_distance_plot(data_folder, save_path)
        # plot_centroid_results_with_shuffle(data_folder, save_path)


# ── STAGE 4 — CORRELATIONS LDA vs PERFORMANCE ───────────────────────────
    if RUN_CORRELATIONS:

        correlation_lda_res_mouse_perf(
            data_folder, data_folder_ripples, save_path,
            pair=pair,
            spindle_coupled_only=spindle_coupled_only,
            cooccurrence_window=cooccurrence_window,
            perf_table_path=perf_table_path,
        )
