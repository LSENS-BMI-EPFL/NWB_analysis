import sys
sys.path.append('/Users/nigro/Desktop/NWB_analysis')
from utils.LDA_loader import *
from utils.LDA_tables import *
from utils.LDA_plotting import *
from utils.ripples_cor import *
from utils.spindle_association import *
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

if __name__ == "__main__":

    # MAIN #
    # PARAMETERS
    task = 'fast-learning'  # 'context' or 'fast-learning'
    trial_types = ['no_stim_trial', 'whisker_trial', 'auditory_trial']

    # DATA FOLDER
    #save_path = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/lda_plot/Centroids_distance")
    #save_path = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/centroids_distance_table")
    #save_path = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/lda_plot/LDA_plot_centroids_shuffle")
    #save_path = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/DBSCAN_results")
    #save_path = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/lda_plot/LDA_plot_binary")
    #save_path = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/lda_plot/LDA_plot_from_big_table")
    #save_path = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/corr_plots")
    #save_path = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/corr_plots/ corr_plots_in_time")
    #save_path = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/corr_plots")
    save_path = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/spindle_swr_coupling")
    #save_path = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/corr_tables")
    #save_path = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/lda_plot/LDA_accuracy_plot")
    #save_path = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/lda_plots/summary_proba_whisker_uniform_prior")
    #save_path = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/lda_plots/summary_proba_whisker")
    #save_path = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/lda_tables_prior_uniform")
    

    #data_folder = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/centroids_distance_table")
    #data_folder = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/lda_tables_prior_uniform")
    #data_folder_ripples = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/ripple_tables_bis")
    #data_folder = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/lda_tables")
    #data_folder = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/ripple_tables_bis")
    data_folder = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/corr_tables")

    # MAIN
    #regions= ['SSp','DMS','MO-ALM','MO-wM1','MO-wM2','DLS','mPFC','PPC']
    #for pairwise in [True, False]:
    #for region in regions:
        #print(f"Processing region: {region}")#', pairwise: {pairwise}")
        ## plots

    #plot_lda_results(data_folder,save_path, brain_regions=['ca1','second'],window_ripple=0.05, window_sensory=0.05,classes_labels=trial_types, index_order=True, shuffle_tot=None)
    #plot_lda_binary_results(data_folder,save_path,pair='auditory_trial-whisker_trial')
    #centroids_distance_plot(data_folder,save_path)
    #plot_centroid_results_with_shuffle(data_folder,save_path)
    #plot_lda_results_with_index_order(data_folder,save_path, brain_regions=['ca1','second'], window_ripple=0.05, window_sensory=0.05, classes_labels=trial_types, shuffle_tot=None)
    #plot_lda_results_from_table(data_folder,save_path,in_filename=f"lda_big_table_all_mice_multiclass_{region}.pkl")
    #plot_tsne(data_folder,save_path, eps=5, min_samples=5)
    #accuracy_plot(data_folder, save_path)
    #plot_correlation_table(data_folder, save_path)
    #plot_correlation_table_in_time(data_folder, save_path)
        #plot_correlation_table_in_time_with_bhv(data_folder_corr_table=data_folder, data_folder_ripples=data_folder_ripples, save_path=save_path,second_target=region)
        #plot_lda_whisker_proba_in_time_with_bhv(data_folder_lda_table=data_folder, data_folder_ripples=data_folder_ripples, save_path=save_path, in_filename=f"lda_big_table_all_mice_pairwise_{region}.pkl", pair='auditory_trial-whisker_trial')
    #plot_mean_whisker_proba_distibution_per_region(data_folder, save_path, pair='no_stim_trial-whisker_trial', spindle_coupled_only=True)
    #plot_delta_whisker_proba_per_region(data_folder, save_path, pair='no_stim_trial-whisker_trial', spindle_coupled_only=True)
    #combine_plots(data_folder,save_path, pair='no_stim_trial-whisker_trial', spindle_coupled_only=True)
        #plot_distibution_spindle_lags_per_region(data_folder,save_path, second_target=region)
    #plot_mean_spindle_lag_distribution_all_regions(data_folder, save_path)
    #plot_spindle_lag_distribution_per_region(data_folder, save_path)
    plot_boxplot_spindle_coupling_per_region(data_folder, save_path,region_order=None)
    plot_boxplot_delta_spindle_coupling_per_region(data_folder, save_path, region_order=None)
    #correlation_lda_res_mouse_perf(data_folder, data_folder_ripples, save_path)
    #correlation_coupling_res_mouse_perf(data_folder, data_folder_ripples, save_path)

    ## tables 
            #make_lda_big_table_all_mice(data_folder,save_path,brain_regions=['ca1',region],window_ripple=0.05,window_sensory=0.05,classes_labels=["no_stim_trial", "whisker_trial", "auditory_trial"], use_multiprocessing=False, project_all_ripples=True, pairwise=pairwise, uniform_priors=True)
    #make_centroid_distance_table_for_all_mice(data_folder,save_path, lda_cols=["LD1","LD2"], group_col='trial_type',classes_labels=["whisker_trial", "auditory_trial", "no_stim_trial"])
    #make_lda_big_table_all_mice(data_folder,save_path,brain_regions=['ca1','mPFC'],window_ripple=0.05,window_sensory=0.05,classes_labels=["no_stim_trial", "whisker_trial", "auditory_trial"], use_multiprocessing=False)
        #make_ripple_correlation_table_all_mice(data_folder, save_path, brain_regions=['ca1',region], window_sensory=0.05, window_ripple=0.05, context_value="active", classes_labels=["no_stim_trial", "whisker_trial", "auditory_trial"], scale_data=True)



