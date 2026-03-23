import sys
sys.path.append('/Users/nigro/Desktop/NWB_analysis')
from utils.LDA_loader import *
from utils.LDA_tables import *
from utils.LDA_plotting import *
from utils.lfp_utils import *
from utils.DBSCAN_model import *
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
    save_path = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/DBSCAN_results")
    #save_path = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/lda_plot/LDA_plot_all_index_order")
    #save_path = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/lda_plot/LDA_plot_from_big_table")
    #save_path = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/lda_tables")

    #data_folder = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/centroids_distance_table")
    #data_folder = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/share_internal/Toni_Nigro/lda_tables")
    data_folder = Path('//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Robin_Dard/ripple_results/fastlearning_task/ripple_tables')

    # MAIN

    ## plots

    #plot_lda_results(data_folder,save_path, brain_regions=['ca1','second'],window_ripple=0.05, window_sensory=0.05,classes_labels=trial_types, index_order=True, shuffle_tot=None)
    #plot_lda_binary_results(data_folder,save_path, brain_regions=['ca1','second'],window_ripple=0.05, window_sensory=0.05,classes_labels=["whisker_trial", "auditory_trial"])
    #centroids_distance_plot(data_folder,save_path)
    #plot_centroid_results_with_shuffle(data_folder,save_path)
    #plot_lda_results_with_index_order(data_folder,save_path, brain_regions=['ca1','second'], window_ripple=0.05, window_sensory=0.05, classes_labels=trial_types, shuffle_tot=None)
    #plot_lda_results_from_table(data_folder,save_path,in_filename="lda_big_table_all_mice.pkl")
    plot_tsne(data_folder,save_path, eps=5, min_samples=5)



    ## tables 
    #make_lda_big_table_all_mice(data_folder,save_path,brain_regions=['ca1','second'],window_ripple=0.05,window_sensory=0.05,classes_labels=["no_stim_trial", "whisker_trial", "auditory_trial"],shuffle_tot=1000, use_multiprocessing=True)
    #make_centroid_distance_table_for_all_mice(data_folder,save_path, lda_cols=["LD1","LD2"], group_col='trial_type',classes_labels=["whisker_trial", "auditory_trial", "no_stim_trial"])
    #make_lda_big_table_all_mice(data_folder,save_path,brain_regions=['ca1','second'],window_ripple=0.05,window_sensory=0.05,classes_labels=["no_stim_trial", "whisker_trial", "auditory_trial"],shuffle_tot=100)



