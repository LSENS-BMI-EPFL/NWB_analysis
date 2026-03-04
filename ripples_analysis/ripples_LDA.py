import sys
sys.path.append('/Users/nigro/Desktop/NWB_analysis')
from utils.LDA_loader import plot_lda_binary_results, plot_lda_results
from utils.lfp_utils import *
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


# MAIN #
# PARAMETERS
task = 'fast-learning'  # 'context' or 'fast-learning'
trial_types = ['no_stim_trial', 'whisker_trial', 'auditory_trial']

# DATA FOLDER
save_path = Path("//sv-nas1.rcp.epfl.ch/Petersen-Lab/z_LSENS/Share/Toni_Nigro/lda_plot/LDA_plot_binary")
data_folder = Path('//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Robin_Dard/ripple_results/fastlearning_task/ripple_tables')

# MAIN

#plot_lda_results(data_folder,save_path, brain_regions=['ca1','second'],window_ripple=0.05, window_sensory=0.05,classes_labels=trial_types)
plot_lda_binary_results(data_folder,save_path, brain_regions=['ca1','second'],window_ripple=0.05, window_sensory=0.05,classes_labels=["whisker_trial", "auditory_trial"])



