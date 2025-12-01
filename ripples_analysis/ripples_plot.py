from utils.lfp_utils import *
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


# MAIN #
# PARAMETERS
task = 'fast-learning'  # 'context' or 'fast-learning'
trial_types = ['no_stim_trial', 'whisker_trial', 'auditory_trial']

# DATA FOLDER
data_folder = Path('//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Robin_Dard/ripple_results/fastlearning_task/ripple_tables')
save_path = Path('//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Robin_Dard/ripple_results/fastlearning_task/ripple_results')

# MAIN
# plot_ripple_frequency_fastlearning(data_folder, trial_types=trial_types, save_path=save_path)

# plot_hist_ripples_time(data_folder, trial_types, save_path, bin_width=0.5)

plot_wh_hit_trial_ripple_content(data_folder, task, window_sensory=0.050, window_ripple=0.025, save_path=save_path)

# plot_ripple_similarity(data_folder, task, window=0.025, save_path=save_path)

# plot_all_trials_data(data_folder, task, save_path)

# plot_single_event_data(data_folder, task, window=0.10, only_average=True, save_path=save_path)

