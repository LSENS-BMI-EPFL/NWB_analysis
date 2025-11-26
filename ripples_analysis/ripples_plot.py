from utils.lfp_utils import *


# MAIN #
# PARAMETERS
task = 'fast-learning'  # 'context' or 'fast-learning'

# DATA FOLDER
data_folder = Path('//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Robin_Dard/ripple_results/fastlearning_task/ripple_tables')
names = os.listdir(data_folder)
save_path = Path('//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Robin_Dard/ripple_results/fastlearning_task/ripple_results')

# MAIN
# plot_ripple_frequency_fastlearning(data_folder, trial_type="no_stim_trial", lick_flag=None, save_path=save_path)
# plot_all_trials_data(data_folder, task, save_path)
# plot_single_event_data(data_folder, task, window=0.10, save_path=save_path)
plot_trial_ripple_content(data_folder, task, window=0.050, save_path=save_path)


