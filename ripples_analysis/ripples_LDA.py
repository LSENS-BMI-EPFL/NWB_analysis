from utils.lfp_utils import *
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


# MAIN #
# PARAMETERS
task = 'fast-learning'  # 'context' or 'fast-learning'
trial_types = ['no_stim_trial', 'whisker_trial', 'auditory_trial']

# DATA FOLDER
data_folder = Path('/Volumes/z_LSENS/Share/Toni_Nigro/ripple_tables')

save_path = Path("/Volumes/z_LSENS/Share/Toni_Nigro/ripple_results")

