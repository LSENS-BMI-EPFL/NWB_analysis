import os
import pathlib
import subprocess
from pathlib import Path
import scipy as sci
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface as si
import spikeinterface.preprocessing as sip
from sklearn.manifold import TSNE
from nwb_utils.utils_misc import find_nearest
from utils.lfp_utils import *



def build_data_for_LDA(data_table, trial_types, save_path):
    """Build data for LDA analysis."""
    window_sensory = 0.05
    new_df = build_table_population_vectors(data_table, window_sensory=window_sensory, window_ripple=0.05, substract_baseline=True)

    df_active = new_df[new_df.context == "active"]