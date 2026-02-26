import os
import pathlib
import subprocess
import sys
sys.path.append('/Users/toninigro/Desktop/NWB_analysis')
from pathlib import Path
import scipy as sci
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface as si
import spikeinterface.preprocessing as sip
from sklearn.manifold import TSNE


def build_sensory_vectors(data_folder, trial_types, save_path):
    """Build sensory vectors for each trial type."""
