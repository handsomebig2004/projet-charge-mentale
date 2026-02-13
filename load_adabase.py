import h5py
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML

# %config InlineBackend.figure_format = "retina"


# Suppress the warning
warnings.filterwarnings('ignore')



# Sampling Rates
SKT_SR =100
ECG_SR = 500
RSP_SR = 250
EMG_SR = 1000
EDA_SR = 500
EYE_SR = 250

SUBJECT_IDX = 0 # You can change this
# Path to folder containing hdf5 files
data_folder = "data/adabase"
# Get list of file names
file_names = os.listdir(data_folder)
file_names = list(filter(lambda x: x.endswith(".h5py"), file_names))
# Select one file
file_name = file_names[SUBJECT_IDX]
file_path = os.path.join(data_folder , file_name)



df_signals = pd.read_hdf(file_path, "SIGNALS", mode="r")
df_performance = pd.read_hdf(file_path, "PERFORMANCE", mode="r")
df_subjective = pd.read_hdf(file_path, "SUBJECTIVE", mode="r")

print(df_signals.info(verbose=True))
