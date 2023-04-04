import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from resampy import resample

# from scipy.signal import resample
from scipy import signal



def read_data(root):
    BCGpath = os.path.join(root, 'BCG_sync.txt')
    Jpeak_AIpath = os.path.join(root, 'Jpeaks_sync.txt')
    Jpeak_DLpath = os.path.join(root, 'Jpeak_DL.txt')
    Rpeaks_syncpath = os.path.join(root, 'Rpeaks_sync.txt')

    BCG = np.array(pd.read_csv(BCGpath, header=None)).reshape(-1)
    # B, A = signal.butter(4, np.array([2,8])*2/1000, 'bandpass')
    # BCG = signal.filtfilt(B, A, orgBCG)
    Jpeak_AI = np.array(pd.read_csv(Jpeak_AIpath)).reshape(-1)
    Jpeak_DL = np.array(pd.read_csv(Jpeak_DLpath, header=None)).reshape(-1)

    Rpeaks_sync = np.loadtxt(Rpeaks_syncpath).astype(int)

    return BCG, Jpeak_AI, Jpeak_DL, Rpeaks_sync
