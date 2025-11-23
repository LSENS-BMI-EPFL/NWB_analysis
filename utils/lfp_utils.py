import scipy as sci


def lfp_filter(data, fs, freq_min=150, freq_max=200):
    nyq = 0.5 * fs
    low = freq_min / nyq
    high = freq_max / nyq
    b, a = sci.signal.butter(3, [low, high], btype='band')
    return sci.signal.filtfilt(b, a, data, axis=0)
