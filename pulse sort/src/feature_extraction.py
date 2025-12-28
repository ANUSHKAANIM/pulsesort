import numpy as np

def extract_features(signal):
    mav = np.mean(np.abs(signal))
    rms = np.sqrt(np.mean(signal ** 2))
    var = np.var(signal)
    wl  = np.sum(np.abs(np.diff(signal)))
    return [mav, rms, var, wl]
