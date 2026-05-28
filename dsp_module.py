import numpy as np
from matplotlib import pyplot as plt
import sys

# Generate sinusoid
# f - Frequency
# start - Start timeline
# end - End timeline
# Fs - Sampling frequency
def sin_generator(f, Fs, timeline):
    # check errors
    if f <= 0:
        print("[dsp_module:sin_generator] Frequency must be a positive number!")
        sys.exit(1)
        
    if Fs <= 0:
        print("[dsp_module:sin_generator] Fs must be a positive number!")
        sys.exit(1)
    
    return np.sin(2*np.pi*f*timeline)

def upsampling(signal, L):
    ups_signal = []
    
    for sample in signal:
        ups_signal.append(sample)
        for _ in range(L-1):
            ups_signal.append(0)
        
    return ups_signal

def create_FIR(Fs, fc, FIR_len=35):
    mid = FIR_len // 2
    
    FIR_timeline = np.arange(-mid, mid)

    # digital slice frequency
    wc = 2.0 * np.pi * fc / Fs    
    
    # create FIR
    FIR = np.sinc(FIR_timeline * wc / np.pi) * (wc / np.pi)
    
    FIR /= np.max(FIR)
    
    return FIR_timeline, FIR
    
    
def decimation(signal, L):
    return signal[::L]