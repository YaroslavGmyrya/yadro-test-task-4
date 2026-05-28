import numpy as np
from matplotlib import pyplot as plt
import dsp_module as dm

# parameters for default signal
f = 400      # frequency
Fs = 1e4     # sampling frequency
Ts = 1.0/Fs  # sampling interval
start = 0    # start time
end = 0.5    # end time

# interpolation parameters
L = 3       # intrepolation order

# 1. Generate signal
timeline = np.arange(start, end, 1.0/Fs)
signal = dm.sin_generator(f=f, Fs=Fs, timeline=timeline)

# 1.1 Visualization in time domain
plt.subplot(2, 1, 1)
plt.plot(timeline, signal)
plt.xlabel("time, s")
plt.ylabel("A, V")
plt.title(f"Base signal")

# 1.2 Visualization in freq domain
plt.subplot(2, 1, 2)
plt.plot(np.fft.fftshift(np.fft.fftfreq(len(signal), Ts)), np.abs(np.fft.fftshift(np.fft.fft(signal))))
plt.xlabel("frequency, Hz")
plt.ylabel("|S(f)|")
plt.title(f"Base spectrum")

plt.show()

# 2. Decimation
decim_Fs = Fs/L

decim_timeline = np.arange(start, end, 1/decim_Fs)
decim_signal = dm.decimation(signal, L)

# 2.1.1 Signal after downsampling in time domain
plt.subplot(2, 1, 1)
plt.plot(decim_timeline, decim_signal)
plt.xlabel("time, s")
plt.ylabel("A, V")
plt.title(f"Signal AFTER downsampling")

# 2.1.1 Signal after downsampling in frequency domain
plt.subplot(2, 1, 2)
plt.plot(np.fft.fftshift(np.fft.fftfreq(len(decim_signal), 1/decim_Fs)), np.abs(np.fft.fftshift(np.fft.fft(decim_signal))))
plt.xlabel("frequency, Hz")
plt.ylabel("|S(f)|")
plt.title(f"Spectrum AFTER downsampling")

plt.show()

# 3. Interpolation

# 3.1 Upsampling
new_timeline = np.linspace(start, end, len(decim_signal)*L, endpoint=False)
ups_signal = dm.upsampling(signal=decim_signal, L=L)

# 3.1.1 Upsampling signal in time domain
plt.subplot(2, 1, 1)
plt.plot(new_timeline, ups_signal)
plt.xlabel("time, s")
plt.ylabel("A, V")
plt.title(f"Signal AFTER upsampling")

# 3.1.2 Upsampling signal in frequency domain
plt.subplot(2, 1, 2)
plt.stem(np.fft.fftshift(np.fft.fftfreq(len(ups_signal), Fs)), np.abs(np.fft.fftshift(np.fft.fft(ups_signal))))
plt.xlabel("frequency, Hz")
plt.ylabel("|S(f)|")
plt.title(f"Spectrum AFTER upsampling")

plt.show()

# 3.2 Create FIR
FIR_timeline, FIR = dm.create_FIR(Fs, decim_Fs//2)

# 3.2.1 FIR visualization
plt.plot(FIR_timeline, FIR)
plt.xlabel("time, s")
plt.ylabel("FIR(s)")
plt.title(f"FIR")
plt.show()

# 3.3 Filtering
interpolate_singal = np.convolve(ups_signal, FIR, mode="same")

# 3.3.1 Interpolate signal in time domain
plt.subplot(2, 1, 1)
plt.plot(new_timeline, interpolate_singal)
plt.xlabel("time, s")
plt.ylabel("A, V")
plt.title(f"Signal AFTER filtering")

# 3.3.2 Interpolate signal in frequency domain
plt.subplot(2, 1, 2)
plt.stem(np.fft.fftshift(np.fft.fftfreq(len(interpolate_singal), Ts)), np.abs(np.fft.fftshift(np.fft.fft(interpolate_singal))))
plt.xlabel("frequency, Hz")
plt.ylabel("|S(f)|")
plt.title(f"Sinusoid spectrum, f={f}, Fs={Fs}")

plt.show()

