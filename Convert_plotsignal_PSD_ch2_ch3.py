"""
Code 4: Draw power spectral density (PSD) for a chosen channel from
a multi-channel NPY file.

Created on Sun Sep 21 2025

@author - writing and design: Jawad_Karim
Reviews and edits: Bara_Kharseh
Lab Supervisor: Dr. William D. Hutchison


- Uses Welch method with nfft=256 and histogram style.
- Allows a time window selection.
***************In spike2, channel 1 is ch2 here and spike2 channel 2 is ch3 here***********************
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from scipy.signal import welch

# File path (fill before running)
input_npy = ""  # Path to multi-channel NPY file
channel_to_plot = "ch2" # Change to 'ch3' for second channel
start_time = 0   # ---->  CAN CHANGE Start time in seconds
end_time = 50    # ---->  CAN CHANGE End time in seconds

# Load NPY
all_channels = np.load(input_npy, allow_pickle=True).item()
entry = all_channels[channel_to_plot]
sig = entry["data"]
fs = entry["sampling_rate"]

# Apply time window
start_idx = int(start_time * fs)
end_idx = int(end_time * fs)
sig_window = sig[start_idx:end_idx]

# Compute PSD (Welch)
nfft = 256 # CHANGE
freqs, Pxx_den = welch(sig_window, fs=fs, nperseg=nfft, nfft=nfft, scaling="density")
df = freqs[1] - freqs[0]
Pxx_bin = Pxx_den * df

# Plot histogram
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(freqs, Pxx_bin, width=df, facecolor="black", edgecolor="white", alpha=0.9)
ax.set_xlim(0, 50) # CHANGE
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power per bin (V²)")
ax.set_title(f"Welch PSD ({start_time}-{end_time}s, {channel_to_plot})")
ax.grid(True, linestyle="--", alpha=0.5)
Cursor(ax, useblit=True, color="red", linewidth=1)
plt.tight_layout()
plt.show()
