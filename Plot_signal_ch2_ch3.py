"""
Code 3: Plot the full raw signals of channels ch2 and ch3 from a
multi-channel NPY dictionary converted from SMR.

Created on Sun Sep 21 2025

@author - writing and design: Jawad_Karim
Reviews and edits: Bara_Kharseh
Lab Supervisor: Dr. William D. Hutchison

***************In spike2, channel 1 is ch2 here and spike2 channel 2 is ch3 here***********************
"""
import numpy as np
import matplotlib.pyplot as plt

# File path (fill before running)
input_npy = ""  # Path to multi-channel NPY file

# Load NPY
all_channels = np.load(input_npy, allow_pickle=True).item()

# Function to plot full signal
def plot_full_signal(signal, fs, channel_name):
    import matplotlib.pyplot as plt
    time = np.arange(len(signal)) / fs
    plt.figure(figsize=(14, 4))
    plt.plot(time, signal, color="blue")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Full Signal ({channel_name})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# Plot ch2 and ch3
for ch_name in ["ch2", "ch3"]:
    entry = all_channels[ch_name]
    sig = entry["data"]
    fs = entry["sampling_rate"]
    plot_full_signal(sig, fs, ch_name)
