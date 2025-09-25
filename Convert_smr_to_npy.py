"""
Code 1: Convert a Spike2 .SMR file to a numpy (.npy) file.

Created on Sun Sep 21 2025

@author - writing and design: Jawad_Karim
Reviews and edits: Bara_Kharseh
Lab Supervisor: Dr. William D. Hutchison

Saves all analog channels into a dictionary with keys ch1, ch2, ..., chN.
Each channel entry contains 'sampling_rate' and 'data'.
***************In spike2, channel 1 is ch2 here and spike2 channel 2 is ch3 here***********************
"""
import neo
import numpy as np

# File paths (fill before running)
smr_file = ""   # Path to your .smr file
output_npy = "" # Path to save .npy file


# Load SMR file
reader = neo.io.Spike2IO(filename=smr_file)
block = reader.read_block()


# Convert analog channels to dictionary
channels_dict = {}
for idx, signal in enumerate(block.segments[0].analogsignals):
    ch = np.asarray(signal).squeeze().astype(float)

    # Fix duplication/stitching (common in Spike2 exports)
    if len(ch) % 2 == 0 and np.allclose(ch[0::2], ch[1::2]):
        ch = ch[::2]

    channels_dict[f"ch{idx+1}"] = {
        "sampling_rate": float(signal.sampling_rate),
        "data": ch
    }


# Save dictionary as NPY
np.save(output_npy, channels_dict, allow_pickle=True)
print(f"Saved NPY file to: {output_npy}")
