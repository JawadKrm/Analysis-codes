import numpy as np
from tkinter import filedialog
import os

def _clean_channel_array(arr):
    return np.asarray(arr, dtype=float).squeeze()

def generate_beta_signal_dict(sampling_rate=1024, duration=1.0):
    """
    Generate a synthetic 1D signal with user-defined powers per frequency
    in the beta range (Hz) for testing PSD tools.
    Returns a dict: {'ch1': {'sampling_rate': fs, 'data': signal}}
    """
    freq_power_map = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 3,
        10: 4,
        11: 5,
        12: 6,
        13: 7,
        14: 8,
        15: 7,
        16: 6,
        17: 0,
        18: 3,
        19: 4,
        20: 5,
        21: 4,
        22: 3,
        23: 0,
        24: 0,
        25: 0,
        26: 0,
        27: 0,
        28: 0,
        29: 0,
        30: 0
    }
    max_freq = max(freq_power_map.keys())
    t = np.arange(0, duration, 1/sampling_rate)
    signal = np.zeros_like(t)

    for f, p in freq_power_map.items():
        if p <= 0:
            continue
        phase = np.random.uniform(0, 2*np.pi)  
        amp = np.sqrt(p)
        signal += amp * np.sin(2 * np.pi * f * t + phase)

    # convert to dictionary form
    signal_dict = {'ch1': {'sampling_rate': float(sampling_rate), 'data': _clean_channel_array(signal)}}
    return signal_dict


def save_signal_dict(signal_dict):
    filepath = filedialog.asksaveasfilename(
        title="Save signal dictionary",
        defaultextension=".npy",
        filetypes=[("NumPy files", "*.npy")]
    )
    if not filepath:
        return
    np.save(filepath, signal_dict)
    print(f"Saved: {os.path.basename(filepath)}")

if __name__ == "__main__":
    signal_dict = generate_beta_signal_dict(sampling_rate=1024, duration=25)
    save_signal_dict(signal_dict)
