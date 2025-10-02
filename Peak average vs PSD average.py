import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import neo
import os


# --------------------------
# Helper functions
# --------------------------
def parabolic_interpolation(f, Pxx, idx):
    """Refines peak location using parabolic interpolation."""
    if idx <= 0 or idx >= len(Pxx) - 1:
        return f[idx], Pxx[idx]
    alpha, beta, gamma = Pxx[idx - 1], Pxx[idx], Pxx[idx + 1]
    p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
    peak_freq = f[idx] + p * (f[1] - f[0])
    peak_power = beta - 0.25 * (alpha - gamma) * p
    return peak_freq, peak_power


def compute_psd(data, fs, nfft, tmin, tmax):
    """Compute PSD for given time window using Welch method."""
    segment = data[int(tmin * fs):int(tmax * fs)]
    f, Pxx = welch(segment, fs=fs, nperseg=nfft)
    return f, Pxx


# --------------------------
# File loading & dict handling
# --------------------------
def load_file():
    filepath = filedialog.askopenfilename(
        title="Select File", filetypes=[("Spike2 SMR", "*.smr"), ("NumPy Dict", "*.npy")]
    )
    if not filepath:
        return None, None, None

    if filepath.endswith(".smr"):
        try:
            reader = neo.io.Spike2IO(filename=filepath)
            block = reader.read(lazy=False)[0]
            seg = block.segments[0]
            analogs = seg.analogsignals
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load SMR: {e}")
            return None, None, None

        data_dict = {}
        for i, sig in enumerate(analogs):
            data_dict[f"ch{i + 1}"] = {
                "sampling_rate": float(sig.sampling_rate),
                "data": np.array(sig).squeeze()
            }

    elif filepath.endswith(".npy"):
        try:
            obj = np.load(filepath, allow_pickle=True).item()
            if not isinstance(obj, dict):
                messagebox.showerror("Error", "NPY file is not a dictionary.")
                return None, None, None
            data_dict = obj
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load NPY: {e}")
            return None, None, None
    else:
        return None, None, None

    # Fill channel dropdown
    channel_menu['values'] = list(data_dict.keys())
    channel_var.set(list(data_dict.keys())[0])

    # Save loaded data globally
    root.loaded_data = data_dict
    root.loaded_path = filepath
    return data_dict, filepath, list(data_dict.keys())


def preview_channel():
    """Preview selected channel before analysis."""
    if not hasattr(root, "loaded_data") or root.loaded_data is None:
        messagebox.showwarning("No file", "Please load a file first.")
        return
    ch_name = channel_var.get()
    ch = root.loaded_data[ch_name]
    data = ch["data"]
    fs = ch["sampling_rate"]
    t = np.arange(len(data)) / fs
    plt.figure(figsize=(8, 4))
    plt.plot(t, data, color="black")
    plt.title(f"Preview of {ch_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()


def export_dict():
    """Export currently loaded channels as dictionary .npy"""
    if not hasattr(root, "loaded_data") or root.loaded_data is None:
        messagebox.showwarning("No file", "Please load a file first.")
        return
    savepath = filedialog.asksaveasfilename(
        defaultextension=".npy", filetypes=[("NumPy Dictionary", "*.npy")]
    )
    if not savepath:
        return
    np.save(savepath, root.loaded_data, allow_pickle=True)
    messagebox.showinfo("Export", f"Dictionary saved to:\n{savepath}")


# --------------------------
# Main Analysis Function
# --------------------------
def run_analysis():
    if not hasattr(root, "loaded_data") or root.loaded_data is None:
        messagebox.showwarning("No file", "Please load a file first.")
        return

    ch_name = channel_var.get()
    if ch_name not in root.loaded_data:
        messagebox.showerror("Error", f"Channel {ch_name} not found.")
        return

    selected_channel = root.loaded_data[ch_name]
    signal = selected_channel["data"]
    fs = selected_channel["sampling_rate"]

    # User params
    fft_size = int(fft_var.get())
    window = float(window_entry.get())
    step = float(step_entry.get())
    t_start = float(start_entry.get())
    t_end = float(end_entry.get())

    psd_list = []
    peak_freqs = []
    peak_powers = []
    times = []

    t = t_start
    while t + window <= t_end:
        f, Pxx = compute_psd(signal, fs, fft_size, t, t + window)
        # restrict to beta
        mask = (f >= 14) & (f <= 30)
        if not np.any(mask):
            t += step
            continue
        idx = np.argmax(Pxx[mask])
        beta_freqs = f[mask]
        beta_powers = Pxx[mask]
        peak_f, peak_p = parabolic_interpolation(beta_freqs, beta_powers, idx)

        psd_list.append(Pxx)
        peak_freqs.append(peak_f)
        peak_powers.append(peak_p)
        times.append(f"{t:.2f}-{t + window:.2f}")

        t += step

    if not psd_list:
        messagebox.showwarning("No data", "No PSDs computed in the given range.")
        return

    # --- Method 1: average peaks
    avg_peak_freq = np.mean(peak_freqs)
    avg_peak_power = np.mean(peak_powers)

    # --- Method 2: average spectra then find peak
    avg_psd = np.mean(psd_list, axis=0)
    mask = (f >= 14) & (f <= 30)
    idx = np.argmax(avg_psd[mask])
    beta_freqs = f[mask]
    beta_powers = avg_psd[mask]
    avg_psd_peak_f, avg_psd_peak_p = parabolic_interpolation(beta_freqs, beta_powers, idx)

    # --- Plot averaged PSD
    plt.figure(figsize=(8, 5))
    plt.semilogy(f, avg_psd, label="Averaged PSD", color="black")
    plt.axvline(avg_psd_peak_f, color="red", linestyle="--", label=f"Avg PSD Peak {avg_psd_peak_f:.2f} Hz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title("Averaged PSD")
    plt.legend()
    plt.show()

    # --- Results popup
    result_text = (
        f"Method 1 (Avg of individual peaks): {avg_peak_freq:.2f} Hz, {avg_peak_power:.2f}\n"
        f"Method 2 (Peak of averaged PSD): {avg_psd_peak_f:.2f} Hz, {avg_psd_peak_p:.2f}"
    )
    messagebox.showinfo("Results", result_text)

    # --- Populate table with per-window peaks
    for row in table.get_children():
        table.delete(row)
    for t_str, f_val, p_val in zip(times, peak_freqs, peak_powers):
        table.insert("", "end", values=(t_str, f"{f_val:.2f}", f"{p_val:.2e}"))


# --------------------------
# GUI Setup
# --------------------------
root = tk.Tk()
root.title("Beta Peak Analysis GUI")

frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0)

# File load buttons
ttk.Button(frame, text="Load File (.smr/.npy)", command=load_file).grid(row=0, column=0, columnspan=2, pady=5)
ttk.Button(frame, text="Preview Channel", command=preview_channel).grid(row=1, column=0, columnspan=2, pady=5)
ttk.Button(frame, text="Export Dictionary", command=export_dict).grid(row=2, column=0, columnspan=2, pady=5)

# Channel dropdown
ttk.Label(frame, text="Channel:").grid(row=3, column=0, sticky="w")
channel_var = tk.StringVar(value="ch1")
channel_menu = ttk.Combobox(frame, textvariable=channel_var, values=[])
channel_menu.grid(row=3, column=1)

# FFT size dropdown
ttk.Label(frame, text="FFT Size:").grid(row=4, column=0, sticky="w")
fft_var = tk.StringVar(value="32768")
fft_menu = ttk.Combobox(frame, textvariable=fft_var,
                        values=["1024", "2048", "4096", "8192", "16384", "32768"])
fft_menu.grid(row=4, column=1)

# Time window controls
ttk.Label(frame, text="Window (s):").grid(row=5, column=0, sticky="w")
window_entry = ttk.Entry(frame)
window_entry.insert(0, "5")
window_entry.grid(row=5, column=1)

ttk.Label(frame, text="Step (s):").grid(row=6, column=0, sticky="w")
step_entry = ttk.Entry(frame)
step_entry.insert(0, "5")
step_entry.grid(row=6, column=1)

ttk.Label(frame, text="Start time (s):").grid(row=7, column=0, sticky="w")
start_entry = ttk.Entry(frame)
start_entry.insert(0, "0")
start_entry.grid(row=7, column=1)

ttk.Label(frame, text="End time (s):").grid(row=8, column=0, sticky="w")
end_entry = ttk.Entry(frame)
end_entry.insert(0, "60")
end_entry.grid(row=8, column=1)

ttk.Button(frame, text="Run Analysis", command=run_analysis).grid(row=9, column=0, columnspan=2, pady=10)

# Table for per-window peaks
table_frame = ttk.Frame(root, padding=10)
table_frame.grid(row=1, column=0, sticky="nsew")

cols = ("Time Window", "Peak Freq (Hz)", "Peak Power")
table = ttk.Treeview(table_frame, columns=cols, show="headings", height=10)
for col in cols:
    table.heading(col, text=col)
    table.column(col, anchor="center", width=120)
table.pack(fill="both", expand=True)

root.mainloop()
