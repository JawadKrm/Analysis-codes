import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
import neo
import os

# --------------------------
# Helper functions
# --------------------------
def parabolic_peak_interpolation(f, Pxx, idx, use_log=True):
    """Refines peak location using parabolic interpolation."""
    if idx <= 0 or idx >= len(Pxx) - 1:
        return f[idx], Pxx[idx]
    alpha, beta, gamma = Pxx[idx - 1], Pxx[idx], Pxx[idx + 1]
    if use_log:
        alpha, beta, gamma = np.log(alpha), np.log(beta), np.log(gamma)
    p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
    peak_freq = f[idx] + p * (f[1] - f[0])
    if use_log:
        peak_power = np.exp(beta - 0.25 * (alpha - gamma) * p)
    else:
        peak_power = beta - 0.25 * (alpha - gamma) * p
    return peak_freq, peak_power


def find_beta_peak(freqs, pxx_bin, beta_band=(14.0, 30.0), use_log=True):
    """Finds the main beta peak using scipy.find_peaks + parabolic refinement."""
    mask = (freqs >= beta_band[0]) & (freqs <= beta_band[1])
    if not np.any(mask):
        return float('nan'), float('nan')
    f_beta = np.asarray(freqs[mask], dtype=float)
    p_beta = np.asarray(pxx_bin[mask], dtype=float)
    if len(p_beta) < 3:
        return float('nan'), float('nan')

    # --- Find local peaks based on prominence ---
    peaks, properties = find_peaks(p_beta, prominence=0.5, width=1)
    if len(peaks) == 0:
        rel_peak_idx = int(np.argmax(p_beta))
    else:
        prominences = properties["prominences"]
        rel_peak_idx = peaks[np.argmax(prominences)]

    return parabolic_peak_interpolation(f_beta, p_beta, rel_peak_idx, use_log=use_log)


def compute_psd(data, fs, nfft, tmin, tmax):
    """Compute PSD for given time window using Welch method."""
    start_idx = int(round(tmin * fs))
    end_idx = int(round(tmax * fs))
    segment = data[start_idx:end_idx]
    if len(segment) <= 0:
        return None, None
    nperseg = min(nfft, len(segment))
    f, Pxx = welch(segment, fs=fs, nperseg=nperseg)
    return f, Pxx


# --------------------------
# File loading & dict handling
# --------------------------
def confirm_sampling_rate(initial_fs):
    """Ask the user to confirm or change the sampling frequency."""
    response = messagebox.askyesno(
        "Sampling Frequency Check",
        f"Detected sampling frequency = {initial_fs:.2f} Hz.\n\nIs this correct?"
    )
    if response:
        return initial_fs
    else:
        user_fs = simpledialog.askfloat(
            "Manual Sampling Frequency Input",
            "Enter the correct sampling frequency (Hz):",
            minvalue=1.0
        )
        if user_fs is None:
            messagebox.showwarning("No input", "Sampling frequency unchanged.")
            return initial_fs
        return user_fs


def load_file():
    filepath = filedialog.askopenfilename(
        title="Select File", filetypes=[("Spike2 SMR", "*.smr"), ("NumPy", "*.npy")]
    )
    if not filepath:
        return None, None, None

    data_dict = {}
    try:
        if filepath.endswith(".smr"):
            reader = neo.io.Spike2IO(filename=filepath)
            block = reader.read(lazy=False)[0]
            seg = block.segments[0]
            analogs = seg.analogsignals
            for i, sig in enumerate(analogs):
                detected_fs = float(sig.sampling_rate)
                fs_checked = confirm_sampling_rate(detected_fs)
                data_dict[f"ch{i + 1}"] = {
                    "sampling_rate": fs_checked,
                    "data": np.array(sig).squeeze()
                }

        elif filepath.endswith(".npy"):
            obj = np.load(filepath, allow_pickle=True)
            if isinstance(obj, np.ndarray) and obj.dtype == np.object_ and obj.shape == ():
                obj = obj.item()
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, dict) and "data" in v and "sampling_rate" in v:
                        detected_fs = float(v["sampling_rate"])
                    else:
                        detected_fs = 30000.0
                    fs_checked = confirm_sampling_rate(detected_fs)
                    data_dict[k] = {
                        "sampling_rate": fs_checked,
                        "data": np.asarray(v["data"] if isinstance(v, dict) else v).squeeze()
                    }
            else:
                arr = np.asarray(obj)
                fs_checked = confirm_sampling_rate(30000.0)
                if arr.ndim == 1:
                    data_dict["ch1"] = {"sampling_rate": fs_checked, "data": arr}
                elif arr.ndim == 2:
                    if arr.shape[0] < arr.shape[1]:
                        nch, ns = arr.shape
                        for i in range(nch):
                            data_dict[f"ch{i+1}"] = {"sampling_rate": fs_checked, "data": arr[i, :]}
                    else:
                        ns, nch = arr.shape
                        for i in range(nch):
                            data_dict[f"ch{i+1}"] = {"sampling_rate": fs_checked, "data": arr[:, i]}
        else:
            messagebox.showerror("Error", "Unsupported file type.")
            return None, None, None

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load file: {e}")
        return None, None, None

    root.loaded_data = data_dict
    root.loaded_path = filepath
    ch_keys = list(data_dict.keys())
    if not ch_keys:
        messagebox.showerror("Error", "No channels found in file.")
        return None, None, None
    channel_menu['values'] = ch_keys
    channel_var.set(ch_keys[0])

    messagebox.showinfo("Loaded", f"Loaded {os.path.basename(filepath)} with {len(ch_keys)} channel(s).")
    return data_dict, filepath, ch_keys

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
    plt.plot(t, data)
    plt.title(f"Preview of {ch_name} (fs={fs:.1f} Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()



# --------------------------
# PSD computation per phase
# --------------------------
def compute_phase_psds(signal, fs, nfft, window, step, phase_start, phase_end, beta_band=(14, 30)):
    psd_list = []
    per_window_info = []
    freqs = None
    t = phase_start

    while t + window <= phase_end:
        f, Pxx = compute_psd(signal, fs, nfft, t, t + window)
        if f is None:
            t += step
            continue
        freqs = f
        peak_f, peak_p = find_beta_peak(f, Pxx, beta_band=beta_band)
        psd_list.append(Pxx)
        per_window_info.append((f"{t:.2f}-{t+window:.2f}", peak_f, peak_p))
        t += step

    return freqs, psd_list, per_window_info


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
    signal = np.asarray(selected_channel["data"]).squeeze()
    fs = float(selected_channel["sampling_rate"])

    try:
        fft_size = int(fft_var.get())
        window = float(window_entry.get())
        step = float(step_entry.get())
        t_start = float(start_entry.get())
        t_end = float(end_entry.get())
        task_start = float(task_start_entry.get())
        task_end = float(task_end_entry.get())
    except Exception as e:
        messagebox.showerror("Error", f"Invalid numeric parameter: {e}")
        return

    if not (t_start < task_start < task_end < t_end):
        messagebox.showerror("Error", "Times must satisfy: start < task_start < task_end < end")
        return

    beta_band = (14, 30)
    freqs_b, psds_b, info_b = compute_phase_psds(signal, fs, fft_size, window, step, t_start, task_start, beta_band)
    freqs_t, psds_t, info_t = compute_phase_psds(signal, fs, fft_size, window, step, task_start, task_end, beta_band)
    freqs_a, psds_a, info_a = compute_phase_psds(signal, fs, fft_size, window, step, task_end, t_end, beta_band)

    freqs = next((arr for arr in [freqs_b, freqs_t, freqs_a] if arr is not None), None)
    if freqs is None:
        messagebox.showwarning("No freq axis", "Unable to compute any PSDs.")
        return

    def average_psd_list(psd_list):
        if not psd_list:
            return np.full_like(freqs, np.nan)
        return np.nanmean(np.vstack(psd_list), axis=0)

    avg_psd_b = average_psd_list(psds_b)
    avg_psd_t = average_psd_list(psds_t)
    avg_psd_a = average_psd_list(psds_a)

    avg_b_f, avg_b_p = find_beta_peak(freqs, avg_psd_b)
    avg_t_f, avg_t_p = find_beta_peak(freqs, avg_psd_t)
    avg_a_f, avg_a_p = find_beta_peak(freqs, avg_psd_a)

    plt.figure(figsize=(9, 6))
    plt.semilogy(freqs, avg_psd_b, label=f"Baseline avg (peak {avg_b_f:.2f} Hz)")
    plt.semilogy(freqs, avg_psd_t, label=f"Task avg (peak {avg_t_f:.2f} Hz)")
    plt.semilogy(freqs, avg_psd_a, label=f"After-task avg (peak {avg_a_f:.2f} Hz)")
    for f_ in [avg_b_f, avg_t_f, avg_a_f]:
        if not np.isnan(f_):
            plt.axvline(f_, linestyle="--")
    plt.xlim(0, min(100, freqs.max()))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title("Averaged PSDs: Baseline vs Task vs After-task")
    plt.legend()
    plt.show()

    results_lines = [
        f"Baseline averaged-PSD peak: {avg_b_f:.4f} Hz, power {avg_b_p:.4e}",
        f"Task averaged-PSD peak:     {avg_t_f:.4f} Hz, power {avg_t_p:.4e}",
        f"After-task averaged-PSD peak: {avg_a_f:.4f} Hz, power {avg_a_p:.4e}",
    ]
    messagebox.showinfo("Averaged PSD peaks", "\n".join(results_lines))

    for row in table.get_children():
        table.delete(row)

    def insert_info_list(phase_name, info_list):
        for t_str, f_val, p_val in info_list:
            f_str = f"{f_val:.4f}" if not np.isnan(f_val) else "nan"
            p_str = f"{p_val:.2e}" if not np.isnan(p_val) else "nan"
            table.insert("", "end", values=(phase_name, t_str, f_str, p_str))

    insert_info_list("Baseline", info_b)
    insert_info_list("Task", info_t)
    insert_info_list("After-task", info_a)


# --------------------------
# GUI Setup
# --------------------------
root = tk.Tk()
root.title("Beta Peak Analysis GUI (find_peaks + Sampling Rate Check)")

frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0)

ttk.Button(frame, text="Load File (.smr/.npy)", command=load_file).grid(row=0, column=0, columnspan=2, pady=5)
ttk.Button(frame, text="Preview Channel", command=preview_channel).grid(row=1, column=0, columnspan=2, pady=5)

ttk.Label(frame, text="Channel:").grid(row=2, column=0, sticky="w")
channel_var = tk.StringVar(value="ch1")
channel_menu = ttk.Combobox(frame, textvariable=channel_var, values=[])
channel_menu.grid(row=2, column=1)

ttk.Label(frame, text="FFT Size:").grid(row=3, column=0, sticky="w")
fft_var = tk.StringVar(value="32768")
fft_menu = ttk.Combobox(frame, textvariable=fft_var,
                        values=["1024","2048","4096","8192","16384","32768","65536","131072","262144"])
fft_menu.grid(row=3, column=1)

ttk.Label(frame, text="Window (s):").grid(row=4, column=0, sticky="w")
window_entry = ttk.Entry(frame)
window_entry.insert(0, "5")
window_entry.grid(row=4, column=1)

ttk.Label(frame, text="Step (s):").grid(row=5, column=0, sticky="w")
step_entry = ttk.Entry(frame)
step_entry.insert(0, "5")
step_entry.grid(row=5, column=1)

ttk.Label(frame, text="Analysis start (s):").grid(row=6, column=0, sticky="w")
start_entry = ttk.Entry(frame)
start_entry.insert(0, "0")
start_entry.grid(row=6, column=1)

ttk.Label(frame, text="Analysis end (s):").grid(row=7, column=0, sticky="w")
end_entry = ttk.Entry(frame)
end_entry.insert(0, "60")
end_entry.grid(row=7, column=1)

ttk.Label(frame, text="Task start (s):").grid(row=8, column=0, sticky="w")
task_start_entry = ttk.Entry(frame)
task_start_entry.insert(0, "15")
task_start_entry.grid(row=8, column=1)

ttk.Label(frame, text="Task end (s):").grid(row=9, column=0, sticky="w")
task_end_entry = ttk.Entry(frame)
task_end_entry.insert(0, "45")
task_end_entry.grid(row=9, column=1)

ttk.Button(frame, text="Run Analysis", command=run_analysis).grid(row=10, column=0, columnspan=2, pady=10)

table_frame = ttk.Frame(root, padding=10)
table_frame.grid(row=1, column=0, sticky="nsew")

cols = ("Phase", "Time Window", "Peak Freq (Hz)", "Peak Power")
table = ttk.Treeview(table_frame, columns=cols, show="headings", height=12)
for col in cols:
    table.heading(col, text=col)
    table.column(col, anchor="center", width=140)
table.pack(fill="both", expand=True)

root.mainloop()
