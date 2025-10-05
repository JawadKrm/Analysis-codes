import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import neo


def load_file():
    filepath = filedialog.askopenfilename(
        title="Select SMR or NPY File",
        filetypes=[("SMR files", "*.smr"), ("NumPy files", "*.npy")]
    )
    if not filepath:
        return

    try:
        if filepath.endswith(".smr"):
            reader = neo.io.Spike2IO(filename=filepath)
            block = reader.read(lazy=False)[0]
            seg = block.segments[0]
            analogs = seg.analogsignals
            if len(analogs) != 1:
                messagebox.showwarning("Warning",
                                       f"File has {len(analogs)} channels. Only single-channel files supported.")
                return
            fs = float(analogs[0].sampling_rate)
        elif filepath.endswith(".npy"):
            data = np.load(filepath, allow_pickle=True)
            if isinstance(data, dict):
                keys = list(data.keys())
                if len(keys) != 1:
                    messagebox.showwarning("Warning",
                                           f"NPY dictionary has {len(keys)} channels. Only single-channel supported.")
                    return
                fs = data[keys[0]]["sampling_rate"]
            else:
                messagebox.showwarning("Warning", "NPY file must be a dictionary with one channel.")
                return
        else:
            messagebox.showerror("Error", "Unsupported file type.")
            return
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load file: {e}")
        return

    messagebox.showinfo("Sampling Rate", f"Sampling rate: {fs} Hz")


# ---------------- GUI ----------------
root = tk.Tk()
root.title("Single-Channel File Samplerate")

frame = tk.Frame(root, padx=10, pady=10)
frame.pack()

tk.Button(frame, text="Browse File", command=load_file, width=20).pack(pady=5)

root.mainloop()
