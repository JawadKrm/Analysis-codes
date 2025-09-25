"""
PeakAnalysis v0.1 (Prototype) with multi-channel selection & preview

Created on Thursday Sep 18 2025

@author - writing and design: Jawad_Karim
Reviews and edits: Bara_Kharseh
Lab Supervisor: Dr. William D. Hutchison

- Intro -> Config -> Channel selection (if multiple) -> Spectrum Explorer
- Peak detection: parabolic interpolation in beta band (13-30 Hz)
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor, TextBox, Button
from scipy.signal import welch
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# -------------------------
# Helper functions
# -------------------------
def list_power_of_two(min_pow=4, max_pow=18):
    return [2 ** p for p in range(min_pow, max_pow + 1)]

def safe_get_channel_name(sig):
    # Try common places for channel names in neo objects
    name = getattr(sig, 'name', None)
    if not name:
        ann = getattr(sig, 'annotations', {})
        name = ann.get('name') if isinstance(ann, dict) else None
    if not name:
        # sometimes channel_name in annotations
        if isinstance(ann, dict):
            name = ann.get('channel_name') or ann.get('label') or ann.get('title')
    return name or ""

# -------------------------
# Intro Window
# -------------------------
def show_intro_then_config():
    intro = tk.Tk()
    intro.title("PeakAnalysis v0.1 (Prototype)")

    w = 720; h = 360
    sw = intro.winfo_screenwidth(); sh = intro.winfo_screenheight()
    x = int((sw - w) / 2); y = int((sh - h) / 3)
    intro.geometry(f"{w}x{h}+{x}+{y}")

    text = (
        "Welcome to PeakAnalysis v0.1 (Prototype)  🧠\n\n"
        "This interactive GUI analyzes power spectra of a signal channel to extract the "
        "beta frequency peak (13–30 Hz) and its amplitude.\n\n"
        "You can navigate through the time block flexibly using the arrow functions and time inputs.\n\n"
        "Designed and written by: Jawad Karim\n"
        "Reviewed & edited by: Bara Kharseh\n"
        "Supervision: Dr. William D. Hutchison (Hutchison Lab, University of Toronto)\n\n"
        "For more code: https://github.com/wdhutchison-neurophysiolab\n"
        "Contact: j.karim@mail.utoronto.ca\n\n"
        "NOTE: This code does not preprocess the channels. You must preprocess channels before analyzing here."
    )

    lbl = tk.Label(intro, text=text, justify="left", wraplength=680, padx=12, pady=12, font=("Segoe UI", 10))
    lbl.pack(fill="both", expand=True)

    def on_continue():
        intro.destroy()
        show_config_window()

    btn = tk.Button(intro, text="Continue to Setup", command=on_continue, font=("Segoe UI", 11), padx=8, pady=6)
    btn.pack(pady=(0, 12))

    intro.mainloop()

# -------------------------
# Config Window
# -------------------------
def show_config_window():
    cfg = tk.Tk()
    cfg.title("PeakAnalysis Setup")

    w = 620; h = 280
    sw = cfg.winfo_screenwidth(); sh = cfg.winfo_screenheight()
    x = int((sw - w) / 2); y = int((sh - h) / 3)
    cfg.geometry(f"{w}x{h}+{x}+{y}")

    frm = tk.Frame(cfg, padx=12, pady=12)
    frm.pack(fill="both", expand=True)

    tk.Label(frm, text="Enter .SMR file path:", anchor="w").grid(row=0, column=0, sticky="w")
    file_entry = tk.Entry(frm, width=58)
    file_entry.grid(row=1, column=0, columnspan=2, sticky="w")
    def browse():
        fp = filedialog.askopenfilename(title="Select .SMR file", filetypes=[("SMR files", "*.smr"), ("All files", "*.*")])
        if fp:
            file_entry.delete(0, tk.END)
            file_entry.insert(0, fp)
    tk.Button(frm, text="Browse...", command=browse).grid(row=1, column=2, padx=(6,0))

    tk.Label(frm, text="FFT window (nperseg):", anchor="w").grid(row=2, column=0, sticky="w", pady=(10,0))
    pow2_list = list_power_of_two(min_pow=4, max_pow=18)  # 16..262144
    fft_combo = ttk.Combobox(frm, values=pow2_list, state="readonly", width=18)
    fft_combo.set(256)
    fft_combo.grid(row=3, column=0, sticky="w")

    tk.Label(frm, text="Initial start time (s):", anchor="w").grid(row=2, column=1, sticky="w", pady=(10,0))
    start_input = tk.Entry(frm, width=14); start_input.insert(0, "0.0"); start_input.grid(row=3, column=1, sticky="w")

    tk.Label(frm, text="Initial end time (s):", anchor="w").grid(row=2, column=2, sticky="w", pady=(10,0))
    end_input = tk.Entry(frm, width=14); end_input.insert(0, "50.0"); end_input.grid(row=3, column=2, sticky="w")

    def start_analysis():
        file_path = file_entry.get().strip()
        try:
            nfft_val = int(fft_combo.get())
        except Exception:
            messagebox.showerror("FFT error", "Pick a valid FFT value from the dropdown.")
            return
        try:
            s0 = float(start_input.get()); e0 = float(end_input.get())
            if s0 >= e0: messagebox.showerror("Time error", "Start must be < End."); return
            if s0 < 0 or e0 <= 0: messagebox.showerror("Time error", "Times must be >= 0."); return
        except Exception:
            messagebox.showerror("Time error", "Enter valid numeric start/end times."); return

        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("File error", "Please provide a valid file path.")
            return

        # Read the SMR and inspect channels
        try:
            import neo
            reader = neo.io.Spike2IO(filename=file_path)
            block = reader.read_block()
            seg = block.segments[0]
            channels = seg.analogsignals
            n_channels = len(channels)
        except Exception as exc:
            messagebox.showerror("SMR read error", f"Failed reading SMR:\n{exc}")
            return

        # If only one channel -> inform and preview, then go to explorer
        cfg.destroy()
        if n_channels == 1:
            messagebox.showinfo("Single channel", "The file you entered has a single channel which will be analyzed.")
            # preview and confirm
            confirmed = preview_single_channel_and_confirm(channels, 0, file_path)
            if confirmed:
                launch_spectrum_explorer(file_path, nfft_val, s0, e0, channel_idx=0)
            else:
                # user canceled: go back to config
                show_config_window()
        else:
            # multiple channels -> open channel selection window
            choose_channel_window(file_path, channels, nfft_val, s0, e0)

    btn_start = tk.Button(frm, text="Start Analysis", command=start_analysis, padx=10, pady=6)
    btn_start.grid(row=5, column=1, pady=(18,0))

    help_text = "Notes: nperseg will be used for Welch nperseg (<= segment length). Use Process to compute from inputs. This tool does NOT preprocess channels."
    tk.Label(frm, text=help_text, wraplength=580, justify="left", fg="gray").grid(row=6, column=0, columnspan=3, pady=(12,0))

    cfg.mainloop()

# -------------------------
# Channel selection + preview
# -------------------------
def preview_single_channel_and_confirm(channels, idx, file_path):
    """Plot single channel snippet and ask the user to confirm yes/no/print all (returns True if confirmed)."""
    # Show plot
    sig = channels[idx]
    sr = float(sig.sampling_rate)
    data = np.asarray(sig).squeeze().astype(float)
    # plot a short snippet: first min(5s, rec_len)
    rec_len = len(data) / sr
    t_end = min(5.0, rec_len)
    t = np.linspace(0, t_end, int(t_end * sr), endpoint=False)
    snippet = data[:len(t)]
    plt.figure(figsize=(8, 2.2))
    plt.plot(t, snippet, color='black')
    ch_name = safe_get_channel_name(sig)
    title = f"Channel {idx+1}" + (f" - {ch_name}" if ch_name else "")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show(block=False)

    # Ask for confirm
    root = tk.Tk(); root.withdraw()
    ans = messagebox.askquestion("Confirm channel", f"Below is a plot of Channel {idx+1}. Is this the channel you want to analyze?")
    plt.close()
    root.destroy()
    return (ans == 'yes')

def choose_channel_window(file_path, channels, nfft_val, s0, e0):
    """Open a Tk window to enter channel number, preview, or browse batches of 5."""
    n_channels = len(channels)
    win = tk.Tk(); win.title("Choose channel to analyze")
    w = 520; h = 220
    sw = win.winfo_screenwidth(); sh = win.winfo_screenheight()
    x = int((sw - w) / 2); y = int((sh - h) / 3)
    win.geometry(f"{w}x{h}+{x}+{y}")

    frm = tk.Frame(win, padx=12, pady=12)
    frm.pack(fill="both", expand=True)

    tk.Label(frm, text=f"This SMR file contains {n_channels} channels. Which channel would you like to analyze? (Enter channel number, 1 = first channel)", wraplength=480, justify="left").grid(row=0, column=0, columnspan=3, sticky="w")

    tk.Label(frm, text="Channel number:", anchor="w").grid(row=1, column=0, sticky="w")
    ch_entry = tk.Entry(frm, width=8); ch_entry.grid(row=1, column=1, sticky="w")
    ch_entry.insert(0, "1")

    def show_preview():
        try:
            chn = int(ch_entry.get())
            if not (1 <= chn <= n_channels):
                messagebox.showerror("Range error", f"Enter 1..{n_channels}")
                return
        except Exception:
            messagebox.showerror("Value error", "Enter a valid integer")
            return
        # plot this channel and ask confirm
        confirmed = preview_single_channel_and_confirm(channels, chn - 1, file_path)
        if confirmed:
            win.destroy()
            launch_spectrum_explorer(file_path, nfft_val, s0, e0, channel_idx=chn - 1)
        else:
            # show options: user can choose Print All or re-enter a number
            pass

    def open_batch_browser():
        # If many channels > 5 we warn
        if n_channels > 5:
            proceed = messagebox.askyesno("Many channels", f"This file has {n_channels} channels. We'll show 5 at a time. Continue?")
            if not proceed:
                return
        win.withdraw()
        browse_batches(channels, file_path)
        win.deiconify()

    btn_preview = tk.Button(frm, text="Show Channel", command=show_preview)
    btn_preview.grid(row=1, column=2, padx=(8,0))

    btn_batch = tk.Button(frm, text="Browse channels (5 at a time)", command=open_batch_browser)
    btn_batch.grid(row=2, column=0, columnspan=2, pady=(12,0))

    tk.Button(frm, text="Cancel / Back", command=lambda: (win.destroy(), show_config_window())).grid(row=3, column=0, pady=(14,0))

    win.mainloop()

def browse_batches(channels, file_path):
    """Show batches of up to 5 channels in a Matplotlib window with Next/Prev navigation.
       After browsing user clicks 'I have found my channel' to return to channel input."""
    n = len(channels)
    batch_size = 5
    start_idx = 0

    def show_batch(si):
        plt.close('batch_fig')  # close previous
        fig = plt.figure(num='batch_fig', figsize=(12, 6))
        figs = []
        end_idx = min(si + batch_size, n)
        for i, ch_idx in enumerate(range(si, end_idx)):
            ax = fig.add_subplot(1, min(batch_size, end_idx-si), i+1)
            sig = channels[ch_idx]
            sr = float(sig.sampling_rate)
            data = np.asarray(sig).squeeze().astype(float)
            # plot short snippet (first few seconds or decimated if long)
            t_end = min(3.0, len(data)/sr)  # 3-second preview
            L = int(max(1, t_end*sr))
            t = np.linspace(0, t_end, L, endpoint=False)
            snippet = data[:L]
            ax.plot(t, snippet, color='black', linewidth=0.6)
            name = safe_get_channel_name(sig)
            label = f"Ch {ch_idx+1}" + (f"\n{name}" if name else "")
            ax.set_title(label, fontsize=9)
            ax.set_xlabel("s")
            ax.set_yticklabels([])
            ax.grid(False)
        fig.suptitle(f"Channels {si+1} to {end_idx} (of {n}) - preview (3 s snippet each)", fontsize=10)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show(block=False)
        return fig

    # Open a small Tk window to navigate
    nav = tk.Tk(); nav.title("Browse channels (preview)")
    nav.geometry("420x120")
    info = tk.Label(nav, text="Previewing channels in batches of 5. Use Next/Prev to move. When you found your channel, click 'I have found my channel' and enter its number in the next dialog.")
    info.pack(padx=8, pady=(6,4))

    frame = tk.Frame(nav)
    frame.pack()

    btn_prev = tk.Button(frame, text="◀ Prev")
    btn_next = tk.Button(frame, text="Next ▶")
    btn_choose = tk.Button(frame, text="I have found my channel")
    btn_close = tk.Button(frame, text="Close / Back", command=lambda: (plt.close('batch_fig'), nav.destroy()))
    btn_prev.grid(row=0, column=0, padx=6)
    btn_choose.grid(row=0, column=1, padx=6)
    btn_next.grid(row=0, column=2, padx=6)
    btn_close.grid(row=0, column=3, padx=6)

    current_si = 0
    fig = show_batch(current_si)

    def go_prev():
        nonlocal current_si, fig
        if current_si - batch_size >= 0:
            current_si -= batch_size
            plt.close('batch_fig')
            fig = show_batch(current_si)

    def go_next():
        nonlocal current_si, fig
        if current_si + batch_size < n:
            current_si += batch_size
            plt.close('batch_fig')
            fig = show_batch(current_si)

    def choose_channel_and_close():
        # ask user to enter channel number in a small dialog
        ch_num = simple_input_box("Select channel", f"Enter channel number (1..{n}) you identified:")
        try:
            chn = int(ch_num)
            if 1 <= chn <= n:
                plt.close('batch_fig')
                nav.destroy()
                # preview that single channel and confirm
                confirmed = preview_single_channel_and_confirm(channels, chn - 1, file_path)
                if confirmed:
                    # if confirmed, launch explorer
                    # Note: need to pick up nfft/start/end from previous flow; simplest: re-open config window and preserve inputs
                    # For simplicity, we will ask user to re-enter settings again.
                    messagebox.showinfo("Proceed", "Channel confirmed. Please re-open setup and press Start Analysis to continue with this channel number.")
                    show_config_window()
                else:
                    # return to nav window for more browsing if desired
                    # reopen browse window
                    browse_batches(channels, file_path)
            else:
                messagebox.showerror("Range", "Invalid channel number.")
        except Exception:
            messagebox.showerror("Value", "Enter a valid integer channel number.")

    btn_prev.config(command=go_prev); btn_next.config(command=go_next); btn_choose.config(command=choose_channel_and_close)

    nav.mainloop()
    plt.close('batch_fig')

def simple_input_box(title, prompt):
    root = tk.Tk(); root.title(title)
    w = 360; h = 120
    sw = root.winfo_screenwidth(); sh = root.winfo_screenheight()
    x = int((sw - w) / 2); y = int((sh - h) / 3)
    root.geometry(f"{w}x{h}+{x}+{y}")
    tk.Label(root, text=prompt, wraplength=320).pack(pady=(10,6))
    ent = tk.Entry(root, width=12); ent.pack()
    ent.focus()
    result = {"val": None}
    def on_ok():
        result["val"] = ent.get(); root.destroy()
    def on_cancel():
        root.destroy()
    btnf = tk.Frame(root); btnf.pack(pady=(8,6))
    tk.Button(btnf, text="OK", command=on_ok).grid(row=0, column=0, padx=6)
    tk.Button(btnf, text="Cancel", command=on_cancel).grid(row=0, column=1, padx=6)
    root.mainloop()
    return result["val"]

# -------------------------
# Spectrum Explorer (Matplotlib)
# -------------------------
def launch_spectrum_explorer(file_path, nfft_pref, start_time_init, end_time_init, channel_idx=0):
    import neo
    # Load the selected channel
    reader = neo.io.Spike2IO(filename=file_path)
    block = reader.read_block()
    seg = block.segments[0]
    if len(seg.analogsignals) <= channel_idx:
        raise IndexError("Channel index out of range.")
    signal = seg.analogsignals[channel_idx]
    sampling_rate = float(signal.sampling_rate)
    lfp = np.asarray(signal).squeeze().astype(float)
    rec_length_s = len(lfp) / sampling_rate

    state = {
        "current_start": float(start_time_init),
        "current_end": float(end_time_init),
        "nfft": int(nfft_pref),
        "sampling_rate": sampling_rate,
        "lfp": lfp,
        "rec_len": rec_length_s
    }

    plt.ion()
    fig, ax = plt.subplots(figsize=(11, 5))
    plt.subplots_adjust(bottom=0.52)
    cursor = Cursor(ax, useblit=True, color="red", linewidth=1)

    def compute_and_plot(start_time, end_time):
        sr = state["sampling_rate"]; lfp_local = state["lfp"]
        if start_time < 0: start_time = 0.0
        if end_time > state["rec_len"]: end_time = state["rec_len"]
        start_idx = int(start_time * sr); end_idx = int(end_time * sr)
        if end_idx <= start_idx + 1:
            ax.clear(); ax.text(0.5,0.5,"Selected window too short", ha='center'); fig.canvas.draw(); return
        seg_data = lfp_local[start_idx:end_idx]
        preferred = state["nfft"]
        nperseg = min(preferred, len(seg_data))
        if nperseg < 16: nperseg = max(8, len(seg_data))
        freqs, pxx_den = welch(seg_data, fs=sr, nperseg=nperseg, nfft=preferred, scaling="density")
        df = freqs[1] - freqs[0]; pxx_bin = pxx_den * df
        beta_mask = (freqs >= 13) & (freqs <= 30)
        if not np.any(beta_mask):
            peak_freq = float('nan'); peak_power = float('nan')
        else:
            freqs_beta = freqs[beta_mask]; p_beta = pxx_bin[beta_mask]; peak_idx = int(np.argmax(p_beta))
            if 0 < peak_idx < len(freqs_beta)-1:
                y0,y1,y2 = np.log(p_beta[peak_idx-1:peak_idx+2]); x0,x1,x2 = freqs_beta[peak_idx-1:peak_idx+2]
                denom = (y0 - 2.0*y1 + y2); delta = 0.5*(y0 - y2)/denom if denom !=0 else 0.0
                peak_freq = x1 + delta*(x2 - x0)/2.0
                peak_power = float(np.exp(y1 - 0.25*(y0 - y2)*delta))
            else:
                peak_freq = float(freqs_beta[peak_idx]); peak_power = float(p_beta[peak_idx])
        state["current_start"] = float(start_time); state["current_end"] = float(end_time)
        ax.clear()
        ax.bar(freqs, pxx_bin, width=df, facecolor="black", edgecolor="white", linewidth=0.7, alpha=0.9)
        if not math.isnan(peak_freq):
            ax.axvline(peak_freq, color="red", linestyle="--", linewidth=1.2)
            ax.scatter(peak_freq, peak_power, color="red", zorder=10)
            annotation = f"Peak: {peak_freq:.2f} Hz\nPower: {peak_power:.9f} V²"
        else:
            annotation = "No beta band data"
        ax.text(0.98, 0.95, annotation, transform=ax.transAxes,
                fontsize=10, color="red", ha="right", va="top",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="red"))
        ax.set_xlim(0, 50)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power per bin (V²)")
        ch_name = safe_get_channel_name(signal)
        title_info = f"Welch Spectrum ({start_time:.3f}–{end_time:.3f} s)  |  File: {os.path.basename(file_path)}  |  Channel: {channel_idx+1}"
        if ch_name: title_info += f" ({ch_name})"
        ax.set_title(title_info)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.canvas.draw_idle()

    compute_and_plot(state["current_start"], state["current_end"])

    # Widgets (text boxes, buttons) - same layout/behavior as previous prototype
    axbox_start = plt.axes([0.12, 0.40, 0.13, 0.055]); tb_start = TextBox(axbox_start, "Start (s):", initial=f"{state['current_start']:.3f}")
    axbox_end = plt.axes([0.32, 0.40, 0.13, 0.055]); tb_end = TextBox(axbox_end, "End (s):", initial=f"{state['current_end']:.3f}")
    ax_proc = plt.axes([0.50, 0.40, 0.11, 0.055]); btn_proc = Button(ax_proc, "Process")
    def on_process(evt):
        try: s = float(tb_start.text); e = float(tb_end.text)
        except Exception: print("Invalid start/end"); return
        if s < 0 or e <= s: print("Invalid time window"); return
        compute_and_plot(s, e)
    btn_proc.on_clicked(on_process)

    # 1-second arrows auto-process (shift from last processed window)
    ax_left1 = plt.axes([0.12, 0.30, 0.09, 0.05]); ax_right1 = plt.axes([0.24, 0.30, 0.09, 0.05])
    btn_left1 = Button(ax_left1, "◀ 1s"); btn_right1 = Button(ax_right1, "1s ▶")
    def left1_cb(evt):
        s = state["current_start"]; e = state["current_end"]
        if s - 1 >= 0:
            s -= 1.0; e -= 1.0; tb_start.set_val(f"{s:.3f}"); tb_end.set_val(f"{e:.3f}"); compute_and_plot(s,e)
        else: print("Start cannot go below 0s.")
    btn_left1.on_clicked(left1_cb)
    def right1_cb(evt):
        s = state["current_start"]; e = state["current_end"]
        if e + 1.0 <= state["rec_len"]:
            s += 1.0; e += 1.0; tb_start.set_val(f"{s:.3f}"); tb_end.set_val(f"{e:.3f}"); compute_and_plot(s,e)
        else: print("End exceeds recording length.")
    btn_right1.on_clicked(right1_cb)

    # Fine-step input + fine arrows auto-process
    ax_step = plt.axes([0.375, 0.30, 0.08, 0.05]); tb_step = TextBox(ax_step, "Step (s)", initial="0.1")
    ax_leftfine = plt.axes([0.465, 0.30, 0.09, 0.05]); ax_rightfine = plt.axes([0.575, 0.30, 0.09, 0.05])
    btn_leftfine = Button(ax_leftfine, "◀ Fine"); btn_rightfine = Button(ax_rightfine, "Fine ▶")
    def leftfine_cb(evt):
        try: step = float(tb_step.text);
        except Exception: print("Invalid step"); return
        s = state["current_start"]; e = state["current_end"]
        if s - step >= 0: s -= step; e -= step; tb_start.set_val(f"{s:.3f}"); tb_end.set_val(f"{e:.3f}"); compute_and_plot(s,e)
        else: print("Start cannot go below 0s.")
    btn_leftfine.on_clicked(leftfine_cb)
    def rightfine_cb(evt):
        try: step = float(tb_step.text);
        except Exception: print("Invalid step"); return
        s = state["current_start"]; e = state["current_end"]
        if e + step <= state["rec_len"]: s += step; e += step; tb_start.set_val(f"{s:.3f}"); tb_end.set_val(f"{e:.3f}"); compute_and_plot(s,e)
        else: print("End exceeds recording length.")
    btn_rightfine.on_clicked(rightfine_cb)

    # Small arrows next to start/end inputs (change values only, no auto-process)
    ax_s_left = plt.axes([0.12, 0.22, 0.05, 0.04]); ax_s_right = plt.axes([0.19, 0.22, 0.05, 0.04])
    btn_s_left = Button(ax_s_left, "◀"); btn_s_right = Button(ax_s_right, "▶")
    def s_left_cb(evt):
        try: v = float(tb_start.text); v_new = max(0.0, v - 1.0); tb_start.set_val(f"{v_new:.3f}")
        except Exception: pass
    btn_s_left.on_clicked(s_left_cb)
    def s_right_cb(evt):
        try: v = float(tb_start.text); tb_start.set_val(f"{v + 1.0:.3f}")
        except Exception: pass
    btn_s_right.on_clicked(s_right_cb)

    ax_e_left = plt.axes([0.32, 0.22, 0.05, 0.04]); ax_e_right = plt.axes([0.39, 0.22, 0.05, 0.04])
    btn_e_left = Button(ax_e_left, "◀"); btn_e_right = Button(ax_e_right, "▶")
    def e_left_cb(evt):
        try: v = float(tb_end.text); v_new = max(0.0, v - 1.0); tb_end.set_val(f"{v_new:.3f}")
        except Exception: pass
    btn_e_left.on_clicked(e_left_cb)
    def e_right_cb(evt):
        try: v = float(tb_end.text); tb_end.set_val(f"{v + 1.0:.3f}")
        except Exception: pass
    btn_e_right.on_clicked(e_right_cb)

    # Status line
    fig.text(0.02, 0.96, f"File: {os.path.basename(file_path)}  |  Recording: {state['rec_len']:.1f}s  |  Fs: {state['sampling_rate']:.1f} Hz  |  nperseg(pref): {state['nfft']}", fontsize=9, color="gray")

    plt.show(block=True)


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    show_intro_then_config()
