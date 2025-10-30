import os
import numpy as np
from scipy import signal
import neo
import quantities as pq
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
from scipy.signal import medfilt, resample

try:
    from specparam import SpectralModel
    HAVE_SPECPARAM = True
except Exception as e:
    SpectralModel = None
    HAVE_SPECPARAM = False
    _SPEC_IMPORT_ERR = str(e)

beta_range = (13.0, 30.0)

def read_smr_single_channel(path):
    reader = neo.io.Spike2IO(filename=path)
    block = reader.read_block()
    seg = block.segments[0]
    a = seg.analogsignals[0]
    sig = np.asarray(a).squeeze()
    sr = float(a.sampling_rate.rescale('Hz').magnitude)
    return sig, sr

def slice_timecourse(sig, sr, t0, t1):
    n0 = int(np.floor(t0 * sr))
    n1 = int(np.floor(t1 * sr))
    n0 = max(0, n0)
    n1 = min(len(sig), n1)
    if n1 <= n0:
        return np.array([]), n0, n1
    return sig[n0:n1], n0, n1

def compute_welch(sig_segment, sr, nfft, window_name):
    len_seg = len(sig_segment)
    if len_seg < 2:
        return np.array([0.]), np.array([0.])
    nperseg = min(len_seg, nfft)
    noverlap = nperseg // 2
    window = signal.get_window(window_name, nperseg, fftbins=True)
    freqs, psd = signal.welch(sig_segment, fs=sr, window=window,
                              nperseg=nperseg, noverlap=noverlap,
                              nfft=nfft, scaling='density', detrend='constant')
    return freqs, psd

def apply_specparam_remove(freqs, psd, fit_range=(1.0, 40.0), gui_parent=None):
    if not HAVE_SPECPARAM:
        return psd
    eps = np.finfo(float).eps
    mask = freqs > 0
    if not np.any(mask):
        return psd
    fq = freqs[mask]
    pw_lin = psd[mask].astype(float).copy()
    pw_lin[~np.isfinite(pw_lin)] = eps
    pw_lin = np.maximum(pw_lin, eps)
    low = float(max(fq[0], fit_range[0]))
    high = float(min(fq[-1], fit_range[1]))
    n_in_range = np.sum((fq >= low) & (fq <= high))
    if n_in_range < 10:
        logp = np.log10(pw_lin)
        k = max(3, 1 + (len(logp)//50))
        if k % 2 == 0: k += 1
        bg = medfilt(logp, kernel_size=k)
        corrected_sub = 10**(logp - bg)
        corrected = psd.copy().astype(float)
        corrected[mask] = corrected_sub
        return corrected
    attempts = [
        {'aperiodic_mode': 'fixed', 'peak_width_limits': [1.0, 8.0], 'max_n_peaks': 6},
        {'aperiodic_mode': 'knee',  'peak_width_limits': [1.0, 8.0], 'max_n_peaks': 6},
        {'aperiodic_mode': 'fixed', 'peak_width_limits': [1.0, 8.0], 'max_n_peaks': 8, 'smooth': True},
        {'aperiodic_mode': 'fixed', 'peak_width_limits': [1.0, 8.0], 'max_n_peaks': 12}
    ]
    ap_fit = None
    fit_freqs = None
    for att in attempts:
        try:
            pw_try = pw_lin
            if att.get('smooth', False):
                k = 7 if len(pw_lin) > 50 else 3
                if k % 2 == 0: k += 1
                logp = medfilt(np.log10(pw_lin), kernel_size=k)
                pw_try = 10**logp
            fm = SpectralModel(peak_width_limits=att['peak_width_limits'],
                               max_n_peaks=att['max_n_peaks'],
                               aperiodic_mode=att['aperiodic_mode'])
            if hasattr(fm, 'fit'):
                fm.fit(fq, pw_try, freq_range=[low, high])
            else:
                try:
                    fm.report(fq, pw_try, [low, high])
                except Exception:
                    try:
                        fm.report(fq, pw_try, freq_range=[low, high])
                    except Exception:
                        pass
            ap_fit = None
            fit_freqs = None
            for name in ('_ap_fit', 'ap_fit', 'ap_fit_', 'aperiodic_fit_', 'aperiodic_fit'):
                if hasattr(fm, name):
                    ap_fit = getattr(fm, name)
                    break
            for name in ('freqs_', 'fit_freqs', 'freqs', 'f_fit'):
                if hasattr(fm, name):
                    fit_freqs = getattr(fm, name)
                    break
            if ap_fit is not None and fit_freqs is not None:
                ap_fit = np.asarray(ap_fit)
                fit_freqs = np.asarray(fit_freqs)
                if ap_fit.size >= 3 and ap_fit.shape[0] == fit_freqs.shape[0] and np.all(np.isfinite(ap_fit)):
                    break
                else:
                    ap_fit = None
                    fit_freqs = None
        except Exception:
            ap_fit = None
            fit_freqs = None
    if ap_fit is None or fit_freqs is None:
        logp = np.log10(pw_lin)
        bg = medfilt(logp, kernel_size=3)
        corrected_sub = 10**(logp - bg)
        corrected = psd.copy().astype(float)
        corrected[mask] = corrected_sub
        return corrected
    try:
        ap_fit = np.asarray(ap_fit).astype(float)
        fit_freqs = np.asarray(fit_freqs).astype(float)
        if ap_fit.shape != fq.shape:
            ap_fit_interp = np.interp(fq, fit_freqs, ap_fit)
        else:
            ap_fit_interp = ap_fit
        corrected_sub = 10**(np.log10(pw_lin) - ap_fit_interp)
        corrected = psd.copy().astype(float)
        corrected[mask] = corrected_sub
        return corrected
    except Exception:
        logp = np.log10(pw_lin)
        bg = medfilt(logp, kernel_size=3)
        corrected_sub = 10**(logp - bg)
        corrected = psd.copy().astype(float)
        corrected[mask] = corrected_sub
        return corrected

def apply_irasa_remove(freqs, psd, sig_segment, sr, nfft, h_count=4):
    eps = np.finfo(float).eps
    mask = freqs > 0
    if not np.any(mask):
        return psd
    fq = freqs[mask]
    psd_lin = psd[mask].astype(float).copy()
    psd_lin[~np.isfinite(psd_lin)] = eps
    psd_lin = np.maximum(psd_lin, eps)
    L = len(sig_segment)
    if L < 10:
        return psd
    h_count = max(2, int(h_count))
    h_vals = np.linspace(1.1, 1.9, h_count)
    geom_means = []
    for h in h_vals:
        try:
            len_down = max(10, int(np.round(L / h)))
            len_up = max(10, int(np.round(L * h)))
            sig_down = resample(sig_segment, len_down)
            sig_up = resample(sig_segment, len_up)
            sr_down = sr * (len_down / L)
            sr_up = sr * (len_up / L)
            f_down, p_down = compute_welch(sig_down, sr_down, nfft, 'hann')
            f_up, p_up = compute_welch(sig_up, sr_up, nfft, 'hann')
            p_down_interp = np.interp(fq, f_down, p_down, left=np.nan, right=np.nan)
            p_up_interp = np.interp(fq, f_up, p_up, left=np.nan, right=np.nan)
            mask_finite = np.isfinite(p_down_interp) & np.isfinite(p_up_interp)
            if not np.any(mask_finite):
                continue
            p_down_interp[~mask_finite] = eps
            p_up_interp[~mask_finite] = eps
            g = np.sqrt(p_down_interp * p_up_interp)
            g = np.maximum(g, eps)
            geom_means.append(g)
        except Exception:
            continue
    if len(geom_means) == 0:
        logp = np.log10(psd_lin)
        bg = medfilt(logp, kernel_size=3)
        corrected_sub = 10**(logp - bg)
        corrected = psd.copy().astype(float)
        corrected[mask] = corrected_sub
        return corrected
    stacked = np.vstack(geom_means)
    fractal = np.median(stacked, axis=0)
    fractal = np.maximum(fractal, eps)
    corrected_sub = psd_lin / fractal
    corrected = psd.copy().astype(float)
    corrected[mask] = corrected_sub
    return corrected

def find_beta_peak(freqs, psd, beta_range):
    fmin, fmax = beta_range
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return None
    freqs_beta = freqs[mask]
    psd_beta = psd[mask]
    peaks, _ = signal.find_peaks(psd_beta)
    if len(peaks) == 0:
        idx_rel = np.argmax(psd_beta)
    else:
        idx_rel = peaks[np.argmax(psd_beta[peaks])]
    return freqs_beta[idx_rel], psd_beta[idx_rel]

class PSDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PSD GUI (specparam / IRASA)")
        self.filepath = ""
        self.sig = None
        self.sr = None
        self.window_name = 'hann'
        self.nfft_options = [2**i for i in range(4,19)]
        top = tk.Frame(root)
        top.pack(fill='x', padx=6, pady=6)
        tk.Button(top, text="Open .smr", command=self.open_file).pack(side='left')
        tk.Label(top, text="Start (s)").pack(side='left', padx=4)
        self.start_var = tk.DoubleVar(value=35.0)
        tk.Entry(top, width=6, textvariable=self.start_var).pack(side='left')
        tk.Label(top, text="End (s)").pack(side='left', padx=4)
        self.end_var = tk.DoubleVar(value=45.0)
        tk.Entry(top, width=6, textvariable=self.end_var).pack(side='left')
        tk.Label(top, text="FFT").pack(side='left', padx=4)
        self.fft_var = tk.StringVar(value=str(131072))
        self.fft_menu = ttk.Combobox(top, values=[str(x) for x in self.nfft_options],
                                     textvariable=self.fft_var, width=10, state='readonly')
        self.fft_menu.pack(side='left')
        self.fft_menu.bind("<<ComboboxSelected>>", self.update_resolution)
        self.res_label = tk.Label(top, text="Res: - Hz")
        self.res_label.pack(side='left', padx=6)
        self.spec_var = tk.IntVar(value=0)
        self.spec_cb = tk.Checkbutton(top, text="Use specparam aperiodic removal",
                                      variable=self.spec_var, command=self.on_spec_toggle)
        self.spec_cb.pack(side='left', padx=6)
        self.spec_status = tk.Label(top, text=("specparam OK" if HAVE_SPECPARAM else "specparam not installed"))
        self.spec_status.pack(side='left', padx=6)
        if not HAVE_SPECPARAM:
            self.spec_cb.config(state='disabled')
        self.irasa_var = tk.IntVar(value=0)
        self.irasa_cb = tk.Checkbutton(top, text="Use IRASA aperiodic removal",
                                       variable=self.irasa_var, command=self.on_irasa_toggle)
        self.irasa_cb.pack(side='left', padx=6)
        self.h_var = tk.StringVar(value='4')
        self.h_menu = ttk.Combobox(top, values=[str(i) for i in range(2,13)], textvariable=self.h_var,
                                   width=4, state='disabled')
        self.h_menu.pack(side='left')
        tk.Label(top, text="Xmin").pack(side='left', padx=4)
        self.xmin_var = tk.StringVar(value='10')
        tk.Entry(top, width=6, textvariable=self.xmin_var).pack(side='left')
        tk.Label(top, text="Xmax").pack(side='left', padx=4)
        self.xmax_var = tk.StringVar(value='40')
        tk.Entry(top, width=6, textvariable=self.xmax_var).pack(side='left')
        tk.Label(top, text="Ymin (log)").pack(side='left', padx=4)
        self.ymin_var = tk.StringVar(value='')
        tk.Entry(top, width=6, textvariable=self.ymin_var).pack(side='left')
        tk.Label(top, text="Ymax (log)").pack(side='left', padx=4)
        self.ymax_var = tk.StringVar(value='')
        tk.Entry(top, width=6, textvariable=self.ymax_var).pack(side='left')
        self.btn_process = tk.Button(top, text="Process", command=self.process)
        self.btn_process.pack(side='right')
        self.status_label = tk.Label(top, text="")
        self.status_label.pack(side='right', padx=6)
        self.plot_frame = tk.Frame(root)
        self.plot_frame.pack(fill='both', expand=True, padx=6, pady=6)
        self.fig, (self.ax1, self.ax2) = plt.subplots(2,1, figsize=(8,6))
        plt.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        try:
            idx = [str(x) for x in self.nfft_options].index(str(131072))
            self.fft_menu.current(idx)
        except Exception:
            pass
        print(f"HAVE_SPECPARAM = {HAVE_SPECPARAM}" + (f"; import error: {_SPEC_IMPORT_ERR}" if not HAVE_SPECPARAM else ""))

    def on_spec_toggle(self):
        if self.spec_var.get():
            self.irasa_var.set(0)
            self.h_menu.config(state='disabled')
        if HAVE_SPECPARAM and self.spec_var.get():
            self.spec_status.config(text="specparam: enabled")
        elif HAVE_SPECPARAM:
            self.spec_status.config(text="specparam: disabled")

    def on_irasa_toggle(self):
        if self.irasa_var.get():
            self.spec_var.set(0)
            self.spec_status.config(text=("specparam OK" if HAVE_SPECPARAM else "specparam not installed"))
            self.h_menu.config(state='readonly')
        else:
            self.h_menu.config(state='disabled')

    def open_file(self):
        p = filedialog.askopenfilename(filetypes=[("Spike2 Files","*.smr"),("All files","*.*")])
        if not p:
            return
        self.filepath = p
        try:
            self.sig, self.sr = read_smr_single_channel(self.filepath)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self.update_resolution()
        self.process()

    def update_resolution(self, event=None):
        if self.sr is None:
            self.res_label.config(text="Res: - Hz")
            return
        try:
            nfft = int(self.fft_var.get())
        except Exception:
            nfft = 131072
        res = self.sr / nfft
        self.res_label.config(text=f"Res: {res:.6f} Hz")

    def _update_plots(self, freqs, psd_proc, sig_seg, t0, t1, beta_peak):
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.plot(freqs, psd_proc)
        self.ax1.set_yscale('log')
        self.ax1.set_xlabel('Frequency (Hz)')
        self.ax1.set_ylabel('PSD (units^2/Hz)')
        self.ax1.set_title('Welch PSD')
        self.ax1.grid(True, which='both', linestyle=':', alpha=0.6)
        try:
            xmin = float(self.xmin_var.get()); xmax = float(self.xmax_var.get())
            self.ax1.set_xlim(xmin, xmax)
        except Exception:
            self.ax1.set_xlim(0, self.sr/2)
        if self.ymin_var.get() != '' and self.ymax_var.get() != '':
            try:
                ymin = float(self.ymin_var.get()); ymax = float(self.ymax_var.get())
                self.ax1.set_ylim(10**ymin, 10**ymax)
            except Exception:
                pass
        self.ax1.axvspan(beta_range[0], beta_range[1], color='gray', alpha=0.15)
        if beta_peak is not None:
            pfreq, pval = beta_peak
            self.ax1.plot(pfreq, pval, 'ro')
            self.ax1.annotate(f'{pfreq:.2f} Hz\n{pval:.3e}', xy=(pfreq, pval), xytext=(pfreq, pval*1.5),
                              arrowprops=dict(arrowstyle='->'))
        t = np.arange(len(sig_seg)) / self.sr + t0
        self.ax2.plot(t, sig_seg)
        self.ax2.set_xlim(t0, t1)
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Amplitude (units)')
        self.ax2.set_title(f'Time Course: {t0}-{t1}s')
        self.ax2.grid(True, linestyle=':', alpha=0.6)
        self.fig.tight_layout()
        self.canvas.draw()

    def _process_worker(self, t0, t1, nfft):
        try:
            sig_seg, n0, n1 = slice_timecourse(self.sig, self.sr, t0, t1)
            if len(sig_seg) == 0:
                raise ValueError("Invalid time window")
            freqs, psd = compute_welch(sig_seg, self.sr, nfft, self.window_name)
            if self.spec_var.get() and HAVE_SPECPARAM:
                psd_proc = apply_specparam_remove(freqs, psd, fit_range=(1.0, 40.0), gui_parent=None)
            elif self.irasa_var.get():
                try:
                    h_count = int(self.h_var.get())
                except Exception:
                    h_count = 4
                psd_proc = apply_irasa_remove(freqs, psd, sig_seg, self.sr, nfft, h_count=h_count)
            else:
                psd_proc = psd
            beta_peak = find_beta_peak(freqs, psd_proc, beta_range)
            def gui_update():
                self._update_plots(freqs, psd_proc, sig_seg, t0, t1, beta_peak)
                self.btn_process.config(state='normal')
                self.status_label.config(text="Done")
            self.root.after(0, gui_update)
        except Exception as e:
            def gui_error():
                self.btn_process.config(state='normal')
                self.status_label.config(text="Error")
                messagebox.showerror("Processing error", str(e), parent=self.root)
            self.root.after(0, gui_error)

    def process(self):
        if self.filepath == "":
            messagebox.showinfo("Info", "Please open a .smr file first")
            return
        try:
            t0 = float(self.start_var.get()); t1 = float(self.end_var.get())
        except Exception:
            messagebox.showerror("Error", "Invalid start/end times"); return
        try:
            nfft = int(self.fft_var.get())
        except Exception:
            nfft = 131072
        self.btn_process.config(state='disabled')
        status_text = ("Running specparam..." if (self.spec_var.get() and HAVE_SPECPARAM)
                       else "Running IRASA..." if self.irasa_var.get()
                       else "Processing...")
        self.status_label.config(text=status_text)
        thr = threading.Thread(target=self._process_worker, args=(t0, t1, nfft), daemon=True)
        thr.start()

if __name__ == '__main__':
    root = tk.Tk()
    app = PSDApp(root)
    root.mainloop()
