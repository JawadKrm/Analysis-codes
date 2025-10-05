import tkinter as tk
from tkinter import messagebox


def compute_windows(total_time, task_start, task_end, window, step):
    """Compute valid window start times while avoiding partial overlaps with task."""
    starts = []
    t = 0
    while t + window <= total_time:
        if (t < task_start < t + window) or (t < task_end < t + window):
            t += step
            continue
        starts.append(t)
        t += step
    return starts


def calculate():
    """Calculate and display the expected number of windows and their ranges."""
    try:
        total_time = int(entry_total.get())
        task_start = int(entry_start.get())
        task_end = int(entry_end.get())
        window = int(entry_window.get())
        step = int(entry_step.get())

        starts = compute_windows(total_time, task_start, task_end, window, step)
        num_windows = len(starts)

        result_text.set(
            f"Expected power spectra: {num_windows}\n\nWindows:\n" +
            ", ".join([f"{s}-{s+window}" for s in starts])
        )
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers.")


def clear_fields():
    """Clear all input and output fields."""
    entry_total.delete(0, tk.END)
    entry_start.delete(0, tk.END)
    entry_end.delete(0, tk.END)
    entry_window.delete(0, tk.END)
    entry_step.delete(0, tk.END)
    result_text.set("")


# --- GUI ---
root = tk.Tk()
root.title("Power Spectrum Window Calculator")

# Labels
labels = ["Total Time (s):", "Task Start (s):", "Task End (s):",
          "Window Length (s):", "Step Size (s):"]
for i, text in enumerate(labels):
    tk.Label(root, text=text).grid(row=i, column=0, sticky="e")

# Entry fields
entry_total = tk.Entry(root)
entry_start = tk.Entry(root)
entry_end = tk.Entry(root)
entry_window = tk.Entry(root)
entry_step = tk.Entry(root)

entries = [entry_total, entry_start, entry_end, entry_window, entry_step]
for i, entry in enumerate(entries):
    entry.grid(row=i, column=1)

# Buttons
btn_calc = tk.Button(root, text="Calculate", command=calculate)
btn_calc.grid(row=5, column=0, pady=5)

btn_clear = tk.Button(root, text="Clear", command=clear_fields)
btn_clear.grid(row=5, column=1, pady=5)

# Result display
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, justify="left", wraplength=400)
result_label.grid(row=6, column=0, columnspan=2, pady=10)

root.mainloop()
