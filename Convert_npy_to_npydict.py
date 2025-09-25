"""
Code: Checks if npy is a dictionary, then converts it to dictionary

@author - writing and design: Jawad_Karim
Date: Sep 24 2025

"""

# save as fix_npy_to_dict.py and run with python
import numpy as np
import os
import sys
import traceback


# Edit these paths (or pass args)
input_npy = r""
# default output filename (won't overwrite original)
output_npy = ""
default_sampling_rate = 125.0  # change if you know the correct fs


# Utility helpers
def is_dict_obj(x):
    return isinstance(x, dict)

def try_item_unwrap(arr):
    """If arr is 0-dim or single-element object array containing the dict, unwrap and return object."""
    try:
        if hasattr(arr, "shape") and arr.shape == ():
            obj = arr.item()
            return obj
        # also if array shape (1,) dtype object
        if hasattr(arr, "ndim") and arr.ndim == 1 and arr.size == 1:
            return arr.reshape(-1)[0]
    except Exception:
        pass
    return None

def fix_channel_data(data):
    """
    Given data (np.ndarray or array-like), return cleaned 1D np.float64 array.
    Apply common dedup/stitch fixes:
      - If 2D with two identical columns -> take first column
      - If 1D and adjacent pairs duplicate -> keep every other
      - If 1D and first half == second half -> keep first half
    """
    arr = np.asarray(data)
    # If 2D with two identical columns -> reduce
    if arr.ndim == 2:
        if arr.shape[1] == 2:
            try:
                if np.allclose(arr[:, 0], arr[:, 1]):
                    arr = arr[:, 0]
                else:
                    # if columns differ but one column full of zeros maybe take non-zero?
                    pass
            except Exception:
                pass
        else:
            # If multi-col and one dimension looks like time (many rows), we flatten to first column by default
            # but we will warn - caller may want different behavior
            # pick the column with largest std (most informative)
            try:
                col_stds = np.std(arr, axis=0)
                idx = int(np.argmax(col_stds))
                arr = arr[:, idx]
            except Exception:
                arr = arr[:, 0]
    # ensure 1D now
    arr = np.asarray(arr).squeeze()
    # if still not 1D, attempt flatten
    if arr.ndim != 1:
        arr = arr.ravel()
    # adjacent duplicate pair check: [a,a,b,b,c,c] -> [a,b,c]
    n = len(arr)
    if n % 2 == 0 and n > 2:
        try:
            if np.allclose(arr[0::2], arr[1::2]):
                arr = arr[0::2]
            else:
                # check half-split duplication: [first_half, first_half]
                half = n // 2
                if np.allclose(arr[:half], arr[half:]):
                    arr = arr[:half]
        except Exception:
            pass
    # final ensure float dtype
    arr = np.asarray(arr, dtype=float)
    return arr


# Main conversion routine
def inspect_and_fix(input_path, output_path, default_fs=None):
    print("Loading:", input_path)
    arr = np.load(input_path, allow_pickle=True)
    print("Loaded object type:", type(arr), " shape:", getattr(arr, "shape", None), " dtype:", getattr(arr, "dtype", None))

    # Case A: arr is ndarray object containing a dict inside (0-dim or 1-element)
    unwrapped = try_item_unwrap(arr)
    if unwrapped is not None and is_dict_obj(unwrapped):
        print("Detected wrapped dictionary (object array). Unwrapping.")
        data_dict = unwrapped
    elif isinstance(arr, dict):
        print("File is already a dict.")
        data_dict = arr
    else:
        # arr not a dict - attempt to interpret
        data_dict = {}
        # if arr is ndarray numeric
        if isinstance(arr, np.ndarray):
            if arr.ndim == 1:
                # Could be 1D channel or object array of elements
                if arr.dtype == object:
                    # object array: each element might be array-like or a dict
                    print("Object array (1D). Attempting to parse elements as channels...")
                    idx = 1
                    for el in arr.reshape(-1):
                        # element could be dict-like
                        if isinstance(el, dict) and 'data' in el:
                            data = np.asarray(el['data'])
                            fs = el.get('sampling_rate', default_fs)
                        else:
                            data = np.asarray(el)
                            fs = default_fs
                        cleaned = fix_channel_data(data)
                        key = f"ch{idx}"
                        data_dict[key] = {'sampling_rate': float(fs) if fs is not None else None, 'data': cleaned}
                        idx += 1
                    if data_dict:
                        print("Converted object-array elements into channels:", list(data_dict.keys()))
                    else:
                        print("Could not interpret object array elements as channels - fallback to single-channel.")
                        data_dict['ch1'] = {'sampling_rate': default_fs, 'data': fix_channel_data(arr)}
                else:
                    # numeric 1D -> single channel
                    print("Numeric 1D array -> creating ch1")
                    data_dict['ch1'] = {'sampling_rate': default_fs, 'data': fix_channel_data(arr)}
            elif arr.ndim == 2:
                # Decide whether rows=time or cols=time using heuristic: longer axis = time
                r, c = arr.shape
                print(f"Numeric 2D array shape {arr.shape}. Heuristic mapping: longer axis is time.")
                if r >= c:
                    # assume rows=time, columns=channels
                    print("Assuming rows=time, columns=channels. Creating channels from columns.")
                    for i in range(c):
                        col = arr[:, i]
                        data_dict[f"ch{i+1}"] = {'sampling_rate': default_fs, 'data': fix_channel_data(col)}
                else:
                    # columns likely time, rows channels -> transpose
                    print("Assuming columns=time, rows=channels. Transposing array.")
                    arr_t = arr.T
                    for i in range(arr_t.shape[1]):
                        col = arr_t[:, i]
                        data_dict[f"ch{i+1}"] = {'sampling_rate': default_fs, 'data': fix_channel_data(col)}
            else:
                raise ValueError("Unsupported ndarray dimensionality: {}".format(arr.ndim))
        else:
            # Not ndarray, maybe loaded object (list/dict)
            if isinstance(arr, (list, tuple)):
                print("Loaded a list/tuple. Treating each element as channel.")
                for i, el in enumerate(arr):
                    data = np.asarray(el)
                    data_dict[f"ch{i+1}"] = {'sampling_rate': default_fs, 'data': fix_channel_data(data)}
            else:
                raise TypeError(f"Unsupported loaded object type: {type(arr)}")

    # Final pass: if data_dict has values that are dicts with 'data' or raw arrays, normalize
    normalized = {}
    idx = 1
    for k, v in list(data_dict.items()):
        if isinstance(v, dict) and 'data' in v:
            data = np.asarray(v['data'])
            fs = v.get('sampling_rate', default_fs)
        elif isinstance(v, np.ndarray):
            data = v
            fs = default_fs
        elif isinstance(v, dict):
            # maybe dict of meta like {'sampling_rate':..., 'values':...}
            data = None
            if 'values' in v:
                data = np.asarray(v['values'])
                fs = v.get('sampling_rate', default_fs)
            else:
                # fallback: attempt to find first array-like item in dict
                found = False
                for subk, subv in v.items():
                    if isinstance(subv, (list, np.ndarray)):
                        data = np.asarray(subv)
                        fs = v.get('sampling_rate', default_fs)
                        found = True
                        break
                if not found:
                    raise ValueError(f"Could not interpret dict entry for key {k}: {v.keys()}")
        else:
            data = np.asarray(v)
            fs = default_fs

        cleaned = fix_channel_data(data)
        # ensure length > 0
        if cleaned.size == 0:
            print(f"Warning: channel {k} produced empty data after cleaning - skipping.")
            continue
        out_key = f"ch{idx}"
        normalized[out_key] = {'sampling_rate': float(fs) if fs is not None else None, 'data': cleaned, 'label': str(k)}
        idx += 1

    if not normalized:
        raise RuntimeError("Conversion failed: produced no channels.")

    # Save normalized dict
    np.save(output_path, normalized, allow_pickle=True)
    print("\nSaved cleaned dictionary to:", output_path)
    # Print summary
    print("Channels saved:", list(normalized.keys()))
    for kk in normalized:
        ent = normalized[kk]
        print(f" {kk}: samples={len(ent['data'])}, sampling_rate={ent['sampling_rate']}, label={ent.get('label')}")
    return normalized


# Run
if __name__ == '__main__':
    try:
        if not os.path.exists(input_npy):
            print("Input file not found:", input_npy)
            sys.exit(1)
        norm = inspect_and_fix(input_npy, output_npy, default_sampling_rate)
        print("\nYou can now point PeakAnalysis v0.3.3 to this new file:\n", output_npy)
    except Exception as e:
        print("Conversion failed with exception:\n", e)
        traceback.print_exc()
        sys.exit(2)
