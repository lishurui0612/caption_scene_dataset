import argparse
import json
import os

import nibabel as nib
import numpy as np


def bids_to_fsl_shifts(slice_timing, repetition_time):
    """Convert BIDS slice times in seconds to FSL shifts in units of TR."""
    timing = np.asarray(slice_timing, dtype=float)
    tr = float(repetition_time)

    if tr <= 0 or not np.isfinite(tr):
        raise ValueError(f"Invalid RepetitionTime: {repetition_time}")
    if timing.ndim != 1 or timing.size == 0:
        raise ValueError("SliceTiming must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(timing)):
        raise ValueError("SliceTiming contains non-finite values")
    if np.any(timing < 0) or np.any(timing >= tr):
        raise ValueError(
            f"SliceTiming values must be in seconds within [0, TR); "
            f"got range [{timing.min()}, {timing.max()}] for TR={tr}"
        )

    shifts = (tr / 2.0 - timing) / tr
    if np.any(shifts < -0.5 - 1e-6) or np.any(shifts > 0.5 + 1e-6):
        raise ValueError(
            f"Converted FSL shifts are outside [-0.5, 0.5]: "
            f"[{shifts.min()}, {shifts.max()}]"
        )
    return shifts


def write_fsl_timing(json_path, timing_path, tr_path):
    nii_path = json_path[:-5] + ".nii.gz"
    if not os.path.isfile(nii_path):
        raise FileNotFoundError(f"Missing NIfTI paired with {json_path}: {nii_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    tr = float(metadata["RepetitionTime"])
    shifts = bids_to_fsl_shifts(metadata["SliceTiming"], tr)
    n_slices = nib.load(nii_path).shape[2]
    if shifts.size != n_slices:
        raise ValueError(
            f"SliceTiming length ({shifts.size}) does not match NIfTI dim3 "
            f"({n_slices}) for {nii_path}"
        )
    np.savetxt(timing_path, shifts, fmt="%.10f")
    with open(tr_path, "w", encoding="utf-8") as f:
        f.write(f"{tr:.10g}\n")

    print(
        f"Prepared {timing_path} from {json_path}: TR={tr:g}s, "
        f"slices={n_slices}, FSL shift range="
        f"[{shifts.min():.6f}, {shifts.max():.6f}]"
    )


def find_single_json(directory, phase):
    matches = sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if phase in name and name.endswith(".json")
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {phase} JSON in {directory}, found {len(matches)}: "
            f"{matches}"
        )
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--ap-json", type=str)
    parser.add_argument("--pa-json", type=str)
    args = parser.parse_args()

    ap_json = args.ap_json or find_single_json(args.path, "AP")
    pa_json = args.pa_json or find_single_json(args.path, "PA")

    write_fsl_timing(
        ap_json,
        os.path.join(args.path, "ST_AP.txt"),
        os.path.join(args.path, "TR_AP.txt"),
    )
    write_fsl_timing(
        pa_json,
        os.path.join(args.path, "ST_PA.txt"),
        os.path.join(args.path, "TR_PA.txt"),
    )


if __name__ == "__main__":
    main()
