import os
import numpy as np
import scipy.io


base_folder = ...  # os.curdir, enter yours here


def main():
    results_files = find_results_paths(base_folder)

    print(f"Found {len(results_files)} result files to check.\n")
    for file in results_files:
        check_file(file)
        print("-" * 80)


def check_file(file):
    try:
        data = scipy.io.loadmat(file)
    except Exception as e:
        print(f"ERROR loading {file}: {e}")
        return

    if "videoDataResults" not in data:
        print(f"{file}: No 'videoDataResults' found")
        return

    video_data = data["videoDataResults"][0][0][0][0]

    total_wells = len(video_data)
    empty_wells = 0
    inaccessible_wells = 0
    empty_bouts = 0
    invalid_bouts = 0
    total_bouts = 0

    frame_lengths = []
    fishnumber_issues = 0
    boutstart_end_issues = 0
    missing_params = 0

    for well_idx in range(total_wells):
        well_data = video_data[well_idx]

        if well_data.size == 0:
            empty_wells += 1
            continue

        try:
            well_bouts = well_data[0][0][0]
        except Exception:
            inaccessible_wells += 1
            continue

        if well_bouts.size == 0:
            empty_wells += 1
            continue

        total_bouts += len(well_bouts)

        for bout in well_bouts:
            if bout.size == 0:
                empty_bouts += 1
                continue

            fish_num = safe_get_field(bout, "FishNumber")
            if fish_num is None or not is_valid_number(fish_num):
                fishnumber_issues += 1

            bout_start = safe_get_field(bout, "BoutStart")
            bout_end = safe_get_field(bout, "BoutEnd")
            if (
                bout_start is None
                or bout_end is None
                or not is_valid_number(bout_start)
                or not is_valid_number(bout_end)
                or bout_start >= bout_end
            ):
                boutstart_end_issues += 1

            if (
                bout_start is not None
                and bout_end is not None
                and is_valid_number(bout_start)
                and is_valid_number(bout_end)
            ):
                length = int(bout_end - bout_start + 1)
                frame_lengths.append(length)

            required_params = ["TailAngle_smoothed", "HeadX", "HeadY"]
            missing_or_invalid = False
            for param in required_params:
                vals = safe_get_field(bout, param)
                if vals is None or not is_valid_numeric_array(vals):
                    missing_or_invalid = True
            if missing_or_invalid:
                missing_params += 1

    if frame_lengths:
        min_len = min(frame_lengths)
        max_len = max(frame_lengths)
        mean_len = np.mean(frame_lengths)
    else:
        min_len = max_len = mean_len = 0

    print(f"{os.path.basename(file)}:")
    print(f"  Wells total: {total_wells}")
    print(f"  Empty wells: {empty_wells}")
    print(f"  Inaccessible wells: {inaccessible_wells}")
    print(f"  Total bouts: {total_bouts}")
    print(f"  Empty bouts: {empty_bouts}")
    print(f"  Bouts with missing/invalid params: {missing_params}")
    print(f"  FishNumber issues: {fishnumber_issues}")
    print(f"  BoutStart/BoutEnd issues: {boutstart_end_issues}")
    print(
        f"  Bout frame length (frames): min={min_len}, max={max_len}, mean={mean_len:.2f}"
    )
    print()


def safe_get_field(bout, field):
    # check if field exists in this bout and if has valid structure
    try:
        if field in bout.dtype.names:
            val = bout[field]
            while isinstance(val, np.ndarray) and val.size == 1:
                val = val[0]
            return val
        else:
            return None
    except Exception:
        return None


def is_valid_numeric_array(arr):
    try:
        numeric_arr = np.array(arr, dtype=float)
        if numeric_arr.size == 0:
            return False
        if np.all(np.isnan(numeric_arr)):
            return False
        return True
    except Exception:
        return False


def is_valid_number(val):
    try:
        num = float(val)
        return not np.isnan(num)
    except Exception:
        return False


def find_results_paths(root_folder):
    mat_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.startswith("results") and filename.endswith(".mat"):
                mat_files.append(os.path.join(dirpath, filename))
    return mat_files


if __name__ == "__main__":
    main()
