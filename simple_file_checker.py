import os
import h5py
import pandas as pd
import cv2

# Ścieżka do katalogu z danymi
DATASET_PATH = ""

# Lista plików do sprawdzenia
files_to_check = {
    # do zastapienia przez pliki danego datasetu
    "csv_files": [
        "VolEst_D1_sucrose_density.csv",
        "VolEst_D2_accuracy_weight_metadata.csv",
        "VolEst_D4_accuracy_nanoliter_metadata.csv",
        "VolEst_D6_sucrose_caffeine_consumption_metadata.csv"
    ],
    "h5_files": [
        "data/DeepLabCut_output.h5"
    ],
    "video_files": [
        "example_video_1.mp4",
        "example_video_2.mp4"
    ]
}

# Sprawdzenie istnienia pliku
def check_file_exists(file_path):
    if os.path.isfile(file_path):
        print(f" Plik istnieje: {file_path}")
        return True
    else:
        print(f" Brak pliku: {file_path}")
        return False

# Walidacja CSV
def check_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f" CSV OK: {file_path} | Wiersze: {len(df)} | Kolumny: {len(df.columns)}")
        if df.isnull().values.any():
            print(f"️  Brakujące dane w CSV: {file_path}")
    except Exception as e:
        print(f" Błąd wczytywania CSV: {file_path} | {e}")

# Walidacja plików H5
def check_h5_file(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            print(f" H5 OK: {file_path} | Klucze: {keys}")
            # Sprawdzenie obecności zbiorów
            for key in keys:
                data = f[key]
                print(f"  Dataset '{key}' shape: {data.shape}")
    except Exception as e:
        print(f" Błąd H5: {file_path} | {e}")

# Walidacja pliku wideo
def check_video_file(file_path):
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f" Nie można otworzyć wideo: {file_path}")
            return
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps else 0
        print(f" Wideo OK: {file_path} | FPS: {fps:.2f} | Ramki: {frame_count} | Długość: {duration:.2f} s")
        cap.release()
    except Exception as e:
        print(f" Błąd wideo: {file_path} | {e}")

# Główna procedura walidacyjna
def main():
    for f in files_to_check["csv_files"]:
        full_path = os.path.join(DATASET_PATH, f)
        if check_file_exists(full_path):
            check_csv_file(full_path)

    print("\n--- Sprawdzam H5 ---")
    for f in files_to_check["h5_files"]:
        full_path = os.path.join(DATASET_PATH, f)
        if check_file_exists(full_path):
            check_h5_file(full_path)

    print("\n--- Sprawdzam WIDEO (MP4) ---")
    for f in files_to_check["video_files"]:
        full_path = os.path.join(DATASET_PATH, f)
        if check_file_exists(full_path):
            check_video_file(full_path)

if __name__ == "__main__":
    main()