import numpy as np
import napari
import cv2 # OpenCV do wczytywania wideo
import os # Do łączenia ścieżek, jeśli plik npy jest w tym samym folderze

# Pamiętaj, aby ZMIENIĆ TE ŚCIEŻKI na swoje rzeczywiste pliki!
npy_file_path = 'seizure_detection_ml_dataset/timepoint2/20221205_AB_PTZ_5DPF_Timepoint2_96wp_20221205_162816_804_well_A11/20221205_AB_PTZ_5DPF_Timepoint2_96wp_20221205_162816_804_well_A11_frames_36619_to_36679.npy'
video_file_path = 'seizure_detection_ml_dataset/timepoint2/20221205_AB_PTZ_5DPF_Timepoint2_96wp_20221205_162816_804_well_A11/20221205_AB_PTZ_5DPF_Timepoint2_96wp_20221205_162816_804_well_A11_frames_36619_to_36679.mp4'
try:
    data = np.load(npy_file_path)
    print(f"Załadowane dane z .npy o kształcie: {data.shape}")

    # Wczytaj wideo
    cap = cv2.VideoCapture(video_file_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Konwertuj BGR (OpenCV) na RGB (dla Napari/Matplotlib)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    video_stack = np.array(frames)
    print(f"Załadowane wideo o kształcie: {video_stack.shape}")

    # Sprawdź, czy liczba klatek w wideo zgadza się z pierwszym wymiarem danych npy
    if video_stack.shape[0] != data.shape[0]:
        print(f"OSTRZEŻENIE: Niezgodność liczby klatek! Wideo: {video_stack.shape[0]}, Dane .npy: {data.shape[0]}")
        # Przytnij dłuższe dane do krótszego zestawu
        min_frames = min(video_stack.shape[0], data.shape[0])
        video_stack = video_stack[:min_frames]
        data = data[:min_frames]
        print(f"Przycięto dane do {min_frames} klatek.")


    # Uruchom Napari
    viewer = napari.Viewer()
    
    # Dodaj warstwę obrazu (wideo)
    # Wymiary obrazu są (T, H, W, C) - czas, wysokość, szerokość, kanały
    viewer.add_image(video_stack, name='Video')

    # Przekształć dane punktów do formatu oczekiwanego przez Napari dla danych czasowych (T, N, D)
    # Pamiętaj, że Napari domyślnie przyjmuje (Y, X) dla współrzędnych 2D.
    # Jeśli Twoje dane to (X, Y), musisz odwrócić kolejność: data[:, :, ::-1]
    # Sprawdź w dokumentacji źródła danych (.npy), czy to (X,Y) czy (Y,X).
    # Dla większości śledzenia (X,Y) jest standardem, więc odwracamy na (Y,X)
    points_for_napari = data[:, :, ::-1] # Konwersja (T, N, X, Y) -> (T, N, Y, X)
    
    # Dodaj warstwę punktów
    # Napari automatycznie rozpoznaje pierwszy wymiar jako czas,
    # jeśli jest on zgodny z wymiarem czasowym w już dodanych warstwach (np. wideo).
    # Nie musisz już jawnie ustawiać ndim=3, Napari to wywnioskuje.
    viewer.add_points(
        points_for_napari.reshape(-1, 2), # Spłaszcz do (liczba_wszystkich_punktow, 2)
        ndim=2, # Wymiary przestrzenne to tylko 2D (Y, X)
        face_color='red',
        edge_color='white',
        size=10,
        name='Śledzone Punkty',
        # Użyj 'properties' i 'text' aby wyświetlić etykiety punktów (opcjonalne)
        # properties={'frame': np.repeat(np.arange(data.shape[0]), data.shape[1]),
        #             'point_id': np.tile(np.arange(data.shape[1]), data.shape[0])},
        # text={'string': '{point_id}', 'color': 'white', 'size': 12},
        # Zapewnij, że punkty są powiązane z klatkami czasowymi
        # W Napari 0.4.x i nowszych, jeśli dodasz warstwę punktów
        # i warstwę obrazu o tym samym wymiarze czasowym,
        # suwak czasowy automatycznie synchronizuje obie warstwy.
    )

    napari.run()

except FileNotFoundError:
    print(f"Błąd: Nie znaleziono pliku .npy lub wideo. Sprawdź ścieżki: {npy_file_path}, {video_file_path}")
except Exception as e:
    print(f"Wystąpił błąd: {e}")