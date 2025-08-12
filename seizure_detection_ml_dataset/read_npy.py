import numpy as np
import os # Do łączenia ścieżek, jeśli plik npy jest w tym samym folderze

# Ustaw ścieżkę do pliku .npy
# Jeśli plik .npy jest w tym samym folderze co skrypt, wystarczy nazwa pliku.
# W przeciwnym razie podaj pełną ścieżkę, np. '/Users/TwojaNazwa/Dokumenty/dane/moje_dane.npy'
npy_filename = 'seizure_detection_ml_dataset/baseline/20221205_AB_Baseline_5DPF_96wp_20221205_160449_352_well_A11/20221205_AB_Baseline_5DPF_96wp_20221205_160449_352_well_A11_frames_847_to_907.npy'
# Możesz też użyć os.path.join, aby utworzyć ścieżkę do pliku w bieżącym katalogu
file_path = os.path.join(os.path.dirname(__file__), npy_filename)


try:
    data = np.load(file_path)
    print("Dane z pliku .npy:")
    print(data)
    print(f"Typ danych: {type(data)}")
    print(f"Kształt danych: {data.shape}")
    print(f"Liczba wymiarów: {data.ndim}")
    print(f"Rozmiar elementu w bajtach: {data.itemsize}")
    print(f"Typ danych elementów: {data.dtype}")

    # Jeśli dane są numeryczne i chcesz zrobić prosty wykres (np. dla tablicy 1D)
    if data.ndim == 1 and data.dtype.kind in ['i', 'f']: # int lub float
        import matplotlib.pyplot as plt
        plt.plot(data)
        plt.title("Wykres danych z .npy")
        plt.xlabel("Indeks")
        plt.ylabel("Wartość")
        plt.show()

    # Jeśli dane są obrazem (np. 2D lub 3D z kanałami RGB)
    elif data.ndim >= 2 and data.dtype.kind in ['u', 'i', 'f']: # unsigned int, int, float
        import matplotlib.pyplot as plt
        # Spróbuj wyświetlić jako obraz
        try:
            plt.imshow(data, cmap='gray') # Możesz zmienić cmap na 'viridis', 'jet' itp.
            plt.title("Obraz z pliku .npy")
            plt.show()
        except TypeError:
            print("\nPlik .npy nie może być bezpośrednio zinterpretowany jako obraz (może ma nieprawidłowy kształt dla imshow).")


except FileNotFoundError:
    print(f"Błąd: Plik '{file_path}' nie został znaleziony. Sprawdź ścieżkę i nazwę pliku.")
except Exception as e:
    print(f"Wystąpił błąd podczas otwierania pliku: {e}")