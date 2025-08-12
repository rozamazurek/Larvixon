import pandas as pd
import re
import os

def analyze_metadata_and_annotations(file_path):
    """
    Analizuje plik CSV zawierający nazwy plików wideo/segmentów
    i ich adnotacje pod kątem jakości i spójności.

    Args:
        file_path (str): Ścieżka do pliku CSV z metadanymi i adnotacjami.
    """
    print(f"--- ROZPOCZĘCIE ANALIZY PLIKU: {os.path.basename(file_path)} ---")

    if not os.path.exists(file_path):
        print(f"BŁĄD: Plik nie został znaleziony pod ścieżką: {file_path}")
        print("Upewnij się, że ścieżka jest poprawna.")
        print("--- ANALIZA ZAKOŃCZONA Z BŁĘDEM ---")
        return

    try:
        # Wczytaj plik CSV
        df = pd.read_csv(file_path)
        print(f"Plik wczytany. Kształt danych: {df.shape}")

        # Sprawdzenie nagłówków kolumn
        expected_columns = ['names', 'annotation']
        if not all(col in df.columns for col in expected_columns):
            print(f"OSTRZEŻENIE: Oczekiwane kolumny '{expected_columns}' nie zostały znalezione.")
            print(f"Dostępne kolumny: {df.columns.tolist()}")
            print("Kontynuuję analizę na podstawie dostępnych kolumn, ale mogą wystąpić błędy.")

        # --- 1. Sprawdzenie brakujących wartości ---
        print("\n--- 1. Analiza brakujących wartości ---")
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print("Znaleziono brakujące wartości:")
            print(missing_values[missing_values > 0])
            print("Wartości NaN w kolumnie 'annotation' mogą wskazywać na brak adnotacji dla danego segmentu.")
        else:
            print("Brak brakujących wartości w pliku. ✅")

        # --- 2. Analiza adnotacji behawioralnych ---
        if 'annotation' in df.columns:
            print("\n--- 2. Analiza adnotacji behawioralnych ---")
            
            # Wyczyść białe znaki wokół adnotacji
            df['annotation'] = df['annotation'].str.strip()

            # Sprawdź unikalne klasy i ich rozkład
            unique_annotations = df['annotation'].dropna().unique()
            print(f"Unikalne klasy zachowań: {unique_annotations.tolist()}")
            print("\nRozkład klas zachowań:")
            annotation_counts = df['annotation'].value_counts(dropna=False) # uwzględnij NaN
            print(annotation_counts)

            # Ocena zbalansowania klas
            if len(unique_annotations) > 1:
                min_count = annotation_counts.min()
                max_count = annotation_counts.max()
                ratio = max_count / min_count if min_count > 0 else float('inf')
                
                print(f"\nNajmniej liczna klasa ({annotation_counts.idxmin()}): {min_count} wystąpień")
                print(f"Najliczniejsza klasa ({annotation_counts.idxmax()}): {max_count} wystąpień")
                if ratio > 5: # Próg do oceny niezbalansowania
                    print(f"OSTRZEŻENIE: Klasy są **mocno niezbalansowane** (stosunek {ratio:.1f}:1). Może to wpłynąć na trening modeli ML. ⚠️")
                elif ratio > 2:
                    print(f"OSTRZEŻENIE: Klasy są **niezbalansowane** (stosunek {ratio:.1f}:1). Warto rozważyć techniki balansowania danych. ⚠️")
                else:
                    print("Klasy są w miarę zbalansowane. ✅")
            else:
                print("Tylko jedna unikalna klasa adnotacji lub brak adnotacji. Brak zróżnicowania.")
            
            # Sprawdzenie niespójności w kolumnie 'annotation' (np. podwójne adnotacje)
            # Jeśli wiersze zostały poprawnie sparsowane, to 'annotation' nie powinno zawierać przecinków
            # Jeśli jednak format CSV był problematyczny, to ta kontrola jest kluczowa.
            problematic_annotations = df[df['annotation'].astype(str).str.contains(',')]
            if not problematic_annotations.empty:
                print("\nOSTRZEŻENIE: Znaleziono niespójne formatowanie adnotacji (np. wiele adnotacji w jednej komórce): ⚠️")
                print(problematic_annotations)
                print("Może to wymagać ręcznej weryfikacji lub poprawy pliku źródłowego.")
        else:
            print("Kolumna 'annotation' nie została znaleziona. Pomijam analizę adnotacji behawioralnych.")


        # --- 3. Analiza metadanych z nazw plików (`names` column) ---
        if 'names' in df.columns:
            print("\n--- 3. Analiza metadanych z nazw plików ---")

            # Wyodrębnienie kluczowych informacji
            # ID studzienki (well_id)
            df['well_id'] = df['names'].str.extract(r'_well_([A-Z]\d+)')
            if not df['well_id'].isnull().all():
                print(f"Unikalne ID studzienek: {df['well_id'].dropna().unique().tolist()}")
                if df['well_id'].nunique() > 1:
                    print("Liczba segmentów na studzienkę:")
                    print(df['well_id'].value_counts())
            else:
                print("Nie udało się wyodrębnić ID studzienek. Sprawdź format nazewnictwa.")

            # Wyodrębnienie zakresu klatek
            def extract_frame_range(filename):
                match = re.search(r'frames_(\d+)_to_(\d+)', str(filename))
                if match:
                    return int(match.group(1)), int(match.group(2))
                return None, None

            df[['start_frame', 'end_frame']] = df['names'].apply(
                lambda x: pd.Series(extract_frame_range(x))
            )

            if not df['start_frame'].isnull().all() and not df['end_frame'].isnull().all():
                print("\nAnaliza ciągłości segmentów wideo:")
                # Sprawdź brakujące segmenty
                # Grupuj po well_id (lub innym ID, jeśli wideo jest od wielu zwierząt)
                for well_id, group in df.groupby('well_id'):
                    sorted_group = group.sort_values('start_frame').reset_index(drop=True)
                    
                    # Sprawdź duplikaty (czy ten sam segment nie jest wpisany wiele razy)
                    if sorted_group.duplicated(['names']).any():
                        print(f"OSTRZEŻENIE dla {well_id}: Znaleziono zduplikowane wpisy segmentów! ⚠️")
                        print(sorted_group[sorted_group.duplicated(['names'], keep=False)])

                    # Sprawdź luki w zakresie klatek
                    gaps_found = False
                    for i in range(1, len(sorted_group)):
                        prev_end = sorted_group.loc[i-1, 'end_frame']
                        current_start = sorted_group.loc[i, 'start_frame']
                        
                        # Zakładamy, że segmenty powinny być ciągłe (np. 0-60, 60-120)
                        if current_start > prev_end + 60:  # Zakładamy, że segmenty powinny być co najmniej 60 klatek od siebie 
                            print(f"OSTRZEŻENIE dla {well_id}: Luka lub nakładanie się klatek między segmentami!")
                            print(f"Poprzedni koniec: {prev_end}, Aktualny początek: {current_start}")
                            print(f"Segmenty: {sorted_group.loc[i-1, 'names']} oraz {sorted_group.loc[i, 'names']} ⚠️")
                            gaps_found = True
                    if not gaps_found:
                        print(f"Dla {well_id}: Segmenty klatek są ciągłe i bez luk. ✅")
            else:
                print("Nie udało się wyodrębnić zakresów klatek z nazw plików. Pomijam analizę ciągłości.")

            # Wyodrębnienie dodatkowych metadanych (np. Data, DPF, Typ eksperymentu)
            df['date'] = df['names'].str.extract(r'^(\d{8})')
            df['age_dpf'] = df['names'].str.extract(r'_(\d+)DPF_')
            df['experiment_type'] = df['names'].str.extract(r'_(AB_PTZ|Control)_') # Dostosuj regex do swoich typów

            print("\nPrzykładowe wyodrębnione metadane:")
            print(df[['well_id', 'date', 'age_dpf', 'experiment_type']].head())

            # Sprawdź, czy są jakieś NaN w wyodrębnionych metadanych, jeśli to kluczowe
            print("\nBrakujące wartości w wyodrębnionych metadanych:")
            print(df[['well_id', 'date', 'age_dpf', 'experiment_type']].isnull().sum())

        else:
            print("Kolumna 'names' nie została znaleziona. Pomijam analizę metadanych z nazw plików.")

    except pd.errors.EmptyDataError:
        print(f"BŁĄD: Plik '{file_path}' jest pusty.")
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd podczas analizy: {e}")

    print("\n--- ANALIZA ZAKOŃCZONA ---")

# --- PRZYKŁADOWE UŻYCIE ---
# Zmień tę ścieżkę na ścieżkę do Twojego pliku CSV
metadata_file_path = 'seizure_detection_ml_dataset/timepoint1/20221205_AB_PTZ_5DPF_Timepoint1_96wp_20221205_161742_024_well_B1/dataset_20221205_AB_PTZ_5DPF_Timepoint1_96wp_20221205_161742_024_well_B1_annotations_60_window.csv'

analyze_metadata_and_annotations(metadata_file_path)