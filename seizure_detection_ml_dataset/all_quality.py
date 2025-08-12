import pandas as pd
import re
import os
from collections import defaultdict

def analyze_metadata_and_annotations(file_path, collect_summary=None):
    """
    Analizuje plik CSV zawierający nazwy plików wideo/segmentów
    i ich adnotacje pod kątem jakości i spójności.
    Może również zbierać dane do podsumowania globalnego.

    Args:
        file_path (str): Ścieżka do pliku CSV z metadanami i adnotacjami.
        collect_summary (dict, optional): Słownik do zbierania danych sumarycznych.
                                         Jeśli None, analiza jest wykonywana bez zbierania podsumowania.
    Returns:
        bool: True, jeśli analiza pliku przebiegła pomyślnie, False w przeciwnym razie.
    """
    file_name = os.path.basename(file_path)
    print(f"\n--- ROZPOCZĘCIE ANALIZY PLIKU: {file_name} ---")
    
    analysis_successful = False

    if not os.path.exists(file_path):
        print(f"BŁĄD: Plik nie został znaleziony pod ścieżką: {file_path}")
        print("Upewnij się, że ścieżka jest poprawna.")
        print("--- ANALIZA ZAKOŃCZONA Z BŁĘDEM ---")
        if collect_summary is not None:
            collect_summary['files_processed_with_errors'].append(file_name)
        return analysis_successful

    try:
        df = pd.read_csv(file_path)
        print(f"Plik wczytany. Kształt danych: {df.shape}")

        if collect_summary is not None:
            collect_summary['total_files_found'] += 1
            collect_summary['total_rows'] += df.shape[0]
            collect_summary['total_columns'] += df.shape[1]

        expected_columns = ['names', 'annotation']
        if not all(col in df.columns for col in expected_columns):
            print(f"OSTRZEŻENIE: Oczekiwane kolumny '{expected_columns}' nie zostały znalezione.")
            print(f"Dostępne kolumny: {df.columns.tolist()}")
            if collect_summary is not None:
                collect_summary['files_with_column_warnings'].append(file_name)
        
        # --- 1. Sprawdzenie brakujących wartości ---
        print("\n--- 1. Analiza brakujących wartości ---")
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print("Znaleziono brakujące wartości:")
            print(missing_values[missing_values > 0])
            print("Wartości NaN w kolumnie 'annotation' mogą wskazywać na brak adnotacji dla danego segmentu.")
            if collect_summary is not None:
                collect_summary['files_with_missing_values'].append(file_name)
                for col, count in missing_values[missing_values > 0].items():
                    collect_summary['missing_values_by_column'][col] += count
        else:
            print("Brak brakujących wartości w pliku. ✅")

        # --- 2. Analiza adnotacji behawioralnych ---
        if 'annotation' in df.columns:
            print("\n--- 2. Analiza adnotacji behawioralnych ---")
            
            df['annotation'] = df['annotation'].astype(str).str.strip() # Upewnij się, że to stringi

            unique_annotations = df['annotation'].dropna().unique()
            print(f"Unikalne klasy zachowań: {unique_annotations.tolist()}")
            print("\nRozkład klas zachowań:")
            annotation_counts = df['annotation'].value_counts(dropna=False)
            print(annotation_counts)

            if collect_summary is not None:
                for ann, count in annotation_counts.items():
                    collect_summary['total_annotations_by_class'][ann] += count

            if len(unique_annotations) > 1:
                min_count = annotation_counts.min()
                max_count = annotation_counts.max()
                ratio = max_count / min_count if min_count > 0 else float('inf')
                
                print(f"\nNajmniej liczna klasa ({annotation_counts.idxmin()}): {min_count} wystąpień")
                print(f"Najliczniejsza klasa ({annotation_counts.idxmax()}): {max_count} wystąpień")
                if ratio > 5:
                    print(f"OSTRZEŻENIE: Klasy są **mocno niezbalansowane** (stosunek {ratio:.1f}:1). Może to wpłynąć na trening modeli ML. ⚠️")
                    if collect_summary is not None:
                        collect_summary['files_with_highly_imbalanced_classes'].append(file_name)
                elif ratio > 2:
                    print(f"OSTRZEŻENIE: Klasy są **niezbalansowane** (stosunek {ratio:.1f}:1). Warto rozważyć techniki balansowania danych. ⚠️")
                    if collect_summary is not None:
                        collect_summary['files_with_imbalanced_classes'].append(file_name)
                else:
                    print("Klasy są w miarę zbalansowane. ✅")
            else:
                print("Tylko jedna unikalna klasa adnotacji lub brak adnotacji. Brak zróżnicowania.")
            
            problematic_annotations = df[df['annotation'].astype(str).str.contains(',')]
            if not problematic_annotations.empty:
                print("\nOSTRZEŻENIE: Znaleziono niespójne formatowanie adnotacji (np. wiele adnotacji w jednej komórce): ⚠️")
                print(problematic_annotations)
                print("Może to wymagać ręcznej weryfikacji lub poprawy pliku źródłowego.")
                if collect_summary is not None:
                    collect_summary['files_with_problematic_annotations'].append(file_name)
        else:
            print("Kolumna 'annotation' nie została znaleziona. Pomijam analizę adnotacji behawioralnych.")
            if collect_summary is not None:
                collect_summary['files_missing_annotation_column'].append(file_name)


        # --- 3. Analiza metadanych z nazw plików (`names` column) ---
        if 'names' in df.columns:
            print("\n--- 3. Analiza metadanych z nazw plików ---")

            df['names'] = df['names'].astype(str) # Upewnij się, że to stringi

            df['well_id'] = df['names'].str.extract(r'_well_([A-Z]\d+)')
            if not df['well_id'].isnull().all():
                unique_well_ids = df['well_id'].dropna().unique().tolist()
                print(f"Unikalne ID studzienek: {unique_well_ids}")
                if collect_summary is not None:
                    collect_summary['unique_well_ids'].update(unique_well_ids)
                if df['well_id'].nunique() > 1:
                    print("Liczba segmentów na studzienkę:")
                    well_counts = df['well_id'].value_counts()
                    print(well_counts)
                    if collect_summary is not None:
                        for well, count in well_counts.items():
                            collect_summary['total_segments_by_well'][well] += count
            else:
                print("Nie udało się wyodrębnić ID studzienek. Sprawdź format nazewnictwa.")
                if collect_summary is not None:
                    collect_summary['files_failed_well_id_extraction'].append(file_name)

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
                for well_id, group in df.groupby('well_id'):
                    sorted_group = group.sort_values('start_frame').reset_index(drop=True)
                    
                    if sorted_group.duplicated(['names']).any():
                        print(f"OSTRZEŻENIE dla {well_id}: Znaleziono zduplikowane wpisy segmentów! ⚠️")
                        print(sorted_group[sorted_group.duplicated(['names'], keep=False)])
                        if collect_summary is not None:
                            collect_summary['files_with_duplicated_segments'].append(file_name)

                    gaps_found = False
                    for i in range(1, len(sorted_group)):
                        prev_end = sorted_group.loc[i-1, 'end_frame']
                        current_start = sorted_group.loc[i, 'start_frame']
                        
                        if current_start > prev_end + 60:
                            print(f"OSTRZEŻENIE dla {well_id}: Luka lub nakładanie się klatek między segmentami!")
                            print(f"Poprzedni koniec: {prev_end}, Aktualny początek: {current_start}")
                            print(f"Segmenty: {sorted_group.loc[i-1, 'names']} oraz {sorted_group.loc[i, 'names']} ⚠️")
                            gaps_found = True
                    if gaps_found and collect_summary is not None:
                        collect_summary['files_with_frame_gaps'].append(file_name)
                    if not gaps_found:
                        print(f"Dla {well_id}: Segmenty klatek są ciągłe i bez luk. ✅")
            else:
                print("Nie udało się wyodrębnić zakresów klatek z nazw plików. Pomijam analizę ciągłości.")
                if collect_summary is not None:
                    collect_summary['files_failed_frame_extraction'].append(file_name)

            df['date'] = df['names'].str.extract(r'^(\d{8})')
            df['age_dpf'] = df['names'].str.extract(r'_(\d+)DPF_')
            df['experiment_type'] = df['names'].str.extract(r'_(AB_PTZ|Control)_')

            print("\nPrzykładowe wyodrębnione metadane:")
            print(df[['well_id', 'date', 'age_dpf', 'experiment_type']].head())

            print("\nBrakujące wartości w wyodrębnionych metadanych:")
            extracted_metadata_missing = df[['well_id', 'date', 'age_dpf', 'experiment_type']].isnull().sum()
            print(extracted_metadata_missing)
            if collect_summary is not None:
                for col, count in extracted_metadata_missing[extracted_metadata_missing > 0].items():
                    collect_summary['missing_extracted_metadata'][col] += count
        else:
            print("Kolumna 'names' nie została znaleziona. Pomijam analizę metadanych z nazw plików.")
            if collect_summary is not None:
                collect_summary['files_missing_names_column'].append(file_name)
        
        analysis_successful = True

    except pd.errors.EmptyDataError:
        print(f"BŁĄD: Plik '{file_path}' jest pusty.")
        if collect_summary is not None:
            collect_summary['files_empty'].append(file_name)
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd podczas analizy: {e}")
        if collect_summary is not None:
            collect_summary['files_processed_with_errors'].append(file_name)

    print(f"--- ANALIZA PLIKU: {file_name} ZAKOŃCZONA ---")
    return analysis_successful

def summarize_project_data_quality(base_dir):
    """
    Przeszukuje podfoldery w podanej ścieżce, znajduje pliki CSV
    pasujące do wzorca 'dataset*.csv' i wykonuje na nich analizę jakości danych.
    Zbiera sumaryczne informacje o jakości danych w całym projekcie.

    Args:
        base_dir (str): Ścieżka do głównego katalogu projektu.
    """
    print(f"*** ROZPOCZĘCIE ANALIZY JAKOŚCI DANYCH DLA PROJEKTU W: {base_dir} ***")

    if not os.path.isdir(base_dir):
        print(f"BŁĄD: Katalog '{base_dir}' nie istnieje lub nie jest katalogiem.")
        print("--- ANALIZA PROJEKTU ZAKOŃCZONA Z BŁĘDEM ---")
        return

    # Słownik do zbierania danych sumarycznych
    project_summary = {
        'total_files_found': 0,
        'total_files_processed': 0,
        'total_rows': 0,
        'total_columns': 0,
        'files_processed_with_errors': [],
        'files_empty': [],
        'files_with_column_warnings': [],
        'files_with_missing_values': [],
        'missing_values_by_column': defaultdict(int),
        'total_annotations_by_class': defaultdict(int),
        'files_with_imbalanced_classes': [],
        'files_with_highly_imbalanced_classes': [],
        'files_with_problematic_annotations': [],
        'files_missing_annotation_column': [],
        'unique_well_ids': set(),
        'total_segments_by_well': defaultdict(int),
        'files_failed_well_id_extraction': [],
        'files_with_duplicated_segments': [],
        'files_with_frame_gaps': [],
        'files_failed_frame_extraction': [],
        'files_missing_names_column': [],
        'missing_extracted_metadata': defaultdict(int),
    }

    found_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith('dataset') and file.endswith('.csv'):
                full_path = os.path.join(root, file)
                found_files.append(full_path)

    if not found_files:
        print(f"Nie znaleziono żadnych plików 'dataset*.csv' w '{base_dir}' i jego podfolderach.")
        print("--- ANALIZA PROJEKTU ZAKOŃCZONA ---")
        return

    print(f"Znaleziono {len(found_files)} plików do analizy.")

    for file_path in found_files:
        if analyze_metadata_and_annotations(file_path, project_summary):
            project_summary['total_files_processed'] += 1

    # --- Generowanie sumarycznego raportu ---
    print("\n" + "="*80)
    print("                 RAPORT JAKOŚCI DANYCH DLA CAŁEGO PROJEKTU                 ")
    print("="*80)

    print(f"\n## 📊 Ogólne Statystyki Projektu")
    print(f"   - Łączna liczba znalezionych plików CSV (dataset*.csv): **{project_summary['total_files_found']}**")
    print(f"   - Łączna liczba przetworzonych plików: **{project_summary['total_files_processed']}**")
    print(f"   - Łączna liczba wierszy (segmentów): **{project_summary['total_rows']}**")
    print(f"   - Łączna liczba unikalnych studzienek (Well ID): **{len(project_summary['unique_well_ids'])}**")
    if project_summary['unique_well_ids']:
        print(f"     (ID: {', '.join(sorted(list(project_summary['unique_well_ids'])))})")
    
    print(f"\n## ⚠️ Podsumowanie Problemów Jakościowych")
    
    if project_summary['files_processed_with_errors']:
        print(f"   - Pliki, które spowodowały błąd podczas przetwarzania: **{len(project_summary['files_processed_with_errors'])}**")
        for f in project_summary['files_processed_with_errors']:
            print(f"     - {f}")
    
    if project_summary['files_empty']:
        print(f"   - Puste pliki: **{len(project_summary['files_empty'])}**")
        for f in project_summary['files_empty']:
            print(f"     - {f}")

    if project_summary['files_with_column_warnings']:
        print(f"   - Pliki z brakującymi oczekiwanymi kolumnami ('names', 'annotation'): **{len(project_summary['files_with_column_warnings'])}**")
        for f in project_summary['files_with_column_warnings']:
            print(f"     - {f}")
            
    if project_summary['files_missing_annotation_column']:
        print(f"   - Pliki bez kolumny 'annotation': **{len(project_summary['files_missing_annotation_column'])}**")
        for f in project_summary['files_missing_annotation_column']:
            print(f"     - {f}")

    if project_summary['files_missing_names_column']:
        print(f"   - Pliki bez kolumny 'names': **{len(project_summary['files_missing_names_column'])}**")
        for f in project_summary['files_missing_names_column']:
            print(f"     - {f}")


    print(f"\n### 1. Brakujące Wartości")
    if project_summary['files_with_missing_values']:
        print(f"   - Pliki zawierające brakujące wartości: **{len(project_summary['files_with_missing_values'])}**")
        for f in project_summary['files_with_missing_values']:
            print(f"     - {f}")
        print(f"   - Sumaryczna liczba brakujących wartości w kolumnach:")
        for col, count in project_summary['missing_values_by_column'].items():
            print(f"     - '{col}': {count}")
    else:
        print("   - Brak brakujących wartości we wszystkich przetworzonych plikach. ✅")

    print(f"\n### 2. Jakość Adnotacji Behawioralnych")
    if project_summary['total_annotations_by_class']:
        print(f"   - Łączny rozkład klas adnotacji w całym projekcie:")
        total_annotations_df = pd.Series(project_summary['total_annotations_by_class']).sort_values(ascending=False)
        print(total_annotations_df)

        if project_summary['files_with_highly_imbalanced_classes']:
            print(f"   - Pliki z **mocno niezbalansowanymi** klasami: **{len(project_summary['files_with_highly_imbalanced_classes'])}** ⚠️")
            for f in project_summary['files_with_highly_imbalanced_classes']:
                print(f"     - {f}")
        
        if project_summary['files_with_imbalanced_classes']:
            print(f"   - Pliki z **niezbalansowanymi** klasami (ale nie mocno): **{len(project_summary['files_with_imbalanced_classes'])}** ⚠️")
            for f in project_summary['files_with_imbalanced_classes']:
                print(f"     - {f}")
        
        if project_summary['files_with_problematic_annotations']:
            print(f"   - Pliki z **problematicznym formatowaniem adnotacji** (np. wiele adnotacji w jednej komórce): **{len(project_summary['files_with_problematic_annotations'])}** ⚠️")
            for f in project_summary['files_with_problematic_annotations']:
                print(f"     - {f}")
    else:
        print("   - Brak danych o adnotacjach (może brak kolumny 'annotation' lub puste pliki).")

    print(f"\n### 3. Jakość Metadanych z Nazw Plików")
    if project_summary['total_segments_by_well']:
        print(f"   - Łączna liczba segmentów na studzienkę w całym projekcie:")
        total_segments_well_df = pd.Series(project_summary['total_segments_by_well']).sort_values(ascending=False)
        print(total_segments_well_df)
    else:
        print("   - Brak danych o segmentach na studzienkę (może brak kolumny 'names' lub brak 'well_id' w nazwach).")

    if project_summary['files_failed_well_id_extraction']:
        print(f"   - Pliki, w których **nie udało się wyodrębnić Well ID**: **{len(project_summary['files_failed_well_id_extraction'])}** ⚠️")
        for f in project_summary['files_failed_well_id_extraction']:
            print(f"     - {f}")
            
    if project_summary['files_with_duplicated_segments']:
        print(f"   - Pliki zawierające **zduplikowane wpisy segmentów**: **{len(project_summary['files_with_duplicated_segments'])}** ⚠️")
        for f in project_summary['files_with_duplicated_segments']:
            print(f"     - {f}")
            
    if project_summary['files_with_frame_gaps']:
        print(f"   - Pliki, w których znaleziono **luki lub nakładanie się klatek** między segmentami: **{len(project_summary['files_with_frame_gaps'])}** ⚠️")
        for f in project_summary['files_with_frame_gaps']:
            print(f"     - {f}")

    if project_summary['files_failed_frame_extraction']:
        print(f"   - Pliki, w których **nie udało się wyodrębnić zakresów klatek**: **{len(project_summary['files_failed_frame_extraction'])}** ⚠️")
        for f in project_summary['files_failed_frame_extraction']:
            print(f"     - {f}")
            
    if project_summary['missing_extracted_metadata']:
        print(f"   - Sumaryczna liczba brakujących wartości w wyodrębnionych metadanych (date, age_dpf, experiment_type):")
        for col, count in project_summary['missing_extracted_metadata'].items():
            print(f"     - '{col}': {count}")
    else:
        print("   - Brak brakujących wartości w wyodrębnionych metadanych (jeśli kolumny 'names' były obecne). ✅")

    print("\n" + "="*80)
    print("                 ANALIZA PROJEKTU ZAKOŃCZONA                 ")
    print("="*80)

# --- PRZYKŁADOWE UŻYCIE ---
if __name__ == "__main__":
    # Ustaw ścieżkę do głównego katalogu Twojego projektu
    # Na przykład, jeśli Twój katalog projektu to 'seizure_detection_ml_dataset'
    # i tam są podfoldery z plikami CSV, ustaw:
    project_base_directory = 'seizure_detection_ml_dataset' 
    # Możesz też użyć '.' jeśli skrypt jest uruchamiany z katalogu głównego projektu

    # Tworzenie przykładowych plików dla testów
    print("Tworzenie przykładowych plików do testów...")
    os.makedirs(os.path.join(project_base_directory, 'timepoint1', 'exp1'), exist_ok=True)
    os.makedirs(os.path.join(project_base_directory, 'timepoint2', 'exp2'), exist_ok=True)

    # Przykładowy plik 1 (dobra jakość)
    pd.DataFrame({
        'names': ['20221205_AB_PTZ_5DPF_Timepoint1_96wp_20221205_161742_024_well_B1_frames_0_to_60.mp4',
                  '20221205_AB_PTZ_5DPF_Timepoint1_96wp_20221205_161742_024_well_B1_frames_60_to_120.mp4'],
        'annotation': ['seizure', 'control']
    }).to_csv(os.path.join(project_base_directory, 'timepoint1', 'exp1', 'dataset_wellB1_annotations.csv'), index=False)

    # Przykładowy plik 2 (z brakującymi wartościami i niezbalansowanymi klasami)
    pd.DataFrame({
        'names': ['20221206_Control_6DPF_Timepoint1_96wp_20221206_100000_001_well_A1_frames_0_to_60.mp4',
                  '20221206_Control_6DPF_Timepoint1_96wp_20221206_100000_001_well_A1_frames_60_to_120.mp4',
                  '20221206_Control_6DPF_Timepoint1_96wp_20221206_100000_001_well_A1_frames_120_to_180.mp4'],
        'annotation': ['control', 'control', None] # Brakująca wartość
    }).to_csv(os.path.join(project_base_directory, 'timepoint2', 'exp2', 'dataset_wellA1_annotations.csv'), index=False)

    # Przykładowy plik 3 (z błędnym formatowaniem adnotacji i luką w klatkach)
    pd.DataFrame({
        'names': ['20221207_AB_PTZ_7DPF_Timepoint1_96wp_20221207_140000_005_well_C3_frames_0_to_60.mp4',
                  '20221207_AB_PTZ_7DPF_Timepoint1_96wp_20221207_140000_005_well_C3_frames_180_to_240.mp4'], # Luka
        'annotation': ['seizure', 'control,seizure'] # Błędne formatowanie
    }).to_csv(os.path.join(project_base_directory, 'timepoint1', 'exp1', 'dataset_wellC3_annotations.csv'), index=False)
    
    # Przykładowy plik 4 (pusty)
    open(os.path.join(project_base_directory, 'timepoint1', 'exp1', 'dataset_empty.csv'), 'a').close()

    # Przykładowy plik 5 (brak kolumn 'names' i 'annotation')
    pd.DataFrame({
        'other_column': [1,2,3]
    }).to_csv(os.path.join(project_base_directory, 'timepoint2', 'exp2', 'dataset_no_columns.csv'), index=False)


    print("\nRozpoczynanie sumarycznej analizy projektu...")
    summarize_project_data_quality(project_base_directory)

    # Opcjonalnie: usunięcie przykładowych plików po zakończeniu
    # import shutil
    # shutil.rmtree(project_base_directory)
    # print(f"\nUsunięto przykładowy katalog: {project_base_directory}")