import sys
import io
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.ingestion import load_multiple, DS1_RECORDS, DS2_RECORDS
from src.preprocessing import preprocess_record
from src.features import extract_beats, beats_to_arrays
from src.model import build_cnn, train_model, predict, encode_labels, AAMI_CLASSES
from src.visualization import plot_raw_vs_filtered, evaluate_and_plot, plot_training_history

OUTPUT_DIR = Path("outputs")
LOCAL_DIR = "mit-bih-arrhythmia-database-1.0.0"

def main():
    print("=" * 65)
    print("  ENTRENAMIENTO MEDICO COMPLETO: N, S, V, F (44 REGISTROS)")
    print("  Usando base de datos local...")
    print("=" * 65 + "\n")

    # Excluir registros que no tienen senal MLII en el canal 0 
    # (El 114 y 202 a veces tienen la senal en otro canal, pero para simplificar los saltamos si fallan, 
    # o mejor usamos todo DS1 y DS2)
    all_records_ids = DS1_RECORDS + DS2_RECORDS
    record_paths = [f"{LOCAL_DIR}/{r}" for r in all_records_ids]
    
    print(f"[1/5] Cargando {len(record_paths)} registros locales...")
    t0 = time.perf_counter()
    # pn_dir=None obliga a wfdb a buscar los archivos localmente
    records = load_multiple(record_paths, pn_dir=None, verbose=True)
    print(f"\n    OK  {len(records)} registros cargados en {time.perf_counter()-t0:.1f}s\n")

    print("[2/5] y [3/5] Preprocesamiento y Extraccion de Espectrogramas...")
    t0 = time.perf_counter()
    all_beats = []
    
    for idx, rec in enumerate(records):
        try:
            raw_signal, clean_signal = preprocess_record(rec)
            beats = extract_beats(rec, clean_signal)
            all_beats.extend(beats)
            if (idx + 1) % 5 == 0:
                print(f"        -> Procesados {idx + 1}/{len(records)} registros...")
        except Exception as e:
            print(f"        -> Error en {rec.record_id}: {e}")
            
    print(f"    OK  Se extrajeron {len(all_beats)} latidos en {time.perf_counter()-t0:.1f}s\n")

    print("[*] Preparando Tensores...")
    X_mel, X_centroid, X_time, y_labels = beats_to_arrays(all_beats)
    y_encoded = encode_labels(y_labels)

    unique, counts = np.unique(y_encoded, return_counts=True)
    print("    Distribucion AAMI:")
    for idx, cnt in zip(unique, counts):
        print(f"      {AAMI_CLASSES[idx]}: {cnt} latidos")

    X_train, X_val, y_train, y_val = train_test_split(
        X_mel, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    print(f"\n    Train: {len(X_train)} latidos  |  Val: {len(X_val)} latidos\n")

    print("[4/5] Entrenamiento de Red Neuronal (30 Epocas)...")
    # Esto tomara tiempo
    model = build_cnn(X_mel.shape[1:])
    history = train_model(
        model, X_train, y_train, X_val, y_val, epochs=30, batch_size=256
    )
    plot_training_history(history, OUTPUT_DIR)

    print("\n[5/5] Evaluacion Final")
    y_pred_idx, _ = predict(model, X_val)
    metrics = evaluate_and_plot(y_val, y_pred_idx, OUTPUT_DIR, tag="val")

    model_path = OUTPUT_DIR / "ecg_cnn_model.keras"
    model.save(model_path)
    print(f"\n    OK  Modelo GUARDADO Y LISTO: {model_path}")
    print("\n" + "=" * 65)
    print(" ENTRENAMIENTO COMPLETADO. LA APLICACION AHORA ES EXPERTA.")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
