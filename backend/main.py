from __future__ import annotations
# -*- coding: utf-8 -*-
"""
Pipeline Principal - main.py
==============================
Orquesta los 5 modulos del sistema de clasificacion de arritmias ECG.

  1. Ingesta          -> src/ingestion.py
  2. Preprocesamiento -> src/preprocessing.py
  3. Feature Eng.     -> src/features.py
  4. Modelado CNN     -> src/model.py
  5. Evaluacion       -> src/visualization.py

Uso:
  python main.py --quick
  python main.py --records 100 101 103 --epochs 30

SRS - Requerimientos No Funcionales:
  < 10 s por registro de 30 min | F1-Score > 90%
"""

import sys
import io
# Forzar UTF-8 en consola Windows (evita UnicodeEncodeError con CP1252)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Modulos del pipeline
from src.ingestion     import load_multiple
from src.preprocessing import preprocess_record
from src.features      import extract_beats, beats_to_arrays
from src.model         import build_cnn, train_model, predict, encode_labels, AAMI_CLASSES
from src.visualization import plot_raw_vs_filtered, evaluate_and_plot, plot_training_history

OUTPUT_DIR = Path("outputs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline ECG CNN - MIT-BIH")
    p.add_argument(
        "--records", nargs="+", default=["100", "101", "103"],
        help="IDs de registros MIT-BIH (default: 100 101 103)"
    )
    p.add_argument("--epochs",    type=int,   default=30,   help="Epocas de entrenamiento")
    p.add_argument("--batch",     type=int,   default=256,  help="Batch size")
    p.add_argument("--test-size", type=float, default=0.20, help="Fraccion de validacion")
    p.add_argument("--quick",     action="store_true",
                   help="Modo rapido: 3 registros, 5 epocas")
    return p.parse_args()


def banner() -> None:
    print("\n" + "=" * 65)
    print("  [ECG]  SISTEMA DE CLASIFICACION DE ARRITMIAS - MIT-BIH")
    print("  SRS #3  |  Clasificacion AAMI: N | S | V | F")
    print("=" * 65 + "\n")


def main() -> None:
    args = parse_args()

    if args.quick:
        args.records = ["100", "101", "103"]
        args.epochs  = 5

    banner()
    t0_total = time.perf_counter()

    # ── FASE 1: Ingesta ────────────────────────────────────────────────────────
    print("[1/5] FASE 1 - Ingesta de senales (wfdb + PhysioNet)")
    t0 = time.perf_counter()
    records = load_multiple(args.records, pn_dir="mitdb", verbose=True)
    print(f"    OK  {len(records)} registros cargados en {time.perf_counter()-t0:.1f}s\n")

    # ── FASES 2 y 3: Preprocesamiento + Features ──────────────────────────────
    print("[2/5] FASE 2 - Preprocesamiento  (Butterworth 0.5-45 Hz + Z-score)")
    print("[3/5] FASE 3 - Feature Engineering (Mel-Spec + Centroide)")
    t0 = time.perf_counter()
    all_beats = []

    for rec in records:
        t_rec = time.perf_counter()

        # RF1 + RF2
        raw_signal, clean_signal = preprocess_record(rec)

        # Figura comparativa cruda vs. filtrada
        plot_raw_vs_filtered(
            raw_signal, clean_signal, rec.fs, rec.record_id,
            duration_s=5.0, output_dir=OUTPUT_DIR,
        )

        # RF3 + RF4: segmentacion + espectrograma Mel + centroides
        beats = extract_beats(rec, clean_signal)
        all_beats.extend(beats)

        elapsed = time.perf_counter() - t_rec
        print(f"    Registro {rec.record_id}: {len(beats)} latidos | {elapsed:.2f}s")

    print(f"    OK  Fases 2+3 completadas en {time.perf_counter()-t0:.1f}s\n")

    # ── Preparar arrays ────────────────────────────────────────────────────────
    print("[*]  Preparando datasets...")
    X_mel, X_centroid, X_time, y_labels = beats_to_arrays(all_beats)
    y_encoded = encode_labels(y_labels)

    unique, counts = np.unique(y_encoded, return_counts=True)
    print("    Distribucion AAMI:")
    for idx, cnt in zip(unique, counts):
        print(f"      {AAMI_CLASSES[idx]}: {cnt} latidos")

    # Desactivar stratify si alguna clase tiene menos de 2 muestras
    min_count = counts.min() if len(counts) > 0 else 0
    use_stratify = y_encoded if (len(unique) > 1 and min_count >= 2) else None
    if use_stratify is None:
        print("    [!] Stratify desactivado (alguna clase tiene < 2 muestras)")

    X_train, X_val, y_train, y_val = train_test_split(
        X_mel, y_encoded,
        test_size    = args.test_size,
        random_state = 42,
        stratify     = use_stratify,
    )
    print(f"    Train: {len(X_train)}  |  Val: {len(X_val)}\n")

    # ── FASE 4: CNN ────────────────────────────────────────────────────────────
    print("[4/5] FASE 4 - Entrenamiento CNN (AAMI: N, S, V, F)")
    model = build_cnn(X_mel.shape[1:])
    model.summary()
    print()

    history = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=args.epochs, batch_size=args.batch,
    )
    plot_training_history(history, OUTPUT_DIR)
    print()

    # ── FASE 5: Evaluacion ─────────────────────────────────────────────────────
    print("[5/5] FASE 5 - Evaluacion y metricas")
    y_pred_idx, _ = predict(model, X_val)
    metrics = evaluate_and_plot(y_val, y_pred_idx, OUTPUT_DIR, tag="val")

    # Guardar modelo entrenado
    model_path = OUTPUT_DIR / "ecg_cnn_model.keras"
    model.save(model_path)
    print(f"\n    OK  Modelo guardado: {model_path}")

    # ── Resumen ────────────────────────────────────────────────────────────────
    total  = time.perf_counter() - t0_total
    f1     = metrics["f1_macro"]
    status = "CUMPLE RNF > 90%" if f1 > 0.90 else "NO cumple (< 90%)"

    print("\n" + "=" * 65)
    print(f"  Pipeline completado en {total:.1f}s")
    print(f"  F1-Score Macro : {f1:.4f}  [{status}]")
    print(f"  Artefactos     : {OUTPUT_DIR.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
