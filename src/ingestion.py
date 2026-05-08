"""
Módulo de Ingesta (RF - Fase 1)
================================
Carga registros del dataset MIT-BIH Arrhythmia Database (.dat / .atr)
usando la librería wfdb y PhysioNet.

Autor  : Pipeline ECG - Antigravity
SRS    : §3 Requerimientos Funcionales
"""

from __future__ import annotations
import wfdb
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field


# ── AAMI beat-type mapping ────────────────────────────────────────────────────
# Mapea los símbolos WFDB originales a las 4 categorías AAMI requeridas (RF5)
AAMI_MAP: dict[str, str] = {
    # Normal
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # Supraventricular ectopic
    "A": "S", "a": "S", "J": "S", "S": "S",
    # Ventricular ectopic
    "V": "V", "E": "V",
    # Fusion
    "F": "F",
}

# Registros canónicos de entrenamiento MIT-BIH (DS1 / DS2 split)
DS1_RECORDS = ["101","106","108","109","112","114","115","116","118","119",
               "122","124","201","203","205","207","208","209","215","220",
               "223","230"]
DS2_RECORDS = ["100","103","105","111","113","117","121","123","200","202",
               "210","212","213","214","219","221","222","228","231","232",
               "233","234"]


@dataclass
class ECGRecord:
    """Contenedor con la señal cruda, frecuencia de muestreo y anotaciones."""
    record_id: str
    signal: np.ndarray          # forma (n_samples, n_leads)
    fs: int                     # Hz — típicamente 360
    annotations: wfdb.Annotation
    aami_labels: list[str] = field(default_factory=list)
    r_peak_indices: list[int]  = field(default_factory=list)


def load_record(record_id: str, pn_dir: str = "mitdb", channel: int = 0) -> ECGRecord:
    """
    Descarga (cache local) y carga un registro MIT-BIH desde PhysioNet.

    Parametros
    ----------
    record_id : str
        Identificador del registro (e.g. "100").
    pn_dir : str
        Base de datos PhysioNet. Por defecto 'mitdb'.
    channel : int
        Canal ECG a usar (0 = MLII, 1 = V1/V5).

    Retorna
    -------
    ECGRecord
        Objeto con senal, fs y anotaciones mapeadas a AAMI.
    """
    # wfdb descarga y cachea automaticamente en ~/.wfdb/
    record = wfdb.rdrecord(record_id, pn_dir=pn_dir)
    ann    = wfdb.rdann(record_id, "atr", pn_dir=pn_dir)

    signal = record.p_signal[:, channel].astype(np.float32)

    # Filtrar picos válidos y mapear a AAMI
    r_peaks, labels = [], []
    for idx, sym in zip(ann.sample, ann.symbol):
        if sym in AAMI_MAP:
            r_peaks.append(idx)
            labels.append(AAMI_MAP[sym])

    return ECGRecord(
        record_id    = record_id,
        signal       = signal,
        fs           = record.fs,
        annotations  = ann,
        aami_labels  = labels,
        r_peak_indices = r_peaks,
    )


def load_multiple(
    record_ids: list[str],
    pn_dir: str = "mitdb",
    channel: int = 0,
    verbose: bool = True,
) -> list[ECGRecord]:
    """Carga una lista de registros con reporte de progreso."""
    records = []
    for rid in record_ids:
        if verbose:
            print(f"  -> Cargando registro {rid} ...", end=" ", flush=True)
        rec = load_record(rid, pn_dir=pn_dir, channel=channel)
        if verbose:
            print(f"OK ({len(rec.r_peak_indices)} latidos validos)")
        records.append(rec)
    return records
