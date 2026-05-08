"""
Módulo de Feature Engineering (RF3, RF4)
==========================================
• RF3 – Espectrograma de Mel por segmento de latido (Librosa)
• RF4 – Centroides espectrales por segmento

Estrategia de segmentación
---------------------------
Para cada pico R se extrae una ventana simétrica de ±200 ms:
  pre  = int(0.2 * fs)   muestras antes del pico
  post = int(0.4 * fs)   muestras después del pico
→ ventana total ≈ 216 muestras @ 360 Hz (PQRST completo)

Autor  : Pipeline ECG - Antigravity
SRS    : §3.2 Análisis de Patrones (Módulo Librosa)
"""

from __future__ import annotations
import numpy as np
import librosa
from dataclasses import dataclass
from src.ingestion import ECGRecord


# ── Parámetros de segmentación ────────────────────────────────────────────────
PRE_SAMPLES  = 72   # ~200 ms @ 360 Hz
POST_SAMPLES = 144  # ~400 ms @ 360 Hz
BEAT_LEN     = PRE_SAMPLES + POST_SAMPLES   # 216 muestras


# ── Parámetros del Espectrograma de Mel (RF3) ─────────────────────────────────
N_FFT      = 128   # ventana FFT - mayor para sr=360 Hz
HOP_LENGTH = 32
N_MELS     = 32    # número de filtros Mel
F_MAX      = 45.0  # igual al corte superior del filtro (RF1)


@dataclass
class BeatFeatures:
    """Características extraídas de un único latido."""
    segment     : np.ndarray   # (BEAT_LEN,)  señal en tiempo
    mel_spec    : np.ndarray   # (N_MELS, T)  espectrograma de Mel
    centroid    : np.ndarray   # (T,)          centroide espectral
    label       : str          # etiqueta AAMI: N, S, V, F


def extract_beats(
    rec: ECGRecord,
    clean_signal: np.ndarray,
    min_margin: int = 10,
) -> list[BeatFeatures]:
    """
    Segmenta latidos y extrae espectrograma de Mel + centroide espectral.

    Parámetros
    ----------
    rec          : ECGRecord   objeto con r_peak_indices y aami_labels
    clean_signal : np.ndarray  señal filtrada y normalizada
    min_margin   : int         margen mínimo de muestras hasta el borde

    Retorna
    -------
    list[BeatFeatures]  lista de objetos con características por latido
    """
    n = len(clean_signal)
    beats: list[BeatFeatures] = []

    for peak, label in zip(rec.r_peak_indices, rec.aami_labels):
        start = peak - PRE_SAMPLES
        end   = peak + POST_SAMPLES

        # Descartar latidos en bordes de la señal
        if start < min_margin or end > n - min_margin:
            continue

        segment = clean_signal[start:end].astype(np.float32)

        # ── RF3: Espectrograma de Mel ─────────────────────────────────────────
        mel_spec = librosa.feature.melspectrogram(
            y          = segment,
            sr         = rec.fs,
            n_fft      = N_FFT,
            hop_length = HOP_LENGTH,
            n_mels     = N_MELS,
            fmax       = F_MAX,
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # ── RF4: Centroide Espectral ──────────────────────────────────────────
        centroid = librosa.feature.spectral_centroid(
            y          = segment,
            sr         = rec.fs,
            n_fft      = N_FFT,
            hop_length = HOP_LENGTH,
        )[0]

        beats.append(BeatFeatures(
            segment  = segment,
            mel_spec = mel_spec_db,
            centroid = centroid,
            label    = label,
        ))

    return beats


def beats_to_arrays(
    beats: list[BeatFeatures],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Convierte la lista de BeatFeatures en arrays NumPy listos para el modelo.

    Retorna
    -------
    X_mel       : (N, N_MELS, T, 1)   espectrogramas (canal extra para CNN)
    X_centroid  : (N, T)               centroides
    X_time      : (N, BEAT_LEN)        señales en tiempo
    y_labels    : list[str]            etiquetas AAMI
    """
    mel_list = [b.mel_spec for b in beats]
    cen_list = [b.centroid for b in beats]
    seg_list = [b.segment  for b in beats]
    lbl_list = [b.label    for b in beats]

    # Padding / truncado para igualar dimensiones temporales
    T_mel = max(m.shape[1] for m in mel_list)
    T_cen = max(c.shape[0] for c in cen_list)

    def pad2d(m: np.ndarray, T: int) -> np.ndarray:
        if m.shape[1] < T:
            return np.pad(m, ((0, 0), (0, T - m.shape[1])))
        return m[:, :T]

    def pad1d(c: np.ndarray, T: int) -> np.ndarray:
        if c.shape[0] < T:
            return np.pad(c, (0, T - c.shape[0]))
        return c[:T]

    X_mel      = np.stack([pad2d(m, T_mel) for m in mel_list])[..., np.newaxis]
    X_centroid = np.stack([pad1d(c, T_cen) for c in cen_list])
    X_time     = np.stack(seg_list)

    return X_mel.astype(np.float32), X_centroid.astype(np.float32), \
           X_time.astype(np.float32), lbl_list
