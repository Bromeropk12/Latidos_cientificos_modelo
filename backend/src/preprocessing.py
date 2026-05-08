"""
Módulo de Preprocesamiento (RF1, RF2)
======================================
• RF1 – Filtro pasabanda Butterworth (0.5 Hz – 45 Hz)
• RF2 – Normalización Z-score por registro

Autor  : Pipeline ECG - Antigravity
SRS    : §3.1 Procesamiento de Señal (Módulo SciPy)
"""

from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt
from src.ingestion import ECGRecord


# ── Parámetros del filtro ──────────────────────────────────────────────────────
LOWCUT_HZ  = 0.5    # RF1: límite inferior del pasabanda
HIGHCUT_HZ = 45.0   # RF1: límite superior del pasabanda
FILTER_ORDER = 4    # orden del filtro Butterworth


def _butter_bandpass_sos(fs: int) -> np.ndarray:
    """
    Diseña el filtro Butterworth pasabanda y retorna los coeficientes
    en formato Second-Order Sections (SOS) para mayor estabilidad numérica.
    """
    nyq  = fs / 2.0
    low  = LOWCUT_HZ  / nyq
    high = HIGHCUT_HZ / nyq
    return butter(FILTER_ORDER, [low, high], btype="band", output="sos")


def bandpass_filter(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Aplica filtro Butterworth pasabanda (RF1) usando filtrado de fase cero
    (sosfiltfilt) para evitar distorsión de fase en el complejo QRS.

    Parámetros
    ----------
    signal : np.ndarray  shape (n_samples,)
    fs     : int          frecuencia de muestreo en Hz

    Retorna
    -------
    np.ndarray  señal filtrada, misma forma que la entrada
    """
    sos = _butter_bandpass_sos(fs)
    return sosfiltfilt(sos, signal).astype(np.float32)


def zscore_normalize(signal: np.ndarray) -> np.ndarray:
    """
    Normalización Z-score (RF2): µ=0, σ=1 por registro.
    Evita división por cero añadiendo ε.

    Parámetros
    ----------
    signal : np.ndarray  señal cruda o filtrada

    Retorna
    -------
    np.ndarray  señal normalizada
    """
    mu    = signal.mean()
    sigma = signal.std() + 1e-8
    return ((signal - mu) / sigma).astype(np.float32)


def preprocess_record(rec: ECGRecord) -> tuple[np.ndarray, np.ndarray]:
    """
    Pipeline completo de preprocesamiento para un ECGRecord.

    Retorna
    -------
    raw_signal      : np.ndarray  señal cruda (para comparación visual)
    clean_signal    : np.ndarray  señal filtrada y normalizada
    """
    raw_signal   = rec.signal.copy()
    filtered     = bandpass_filter(raw_signal, rec.fs)
    clean_signal = zscore_normalize(filtered)
    return raw_signal, clean_signal
