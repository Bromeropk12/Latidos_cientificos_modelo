"""
Módulo de Visualización y Evaluación
======================================
• Señal cruda vs. filtrada (Matplotlib)
• Matriz de confusión
• Reporte de clasificación F1 / Precision / Recall
• Curva de aprendizaje del entrenamiento

Autor  : Pipeline ECG - Antigravity
SRS    : §4 Requerimientos No Funcionales — Precisión y Modularidad
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")           # backend sin pantalla para entornos headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    ConfusionMatrixDisplay,
)
from pathlib import Path
from src.model import AAMI_CLASSES


# ── Paleta de colores ─────────────────────────────────────────────────────────
COLORS = {
    "raw"        : "#64B5F6",   # azul suave
    "filtered"   : "#EF5350",   # rojo vibrante
    "accent"     : "#AB47BC",   # púrpura
    "bg"         : "#1A1A2E",   # fondo oscuro
    "surface"    : "#16213E",
    "text"       : "#E0E0E0",
}


def _apply_dark_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor"  : COLORS["bg"],
        "axes.facecolor"    : COLORS["surface"],
        "axes.edgecolor"    : COLORS["text"],
        "axes.labelcolor"   : COLORS["text"],
        "xtick.color"       : COLORS["text"],
        "ytick.color"       : COLORS["text"],
        "text.color"        : COLORS["text"],
        "grid.color"        : "#2A2A4A",
        "grid.linewidth"    : 0.5,
        "font.family"       : "DejaVu Sans",
    })


# ── RF Visualización: señal cruda vs. filtrada ────────────────────────────────

def plot_raw_vs_filtered(
    raw_signal    : np.ndarray,
    clean_signal  : np.ndarray,
    fs            : int,
    record_id     : str,
    duration_s    : float = 5.0,
    output_dir    : Path | str = "outputs",
) -> Path:
    """
    Genera una figura comparativa de la señal cruda vs. filtrada/normalizada
    para los primeros `duration_s` segundos del registro.

    Parámetros
    ----------
    output_dir : Path  carpeta donde se guarda la figura
    """
    _apply_dark_theme()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = int(duration_s * fs)
    t = np.arange(n_samples) / fs

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle(
        f"MIT-BIH  |  Registro {record_id}  |  Señal cruda vs. Filtrada (Butterworth 0.5-45 Hz)",
        fontsize=13, fontweight="bold", color=COLORS["text"],
    )

    # Señal cruda
    axes[0].plot(t, raw_signal[:n_samples], color=COLORS["raw"], lw=0.8, alpha=0.9)
    axes[0].set_ylabel("Amplitud (mV)", fontsize=10)
    axes[0].set_title("Señal Cruda (RAW)", fontsize=10)
    axes[0].grid(True, ls="--")

    # Señal filtrada
    axes[1].plot(t, clean_signal[:n_samples], color=COLORS["filtered"], lw=0.8, alpha=0.9)
    axes[1].set_ylabel("Amplitud (Z-score)", fontsize=10)
    axes[1].set_xlabel("Tiempo (s)", fontsize=10)
    axes[1].set_title("Señal Filtrada + Normalizada Z-score", fontsize=10)
    axes[1].grid(True, ls="--")

    plt.tight_layout()
    out_path = output_dir / f"signal_comparison_{record_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Figura guardada: {out_path}")
    return out_path


# ── Evaluación: Matriz de confusión y reporte ────────────────────────────────

def evaluate_and_plot(
    y_true     : np.ndarray,
    y_pred     : np.ndarray,
    output_dir : Path | str = "outputs",
    tag        : str = "",
) -> dict:
    """
    Genera y guarda la matriz de confusión + reporte de clasificación.

    Retorna
    -------
    dict  con las métricas resumidas (accuracy, macro F1, per-class F1)
    """
    _apply_dark_theme()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clases presentes en este split (puede que no esten las 4 AAMI)
    present_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    present_names  = [AAMI_CLASSES[i] for i in present_labels]

    cm      = confusion_matrix(y_true, y_pred, labels=present_labels)
    report  = classification_report(
        y_true, y_pred,
        labels       = present_labels,
        target_names = present_names,
        output_dict  = True,
        zero_division = 0,
    )
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # -- Figura: Matriz de confusion ------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["surface"])

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=present_names)
    disp.plot(ax=ax, colorbar=True, cmap="plasma", values_format="d")
    ax.set_title(
        f"Matriz de Confusión AAMI  |  F1-Macro: {f1_macro:.3f}",
        fontsize=12, fontweight="bold", color=COLORS["text"],
    )
    ax.tick_params(colors=COLORS["text"])
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])

    suffix = f"_{tag}" if tag else ""
    cm_path = output_dir / f"confusion_matrix{suffix}.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Matriz de confusion: {cm_path}")

    # ── Imprimir reporte en consola ──────────────────────────────────────────
    print("\n" + "-" * 60)
    print(classification_report(
        y_true, y_pred,
        labels        = present_labels,
        target_names  = present_names,
        zero_division = 0,
    ))
    status = 'CUMPLE > 0.90' if f1_macro > 0.90 else 'NO cumple > 0.90'
    print(f"  F1-Score Macro: {f1_macro:.4f}  ({status})")
    print("─" * 60)

    return {
        "f1_macro"      : f1_macro,
        "report"        : report,
        "confusion_matrix": cm.tolist(),
    }


# ── Curva de aprendizaje ──────────────────────────────────────────────────────

def plot_training_history(
    history    : object,                # keras.callbacks.History
    output_dir : Path | str = "outputs",
) -> Path:
    """Guarda curvas de accuracy y loss del entrenamiento."""
    _apply_dark_theme()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Curvas de Entrenamiento CNN", fontsize=13,
                 fontweight="bold", color=COLORS["text"])

    # Accuracy
    ax1.plot(history.history["accuracy"],     label="Train", color=COLORS["raw"])
    ax1.plot(history.history["val_accuracy"], label="Val",   color=COLORS["filtered"])
    ax1.set_title("Accuracy", color=COLORS["text"])
    ax1.set_xlabel("Época"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(True, ls="--")

    # Loss
    ax2.plot(history.history["loss"],     label="Train", color=COLORS["raw"])
    ax2.plot(history.history["val_loss"], label="Val",   color=COLORS["filtered"])
    ax2.set_title("Loss", color=COLORS["text"])
    ax2.set_xlabel("Época"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(True, ls="--")

    plt.tight_layout()
    out_path = output_dir / "training_history.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Curvas de entrenamiento: {out_path}")
    return out_path
