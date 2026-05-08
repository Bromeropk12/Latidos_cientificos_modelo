"""
Módulo de Modelado CNN (RF5)
==============================
Arquitectura ligera de Red Neuronal Convolucional para clasificación AAMI:
  N (Normal) | S (Supraventricular) | V (Ventricular) | F (Fusión)

Diseño orientado a:
  • Alta precisión  → F1-Score > 90 %  (RNF – §4)
  • Baja latencia   → < 10 s por registro de 30 min (RNF – §4)

Autor  : Pipeline ECG - Antigravity
SRS    : §3.3 Clasificación de Arritmias
"""

from __future__ import annotations
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Etiquetas AAMI en el orden fijo del modelo
AAMI_CLASSES = ["N", "S", "V", "F"]
NUM_CLASSES  = len(AAMI_CLASSES)
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(AAMI_CLASSES)}


# ── Arquitectura CNN ──────────────────────────────────────────────────────────

def build_cnn(input_shape: tuple[int, int, int]) -> keras.Model:
    """
    Construye una CNN ligera de 4 bloques convolucionales para clasificación
    de espectrogramas de Mel (input_shape = (N_MELS, T, 1)).

    Arquitectura
    ------------
    Conv2D(32, 3×3) → BN → ReLU → MaxPool
    Conv2D(64, 3×3) → BN → ReLU → MaxPool
    Conv2D(128, 3×3)→ BN → ReLU → GlobalAvgPool
    Dense(128) → Dropout(0.4) → Dense(NUM_CLASSES, softmax)

    Parámetros totales ≈ 150 K  (diseño ligero para inferencia rápida)
    """
    inp = keras.Input(shape=input_shape, name="mel_input")

    x = layers.Conv2D(32, (3, 3), padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="aami_class")(x)

    model = keras.Model(inputs=inp, outputs=out, name="ECG_CNN_Lite")
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=1e-3),
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"],
    )
    return model


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def encode_labels(labels: list[str]) -> np.ndarray:
    """Convierte etiquetas AAMI a índices enteros."""
    return np.array([LABEL_TO_IDX[l] for l in labels], dtype=np.int32)


def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 30,
    batch_size: int = 256,
    class_weight: dict | None = None,
) -> keras.callbacks.History:
    """
    Entrena el modelo con callbacks de Early Stopping y ReduceLR.

    Parámetros
    ----------
    class_weight : dict | None
        Pesos por clase para manejar el desbalance N >> S,V,F.
        Si es None se calcula automáticamente.
    """
    if class_weight is None:
        counts = np.bincount(y_train, minlength=NUM_CLASSES)
        total  = counts.sum()
        class_weight = {
            i: total / (NUM_CLASSES * c) if c > 0 else 1.0
            for i, c in enumerate(counts)
        }
        print(f"  → Pesos de clase calculados: {class_weight}")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = epochs,
        batch_size      = batch_size,
        class_weight    = class_weight,
        callbacks       = callbacks,
        verbose         = 1,
    )
    return history


# ── Inferencia ────────────────────────────────────────────────────────────────

def predict(model: keras.Model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Retorna etiquetas predichas e índices de clase.

    Retorna
    -------
    y_pred_idx  : np.ndarray int     índices de clase predichos
    y_pred_prob : np.ndarray float   probabilidades softmax
    """
    probs      = model.predict(X, batch_size=512, verbose=0)
    y_pred_idx = probs.argmax(axis=1)
    return y_pred_idx, probs
