# Software Requirements Specification (SRS) - Sistema de Clasificación de Arritmias ECG

## 1. Introducción
Este documento detalla los requerimientos para el desarrollo de un sistema de análisis de electrocardiogramas (ECG) mediante redes neuronales profundas. El proyecto utiliza la base de datos de referencia **MIT-BIH Arrhythmia Database** y se apoya en el procesamiento avanzado de señales con Python.

### 1.1 Propósito
Definir los aspectos funcionales y técnicos para construir un modelo capaz de clasificar latidos cardíacos en tiempo récord (2 horas de desarrollo asistido) utilizando agentes inteligentes.

### 1.2 Alcance del Proyecto
* **Extracción de señales:** Desde archivos `.dat` / `.atr` (PhysioNet).
* **Preprocesamiento:** Filtrado de ruido mediante **SciPy**.
* **Análisis:** Características de frecuencia y tiempo con **Librosa**.
* **Clasificación:** Mediante Redes Neuronales (IA).

---

## 2. Descripción General

### 2.1 Flujo de Trabajo (Workflow Agéntico)
El desarrollo se divide en tres fases críticas gestionadas por el agente de automatización:

| Fase | Descripción | Herramientas |
| :--- | :--- | :--- |
| **Ingesta y Limpieza** | Carga de señales y eliminación de ruidos de línea base y electromiográficos. | **SciPy** (signal) |
| **Feature Engineering** | Segmentación de latidos y extracción de espectrogramas/MFCCs. | **Librosa** |
| **Inferencia** | Evaluación de la señal en la red neuronal pre-entrenada. | **Open Source Models** |

---

## 3. Requerimientos Funcionales

### 3.1 Procesamiento de Señal (Módulo SciPy)
* **RF1:** El sistema debe aplicar un filtro pasabanda (**Butterworth**) entre **0.5Hz** y **45Hz** para aislar el complejo QRS.
* **RF2:** Implementar normalización **Z-score** para estandarizar la amplitud entre diferentes pacientes.

### 3.2 Análisis de Patrones (Módulo Librosa)
* **RF3:** Generar una representación tiempo-frecuencia (**Espectrograma de Mel**) de cada segmento de latido.
* **RF4:** Extraer los centroides espectrales para identificar cambios bruscos en la morfología del latido.

### 3.3 Clasificación de Arritmias
* **RF5:** Clasificar los latidos en al menos 4 categorías de la AAMI: Normal (**N**), Supraventricular (**S**), Ventricular (**V**) y Fusión (**F**).

---

## 4. Requerimientos No Funcionales
* **Eficiencia:** El tiempo de procesamiento por registro de 30 minutos debe ser inferior a **10 segundos**.
* **Precisión:** Se busca un **F1-Score superior al 90%** en el conjunto de prueba de MIT-BIH.
* **Modularidad:** El código debe ser compatible con entornos de ejecución rápida (Notebooks/Scripts).

---

## 5. Stack Tecnológico
* **Lenguaje:** Python 3.10+
* **Procesamiento:** **SciPy** (filtros digitales) y **NumPy** (matrices).
* **Audio/Señal:** **Librosa** (espectrogramas de señales biomédicas).
* **Dataset:** MIT-BIH Arrhythmia Database (vía librería **WFDB**).