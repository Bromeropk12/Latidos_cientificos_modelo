from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sqlite3
import shutil
import os
from datetime import datetime
import numpy as np
from tensorflow import keras
import matplotlib
matplotlib.use('Agg') # Modo headless para no abrir ventanas
import matplotlib.pyplot as plt
import io
import base64

from src.ingestion import load_record
from src.preprocessing import preprocess_record
from src.features import extract_beats, beats_to_arrays

app = FastAPI(title="ECG Analysis API")

# Habilitar CORS para que el frontend Electron pueda comunicarse sin problemas
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo ya entrenado (Cerebro de la aplicacion)
MODEL_PATH = "outputs/ecg_cnn_model.keras"
model = None

def init_db():
    conn = sqlite3.connect("database.sqlite")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS analyses
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  patient_name TEXT, 
                  date TEXT, 
                  total_beats INTEGER, 
                  abnormal_beats INTEGER, 
                  diagnosis TEXT)''')
    conn.commit()
    conn.close()

@app.on_event("startup")
def startup_event():
    global model
    try:
        model = keras.models.load_model(MODEL_PATH)
        print(f"✅ Modelo cargado correctamente desde {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️  No se encontro el modelo en {MODEL_PATH}. Ejecuta main.py primero.")
    init_db()

import traceback

@app.post("/analyze")
async def analyze_ecg(
    patient_name: str = Form(...),
    dat_file: UploadFile = File(...),
    atr_file: UploadFile = File(...),
    hea_file: UploadFile = File(...)
):
    if not model:
        raise HTTPException(status_code=500, detail="Modelo no disponible.")

    # Crear directorio temporal para guardar los archivos subidos
    os.makedirs("temp", exist_ok=True)
    
    # Extraer el nombre base (ej: "223" de "223.hea") 
    # Esto es CRÍTICO porque el .hea tiene escrito internamente "223.dat" y buscará ese nombre exacto.
    original_base = hea_file.filename.split('.')[0]
    base_name = f"temp/{original_base}"
    
    # Guardar los 3 archivos necesarios (.dat, .atr, .hea)
    with open(base_name + ".dat", "wb") as f:
        shutil.copyfileobj(dat_file.file, f)
    with open(base_name + ".atr", "wb") as f:
        shutil.copyfileobj(atr_file.file, f)
    with open(base_name + ".hea", "wb") as f:
        shutil.copyfileobj(hea_file.file, f)
        
    try:
        # Usar el pipeline desarrollado previamente
        # pn_dir=None asegura que lea los archivos locales que acabamos de guardar
        paciente = load_record(base_name, pn_dir=None)
        _, signal_limpia = preprocess_record(paciente)
        latidos = extract_beats(paciente, signal_limpia)
        
        if len(latidos) == 0:
            return JSONResponse({"error": "No se encontraron latidos en los archivos."}, status_code=400)
            
        X_mel, _, X_time, _ = beats_to_arrays(latidos)
        predicciones = model.predict(X_mel, verbose=0)
        indices_predichos = predicciones.argmax(axis=1)
        
        anormales_idx = np.where(indices_predichos != 0)[0]
        anormales = int(len(anormales_idx)) # Clase 0 es 'Normal (N)'
        total = len(latidos)
        
        clases_nombres = ["Normal (N)", "Supraventricular (S)", "Ventricular (V)", "Fusión (F)"]
        
        # Generar hasta 24 imagenes de los latidos anormales (Señal + Espectrograma)
        plots_b64 = []
        for idx in anormales_idx[:24]:
            clase_pred = clases_nombres[indices_predichos[idx]]
            senial = X_time[idx]
            mel = X_mel[idx, :, :, 0] # El espectrograma 2D
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), facecolor='#151A22')
            
            # Grafico 1: Señal en el tiempo
            ax1.set_facecolor('#0B0E14')
            ax1.plot(senial, color='#EF4444', linewidth=2)
            ax1.set_title(f"ECG (Tipo: {clase_pred})", color='#F8FAFC', fontsize=10, weight='bold')
            ax1.axis('off')
            
            # Grafico 2: Espectrograma de Mel (Lo que "ve" la IA)
            ax2.set_facecolor('#0B0E14')
            ax2.imshow(mel, aspect='auto', origin='lower', cmap='magma')
            ax2.set_title("Espectrograma de Frecuencias", color='#F8FAFC', fontsize=10, weight='bold')
            ax2.axis('off')
            
            plt.tight_layout()
            
            # Guardar en buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=False)
            plt.close(fig)
            buf.seek(0)
            b64_str = base64.b64encode(buf.read()).decode('utf-8')
            plots_b64.append(f"data:image/png;base64,{b64_str}")
            
        # Logica basica de recomendaciones medicas
        if anormales == 0:
            diagnosis = "Ritmo cardíaco regular. No se detectaron arritmias significativas."
        elif anormales < (total * 0.05): # Menos del 5%
            diagnosis = "Presencia de latidos ectópicos aislados. Sugerir monitoreo preventivo."
        else:
            diagnosis = "Arritmia detectada. Requiere revisión urgente por cardiología."
            
        # Almacenar en la Base de Datos local SQLite
        conn = sqlite3.connect("database.sqlite")
        c = conn.cursor()
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO analyses (patient_name, date, total_beats, abnormal_beats, diagnosis) VALUES (?, ?, ?, ?, ?)",
                  (patient_name, date_str, total, anormales, diagnosis))
        conn.commit()
        conn.close()
        
        # Limpiar archivos temporales
        os.remove(base_name + ".dat")
        os.remove(base_name + ".atr")
        os.remove(base_name + ".hea")
        
        return {
            "patient_name": patient_name,
            "total_beats": total,
            "abnormal_beats": anormales,
            "diagnosis": diagnosis,
            "date": date_str,
            "plots": plots_b64
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history():
    conn = sqlite3.connect("database.sqlite")
    c = conn.cursor()
    c.execute("SELECT * FROM analyses ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    
    history = []
    for r in rows:
        history.append({
            "id": r[0], 
            "patient_name": r[1], 
            "date": r[2], 
            "total_beats": r[3], 
            "abnormal_beats": r[4], 
            "diagnosis": r[5]
        })
    return history
