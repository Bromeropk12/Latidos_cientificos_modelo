@echo off
title SCA - Sistema de Clasificacion de Arritmias
echo =======================================================
echo     Iniciando Motor IA (Python FastAPI) y Base de Datos
echo =======================================================
start cmd /k "cd backend && title Backend FastAPI && uvicorn api:app --reload --port 8000"

echo Esperando 3 segundos a que el motor arranque...
timeout /t 3 /nobreak > nul

echo =======================================================
echo     Iniciando Aplicacion de Escritorio (Electron)
echo =======================================================
start cmd /c "cd frontend && title Interfaz Electron && npm start"
