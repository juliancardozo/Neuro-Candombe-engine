# Neuro-Candombe-engine — MVP scaffold

Este repositorio contiene un MVP para generar patrones simbólicos de Candombe (chico, repique, piano) con un modelo tipo Transformer y un demo web sencillo.

## Quick start (local)
1. Crear un entorno virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Entrenar (si ya tenés `.npz` en `data/processed/`):
   ```bash
   python src/train.py --epochs 20 --lr 3e-4
   ```
4. Levantar servidor:
   ```bash
   uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
   ```
5. Abrir `http://localhost:8000`.

## Cómo sacar mejores canciones/patrones
- **Dataset primero**: usar MIDI limpios y bien cuantizados por instrumento.
- **Más épocas + LR moderado**: empezar en `--epochs 20 --lr 3e-4` y ajustar.
- **Control en generación**:
  - `temperature` bajo (0.6-0.9) = más estable.
  - `temperature` alto (1.1-1.4) = más variación.
  - `density` bajo (0.30-0.45) = aire/espacio.
  - `density` alto (0.60-0.80) = más actividad.
- **Semillas** (`seed`): `clave` (base) o `energia` (más movimiento).

Ejemplo API:
```bash
curl "http://localhost:8000/generate?tempo=96&bars=8&seed=clave&temperature=0.85&density=0.55"
```

## Qué incluye
- `src/`: preprocesado, modelo, entrenamiento, generación y FastAPI.
- `web/`: frontend mínimo para reproducir eventos.
- `models/`: checkpoints del modelo.
- `data/`: entrada y datos procesados.

## Notas
- Si no hay modelo entrenado, `/generate` usa un patrón rule-based de fallback.
- Reemplazá los WAV de `web/samples/` por muestras cortas de buena calidad para una mejora audible inmediata.
