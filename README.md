# Neuro-Candombe-engine - MVP scaffold

Este repositorio contiene un MVP para generar patrones simbolicos de Candombe (`chico`, `repique`, `piano`) con un modelo tipo Transformer y un demo web sencillo.

## Quick start (local)
1. Crear un entorno virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Instalar dependencias:
   ```bash
   pip install -r requirements-api.txt
   ```
3. Entrenar (si ya tenes `.npz` en `data/processed/`):
   ```bash
   python src/train.py --epochs 20 --lr 3e-4
   ```
4. Levantar servidor:
   ```bash
   uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
   ```
5. Abrir `http://localhost:8000`.

## Como sacar mejores canciones/patrones
- Dataset primero: usar MIDI limpios y bien cuantizados por instrumento.
- Mas epocas + LR moderado: empezar en `--epochs 20 --lr 3e-4` y ajustar.
- Control en generacion:
  - `temperature` bajo (`0.6-0.9`) = mas estable.
  - `temperature` alto (`1.1-1.4`) = mas variacion.
  - `density` bajo (`0.30-0.45`) = aire/espacio.
  - `density` alto (`0.60-0.80`) = mas actividad.
- Semillas (`seed`): `clave` (base) o `energia` (mas movimiento).

Ejemplo API:
```bash
curl "http://localhost:8000/generate?tempo=96&bars=8&seed=clave&temperature=0.85&density=0.55"
```

## Que incluye
- `src/`: preprocesado, modelo, entrenamiento, generacion y FastAPI.
- `web/`: frontend para reproducir eventos.
- `models/`: checkpoints del modelo.
- `data/`: entrada y datos procesados.

## Analizar una referencia de audio
Si queres extraer una aproximacion de `chico`, `repique` y `piano` desde un audio real:

```bash
python src/analyze_reference.py --input C:\Users\julia\Desktop\candombe\references.wav --bars 4

# Con tempo conocido
# python src/analyze_reference.py --input path/to/references.wav --bars 4 --tempo 92
```

Salidas:
- `data/analysis/reference_events.json`: eventos con `instrument`, `step`, `offset_ms`, `velocity`, `hit_type`.
- `data/processed/reference_from_audio.npz`: version cuantizada compatible con el entrenamiento actual.

Notas del analizador:
- Es una heuristica inicial por bandas y onsets, no separacion perfecta de fuentes.
- Funciona mejor con fragmentos cortos y limpios donde la cuerda se escuche clara.
- Si ya conoces el tempo, pasalo con `--tempo` para mejorar la cuantizacion.

## Generar con los sonidos obtenidos de la referencia
Despues de analizar la referencia, podes recortar una muestra por tambor y dejarla lista para la web:

```bash
python src/extract_reference_samples.py --analysis-json data/analysis/reference_events.json
```

Salidas:
- `web/samples/chico.wav`
- `web/samples/repique.wav`
- `web/samples/piano.wav`
- `data/analysis/reference_samples.json`: manifiesto con el golpe elegido por instrumento.

Con esas muestras presentes, el frontend las usa automaticamente durante `/generate` y deja de caer en el sintetizador de fallback para esos instrumentos.

## Notas
- Si no hay modelo entrenado, `/generate` usa un patron rule-based de fallback.
- Reemplaza los WAV de `web/samples/` por muestras cortas de buena calidad para una mejora audible inmediata.

## Deploy: Netlify + Render
Este repo queda preparado para despliegue hibrido:
- `web/` en Netlify
- `src/app.py` en Render

### Backend en Render
1. Conecta este repo en Render.
2. Usa el archivo `render.yaml`, que instala dependencias desde `requirements-api.txt`.
3. El servicio levanta FastAPI con:
   ```bash
   uvicorn src.app:app --host 0.0.0.0 --port $PORT
   ```

### Frontend en Netlify
1. Conecta este repo en Netlify.
2. Netlify usa `netlify.toml`:
   - `publish = "web"`
   - `command = "node scripts/write-runtime-config.mjs"`
   - frontend-only, sin instalar dependencias Python
3. Define la variable de entorno `API_BASE_URL` con la URL publica del backend, por ejemplo:
   ```bash
   API_BASE_URL=https://neuro-candombe-api.onrender.com
   ```

Durante el build, Netlify genera `web/runtime-config.js` con esa URL y el frontend pasa a llamar a la API remota.

### Desarrollo local
Si `API_BASE_URL` esta vacia, el frontend sigue usando same-origin, asi que localmente podes seguir corriendo:
```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```
