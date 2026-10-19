# Neuro-Candombe-engine

Genera candombe simbolico (`chico`, `repique`, `piano`) con un Transformer enmascarado que aprende golpes, intensidad, **microtiming** y **articulacion**, condicionado por seccion, tempo, densidad y estilo. Incluye un demo web, analisis de grabaciones reales, evaluacion objetiva y pruebas de escucha a ciegas.

## Como funciona (v2)

- **Representacion** (`src/representation.py`): grilla de semicorcheas, 3 tambores. Cada celda tiene clase de intensidad (silencio + 4), desfase de microtiming (fraccion de semicorchea, 13 bins) y articulacion (`chico`: mano/palo, `repique`: palo/mano/madera, `piano`: bass_open/bass_muted/stick_open/stick_closed).
- **Modelo** (`src/model.py`): Transformer bidireccional enmascarado. Sabe en que semicorchea del compas, compas y seccion esta cada paso. Solo se puntuan las celdas ocultas, asi que el objetivo nunca esta en la entrada (el modelo v1 lo veia y por eso no aprendia).
- **Generacion** (`src/generate.py`): decodificacion iterativa por confianza. Las reglas del toque no pisan la salida: entran como celdas fijas (chico) y como mascaras de clases permitidas (`src/candombe_rules.py`). Con el mismo modelo se puede generar desde cero, completar un tambor sobre otros fijos o humanizar el timing.
- **Reglas** (`src/candombe_rules.py`): patron canonico del chico, motor por reglas con secciones (fallback de la API y corpus sintetico) y restricciones.

## Quick start (local)
```bash
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -r requirements-dev.txt      # o: pip install -r requirements-dev.txt
pytest                                      # tests
uvicorn src.app:app --reload --port 8000    # abrir http://localhost:8000
```
Sin checkpoint en `MODEL_PATH` (por defecto `models/candombe_v2.pt`), la API usa el motor por reglas.

## Datos

Todos los `.npz` comparten formato (`data` con velocities 0-127 y, opcionales, `offsets_frac`, `articulation`, `sections`, `tempo`, `style`). Los `.npz` viejos solo con `data` siguen funcionando.

| Fuente | Comando | Que aporta |
|---|---|---|
| Sintetico (reglas) | `python src/build_synthetic_corpus.py --count 1200` | Secciones, articulacion y microtiming etiquetados. Techo: lo que saben las reglas. |
| Dataset Candombe (UdelaR, Zenodo 6533068) | `python src/import_candombe_dataset.py --download` | 35 interpretaciones reales (~2 h) con pulso y compas anotados por un experto. **Licencia CC BY-NC-SA 4.0** (no comercial). |
| Grabaciones propias, multipista | `python src/analyze_reference.py --stems chico=c.wav,repique=r.wav,piano=p.wav --downbeat 1.23 --out-npz data/processed/propias/toque1.npz` | Lo mejor: cada tambor aislado, sin heuristicas de separacion. |
| Grabaciones propias, mezcla | `python src/build_reference_corpus.py --input-glob "data/analysis/muestras/*.wav" [--beats-dir anotaciones/]` | Training pieces + banco de muestras por articulacion. |
| MIDI | `python src/data_preprocess.py toque.mid out.npz --note-map 36=piano,38=repique,42=chico` | Conserva el microtiming de los ticks. |

Notas del analisis de audio (`src/analyze_reference.py`):
- La grilla sigue los pulsos reales (anotados o detectados), asi que acompaña las aceleraciones de la llamada.
- El inicio de compas sale de las anotaciones, de `--downbeat`, o se estima (el chico nunca toca la primera semicorchea del tiempo + plantilla de piano/repique). Revisar `phase_confidence` en la salida.
- Cada onset se refina a nivel de muestra para no sumar latencia falsa al microtiming.
- Con una mezcla, los tambores se separan por bandas y se filtra la filtracion entre bandas (`BLEED_RATIO`). Es una heuristica: los golpes fantasma muy suaves pueden perderse. Con stems no se aplica.

Perfil de microtiming medido (reemplaza el provisorio de `src/microtiming.py`):
```bash
python src/microtiming.py --data-glob "data/processed/candombe_dataset/*.npz" --out data/analysis/microtiming_profile.json
python src/build_synthetic_corpus.py --microtiming-profile data/analysis/microtiming_profile.json
```

## Entrenamiento
```bash
python src/train.py --data-glob "data/processed/synthetic/*.npz@1,data/processed/candombe_dataset/*.npz@4" --epochs 30
cp models/candombe_v2_best.pt models/candombe_v2.pt
```
- `patron@peso` sobremuestrea una fuente (dar mas peso a lo real que a lo sintetico).
- Separacion train/val por archivo, AdamW con warmup + coseno, early stopping; guarda `_best.pt` (con su config) y `_history.json`.
- Opciones de tamaño: `--d-model`, `--layers`, `--heads`, `--max-bars`.

## Evaluacion
```bash
python src/evaluate.py --checkpoint models/candombe_v2.pt --reference-glob "data/processed/candombe_dataset/*.npz" \
    --corpus-glob "data/processed/synthetic/*.npz" --n 64 --bars 8 --out reports/eval.json
```
Compara modelo, reglas y referencia: conformidad del chico, apoyo del piano, llamadas, divergencias de posicion/intensidad/articulacion, distancia de microtiming, diversidad y **copy_rate** (compases copiados textuales del corpus: cerca de 1 = memoriza en vez de componer).

## Pruebas de escucha a ciegas
```bash
python src/listening_test.py build --checkpoint models/candombe_v2.pt --reference-glob "data/processed/candombe_dataset/*.npz"
# compartir data/listening_test/clips/ y ratings.csv (groove, autenticidad, interes de 1 a 5); guardar key.json
python src/listening_test.py score --ratings oyente1.csv oyente2.csv
```
Para escuchar una generacion sin el navegador: `python src/render_audio.py --checkpoint models/candombe_v2.pt --bars 8 --out toque.wav`.

## API
```bash
curl "http://localhost:8000/generate?tempo=104&bars=8&density=0.6&structure=intro,base,llamada_repique,base,llamada_piano,base,base,cierre"
```
| Parametro | Por defecto | Que hace |
|---|---|---|
| `tempo`, `bars` | 90, 4 | BPM (50-220), compases (4-32) |
| `seed` | `clave` | `clave`: el repique abre con la clave en la madera; `energia`: +densidad; `base`: -densidad |
| `temperature`, `top_p` | 0.9, 0.92 | Variacion del muestreo |
| `density` | 0.55 | Carga ritmica (condiciona al modelo, no es una moneda al azar) |
| `structure` | forma de llamada | Secciones por compas, separadas por coma: `base`, `intro`, `llamada_repique`, `llamada_piano`, `cierre` |
| `style` | - | `montevideo`, `cuareim`, `ansina`, `cordon` (sirve cuando haya datos etiquetados) |
| `lock_chico`, `strict_rules` | true | Chico fijo, restricciones del toque |
| `palmas` | false | Palmas de clave encima (no son parte de una llamada) |
| `repiques` | 1 | Lineas de repique distintas (1-4), una por repiquero; los eventos traen `voice` |
| `timing_temperature` | 0.5 | Que tan suelto se muestrea el microtiming |
| `guidance` | 1.0 | >1 refuerza el condicionamiento (classifier-free guidance) |
| `random_seed` | - | Reproducibilidad |

La respuesta (`version: 2`) trae por evento `offset_ms` (microtiming) y `articulation`, y la `random_seed` usada (pasarla de nuevo repite el resultado). `/health` informa si hay modelo cargado.

`POST /regenerate` rehace una parte de un resultado anterior y deja fijo el resto: mismo cuerpo que los parametros de `/generate` (en JSON) mas `events` (los del resultado anterior), `keep_instruments` (tambores a mantener) y/o `regenerate_bars` (indices de compases a rehacer, desde 0).

## Frontend (desfile de llamadas)
- Un boton **Tocar/Parar** y cuatro controles: tempo, duracion (8/16/32 compases), tamaño de la cuerda (chica 3/2/1, mediana 6/3/2, grande 10/4/3 chicos/repiques/pianos) e intensidad.
- **Forma de llamada automatica**: los repiques marcan la clave en la madera solos, la llamada de repique arranca el toque, la base sigue con una llamada cada 8 compases (y una de piano en piezas largas) y cierra.
- **Cuerda real**: cada tamborilero tiene su lugar en el estereo, su afinacion y su forma de caer en el tiempo; cada repiquero improvisa su propia linea (el modelo completa repiques distintos sobre el mismo chico y piano).
- **La cuerda pasa frente a vos** (se acerca, pasa y se aleja) y **subir la llamada** (acelera 5% de a poco). Reverberacion de calle, compresion suave y limitador.
- **Otra llamada**, **Nuevos repiques** (mantiene chico y piano), **Descargar WAV** (mismo sonido que se escucha) y **MIDI** (con mapa de tempo, reimportable con `data_preprocess.py`).
- "Mas opciones": improvisacion, swing (microtiming), calle, mezcla por tambor, chico fijo/libre y semilla.
- Los golpes se programan justo antes de sonar (lookahead), no todos al principio.

### Sonidos
`web/samples/llamadas/` tiene golpes sintetizados por modelo fisico (modos de un parche circular, palo, mano y madera), 4 tomas por articulacion. Se regeneran con `python src/synthesize_drum_samples.py`, que tambien escribe `web/samples/manifest.json`. Con grabaciones reales de buena calidad, `build_reference_corpus.py` arma un manifest con ellas y el player las usa sin cambios.

## Deploy: Netlify + Render
- `web/` en Netlify (`netlify.toml`, variable `API_BASE_URL` con la URL del backend).
- `src/app.py` en Render (`render.yaml`, `MODEL_PATH=models/candombe_v2.pt`).

## Limitaciones conocidas
- El checkpoint incluido esta entrenado solo con el corpus sintetico: suena a las reglas con variaciones. El salto de calidad viene de entrenar con el dataset real y grabaciones propias multipista.
- El perfil de microtiming por defecto es provisorio (derivado de los presets del frontend).
- Las articulaciones de repique y piano en audio mezclado se agrupan por clustering heuristico: escuchar el banco y corregir el manifest.
