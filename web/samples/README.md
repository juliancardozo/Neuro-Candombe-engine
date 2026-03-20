Coloca en esta carpeta las muestras reales que quieras usar en la app.

Los `404 Not Found` que veias antes pasaban porque el frontend intentaba adivinar archivos como:

- `chico.wav`
- `repique.wav`
- `piano.wav`
- `clave.wav`

Ahora la carga se controla con `manifest.json`, asi que la app solo intenta abrir las muestras que declares ahi.
Tambien soporta `multisamples` por articulacion con `round-robin`, para que no repita siempre la misma toma.

Formato recomendado de `manifest.json`:

```json
{
  "chico": {
    "default": [
      "/samples/chico_palo_01.wav",
      "/samples/chico_palo_02.wav"
    ],
    "mano": [
      "/samples/chico_mano_01.wav",
      "/samples/chico_mano_02.wav",
      "/samples/chico_mano_03.wav"
    ],
    "palo": [
      "/samples/chico_palo_01.wav",
      "/samples/chico_palo_02.wav",
      "/samples/chico_palo_03.wav"
    ]
  },
  "repique": {
    "default": [
      "/samples/repique_base_01.wav",
      "/samples/repique_base_02.wav"
    ],
    "talk_open": [
      "/samples/repique_talk_open_01.wav",
      "/samples/repique_talk_open_02.wav"
    ],
    "talk_support": [
      "/samples/repique_madera_01.wav",
      "/samples/repique_madera_02.wav",
      "/samples/repique_madera_03.wav"
    ],
    "talk_answer": [
      "/samples/repique_parche_01.wav",
      "/samples/repique_parche_02.wav"
    ],
    "talk_close": [
      "/samples/repique_close_01.wav",
      "/samples/repique_close_02.wav"
    ],
    "call_open": [
      "/samples/repique_call_open_01.wav",
      "/samples/repique_call_open_02.wav"
    ],
    "call_support": [
      "/samples/repique_call_support_01.wav",
      "/samples/repique_call_support_02.wav"
    ],
    "call_close": [
      "/samples/repique_call_close_01.wav",
      "/samples/repique_call_close_02.wav"
    ],
    "call_tail": [
      "/samples/repique_call_tail_01.wav"
    ]
  },
  "piano": {
    "default": [
      "/samples/piano_open_01.wav",
      "/samples/piano_open_02.wav"
    ],
    "bass_open": [
      "/samples/piano_open_01.wav",
      "/samples/piano_open_02.wav"
    ],
    "bass_muted": [
      "/samples/piano_muted_01.wav",
      "/samples/piano_muted_02.wav"
    ],
    "stick_open": [
      "/samples/piano_stick_open_01.wav",
      "/samples/piano_stick_open_02.wav"
    ],
    "stick_closed": [
      "/samples/piano_stick_closed_01.wav",
      "/samples/piano_stick_closed_02.wav"
    ]
  },
  "clave": {
    "default": [
      "/samples/clave_palmas_01.wav",
      "/samples/clave_palmas_02.wav"
    ]
  }
}
```

Tambien podes declarar `.ogg` o `.mp3` y mezclar formatos dentro del mismo banco si hace falta:

```json
{
  "chico": {
    "mano": ["/samples/chico_mano_01.ogg", "/samples/chico_mano_02.mp3"]
  },
  "repique": {
    "talk_support": ["/samples/repique_madera_01.ogg"]
  },
  "piano": {
    "bass_open": []
  },
  "clave": {
    "default": []
  }
}
```

Reglas practicas:

- Usa rutas absolutas dentro de `web/`, por ejemplo `"/samples/chico.wav"`.
- Si un instrumento o articulacion no tiene muestra, dejalo como `[]`.
- Cada clave del manifest puede ser un solo archivo o un array de archivos.
- Si declaras varios archivos en una misma articulacion, la app hace `round-robin` automaticamente.
- Para el `chico`, conviene separar `mano` y `palo`.
- Para el `repique`, conviene separar `madera`, `parche`, `abierto` y cierres.
- Para el `piano`, conviene separar `bass_open`, `bass_muted`, `stick_open` y `stick_closed`.
- Si falta una articulacion puntual, la app intenta usar `default` de ese instrumento antes de volver a la sintesis.
- La `clave` puede ser una muestra seca o palmas, segun lo que quieras superponer sobre la rueda.
- Mono alcanza.
- 44.1 kHz o 48 kHz funciona bien.
- Conviene que cada golpe sea corto, limpio y sin silencio al inicio.
- Si queres sonar mas a cuerda real, junta 3 a 6 tomas por articulacion antes que una sola.

Si no declaras muestras, la app usa sintesis interna y no deberia llenar el log con `404`.
