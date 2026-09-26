# Sport from motion — ¿qué deporte es?

¿Puede un modelo reconocer un deporte de equipo **solo por cómo se mueven los
jugadores**, sin campo, líneas, superficie, balón ni equipamiento? Es el experimento
hermano de [*Where's the ball?*](https://github.com/JaviMaligno/wheres-the-ball).

- Diseño completo, controles de fugas y condiciones: [`docs/design.md`](docs/design.md)
- Qué datasets usar y qué hay que construir (rugby): [`docs/datasets.md`](docs/datasets.md)

> ## ⚠️ Vive aquí de forma temporal: hay que extraerlo
>
> Este directorio está dentro de `personal-website` **solo mientras no exista su
> propio repo**. Está aislado a propósito, así que moverlo es copiar la carpeta:
>
> - no importa nada de fuera de `experiments/sport-from-motion/`;
> - no depende de `wheres-the-ball`. Los loaders de Metrica y SportVU replican los
>   formatos de `wheres_the_ball/data/field_tracking.py`, pero no los importan;
> - no lo toca el build de Astro, y datos y resultados (`data/`, `runs/`) están en
>   `.gitignore`.
>
> **Destino pendiente de decidir:** subpaquete de `wheres-the-ball` (si acaba
> compartiendo datos y loaders) o repo propio. Al extraerlo hay que mover también las
> referencias a este directorio en los borradores del blog, si los hay.

## Estado

Harness completo y probado; **no hay resultados**. `loaders/synthetic.py` genera
datos de juguete solo para probar el pipeline, y nada de lo que sale de ahí es un
resultado.

## Instalación

```bash
cd experiments/sport-from-motion
uv venv && uv pip install -e ".[dev]"      # núcleo: numpy, pillow, scikit-learn
uv pip install -e ".[probe]"               # opcional: torch + transformers (V-JEPA 2)
uv run pytest
```

## Flujo

```bash
# 1. Fuente -> clips (metros, sin balón, diezmados a 5 Hz)
motion-sport ingest --source metrica --input data/raw/Sample_Game_1 --out data/clips
motion-sport ingest --source nfl --input data/raw/tracking_week_1.csv --out data/clips --max-plays 300
# CSV largo genérico (TeamTrack, extracción propia de rugby, kloppy...). Los nombres de
# columna son un ejemplo: se mapean a los del export real.
motion-sport ingest --source long-csv --input data/raw/teamtrack_handball.csv --out data/clips \
    --sport handball --fps 25 --name teamtrack \
    --columns '{"match":"video","frame":"frame","track":"id","x":"x","y":"y"}'

# 2. Controles + renders + prompts
motion-sport prepare --clips data/clips --out runs/pilot-strict --preset strict --per-sport 200
motion-sport prepare --clips data/clips --out runs/pilot-raw --preset raw --conditions motion --reprs sheet

# 3. Comprobar que los controles funcionan (nuisance debe salir al azar en strict)
motion-sport baseline --items runs/pilot-strict --features nuisance
motion-sport baseline --items runs/pilot-strict --features kinematic

# 4. Modelos (reanudable: se puede cortar y relanzar)
motion-sport run --items runs/pilot-strict --model azure-anthropic:claude-opus-5-5 \
    --condition motion --repr sheet --limit 50
motion-sport run --items runs/pilot-strict --model azure-openai:gpt-5.4 --condition motion_shuffled --repr sheet
motion-sport probe --items runs/pilot-strict --encoder facebook/vjepa2-vitl-fpc64-256

# 5. Informe con IC agrupados por partido y contrastes emparejados
motion-sport report --items runs/pilot-strict
motion-sport report --items runs/pilot-strict --tag pre_snap     # subcaso formación
```

Prueba en seco, sin datos ni claves:

```bash
motion-sport ingest --source toy --out runs/toy/clips
motion-sport prepare --clips runs/toy/clips --out runs/toy/items
motion-sport run --items runs/toy/items --model dummy:first --condition motion --repr sheet
motion-sport report --items runs/toy/items
```

## Modelos en Azure

Mismas variables que `experiments/judge-bias`; ver [`.env.example`](.env.example).

| Id | Ruta | Variables |
|---|---|---|
| `azure-anthropic:<deployment>` | Claude en Foundry, `/anthropic/v1/messages` | `AZURE_ANTHROPIC_ENDPOINT` + `AZURE_ANTHROPIC_KEY` (o `ANTHROPIC_FOUNDRY_RESOURCE` + `ANTHROPIC_FOUNDRY_API_KEY`) |
| `azure-openai:<deployment>` | GPT/Grok, `/openai/v1/chat/completions` | `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_KEY` |
| `azure-foundry:<deployment>` | Llama, Mistral, Qwen..., `/models/chat/completions` | `AZURE_FOUNDRY_ENDPOINT` + `AZURE_FOUNDRY_KEY` |
| `openai:` / `anthropic:` | APIs directas | `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` |
| `dummy:first` / `dummy:uniform` | sin red | — |

`<deployment>` es el nombre del despliegue en Foundry (por defecto, el id del modelo).

## Mapa del código

```
src/motion_sport/
  schema.py        Clip: [T, N, 2] en metros + metadatos; registro de deportes y tamaños de campo
  loaders/         metrica, sportvu, nfl, long_csv (genérico), synthetic (solo pruebas)
  controls.py      controles de fugas y presets (raw / strict / strict_tempo / field_scaled)
  conditions.py    formation / motion / motion_shuffled / kinematics / kinematics_solo
  render.py        point-light: puntos grises, lienzo cuadrado, hojas de contacto, estelas, GIF
  serialize.py     la misma vista como texto
  prompts.py       prompt de opciones cerradas (barajadas por ítem) y parseo de la respuesta
  backends/chat.py VLM por Azure y APIs directas (solo stdlib)
  backends/probe.py  encoder de vídeo congelado (V-JEPA 2 / VideoMAE) + probe logístico
  baselines.py     clasificadores de atajos (nuisance, tempo) y cinemático
  evaluate.py      métricas, bootstrap agrupado por partido, contrastes emparejados
  pipeline.py      prepare / run_model / run_baseline / run_probe / report
  cli.py           `motion-sport ...`
```
