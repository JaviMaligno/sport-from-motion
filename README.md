# Sport from motion — ¿qué deporte es?

¿Puede un modelo reconocer un deporte de equipo **solo por cómo se mueven los
jugadores**, sin campo, líneas, superficie, balón ni equipamiento? Es el experimento
hermano de [*Where's the ball?*](https://github.com/JaviMaligno/wheres-the-ball).

- Diseño completo, controles de fugas y condiciones: [`docs/design.md`](docs/design.md)
- Qué datasets usar y qué hay que construir (rugby): [`docs/datasets.md`](docs/datasets.md)
- Pre-registro de la corrida final de modelos: [`docs/preregistration.md`](docs/preregistration.md)
- Resultados de la corrida final: [`docs/results-final.md`](docs/results-final.md)
- Huecos abiertos y cerrados: [`docs/gaps.md`](docs/gaps.md)

> **Origen.** El experimento nació dentro del repo del blog (`personal-website`,
> `experiments/sport-from-motion/`) y se extrajo aquí con su historia completa el
> 2026-09-29. No depende de `wheres-the-ball`: los loaders de Metrica y SportVU replican
> sus formatos, pero no los importan.
>
> **Datos.** No se redistribuye ningún dato (`data/` y `runs/` están en `.gitignore`).
> Cada fuente tiene su licencia (NFL Big Data Bowl, EIGD-H CC BY-NC-SA, TeamTrack MIT,
> SportVU, Metrica, SkillCorner); cómo obtenerlas está en [`docs/datasets.md`](docs/datasets.md)
> y en los scripts de `scripts/`.

## Estado

**Corrida final pre-registrada hecha y analizada** (2026-09-28, `runs/final` y
`runs/final-d8`; resultados en [`docs/results-final.md`](docs/results-final.md)).
400 clips de 151 partidos y 4 deportes, 5 modelos de chat más Jev, familia de Holm de
20 contrastes:

- Con la regla del §6, **Claude Opus 5.5 es el único que «lee el orden temporal»**:
  barajar los instantes le quita 10 puntos [5; 16] (p_holm 0,004). El efecto se
  concentra en fútbol americano y baloncesto, y reaparece a 8 s sin fútbol americano
  (exploratorio).
- gpt-5.6-sol, gpt-5.6-terra, Claude Sonnet 5 y Gemini 3.1 Pro quedan «sin evidencia de
  que vea movimiento». Es ausencia de evidencia: el IC admite hasta 3-6 puntos de efecto
  del orden.
- Todos muy por debajo de los especialistas sobre los mismos clips (0,47 como máximo,
  frente a 0,83 de MiniRocket).
- Una desviación posterior a los datos, **D21** (margen de razonamiento de la ruta
  Anthropic): la sensibilidad no cambia ninguna decisión de Holm ni ninguna
  clasificación, pero la re-corrida entera de la celda afectada queda abierta (A17).

Antes: piloto 1 (fútbol vs baloncesto) en
[`docs/pilot-2026-09-27.md`](docs/pilot-2026-09-27.md). `loaders/synthetic.py` sigue
siendo solo para pruebas.

## Instalación

```bash
cd experiments/sport-from-motion
uv venv && uv pip install -e ".[dev]"      # núcleo: numpy, pillow, scikit-learn
uv pip install -e ".[series]"              # opcional: aeon (MiniRocket)
uv pip install -e ".[probe]"               # opcional: torch + transformers (V-JEPA 2, DeepSets)
uv pip install -e ".[video]"               # opcional: imageio-ffmpeg (--reprs video)
uv run pytest
```

En Mac Intel (x86_64) numba ya no publica wheels: `uv venv -p 3.12`, luego
`uv pip install "numba<0.62" "llvmlite<0.45" "numpy==2.2.6"`, y usar `.venv/bin/...`
en vez de `uv run` (que re-sincroniza y vuelve a subir numpy, rompiendo MiniRocket).

## Flujo

```bash
# 1. Fuente -> clips (metros, sin balón, diezmados a 5 Hz)
motion-sport ingest --source metrica --input data/raw/Sample_Game_1 --out data/clips
motion-sport ingest --source nfl --input data/raw/tracking_week_1.csv --out data/clips --max-plays 300
# NFL: un clip por jugada en una fase aleatoria tras el snap (dataset final); imprime
# cuántas jugadas se descartan por no tener snap o ser demasiado cortas
motion-sport ingest --source nfl --input data/raw/nfl2023/week1.csv --out data/clips_nflrand \
    --nfl-phase random --nfl-min-after-snap 0.5
# CSV largo genérico (TeamTrack, extracción propia de rugby, kloppy...). Los nombres de
# columna son un ejemplo: se mapean a los del export real.
motion-sport ingest --source long-csv --input data/raw/teamtrack_handball.csv --out data/clips \
    --sport handball --fps 25 --name teamtrack \
    --columns '{"match":"video","frame":"frame","track":"id","x":"x","y":"y"}'

# 2. Controles + renders + prompts
motion-sport prepare --clips data/clips --out runs/pilot-strict --preset strict --per-sport 200
motion-sport prepare --clips data/clips --out runs/pilot-raw --preset raw --conditions motion --reprs sheet
# vídeo mp4 para Gemini (necesita .[video]) y N jugadores al azar en vez de los centrales
motion-sport prepare --clips data/clips --out runs/pilot-video --reprs sheet,video --player-mode random

# 3. Comprobar el CV y la señal (nuisance debe salir al azar en strict: con los controles
#    sus variables son constantes, así que comprueba los folds, no los atajos)
motion-sport baseline --items runs/pilot-strict --features nuisance
motion-sport baseline --items runs/pilot-strict --features kinematic

# 4. Clasificadores ajustados sobre nuestros clips (CV agrupada por partido),
#    en las mismas condiciones que los VLM para poder contrastarlas
for c in motion motion_shuffled kinematics kinematics_solo; do
  motion-sport fit --items runs/pilot-strict --learner minirocket --condition $c
done
motion-sport fit --items runs/pilot-strict --learner deepsets --condition formation

# 5. Modelos sin entrenar (reanudable: se puede cortar y relanzar)
motion-sport run --items runs/pilot-strict --model azure-anthropic:claude-opus-5-5 \
    --condition motion --repr sheet --limit 50
motion-sport run --items runs/pilot-strict --model azure-openai:gpt-5.6-sol --condition motion_shuffled --repr sheet
motion-sport probe --items runs/pilot-strict --encoder facebook/vjepa2-vitl-fpc64-256
# réplicas (fichero propio __rK), prompt informado (__informed) y vídeo (solo vertex:/Gemini)
motion-sport run --items runs/pilot-strict --model vertex:gemini-2.5-pro --condition motion \
    --repr sheet --limit 100 --replicate 2
motion-sport run --items runs/pilot-strict --model azure-openai:gpt-5.6-sol --condition motion \
    --repr sheet --prompt-style informed
motion-sport run --items runs/pilot-strict --model vertex:gemini-2.5-pro --condition motion --repr video

# 6. Informe con IC agrupados por partido: exactitud, balanceada, macro-F1, kappa,
#    exactitud corregida por prior, recall por clase, réplicas, contrastes primarios
#    (Holm) y secundarios (exploratorios). Ignora predicciones de clips que ya no están.
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

Ver [`.env.example`](.env.example). Los lanzadores (`scripts/final_run.sh`,
`scripts/pilot_models.sh`) leen además:

| Variable | Para qué |
|---|---|
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY_FILE` (por defecto `~/.azure-openai-key`) | GPT en Azure OpenAI; la clave se lee del fichero y nunca se imprime |
| `VERTEX_PROJECT`, `GCLOUD_BIN` | Gemini y Claude en Vertex; sin `VERTEX_PROJECT` esas rutas fallan con un error claro |
| `~/.openrouter-key` | Jev por OpenRouter |

| Id | Ruta | Variables |
|---|---|---|
| `azure-anthropic:<deployment>` | Claude en Foundry, `/anthropic/v1/messages` | `AZURE_ANTHROPIC_ENDPOINT` + `AZURE_ANTHROPIC_KEY` (o `ANTHROPIC_FOUNDRY_RESOURCE` + `ANTHROPIC_FOUNDRY_API_KEY`) |
| `azure-openai:<deployment>` | GPT/Grok, `/openai/v1/chat/completions` | `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_KEY` |
| `azure-foundry:<deployment>` | Llama, Mistral, Qwen..., `/models/chat/completions` | `AZURE_FOUNDRY_ENDPOINT` + `AZURE_FOUNDRY_KEY` |
| `openai:` / `anthropic:` | APIs directas | `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` |
| `vertex:<model>` | Gemini en Vertex AI (`generateContent`), token de `gcloud` | `VERTEX_PROJECT` (+ `VERTEX_LOCATION`, por defecto `global`; `GCLOUD_BIN`) |
| `vertex-anthropic:<model>` | Claude en Vertex AI (`rawPredict`) | las mismas |
| `jev-openrouter:~typesafe/jev-latest` | Jev servido por OpenRouter (`/api/alpha/decisions`). Solo `--repr text` | `OPENROUTER_API_KEY` |
| `jev:<model>` | Jev (TypeSafe), `POST /v1/systemone`. Solo `--repr text` | `TYPESAFE_API_KEY` (+ `TYPESAFE_BASE_URL`) |
| `laya-http:<checkpoint>` | Laya tras `laya-serve` (mismo protocolo; local o contenedor en Azure) | `LAYA_BASE_URL` (+ `LAYA_API_KEY`) |
| `laya:<checkpoint>` | Laya en proceso (`pip install -e ".[laya]"`) | `LAYA_MAX_LEN` (8192 por defecto) |
| `dummy:first` / `dummy:uniform` | sin red | — |

`<deployment>` es el nombre del despliegue en Foundry (por defecto, el id del modelo).
Jev y Laya son modelos de decisión tipada: reciben las coordenadas como texto
(`--state-format text`) o JSON (`--state-format json`) y devuelven una probabilidad por
deporte. Checkpoints de Laya: `english` (512 tokens, se queda corto),
`multilingual` (hasta 8.192) y `typed-decisions`.

```bash
motion-sport run --items runs/pilot-strict --model jev:jev-latest --condition motion --repr text
motion-sport run --items runs/pilot-strict --model laya:multilingual --condition motion --repr text --state-format json
```

## Mapa del código

```
src/motion_sport/
  schema.py        Clip: [T, N, 2] en metros + metadatos; registro de deportes y tamaños de campo
  loaders/         metrica, sportvu, nfl, long_csv (genérico), synthetic (solo pruebas)
  controls.py      controles de fugas y presets (raw / strict / strict_tempo / field_scaled)
  conditions.py    formation / motion / motion_shuffled / kinematics / kinematics_solo
  render.py        point-light: puntos grises, lienzo cuadrado, hojas de contacto, estelas, GIF, mp4
  serialize.py     la misma vista como texto
  prompts.py       prompt de opciones cerradas (barajadas por ítem), neutro o informado, y parseo
  backends/chat.py VLM por Azure y APIs directas (solo stdlib)
  backends/decision.py  Jev / Laya (protocolo /v1/systemone o Laya en proceso)
  backends/probe.py  encoder de vídeo congelado (V-JEPA 2 / VideoMAE) + probe logístico
  baselines.py     clasificadores de atajos (nuisance, tempo) y cinemático
  learners.py      CV agrupada por partido; MiniRocket (series invariantes) y DeepSets
  evaluate.py      métricas, bootstrap agrupado por partido, contrastes emparejados, Holm, réplicas
  pipeline.py      prepare / run_model / run_baseline / run_probe / report
  cli.py           `motion-sport ...`
```
