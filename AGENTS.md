# AGENTS.md — sport-from-motion

Instrucciones para agentes de código (Codex, Claude Code, etc.). Este fichero es la
fuente canónica; `CLAUDE.md` solo lo importa.

## Qué es

Experimento de investigación: ¿puede un modelo reconocer un deporte de equipo **solo por
cómo se mueven los jugadores** (sin campo, líneas, balón ni equipamiento)? Paquete
Python `motion-sport` (`src/motion_sport/`, CLI `motion-sport`). Hermano de
*Where's the ball?*, pero sin dependencia de código con él. La documentación del
experimento está en español (`README.md`, `docs/`); el código, los comentarios y los
mensajes de commit, en inglés.

## Estructura

- `src/motion_sport/` — `schema.py` (Clip `[T, N, 2]` en metros), `loaders/`
  (`sources.py`: Metrica, SportVU, NFL; `long_csv.py` genérico; `synthetic.py` solo
  para pruebas), `controls.py` (presets raw / strict / strict_tempo / field_scaled),
  `conditions.py`, `render.py`, `serialize.py`, `prompts.py`, `backends/`
  (`chat.py` VLM solo stdlib, `decision.py` Jev/Laya, `probe.py` encoders de vídeo),
  `baselines.py`, `learners.py` (MiniRocket, DeepSets), `evaluate.py`, `pipeline.py`,
  `cli.py`.
- `scripts/` — lanzadores (`final_run.sh`, `pilot_models.sh`), conversores de fuentes a
  CSV largo (`*_to_long_csv.py`), `analysis/` (tablas y análisis de resultados) y
  `laya/` (export, entrenamiento y runner de Kaggle del brazo Laya afinado).
- `tests/` — pytest (`testpaths = ["tests"]`, `pythonpath = ["src"]`).
- `docs/` — `design.md`, `datasets.md`, `preregistration.md`, `preregistration-laya.md`,
  `results-final.md`, `gaps.md`, piloto.
- `data/` y `runs/` — datos descargados, clips, ítems y predicciones. **Nunca se
  commitean** (están en `.gitignore`; en local pueden ser symlinks a otro directorio).

## Comandos

Desde la raíz del repo (el README todavía dice `cd experiments/sport-from-motion`, de
cuando vivía en otro repo; ignóralo):

```bash
uv venv && uv pip install -e ".[dev]"     # núcleo: numpy, pillow, scikit-learn
uv pip install -e ".[series]"             # MiniRocket (aeon)
uv pip install -e ".[probe]"              # torch + transformers (V-JEPA 2, DeepSets)
uv pip install -e ".[video]"              # --reprs video
uv pip install -e ".[laya]"               # Laya en proceso
uv run pytest
```

Prueba en seco, sin datos ni claves:

```bash
motion-sport ingest --source toy --out runs/toy/clips
motion-sport prepare --clips runs/toy/clips --out runs/toy/items
motion-sport run --items runs/toy/items --model dummy:first --condition motion --repr sheet
motion-sport report --items runs/toy/items
```

Subcomandos: `ingest`, `prepare`, `run`, `baseline`, `fit`, `probe`, `report`. El flujo
completo con ejemplos está en `README.md` («Flujo»).

Lanzadores: `DRY_RUN=1 bash scripts/final_run.sh` imprime los comandos sin ejecutar
nada; `bash scripts/pilot_models.sh runs/pilot-strict 200` para el piloto. Ambos son
reanudables (saltan ítems ya respondidos).

## Modelos y credenciales

Ids de modelo `<ruta>:<deployment>` (`azure-anthropic:`, `azure-openai:`,
`azure-foundry:`, `openai:`, `anthropic:`, `vertex:`, `vertex-anthropic:`, `jev:`,
`jev-openrouter:`, `laya:`, `laya-http:`, `dummy:`); la tabla de variables está en
`README.md` y la plantilla en `.env.example`. Las claves se leen del entorno o de
ficheros (`AZURE_OPENAI_KEY_FILE`, por defecto `~/.azure-openai-key`) y no se imprimen
nunca. No escribas claves en el repo ni en los logs. Jev y Laya solo aceptan
`--repr text`.

## Reglas del experimento

- **Pre-registro**: la corrida final está fijada en `docs/preregistration.md` (y el
  brazo Laya en `docs/preregistration-laya.md`). No cambies celdas, N ni modelos de
  `scripts/final_run.sh` ni de los análisis primarios sin añadir una desviación fechada
  en la sección «Desviaciones» del pre-registro correspondiente.
- Lo que no es análisis primario se etiqueta como secundario o exploratorio.
- `loaders/synthetic.py` y `--source toy` son solo para pruebas: nada de lo que producen
  es un resultado.
- No se redistribuye ningún dato; cada fuente tiene su licencia (ver `docs/datasets.md`).

## Gotchas

- Mac Intel (x86_64): numba ya no publica wheels. Usar `uv venv -p 3.12`, luego
  `uv pip install "numba<0.62" "llvmlite<0.45" "numpy==2.2.6"`, y ejecutar con
  `.venv/bin/...` en vez de `uv run` (que re-sincroniza, sube numpy y rompe MiniRocket).
- `.venv-laya/` es un entorno aparte para el brazo Laya (ignorado en git).
- `scripts/laya/kaggle_run.py` corre offline en Kaggle: ruedas y checkpoint vienen de un
  dataset privado con versiones fijadas (ver la cabecera del script).
