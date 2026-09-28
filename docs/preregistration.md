# Pre-registro: corrida final de modelos

> Escrito el 2026-09-27, **antes de que exista `runs/final`** y, por tanto, antes de ver
> ninguna respuesta sobre el dataset final. El commit que añade este fichero es la
> marca temporal. Cualquier cambio posterior va a la sección 11 («Desviaciones»),
> fechado y con motivo; lo de arriba no se reescribe.

Lanzador: [`scripts/final_run.sh`](../scripts/final_run.sh). Vista del análisis primario:
[`scripts/primary_view.py`](../scripts/primary_view.py). Coste y tiempo:
[`scripts/estimate_run.py`](../scripts/estimate_run.py).

## 0. Qué se ha visto antes

Los pilotos 1 y 2 ([`pilot-2026-09-27.md`](pilot-2026-09-27.md)) se hicieron sobre
otros ítems (preset `strict`, NFL alineada con el snap, balonmano solo TeamTrack, sin
prompt informado, sin vídeo ni `trails`). Se han usado **solo para dimensionar** la
corrida: número de clips, anchura esperada de los IC, tokens y tiempo por llamada. Las
hipótesis son las preguntas de [`design.md`](design.md) §1 y §7, escritas antes de
los pilotos. No se reformulan a la vista de ellos. En particular, el piloto 2 no
encontró efecto del orden, y H1 se mantiene igual. Todos los contrastes son
bilaterales.

## 1. Preguntas

- **RQ1** (formación frente a movimiento): ¿añade algo ver varios instantes frente a uno?
- **RQ1'** (lectura temporal): ¿usa el modelo el **orden** de los instantes, o solo
  las formas que aparecen?
- **RQ2** (colectivo frente a individual): `kinematics` frente a `kinematics_solo`
  (exploratoria en esta corrida).
- **RQ3** ¿Ven los modelos frontera la señal que un especialista sí ve?
- **RQ4** ¿Cambia algo si la vista llega como imagen o como texto?
- **Prompt**: ¿mejora el modelo si se le dice qué mirar (prompt informado)?
- **RQ5** ¿Qué deportes se confunden? (exploratoria)

## 2. Hipótesis (antes de los datos)

Las direcciones son las que se esperaban en el diseño. El test es bilateral: un
efecto en la dirección contraria también se detecta y se informa como tal.

| | Hipótesis | Dirección esperada |
|---|---|---|
| H1 | El orden temporal importa | `motion/sheet` > `motion_shuffled/sheet` |
| H2 | El movimiento aporta sobre la forma | `motion/sheet` > `formation/sheet` |
| H3 | El texto empeora frente a la imagen (lo que pasó en la Parte 2) | `motion/text` < `motion/sheet` |
| H4 | Decir qué mirar ayuda | `motion/sheet` informado > neutro |

Para Jev no hay hipótesis confirmatoria: está fuera de su dominio y se espera que
quede al azar (exploratorio).

## 3. Ítems

- **Conjunto**: `runs/final`, construido por separado: preset `strict_smooth`, 4
  deportes (fútbol, baloncesto, balonmano, fútbol americano), NFL a mitad de jugada,
  balonmano EIGD + TeamTrack, baloncesto SportVU (31 partidos), SkillCorner solo con
  posiciones detectadas (`--detected-only`). Los parámetros exactos de `prepare`
  quedan en `runs/final/config.json`. El lanzador se niega a arrancar (preflight) si
  el preset no es `strict_smooth`, si algún deporte tiene menos de 100 clips, si
  faltan los prompts informados o si falta una representación planificada.
- **Opciones**: las de `config.json` (`candidates`). Se esperan los 4 deportes sin
  distractores; azar = 1/|candidates| = 0,25.
- **Muestra**: los **400 primeros clips** del orden `interleave` (reparto por
  deporte y, dentro de cada deporte, por partido; la clave depende solo del clip). Son
  100 por deporte y **los mismos clips en todas las celdas**. Las réplicas 2 y 3 usan
  los 200 primeros (50 por deporte), que son un subconjunto de los 400.

## 4. Celdas y n

| Modelo | Celdas (prompt neutro salvo indicación) | n por celda | Llamadas |
|---|---|---|---|
| gpt-5.6-sol, gpt-5.6-terra (Azure), Claude Sonnet 5, Claude Opus 5.5 (Vertex) | `motion`, `motion_shuffled`, `formation`, `kinematics`, `kinematics_solo` en `sheet`; `motion/text`; `motion/trails`; `motion/sheet` **informado** | 400 | 3.200 |
| | réplicas 2 y 3 de `motion/sheet` y `motion_shuffled/sheet` | 200 | 800 |
| Gemini 3.1 Pro (Vertex) | lo mismo + `motion/video` y `motion_shuffled/video` | 400 | 4.000 + 800 |
| Jev (OpenRouter), `--state-format text` y `json` | las 5 condiciones en `text` + `motion/text` informado | 400 | 2 × 2.400 |

Total: 25.600 llamadas. Orden de ejecución por modelo (primarias primero):
`motion/sheet`, `motion_shuffled/sheet`, `formation/sheet`, `motion/text`,
`motion/sheet` informado; luego `kinematics/sheet`, `kinematics_solo/sheet`,
`motion/trails`, vídeo (Gemini) y réplicas. Si la corrida se corta, lo que falte es
secundario.

Ajustes: `--max-tokens 1024` (más el margen de razonamiento del backend, 16.384),
`--temperature 0` pedido. Los modelos que rechazan `temperature` (gpt-5.6, Sonnet 5 en
los pilotos) usan la suya por defecto, y el log lo registra. Por eso hay réplicas.
Workers: 6 por modelo, 12 para Gemini. No cambian las respuestas, solo el ritmo.

## 5. Análisis primario

**Vista.** `report` fusiona las réplicas de una celda en una sola, promediando la
corrección por ítem sobre los clips que responden *todas* las réplicas. Con las
réplicas 2 y 3 en solo 200 clips, sobre el directorio completo cada contraste
primario que toca `motion/sheet` se calcularía con 200 clips en vez de 400
(comprobado con datos de juguete: n = 4 de 8). Por eso el análisis primario lee
**solo la réplica 1** de los modelos pre-registrados:

```bash
.venv/bin/python scripts/primary_view.py runs/final runs/final-primary
.venv/bin/motion-sport report --items runs/final-primary --n-boot 10000 --json > runs/final/report_primary.json
```

`--n-boot 10000` en vez de 2.000: con 20 contrastes, el primer escalón de Holm exige
p < 0,0025, y con 2.000 remuestreos el p mínimo es 2/2.001 ≈ 0,001. La resolución no
alcanzaría.

**Contrastes**, exactamente los de `pipeline.PRIMARY_CONTRASTS`. Cada celda es
(condición, representación, prompt):

| Nombre | a | b | Hipótesis |
|---|---|---|---|
| `order` | `motion`, `sheet`, neutro | `motion_shuffled`, `sheet`, neutro | H1 |
| `motion_over_shape` | `motion`, `sheet`, neutro | `formation`, `sheet`, neutro | H2 |
| `text_vs_image` | `motion`, `text`, neutro | `motion`, `sheet`, neutro | H3 |
| `prompt` | `motion`, `sheet`, informado | `motion`, `sheet`, neutro | H4 |

**Estadístico** (`evaluate.paired_difference`): sobre los clips que responden las dos
celdas, d = media por clip de (acierto_a − acierto_b), con acierto ∈ {0, 1} y los
errores contados como fallo. IC del 95 % por percentiles de un **bootstrap agrupado
por partido** (`match_id`): se remuestrean partidos con reemplazo, lo que equivale a
pesar cada clip por las veces que sale su partido; semilla 0. El p es bilateral y sale
del mismo bootstrap: p = min(1, 2 · min(#{d* ≤ 0} + 1, #{d* ≥ 0} + 1) / (B + 1)).

**Familia y corrección.** Una sola familia: todos los contrastes primarios del
informe, es decir, 4 contrastes × 5 modelos de chat = **20**. Jev no tiene celdas
`sheet` y no aporta ninguno. Corrección de **Holm** (step-down), **α = 0,05**. Un
contraste es significativo si p_holm < 0,05. Si un contraste tiene menos de 5 partidos
(`evaluate.MIN_MATCHES`), IC y p salen NaN y Holm no lo cuenta. No se espera: los 400
clips cubren decenas de partidos. Si un modelo no puede correr (por ejemplo, Opus 5.5
sin habilitar en Vertex), la familia es la que exista (16 contrastes). Se dice así, y
el modelo no se sustituye por otro.

## 6. Qué cuenta como «el modelo ve movimiento»

Por modelo, con los resultados del análisis primario:

- **Lee el orden temporal** si `order` es **positivo con p_holm < 0,05** y además la
  **exactitud corregida por prior** de `motion/sheet` tiene el límite inferior de su
  IC del 95 % **por encima del azar** (0,25). Las dos condiciones son necesarias. La
  primera dice que el orden importa. La segunda, que lo que hace no es solo tener un
  favorito.
- **Le sirven varios instantes, pero no su orden** si `motion_over_shape` es positivo
  con p_holm < 0,05 y la exactitud corregida de `motion/sheet` supera el azar, pero
  `order` no es significativo.
- **Sin evidencia de que vea movimiento** en otro caso. Un `order` no significativo no
  prueba que el efecto sea cero: se da el límite superior de su IC («el orden aporta
  como mucho X puntos»).
- Un contraste significativo **negativo** se informa como tal. Nunca cuenta como «ve
  movimiento».
- Comprobación de robustez (secundaria, no cambia la clasificación): el contraste
  `order` sobre la media de las tres réplicas (200 clips, sección 7) debe tener el
  mismo signo. Si no lo tiene, se dice.

## 7. Análisis secundarios y exploratorios

Con p sin corregir y marcados como exploratorios. No se usan para las conclusiones de
la sección 6.

- **Todas las celdas**: exactitud, exactitud balanceada, macro-F1, kappa de Cohen,
  exactitud corregida por prior (prior de los *otros* partidos) y log-loss, con IC
  agrupado. También `predicted_share` (sesgo de respuesta).
- **Recall por clase** y matriz de confusión (RQ5): qué deportes se confunden, y si
  el orden sigue al tamaño del campo o a otra cosa.
- **Cinemática** (RQ2): `motion − kinematics`, `motion − kinematics_solo` y
  `kinematics − kinematics_solo` en `sheet` (los calcula `report`).
- **Vídeo** (Gemini): `motion/video − motion_shuffled/video` (lo calcula `report`) y
  `motion/video − motion/sheet` (con `evaluate.paired_difference`, a mano).
- **`trails`**: descriptiva (métricas de la celda). `motion/trails − motion/sheet`, con
  `paired_difference`, a mano.
- **Réplicas**: `report` sobre `runs/final` completo da, para `motion/sheet` y
  `motion_shuffled/sheet`, la exactitud media, la DE entre réplicas, el acuerdo por
  ítem (las tres etiquetas iguales) y los contrastes sobre la corrección media (200
  clips). La sección «primary contrasts» de *ese* informe **no** es el análisis
  primario: es la comprobación de robustez de la sección 6.
- **Jev**: todas sus celdas y contrastes en `text` (condiciones, `kinematics −
  kinematics_solo`, informado − neutro), texto plano frente a JSON (descriptivo) y
  `max_input_tokens` para detectar truncados.
- **Cortes**: por deporte, por fuente (p. ej. balonmano EIGD frente a TeamTrack) y
  `--tag static`. Con menos de 5 partidos no hay IC (`too_few_matches`).
- **Especialistas** (MiniRocket, baselines `nuisance`/`tempo`/`kinematic`) sobre los
  mismos ítems, como referencia de «la señal está». No forman parte de los
  contrastes de este pre-registro.

## 8. Errores y exclusiones

- **Ningún clip se excluye después de ver respuestas.** El conjunto lo fijan
  `interleave` y `--limit`.
- Un error de la API o una respuesta que no se puede interpretar (sin JSON, o sin
  una opción válida) queda registrado con `error` y sin etiqueta. Al reanudar, `run`
  **reintenta solo esos ítems**. El lanzador hace dos pasadas por celda
  (`PASSES=2`) y se puede relanzar las veces que haga falta, siempre con los mismos
  ajustes.
- Los errores que queden al final **cuentan como fallo** en todas las métricas y
  contrastes (etiqueta `__none__`, como ya hace `evaluate`). Nunca se quitan.
- **Una celda con más de un 2 % de errores no recuperados** (más de 8 de 400, o de 4
  de 200) se marca en las tablas con su recuento, y lo mismo sus contrastes. No se
  descarta en silencio. El lanzador imprime ese resumen al terminar.
- Negativas por política de contenido: se tratan como errores.

## 9. Qué no se toca

No se añaden clips, modelos a la familia ni celdas. No se cambian el prompt ni los
ajustes, y no se repite una celda con otros parámetros a la vista de los resultados.
Si aparece un bug del harness después de ver datos, se arregla, se repite **entera**
la celda afectada y se informan las dos versiones en «Desviaciones».

## 10. Qué se informa sea cual sea el resultado

- La tabla completa de celdas × modelos (n, exactitud e IC, balanceada, macro-F1,
  kappa, corregida por prior e IC, log-loss, errores no recuperados).
- **Los 20 contrastes primarios**: d, IC, p, p_holm y si son significativos, incluidos
  los nulos y los negativos.
- La clasificación de la sección 6 por modelo, también si todos quedan en «sin
  evidencia». Que no lo vean es un resultado.
- Recall por clase, réplicas, errores por celda, coste medido y desviaciones.

## 11. Desviaciones

Todas las de 2026-09-28 son **anteriores a cualquier dato de modelo sobre el dataset
final**: ningún modelo ha respondido todavía sobre `runs/final`. Cambian cómo se construye
el conjunto de ítems, no las hipótesis, las celdas, los modelos ni la familia de Holm.

### D1 (2026-09-28). Fase de la jugada NFL aleatoria respecto al snap

- **Qué cambia.** Los clips de fútbol americano dejan de cortarse a un desfase fijo desde
  el inicio de la grabación (`--trim-start-seconds 1.5`, C6). Pasan a ser **un clip por
  jugada** que empieza en un instante uniforme en [snap + 0,5 s, fin de la jugada − 4 s]
  (`ingest --source nfl --nfl-phase random --nfl-min-after-snap 0.5`). El snap es el
  primer evento `ball_snap` o `autoevent_ballsnap`. La semilla sale de (gameId, playId).
  Etiquetas `mid_play` y `random_phase`; el desfase respecto al snap queda en
  `meta.snap_offset_s`. Las jugadas sin snap o demasiado cortas se descartan y el ingest
  imprime el recuento.
- **Por qué.** Con el recorte fijo, todos los clips NFL empiezan 1 s después del snap: es
  la misma fase de la jugada en todos, y ningún otro deporte tiene esa alineación. Es un
  atajo posible que el suavizado no toca (piloto 2).
- **Coste conocido.** En BDB 2023 el tracking acaba poco después del pase, así que la
  jugada dura una mediana de 3,2 s tras el snap (semana 1). Con clips de 4 s solo cabe
  ~15-17 % de las jugadas: 29 de 200 en la semana 1, y ~1.400 en las 8 semanas. Basta
  para 400 clips, pero es un subconjunto sesgado hacia jugadas largas (scrambles, sacks,
  pases tardíos), y el desfase se concentra cerca del mínimo (mediana 0,7 s). Se informa
  como limitación de la fuente.

## Anexo A. Dimensionado (del piloto 4, 400 clips, 55 partidos)

La semianchura del IC al 95 % de los contrastes primarios fue de 0,04 a 0,10 (EE ≈
0,02-0,05). El dataset final tiene más partidos por deporte, así que se esperan IC algo
más estrechos. Con esos EE, el primer escalón de Holm (α/20 = 0,0025, z ≈ 3,0)
detecta diferencias de unos 0,07-0,15 y el último (0,05) de unos 0,04-0,10. Las
réplicas (200 clips) solo sirven para estimar el ruido de muestreo del modelo, no para
el análisis primario.

## Anexo B. Coste y tiempo estimados

`scripts/estimate_run.py` con los tokens medidos en `runs/pilot4-strict` (y en
`runs/pilot-strict` para Gemini 3.1) y precios de lista: unos **510 USD** en total.
Gemini 3.1 se lleva unos 315, por sus ~5.000 tokens de razonamiento por ítem; siguen
gpt-5.6-sol ~95, Opus 5.5 ~52 (estimado con los tokens de Sonnet 5), Sonnet 5 ~31,
terra ~14 y Jev < 1. Unas **11 h** de reloj, marcadas por Gemini (los modelos corren en
paralelo). Los precios son supuestos, no facturas.

## Anexo C. Requisitos antes de lanzar

- `runs/final` construido (con `--reprs sheet,trails,text,video`, que necesita
  `.[video]`), y el preflight en verde.
- Opus 5.5 habilitado en el Model Garden de Vertex (hueco A6; lo hace Javier).
- `gcloud` autenticado. Claves de Azure y OpenRouter en sus ficheros, que el
  lanzador lee sin imprimirlas.

```bash
DRY_RUN=1 bash scripts/final_run.sh     # comprobar los comandos
bash scripts/final_run.sh               # la corrida (reanudable)
```

### D2 (2026-09-28). Techo físico de velocidad: fuera los «teletransportes» del tracker

- **Qué cambia.** `ControlConfig.max_speed_ms = 12` (m/s) en todos los presets. Un clip se
  rechaza si algún jugador **conservado** (tras `fix_player_count`) da un paso más rápido
  que eso a la frecuencia del clip. Se mide sobre las coordenadas en metros, antes del
  suavizado y de normalizar. `prepare --max-speed-ms X` lo cambia (0 lo apaga). `prepare`
  guarda en `config.json` los rechazos por motivo (`players`, `teleport`, `window`), por
  deporte (`rejected_by_sport`) y por fuente (`rejected_by_source`), con entradas y
  supervivientes por fuente. Avisa si una fuente pierde más del 10 % de sus clips.
- **Por qué.** Un paso de 20 m en 0,2 s no es movimiento: es un cambio de ID o una pista
  que se recupera. Y se reparte muy desigual entre fuentes. En una muestra de `pool_final`
  (600 clips por fuente, N = 10 aleatorio), la velocidad máxima por clip supera 12 m/s en
  el 11 % de TeamTrack (fútbol 14 %, balonmano 9 %, baloncesto 3 %; máximo 10.740 m/s),
  el 2,5 % de Metrica (máximo 504 m/s), el 0,3 % de NFL y SportVU, y el 0 % de EIGD y
  SkillCorner. Es otra firma del tracker, como la de C4, y el suavizado la reparte por el
  clip en vez de quitarla.
- **Coste conocido.** TeamTrack pierde ~11 % y Metrica ~2,5 % antes de muestrear, y
  `prepare` lo avisa. El techo (12 m/s) está por encima del sprint humano (~10-11 m/s) y del
  p99 de todas las fuentes limpias (≤ 9 m/s).

### D3 (2026-09-28). El balonmano de TeamTrack es un partido, no dos

- **Qué cambia.** Las dos partes (`1st`, `2nd`) del único partido de balonmano de TeamTrack
  comparten un `match_id` (`tt-handball`). La parte sigue en el segmento.
  `scripts/teamtrack_to_long_csv.py` lo hace así en las ingestas nuevas. Para los clips ya
  ingestados, `scripts/relabel_match.py` reescribe `match_id` sin volver a descargar. Se ha
  aplicado sobre una copia de enlaces, `data/clips_tt_regroup` (434 clips; `data/clips` no se
  toca). El loader `long-csv` ya no parte el id de partido por «-».
- **Por qué.** El CV agrupado y el bootstrap por partido suponen que los grupos son
  independientes. Dos mitades del mismo partido no lo son (mismos jugadores, mismo
  tracker), y contarlas como dos partidos estrecha los IC del balonmano. El balonmano pasa
  de 7 grupos a 6 (5 EIGD + 1 TeamTrack), que es lo que dice C5.
- **Coste conocido.** Un grupo menos de balonmano: IC algo más anchos en los cortes por
  deporte.

### D4 (2026-09-28). Robustez del informe y de la vista primaria

No cambia ningún contraste ni el estadístico. Cierra tres casos que el pre-registro no
resolvía de forma mecánica:

- **Modelo que no corre.** Si **todas** las filas de un modelo son errores no recuperados
  (p. ej. Opus 5.5 sin habilitar en Vertex), `report` lo saca de todos los contrastes y de la
  familia de Holm, e imprime una nota (`excluded_models` en el JSON). Es lo que ya decía la
  sección 5 («la familia es la que exista»); antes habría entrado con contrastes 0 − 0. Sus
  celdas siguen en la tabla. Un modelo con alguna celda que funciona no se excluye: sus
  errores cuentan como fallo (sección 8).
- **Filas que faltan.** Se elige un **plan explícito**: `final_run.sh` escribe
  `runs/final/plan.json` = `{"cells": {<fichero de predicciones>: filas planificadas}}`,
  sacado de sus propios comandos (`--limit`), con `scripts/run_plan.py write`. Se descarta
  la alternativa de guardar el `--limit` en cada registro de predicción, porque no detecta
  una celda que nunca arrancó (sin fichero no hay registros). El resumen de errores del
  lanzador (`scripts/run_plan.py check`) y `report` (`incomplete`) marcan las celdas con
  menos filas que las planificadas, y las que superan el 2 % de errores sobre lo
  planificado. Un relanzamiento con `MODELS=<uno>` fusiona su plan con el existente.
- **Especialistas comparables por clip.** `scripts/primary_view.py --clips-from-models`
  restringe los especialistas de la vista a los clips que se preguntaron a los modelos
  (unión de sus ficheros; avisa si no coinciden). Así, una comparación por clip entre un
  especialista y un modelo va sobre los mismos clips. La vista recibe además un
  `plan.json` filtrado (réplica 1 de los modelos pre-registrados). Los especialistas no
  entran en los contrastes primarios (sección 7), así que esto no toca la familia.

```bash
.venv/bin/python scripts/primary_view.py runs/final runs/final-primary --clips-from-models
.venv/bin/python scripts/run_plan.py check runs/final
```

