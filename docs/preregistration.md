# Pre-registro: corrida final de modelos

> Escrito el 2026-09-27, **antes de que exista `runs/final`** y, por tanto, antes de ver
> ninguna respuesta sobre el dataset final. El commit que añade este fichero es la
> marca temporal. Cualquier cambio posterior va a la sección 11 («Desviaciones»),
> fechado y con motivo; lo de arriba no se reescribe.

> **Nota (2026-09-28).** Antes de la primera llamada a un modelo sobre `runs/final` se
> reconstruyó el conjunto de ítems, se añadió una celda exploratoria y se corrigieron
> algunas frases. Las secciones 3, 4, 5, 7, 8 y 10 y los anexos B y C describen ya lo que
> se va a correr. Cada cambio está en la sección 11, fechado, con su motivo y su
> evidencia, y el texto original sigue en el commit `745f699`. A partir de la primera
> llamada a un modelo sobre `runs/final`, la regla de arriba vuelve a ser estricta.

> **Nota (2026-09-29).** La corrida terminó el 2026-09-28 a las 23:26. La única
> desviación posterior a los datos es **D21** (sección 11): un bug del harness en la ruta
> Anthropic, arreglado después de la corrida. No cambia ningún número pre-registrado, y lo
> corregido se informa solo como sensibilidad. Resultados en
> [`results-final.md`](results-final.md).

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

- **Conjunto**: `runs/final`, reconstruido por cuarta vez el 2026-09-28 desde
  `data/pool_final4` (37.052 clips; D8, D17, D18 y D19), antes de cualquier llamada a un
  modelo.
  Preset `strict_smooth` (suavizado gaussiano, σ = 2 fotogramas), ventanas de 4 s a 5 Hz
  (20 fotogramas), `--per-sport 400`, las 5 condiciones y las representaciones `sheet`,
  `trails`, `text` y `video`. Además del preset:
  - **Jugadores**: N = 10 **elegidos al azar** en cada clip (`--player-mode random
    --n-players 10`), no los 10 más centrales (hueco A14; D5 y D8).
  - **Techo de velocidad**: se rechaza el clip si algún jugador conservado da un paso de
    más de 12 m/s (`max_speed_ms`; D2).
  - **Pistas congeladas**: una pista cuya posición es exactamente constante en toda la
    ventana no es un jugador (un valor retenido o de relleno) y se descarta antes de
    elegir los 10 (`drop_frozen`; D18). En TeamTrack, un jugador no detectado viene como
    (0, 0) exacto y se lee como no observado (D18).
  - **Tramos interpolados**: una pista que se mueve en línea exactamente recta a velocidad
    exactamente constante (segunda diferencia < 0,1 mm por paso; quieta incluida) durante
    **4 s o más** de la ventana está interpolada ahí, no medida, y se descarta antes de
    elegir los 10 (`drop_linear`, `linear_min_s = 4.0`; D19). En la ventana de 4 s es la
    ventana entera.
  - **Pistas duplicadas**: dos pistas que no se separan más de **5 cm** en toda la ventana
    (la distancia máxima, no la media) son un jugador con dos identificadores; se descarta
    la copia de índice mayor antes de elegir los 10 (`drop_duplicates`, `duplicate_tol_m =
    0.05`; D19).
  - **Fútbol americano**: un clip por jugada, que empieza en una fase **aleatoria** entre
    **1,0 s** después del snap y el final de la jugada menos 4 s (D1 y D16).
  - **Balonmano**: 6 partidos. Las dos partes del partido de TeamTrack forman un solo
    grupo, `tt-handball` (D3).
  - **Fútbol de TeamTrack**: un solo partido, `tt-soccer`. Sus dos ficheros son una
    misma grabación con una fecha mal escrita (D10).

  Los parámetros exactos quedan en `runs/final/config.json`: `controls`, clips de entrada
  y conservados por deporte y por fuente, y rechazos por motivo.
- **Fuentes** (clips conservados en `runs/final` y partidos):

  | Deporte | Fuentes (clips) | Clips | Partidos |
  |---|---|---|---|
  | Fútbol americano | NFL Big Data Bowl 2023, semanas 1-8, fase aleatoria desde 1,0 s tras el snap (D1, D16) | 400 | 122 |
  | Baloncesto | NBA SportVU 2015-16 (382) + TeamTrack (9) | 391 | 31 + 1 |
  | Balonmano | EIGD-H, 5 partidos de la HBL medidos con Kinexon (332) + TeamTrack (66) | 398 | 5 + 1 |
  | Fútbol | SkillCorner Open Data, solo posiciones detectadas (248) + Metrica (62) + TeamTrack (27) | 337 | 10 + 2 + 1 |

  De 1.600 clips de entrada (400 por deporte) se conservan 1.526, de 173 partidos.
  Rechazos: SkillCorner 60 de 308 por número de jugadores (19 %, el único aviso de
  `prepare`); TeamTrack 6 de 108 (2 por teletransporte y 4 por jugadores: 3 clips de
  baloncesto se quedan con 9 al quitar una pista interpolada, D19); SportVU 6 (2 por
  teletransporte y 4 por jugadores: 3 son un jugador con dos identificadores, que deja 9
  jugadores reales, D19); EIGD 2 por jugadores, porque todas sus pistas están congeladas
  (el arranque congelado de dos secuencias, D18). Pistas descartadas antes de elegir:
  congeladas, 28 en esos 2 clips de EIGD y 1 en un clip de TeamTrack
  (`frozen_dropped_by_source`); interpoladas, 10 de TeamTrack en 10 clips
  (`linear_dropped_by_source`); duplicadas, 3 de SportVU en 3 clips
  (`duplicate_dropped_by_source`). Ningún jugador conservado está quieto, en (0, 0),
  interpolado 4 s ni a 5 cm de otro en toda la ventana
  (`scripts/analysis/duplicate_tracks.py`, `frozen_tracks.py`, `linear_tracks.py`).
  `prepare` tardó 27 minutos (`runs/prepare_final5.log`).
- **Preflight**. El lanzador se niega a arrancar (`scripts/run_plan.py preflight`) si el
  preset no es `strict_smooth`; si en `controls` no están exactamente `player_mode =
  random`, `n_players = 10`, `n_frames = 20`, `smooth = 2.0` y `max_speed_ms = 12.0`; si
  `candidates` no son los 4 deportes; si algún clip de fútbol americano no lleva la
  etiqueta `random_phase`; si algún deporte tiene menos de 100 clips; si faltan los
  prompts informados; si falta una representación planificada, o si en alguna de ellas
  falta una de las 5 condiciones que admite. El de A7b (`preflight-a7b`) fija igual
  `smooth = 2.0` y `max_speed_ms = 12.0`, con `n_frames = 40`. Los valores booleanos no
  valen como números (D13). Los dos exigen además `drop_frozen = True` y rechazan el
  conjunto si algún jugador conservado no se mueve en ningún fotograma (D18), y exigen
  `drop_linear = True`, `linear_min_s = 4.0`, `drop_duplicates = True` y `duplicate_tol_m
  = 0.05`, y rechazan el conjunto si dos jugadores conservados no se separan más de 5 cm en
  toda la ventana (D19).
- **Opciones**: las de `config.json` (`candidates`), es decir, los 4 deportes sin
  distractores. Azar = 1/|candidates| = 0,25.
- **Muestra**: los **400 primeros clips** del orden `interleave` (reparto por
  deporte y, dentro de cada deporte, por partido; la clave depende solo del clip). Son
  100 por deporte y **los mismos clips en todas las celdas**. Cubren 151 partidos
  (fútbol americano 100, baloncesto 32, fútbol 13, balonmano 6). Por fuente: NFL 100,
  SportVU 97, EIGD 84, SkillCorner 77, Metrica 15 y TeamTrack 27 (3 de baloncesto, 16 de
  balonmano y 8 de fútbol). Las réplicas 2 y 3 usan los 200 primeros (50 por deporte),
  que son un subconjunto de los 400. Frente a D18 cambian 2 de los 400, en la misma
  posición, del mismo deporte y la misma fuente (D19): sale el 381
  (`teamtrack-P3-val-P3_fisheye_30-60-00004`, con una pista interpolada) y entra
  `teamtrack-P3-test-P3_fisheye_60-90-00000`; sale el 389
  (`sportvu-0021500368-s012-00000`, un jugador con dos identificadores) y entra
  `sportvu-0021500368-s107-00014`. Los otros 398 son idénticos.
- **Celda A7b, clips de 8 s (secundaria y exploratoria; D7)**: `runs/final-d8`, con los
  mismos controles (`strict_smooth`, N = 10 al azar, 12 m/s) pero ventanas de **8 s**
  (`--n-frames 40`) y **3 deportes**: baloncesto, balonmano y fútbol (azar 1/3). No hay
  fútbol americano porque solo 42 clips NFL de mitad de jugada llegan a 8 s. Sale de
  `data/pool_d8_final4`, que son los clips de 8 s del barrido A7 con el balonmano y el
  fútbol de TeamTrack reagrupados como en D3 y D10, y con los (0, 0) de TeamTrack como no
  observados (D18). De 1.200 clips de entrada se conservan 1.064
  (baloncesto 386, balonmano 397, fútbol 281; por fuente SportVU 375, EIGD 333,
  SkillCorner 195, Metrica 60 y TeamTrack 101) de 51 partidos. `drop_linear` quita 24
  pistas de TeamTrack en 18 clips (uno sale); ninguna pista duplicada entra en el sorteo
  (D19). Muestra: los
  **300 primeros** del orden `interleave` (100 por deporte, 51 partidos). Celdas:
  `motion/sheet` y `motion_shuffled/sheet`, con prompt neutro. La hoja sigue teniendo 8
  fotogramas: a 8 s quedan a unos 1,1 s uno de otro (a 4 s, a unos 0,55 s). Tiene su
  propio preflight: `scripts/run_plan.py preflight-a7b`.

## 4. Celdas y n

| Modelo | Celdas (prompt neutro salvo indicación) | n por celda | Llamadas |
|---|---|---|---|
| gpt-5.6-sol, gpt-5.6-terra (Azure), Claude Sonnet 5, Claude Opus 5.5 (Vertex) | `motion`, `motion_shuffled`, `formation`, `kinematics`, `kinematics_solo` en `sheet`; `motion/text`; `motion/trails`; `motion/sheet` **informado** | 400 | 3.200 |
| | réplicas 2 y 3 de `motion/sheet` y `motion_shuffled/sheet` | 200 | 800 |
| Gemini 3.1 Pro (Vertex) | lo mismo + `motion/video` y `motion_shuffled/video` | 400 | 4.000 + 800 |
| Jev (OpenRouter), `--state-format text` y `json` | las 5 condiciones en `text` + `motion/text` informado | 400 | 2 × 2.400 |
| Los 5 modelos de chat | **A7b** (D7): `motion/sheet` y `motion_shuffled/sheet` sobre `runs/final-d8` (8 s, 3 deportes) | 300 | 600 por modelo |

Total: 28.600 llamadas, las 25.600 pre-registradas el 2026-09-27 más las 3.000 de A7b.
Orden de ejecución por modelo (primarias primero): `motion/sheet`,
`motion_shuffled/sheet`, `formation/sheet`, `motion/text`, `motion/sheet` informado;
luego las secundarias pre-registradas, `kinematics/sheet`, `kinematics_solo/sheet`,
`motion/trails`, vídeo (Gemini) y réplicas; y **al final** A7b (`motion/sheet` y
`motion_shuffled/sheet` a 8 s, todos menos Jev), que es exploratoria. Si la corrida se
corta, lo que falte es secundario o exploratorio. Si falla algo antes de la primera
llamada (preflight o `plan.json`), el lanzador sale con error y no llama a ningún modelo;
una celda que falla no para las demás del mismo modelo (D14).

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
.venv/bin/python scripts/primary_view.py runs/final runs/final-primary --clips-from-models
.venv/bin/motion-sport report --items runs/final-primary --n-boot 10000 --json > runs/final/report_primary.json
# A7b (D7), exploratoria: su propio directorio, fuera de la familia de Holm
.venv/bin/python scripts/primary_view.py runs/final-d8 runs/final-d8-view --clips-from-models
.venv/bin/motion-sport report --items runs/final-d8-view --n-boot 10000 --exploratory --json > runs/final-d8/report_a7b.json
```

`--clips-from-models` restringe los especialistas de la vista a los clips que se
preguntaron a los modelos (D4).

El informe de A7b es **exploratorio** entero: el lanzador marca `runs/final-d8/plan.json`
con `"exploratory": true`, la vista lo hereda y `report` lo lee (`--exploratory` lo
fuerza). Así no hay ningún contraste primario ni familia de Holm: `order` sale, con su
nombre, en «exploratory contrasts (not pre-registered; raw p)», sin p_holm y sin
asteriscos (D15).

`--n-boot 10000` en vez de 2.000: con 20 contrastes, el primer escalón de Holm exige
p < 0,0025. Con 2.000 remuestreos el p mínimo es 2/2.001 ≈ 0,001, que ya está por debajo,
así que 2.000 bastarían para alcanzarlo. Se eligen 10.000 por resolución: el p mínimo
baja a 2/10.001 ≈ 0,0002 y los p cercanos a los umbrales de Holm se distinguen con un paso
cinco veces más fino (corregido el 2026-09-28, D9).

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
el modelo no se sustituye por otro. `report` lo hace solo: un modelo con **todas** sus
filas en error sale de los contrastes y de la familia (`excluded_models`, D4), y una celda
con más de un 50 % de errores no recuperados, también (sección 8, D12). Las celdas
A7b (8 s) tampoco entran en la familia: van en otro directorio y se analizan aparte, como
exploratorias (sección 7, D7).

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
- **Especialistas** (MiniRocket, DeepSets, baselines `nuisance`/`tempo`/`kinematic`),
  como referencia de «la señal está». Con estos controles `nuisance` es degenerado (sus
  variables son constantes, D11) y solo sirve para comprobar el CV. Se entrenan con CV agrupada por partido sobre
  todos los clips del conjunto, pero **se comparan sobre los mismos clips que los
  modelos**: la vista primaria (`primary_view.py --clips-from-models`, D4) los restringe
  a los 400 clips preguntados. No forman parte de los contrastes de este pre-registro.
- **A7b, clips de 8 s** (D7): por modelo, exactitud (azar 1/3), recall por clase y
  `motion − motion_shuffled` sobre los 300 clips de `runs/final-d8`, con p sin corregir.
  La comparación con 4 s es descriptiva: se dice si `order` a 8 s es distinto de cero y
  si es mayor que a 4 s, pero no se calcula un contraste 8 s − 4 s, porque los clips y el
  número de opciones son distintos. No cambia la clasificación de la sección 6.
- **Fase de la jugada NFL** (D1): recall de fútbol americano según el desfase del clip
  respecto al snap (`meta.snap_offset_s`), descriptivo.

## 8. Errores y exclusiones

- **Ningún clip se excluye después de ver respuestas.** El conjunto lo fijan
  `interleave` y `--limit`.
- Un error de la API o una respuesta que no se puede interpretar (sin JSON, o sin
  una opción válida) queda registrado con `error` y sin etiqueta. Al reanudar, `run`
  **reintenta solo esos ítems**. El lanzador hace dos pasadas por celda
  (`PASSES=2`) y se puede relanzar las veces que haga falta, siempre con los mismos
  ajustes.
- Los errores que queden al final **cuentan como fallo** en todas las métricas y
  contrastes (etiqueta `__none__`, como ya hace `evaluate`). Nunca se quitan filas.
- **Una celda con más de un 2 % y hasta un 50 % de errores no recuperados** (más de 8 de
  400, o de 4 de 200) sigue en el análisis, con sus errores como fallo, y se marca en las
  tablas con su recuento, y lo mismo sus contrastes (`!` en `report`). No se descarta en
  silencio. El lanzador imprime ese resumen al terminar.
- **Una celda con más de un 50 % de errores no recuperados** no mide al modelo: es un
  fallo sistémico (ruta caída, filtro de contenido, respuestas que no se pueden leer).
  Sale de **todos** los contrastes en los que participa, primarios y secundarios, y de la
  familia de Holm; su fila sigue en la tabla, con su recuento y la marca `cell excluded:
  systemic failure` (`excluded_cells` en el JSON). Es la misma regla que la de un modelo
  sin ninguna respuesta (D4), aplicada celda a celda. El umbral se cuenta sobre las filas
  presentes de la celda (todas sus réplicas), porque las filas que faltan no son errores
  (siguiente punto). Se fija antes de cualquier dato de modelo (D12). Si la celda excluida
  es `motion/sheet` neutro, ese modelo se queda sin ninguno de sus 4 contrastes primarios
  (los cuatro la usan) y la familia de Holm pierde esos 4. En la sección 6 queda como
  «sin evidencia de que vea movimiento», con la causa dicha: no se pudo medir.
- **Filas que faltan** (D4). Una celda planificada con menos filas que su `--limit`, o
  sin fichero (p. ej. una corrida cortada), no es un error registrado: no hay respuesta
  que contar. El lanzador es reanudable y se relanza con los mismos ajustes hasta que
  `scripts/run_plan.py check` no liste filas que falten. Solo se analiza después. Si una
  celda no se puede completar (p. ej. un modelo retirado), se informa marcada con sus
  filas presentes y planificadas (`incomplete` en `report`). Sus métricas y contrastes van
  sobre los clips presentes, con su n en la tabla, y se dice. Nunca se completa con otros
  clips ni con otra celda.
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
- A7b (8 s): exactitud y `motion − motion_shuffled` de cada modelo, marcados como
  exploratorios, salga lo que salga.
- Recall por clase, réplicas, errores por celda, coste medido y desviaciones.

## 11. Desviaciones

Todas las de 2026-09-28 son **anteriores a cualquier dato de modelo sobre el dataset
final**. Cuando se escribieron, ningún modelo había respondido sobre `runs/final` ni
`runs/final-d8`, ni sobre los conjuntos anteriores (`runs/final-v1`, `runs/final-v2`,
`runs/final-v3`, `runs/final-v4`, `runs/final-d8-v1`, `runs/final-d8-v2` y
`runs/final-d8-v3`): los nueve directorios `predictions/` contienen solo especialistas
(comprobado de nuevo al escribir D10-D19). Los especialistas (MiniRocket, DeepSets y los
baselines) sí se corrieron, sobre los conjuntos viejos y sobre los nuevos. No son datos de
modelo en el sentido de este pre-registro (sección 7): sirvieron para encontrar los
problemas y para comprobar los conjuntos reconstruidos, y sus números están abajo. Las
desviaciones cambian cómo se construye el conjunto de ítems, añaden una celda exploratoria
(A7b), fijan reglas mecánicas del informe y del lanzador, y corrigen texto. No cambian las
hipótesis, las celdas primarias ni los modelos. La familia de Holm solo cambia por la
regla de fallo sistémico (D12), que saca contrastes que no miden al modelo.

> **Excepción: D21 es POSTERIOR a los datos.** Se escribió al terminar la corrida, con
> todas las respuestas de `runs/final` y `runs/final-d8` ya vistas. No cambia ningún
> número pre-registrado: los resultados primarios siguen saliendo de `runs/final` tal
> como quedó, y lo corregido es solo una sensibilidad.

Todos los cambios desde el commit del pre-registro (`745f699`, 2026-09-27 21:00):

| # | Qué cambia | Por qué | Evidencia | Commit |
|---|---|---|---|---|
| D1 | NFL: un clip por jugada, en fase aleatoria tras el snap | con el recorte fijo, casi todos los clips NFL empezaban en la misma fase de la jugada | verificador: 381 de 400 clips de `runs/final-v1` empezaban justo 1,0 s después del snap | `eda4e4a` |
| D2 | Techo de velocidad de 12 m/s | los teletransportes del tracker son una firma de la fuente, muy desigual | barrido de velocidades por fuente; 7 clips graves en `runs/final-v1` (hasta 6.797 m/s), 1 dentro de los 400 | `02e0d0c` |
| D3 | El balonmano de TeamTrack es un partido | sus dos partes no son grupos independientes | verificador: al fusionarlas, el recall de MiniRocket en ese corte baja de 0,614 a 0,544 | `cd52ef4` |
| D4 | `report` y vista primaria: modelos que no corren, filas que faltan, especialistas sobre los mismos clips | casos que el pre-registro no resolvía de forma mecánica | verificador del pre-registro | `5bbdd24` |
| D5 | El preflight exige el conjunto nuevo | que no dependa de la memoria | `runs/final-v1` no lo pasa | `b2dffa2` |
| D6 | `prepare --n-frames`, relleno de series de MiniRocket; ablaciones A7 y A8 | herramientas del barrido de duración | sin efecto sobre la corrida (20 fotogramas) | `a722feb`, `b1306f5` |
| D7 | Celda A7b: clips de 8 s, exploratoria | a 8 s los especialistas mejoran (C11) | especialistas sobre `runs/final-d8` | `75feb4e` |
| D8 | Conjunto reconstruido con N = 10 al azar; secciones 3, 4, 7, 8 y 10 al día | la selección central quitaba señal de forma desigual (A8) y la sección 3 no describía las fuentes reales | A8 (C10); especialistas sobre el conjunto nuevo | `75feb4e` |
| D9 | Corregida la frase sobre la resolución del bootstrap | era falsa | 2/2.001 ≈ 0,001 < 0,0025 | `75feb4e` |
| D10 | El fútbol de TeamTrack es un partido (`tt-soccer`) | sus dos ficheros son una grabación continua con la fecha mal escrita | verificador; costura de 900 s: mismas 22 pistas a 0,61 m de mediana | `eec85c4` |
| D11 | Corrección: el baseline `nuisance` es degenerado y no es evidencia; se retira la «firma de TeamTrack» de D7 y D8 | tras los controles sus 4 variables son constantes | `baselines.nuisance_features`; spread = 1 ± ruido de float32 | `78f144d` |
| D12 | Regla de fallo sistémico por celda (> 50 % de errores) | una celda caída no mide al modelo y metería un contraste falso en Holm | tests de `report` | `5cddffe` |
| D13 | El preflight fija fotogramas, suavizado, techo, candidatos y celdas | que un conjunto mal construido no pase | tests de mutación | `a62d4e4` |
| D14 | Lanzador: aborta si falla la contabilidad; A7b al final | una celda exploratoria no debe ir antes que las secundarias pre-registradas, y un `plan.json` fallido no puede dejar llamar | tests del lanzador con `python` falso; el script viejo falla 4 | `c6b355d` |
| D15 | Informe exploratorio para A7b (sin Holm) | en D7 `order` salía como primario con Holm propio | tests de `report` | `95de80b` |
| D16 | NFL: fase aleatoria desde 1,0 s tras el snap (no 0,5 s) | los clips seguían pegados al arranque de la jugada | regla fijada antes del ingest: ≥ 400 clips de ≥ 80 partidos; salen 1.125 de 122 | `78f144d` (datos: `runs/ingest_nflrand10.log`) |
| D17 | `runs/final` y `runs/final-d8` reconstruidos; especialistas; test de fuga por fuente dentro del fútbol | D10 y D16 cambian los clips | preflights en verde; especialistas y `source_id.json` | `78f144d` |
| D18 | TeamTrack: el (0, 0) exacto es un jugador no detectado (NaN); una pista exactamente constante en la ventana no es un jugador (`drop_frozen`); conjuntos reconstruidos desde `pool_final4` / `pool_d8_final4` | puntos quietos en una esquina contaban como jugadores | verificador: 19 de 63 clips de balonmano de TeamTrack en `runs/final-v3`; auditoría por fuente; 0 jugadores quietos en los conjuntos nuevos | `fc42605`, `faaac76`, `86806f4`, `381d14e` |
| D19 | Dos pistas a ≤ 5 cm en toda la ventana son un jugador con dos identificadores (`drop_duplicates`); un tramo exactamente recto de ≥ 4 s es interpolado, faltante (`drop_linear`); conjuntos reconstruidos | el modelo veía 9 jugadores dibujados como 10, y en TeamTrack pistas interpoladas como jugadores | verificador: SportVU 0021500368 segmento 12, posición 389 de los 400; par real más cercano en 4 s a 0,31 m (NFL 0,53 m); 0 de 434.859 pistas de otras fuentes rectas 4 s, 94 de TeamTrack | `f42e8ea`, `3a80332`, `20f421e` |
| D20 | A16 decidido: no se aplica ninguna regla nueva; matices de D19; paso real entre instantáneas | ninguna regla por fotogramas cambia un clip de las muestras pagadas | máx. 6 de 20 fotogramas a ≤ 5 cm en los 400 y 9 de 40 en los 300 de A7b | `7801b55` |
| D21 (**POST-datos**) | Ruta Anthropic: margen de razonamiento que el §4 pre-registraba y no se aplicaba; sensibilidad sobre los 40 errores de Sonnet 5 `motion/text` | bug del harness (§9): Sonnet 5 piensa de forma adaptativa y el razonamiento cuenta contra `max_tokens` = 1024 | 40 de 40 errores acaban exactamente en 1.024 tokens de salida; ninguna otra celda llega al tope | `d6a3833` |

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
  atajo posible que el suavizado no toca (piloto 2). El verificador lo midió en
  `runs/final-v1` contra los eventos de snap de los CSV de BDB 2023: 381 de 400 clips
  empezaban exactamente 1,0 s después del snap (los 1,5 s de recorte menos los 0,5 s que
  la grabación tiene antes del snap), 12 entre 5 y 13 s después y 1 a 1,2 s. Otros 3
  contradecían la etiqueta `mid_play`: en dos el snap caía dentro de la ventana y el
  tercero era entero anterior al snap. Y 4 jugadas no tenían evento de snap.
- **Coste conocido** (medido en el ingest de las 8 semanas, `runs/ingest_nflrand.log`;
  sustituye a la estimación de ~1.400 clips y mediana de 0,7 s que se escribió antes del
  ingest). En BDB 2023 el tracking acaba poco después del pase: la jugada dura una
  mediana de 3,2 s tras el snap (semana 1). De 8.557 jugadas, 6.847 son demasiado cortas
  para un clip de 4 s y 24 no tienen snap. Quedan **1.686 clips** (19,7 %) de 122
  partidos. Bastan para 400, pero son un subconjunto sesgado hacia jugadas largas
  (scrambles, sacks, pases tardíos).
- **Limitación: los clips siguen cerca del snap.** El desfase tiene mínimo 0,5 s,
  cuartiles 0,6 / 0,8 / 1,3 s (media 1,1) y máximo 15,7 s. El 46 % de los clips empieza a
  menos de 0,7 s del snap y el 67 % a menos de 1,0 s; el recorte fijo empezaba a 1,0 s.
  La fase aleatoria quita la alineación exacta (todos los clips en la misma fase), pero
  no aleja los clips del arranque de la jugada: la explosión inicial sigue dentro de casi
  todos. Es una propiedad de la fuente (jugadas cortas) que no se puede corregir sin otra
  fuente (A12), y se informa como limitación. Como cada clip lleva `meta.snap_offset_s`,
  el recall de fútbol americano por desfase se da como análisis exploratorio (sección 7).
- **Actualizado por D16 (2026-09-28).** El mínimo tras el snap sube de 0,5 s a 1,0 s. Los
  números de «Coste conocido» y «Limitación» son los del ingest a 0,5 s
  (`data/clips_nflrand`), que ya no está en `runs/final`. Corrección de redacción: el
  «67 % a menos de 1,0 s» es el 66,7 % que empieza **a 1,0 s o antes**; estrictamente
  antes de 1,0 s es el 61 %.

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
  clip en vez de quitarla. El verificador encontró en `runs/final-v1` unos 7 clips con
  teletransportes graves que sobrevivían a los controles: hasta 6.797 m/s en el fútbol de
  TeamTrack, 98-111 m/s en 4 clips de su balonmano y 51 m/s en uno de SportVU. Uno estaba
  entre los 400 primeros, y MiniRocket y DeepSets clasificaban el peor como fútbol
  americano (un punto sale disparado y los demás se aplastan en el render).
- **Coste conocido.** TeamTrack pierde ~11 % y Metrica ~2,5 % antes de muestrear, y
  `prepare` lo avisa. El techo (12 m/s) está por encima del sprint humano (~10-11 m/s) y del
  p99 de todas las fuentes limpias (≤ 9 m/s). En el conjunto reconstruido (D8) los
  rechazos por teletransporte fueron: TeamTrack 12 de 134 (9 %), SportVU 1 y Metrica 1.

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
  de 7 grupos a 6 (5 EIGD + 1 TeamTrack), que es lo que dice C5. El verificador midió el
  efecto en `runs/final-v1`: con una parte entrenando y la otra en test, el recall de
  MiniRocket en el balonmano de TeamTrack era 0,614; con las dos partes en un grupo, 0,544
  (cinemático 0,614 → 0,605; 114 clips). La exactitud global no baja (0,753 → 0,762,
  porque cambian los folds): la inflación es pequeña y está en ese corte.
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

### D5 (2026-09-28). El preflight exige el conjunto de ítems nuevo

- **Qué cambia.** Además de lo que ya comprobaba (sección 3), el lanzador se niega a
  arrancar si en `config.json` (`controls`) no están `player_mode = random` y
  `n_players = 10` (hueco A14, decidido: selección aleatoria), si `max_speed_ms` no está
  fijado (D2), o si algún clip de fútbol americano de los ítems no lleva la etiqueta
  `random_phase` (D1). El preflight pasa a `scripts/run_plan.py preflight` para poder
  probarlo. `final_run.sh` lo llama con los mismos argumentos.
- **Por qué.** El `runs/final` construido el 2026-09-27 (selección central, sin techo de
  velocidad, NFL con desfase fijo) no pasa: hay que reconstruirlo antes de la corrida. El
  preflight lo garantiza en vez de fiarlo a la memoria.

### D6 (2026-09-27/28). Herramientas del barrido de duración; ablaciones A7 y A8

Es la primera en el tiempo: va entre el pre-registro y D1.

- **Qué cambia.** `prepare --n-frames` fija la ventana en fotogramas (10, 20 o 40 = 2, 4 u
  8 s a 5 Hz). Para que MiniRocket acepte las series de 8 puntos de los clips de 2 s,
  `learners.pad_series` repite el último valor hasta 9 puntos (commit `a722feb`,
  2026-09-27 22:17, después del pre-registro). Con eso se corrieron las ablaciones de
  especialistas A7 (duración del clip) y A8 (asimetría de N), commit `b1306f5`,
  documentadas en `gaps.md` (C10 y C11).
- **Efecto sobre la corrida.** Ninguno directo. `runs/final` usa la ventana del preset (20
  fotogramas), así que sus series tienen 20 puntos y no se rellenan, y las ablaciones solo
  tocan especialistas. Sus resultados motivan la celda A7b (D7) y la selección aleatoria
  de jugadores (D8).

### D7 (2026-09-28). Celda A7b: los modelos con clips de 8 s (secundaria, exploratoria)

- **Qué cambia.** Los 5 modelos de chat responden también `motion/sheet` y
  `motion_shuffled/sheet`, con prompt neutro, sobre los 300 primeros clips de
  `runs/final-d8` (sección 3), ~~justo después de sus celdas primarias~~ **al final de
  todas sus celdas** (sustituido por D14). Son 3.000 llamadas
  más, unos 68 USD (anexo B). Jev no entra: solo lee texto y A7b es de imagen.
  `final_run.sh` las lanza con `--items runs/final-d8 --limit 300`, las planifica en
  `runs/final-d8/plan.json` y comprueba el conjunto con `run_plan.py preflight-a7b`
  (preset `strict_smooth`, N = 10 al azar, techo de velocidad, 40 fotogramas, los 3
  deportes, al menos 100 clips por deporte y las dos celdas).
- **Por qué.** En los especialistas, 8 s gana a 4 s: con 3 deportes, MiniRocket `motion`
  pasa de 0,72 a 0,82 (+0,09 [0,06; 0,13], C11). Con los controles nuevos,
  `runs/final-d8` da MiniRocket `motion` 0,87 [0,84; 0,90] sobre todos los clips y 0,88
  [0,85; 0,92] sobre los 300 primeros, y `motion_shuffled` 0,81 (0,80 en los 300). Si un
  modelo no lee el orden a 4 s, la objeción obvia es que 4 s son pocos, y A7b la contesta
  con datos.
- **Qué no es.** No entra en la familia de Holm ni en la clasificación de la sección 6.
  Tampoco es un contraste 8 s − 4 s emparejado: los clips son otros y las opciones son 3
  en vez de 4. Se analiza con `report` sobre su propio directorio (sección 5), con p sin
  corregir. ~~En ese informe `order` sale con la etiqueta de contraste primario y con un
  Holm propio solo porque es la misma pareja de celdas; esa etiqueta no aplica aquí.~~
  **Sustituido por D15:** el informe de A7b se genera en modo exploratorio, sin
  contrastes primarios ni Holm, y `order` sale entre los exploratorios con p sin corregir.
- **Límites conocidos.** ~~El `nuisance` a 8 s acierta al azar (0,35 [0,20; 0,47], kappa
  0,00), pero porque contesta «balonmano» en 829 de 1.069 clips: es un clasificador
  degenerado, y eso es una prueba más débil de que no quedan atajos que un nuisance que
  reparte sus respuestas.~~ **Retirado (D11):** tras los controles las variables de
  `nuisance` son constantes, así que su resultado no dice nada sobre atajos, ni a favor ni
  en contra. En los especialistas, `motion − motion_shuffled` a 8 s es +0,07
  [0,03; 0,10]; en `abl-d8_noaf`, con selección central y sin techo de velocidad, era
  +0,12.
- **Números de un conjunto anterior.** Los especialistas de este apartado son de
  `runs/final-d8-v1` (ahora movido). Los del conjunto reconstruido están en D17.

### D8 (2026-09-28). Conjunto de ítems reconstruido; secciones 3, 4, 7, 8 y 10 al día

> Este conjunto es ahora `runs/final-v2`. Se reconstruyó otra vez (D17) tras D10 y D16;
> las cifras vigentes están en la sección 3 y en D17.

- **Qué cambia.**
  - El `runs/final` del 2026-09-27 pasa a `runs/final-v1` (ningún modelo llegó a
    usarlo) y se reconstruye desde `data/pool_final2`: los clips de `data/clips` menos
    los de NFL (fuera los 1.262 clips de desfase fijo), más SkillCorner detectado
    (7.456), el balonmano de TeamTrack reagrupado (434, D3) y los 1.686 clips NFL de fase
    aleatoria (D1). Son 37.613 clips. `prepare` tardó 21 minutos
    (`runs/prepare_final2.log`) y el preflight pasa.
  - **Selección de jugadores al azar** (hueco A14): N = 10 elegidos al azar en cada clip
    en vez de los 10 más centrales.
  - La sección 3 describe las fuentes reales. La versión del 2026-09-27 no nombraba
    TeamTrack en baloncesto y fútbol ni Metrica en fútbol, que ya estaban en el conjunto
    v1 (lo señaló el verificador del pre-registro).
  - Sección 4 y 10: la celda A7b (D7). Sección 5: la vista primaria restringe los
    especialistas a los clips de los modelos. Sección 7: especialistas sobre los mismos
    clips que los modelos (D4), corte por desfase del snap (D1) y A7b. Sección 8: regla
    para las filas que faltan (D4). Anexos B y C: coste y requisitos con A7b.
- **Por qué la selección al azar.** En A8 (C10), con N = 10 al azar, MiniRocket `motion`
  sube de 0,75 a 0,82 sobre los mismos clips (+0,07 [0,04; 0,09]). El baloncesto, cuyos
  clips son idénticos con las dos reglas (399 de 399), sube de 0,70 a 0,80 (+0,10 [0,06;
  0,14]), y el balonmano +0,12. Los 10 centrales de fútbol y balonmano se parecían al
  baloncesto: la selección central quitaba señal, y de forma desigual entre deportes.
  La selección al azar trata igual a todos los deportes y no depende de una regla
  nuestra.
- **Comprobación con especialistas** sobre el conjunto nuevo (todos los clips; azar
  0,25; es `runs/final-v2`, reconstruido otra vez en D17): MiniRocket `motion` 0,82 [0,79; 0,85] (v1: 0,75), `motion_shuffled` 0,70,
  `kinematics` 0,75, `kinematics_solo` 0,72; DeepSets `motion` 0,81 y `formation` 0,54;
  baseline cinemático 0,69, `tempo` 0,53 y `nuisance` 0,29 [0,21; 0,39] (kappa 0,04;
  degenerado, D11). En los 400 primeros: MiniRocket `motion` 0,80 [0,75; 0,85] y
  `nuisance` 0,31 [0,22; 0,41]. MiniRocket `motion − motion_shuffled` +0,12 [0,09; 0,15]. Entre fuentes
  (TeamTrack fuera del entrenamiento, `runs/final2-xs`), MiniRocket da 0,97 dentro de las
  fuentes y **0,95** en TeamTrack, frente a 0,94 → 0,86 en el conjunto viejo.
- **Límites conocidos.**
  - ~~El `nuisance` nunca contesta «fútbol» y contesta «balonmano» en la mayoría de los
    clips de TeamTrack, sea cual sea su deporte (46 de 59 de balonmano, 9 de 12 de
    baloncesto, 13 de 51 de fútbol). Queda una firma de TeamTrack en las variables de
    nuisance aunque su balonmano sea un solo grupo. Un modelo que leyera esa firma
    acertaría más en el balonmano de TeamTrack y menos en su fútbol y su baloncesto. En
    los 400 primeros, TeamTrack son 34 clips (16 de balonmano). El corte por fuente
    (sección 7) lo deja ver, y se informa como limitación.~~ **Retirado (D11): era falso.**
    Las 4 variables de `nuisance` son constantes tras los controles, y lo que parecía una
    firma de TeamTrack era ruido de redondeo de float32 que `StandardScaler` amplifica. La
    pregunta de si queda una firma de la fuente se contesta con otro test, dentro del
    fútbol y con variables de movimiento (D17).
  - El `tempo` sube de 0,43 a 0,53, sobre todo por EIGD (recall 0,77): el ritmo del
    balonmano medido con Kinexon es distinto. `tempo` es un baseline de atajo que
    `strict_smooth` no pretende quitar (lo quitaría `strict_tempo`). Queda en la tabla de
    especialistas como referencia.

### D9 (2026-09-28). Corrección: la resolución del bootstrap

- **Qué cambia.** La sección 5 decía que con 2.000 remuestreos «la resolución no
  alcanzaría» el primer escalón de Holm. Es falso. El p mínimo con B = 2.000 es 2/2.001 ≈
  0,0010, que ya está por debajo de 0,0025, y el contraste más fuerte tendría p_holm = 20 ×
  0,001 = 0,02 < 0,05. Se mantienen los 10.000 remuestreos por el motivo correcto: más
  resolución (p mínimo ≈ 0,0002) cerca de los umbrales de Holm. No cambia ningún cálculo.
- **Evidencia.** El verificador del pre-registro (2026-09-28), comprobado a mano con la
  fórmula de la sección 5.

### D10 (2026-09-28). El fútbol de TeamTrack es un partido, no dos

- **Qué cambia.** Todos los segmentos de fútbol de TeamTrack comparten el `match_id`
  `tt-soccer`. `scripts/teamtrack_to_long_csv.py` lo hace así en las ingestas nuevas, y
  `scripts/relabel_match.py` documenta el reetiquetado de los clips ya ingestados. Se ha
  aplicado sobre copias: `data/clips_tt_regroup` (390 clips: 174 de `20200220` y 216 de
  `20220220`) y `data/pool_d8_final3`, una copia de enlaces de `pool_d8_final2` con 192
  clips reetiquetados (84 + 108). `data/clips`, `pool_final2` y `pool_d8_final2` no se
  tocan.
- **Por qué.** TeamTrack publica su único partido de fútbol como dos ficheros,
  `F_20200220_1` (0-900 s) y `F_20220220_1` (900-1.980 s). La segunda fecha es una errata:
  es la misma grabación, continua. El verificador lo detectó por la fecha. En la costura
  de los 900 s, los mismos 22 índices de pista están a una mediana de 0,61 m (máximo
  1,70 m) de donde los dejó el primer fichero, más cerca que entre dos segmentos del mismo
  fichero (mediana 3,97 m). Como en D3, dos grupos que no son independientes estrechan los
  IC e inflan el CV agrupado. El verificador midió el efecto en el conjunto anterior: el
  recall del baseline cinemático en el fútbol de TeamTrack era 0,608 con dos grupos y
  0,569 con uno.
- **Coste conocido.** El fútbol pierde un grupo (13 partidos en los 400 primeros, antes
  14). El reparto por partido de `interleave` da ahora a TeamTrack un turno en vez de dos,
  así que su fútbol pesa menos: 27 clips en `runs/final` (antes 51) y 8 en los 400
  primeros (antes 15).

### D11 (2026-09-28). Corrección: el baseline `nuisance` es degenerado

- **Qué cambia.** Nada en el código ni en el análisis. Se corrige cómo se lee un número.
  El baseline `nuisance` usa 4 variables por clip: número de jugadores, fps, duración y
  dispersión (`baselines.nuisance_features`). Con los controles de `runs/final` las cuatro
  son **constantes por construcción**: N = 10, 5 fps, 4 s (8 s en `runs/final-d8`) y
  dispersión = 1, porque la normalización `spread` la fija (desviación típica 4,4 × 10⁻⁸
  en `runs/final`, ruido de float32). Un clasificador sobre variables constantes no
  puede aprender nada: cualquier diferencia entre sus respuestas viene de ese ruido de
  redondeo, que `StandardScaler` amplifica al dividir por una desviación típica casi nula.
  Medido por fuente, el ruido es de −25 ± 45 × 10⁻⁹ en todas.
- **Qué se retira.**
  - D8, «límites conocidos»: la «firma de TeamTrack en las variables de nuisance». Era
    falsa. Se deja tachada en D8.
  - D7, «límites conocidos»: que el `nuisance` a 8 s al azar fuera «una prueba más débil
    de que no quedan atajos». No es prueba de nada. Se deja tachada en D7.
  - `gaps.md` C4, C11 y C15: la misma lectura, corregida allí de forma explícita.
- **Qué significa.** Que `nuisance` salga al azar con estos controles no demuestra que no
  queden atajos: lo que demuestra es que los controles fijan N, fps, duración y escala,
  cosa que ya se sabe por construcción. Sigue siendo útil como comprobación del CV: un
  clasificador sin información tiene que salir al azar, y si no saliera habría un
  problema de folds (como en C3). Donde sí se busca una firma de la fuente es con
  variables de movimiento, dentro de un mismo deporte (D17) y entre fuentes (C4).

### D12 (2026-09-28). Regla de fallo sistémico por celda

- **Qué cambia.** `report` calcula, por celda, la proporción de errores no recuperados
  sobre sus filas **presentes**, sumando todas sus réplicas. Las filas que faltan no
  cuentan como error (D4).
  - **Más de un 50 %**: la celda sale de todos los contrastes primarios y secundarios en
    los que participa, y por tanto de la familia de Holm. Su fila sigue en la tabla con
    «cell excluded: systemic failure (k/n errors)» y en `excluded_cells` del JSON.
  - **Entre un 2 % y un 50 %**: la celda sigue, sus errores cuentan como fallo, y se marca
    ella y sus contrastes (`!` y `flagged_cells`).
  - `scripts/run_plan.py check` nombra las celdas de más del 50 % en el resumen del
    lanzador.
- **Por qué.** El pre-registro solo excluía un modelo con **todas** sus filas en error
  (D4). Una celda con, por ejemplo, un 90 % de negativas por filtro de contenido metería en
  Holm un contraste que mide el filtro, no al modelo. Es la misma regla que D4, celda a
  celda.
- **Consecuencia para la sección 6.** Los 4 contrastes primarios usan `motion/sheet`
  neutro. Si esa celda se excluye, el modelo se queda sin contrastes primarios, la familia
  pierde 4, y en la sección 6 el modelo queda como «sin evidencia», diciendo que fue
  porque no se pudo medir. Cambia un test de D4: un modelo con una sola celda entera en
  error ya no aporta el `order` de esa celda a la familia.

### D13 (2026-09-28). El preflight fija los valores exactos

- **Qué cambia.** `run_plan.py preflight` exige `n_frames == 20`, `smooth == 2.0`,
  `max_speed_ms == 12.0`, `candidates` igual a los 4 deportes, y que cada representación
  planificada (`sheet`, `text`, `trails`, y `video` si hay un Gemini en el plan) tenga
  todas las condiciones que `pipeline.VALID` admite en ella. `preflight-a7b` exige
  `smooth == 2.0` y `max_speed_ms == 12.0`. Se rechazan los booleanos (`True == 1` en
  Python). Hay un test de mutación por cada valor fijado, por los candidatos y por las
  celdas que faltan.
- **Por qué.** D5 comprobaba que `max_speed_ms` estuviera fijado, pero no su valor. El
  preflight principal miraba el nombre del preset, pero no el suavizado, la ventana, los
  candidatos ni las celdas. Un conjunto preparado con `--n-frames 40`, otro techo o un
  preset editado habría pasado.

### D14 (2026-09-28). Lanzador: abortar sin llamar y A7b al final

- **Qué cambia.** `scripts/final_run.sh` corre con `set -euo pipefail`. Si `write_plan`
  falla, imprime «write_plan failed: no model was called» y sale con 1. Dentro de cada
  modelo (`set +e` en su subshell), una celda que falla no para las siguientes. Orden por
  modelo: las 5 celdas primarias; luego `kinematics`, `kinematics_solo`, `trails`, vídeo
  (Gemini) y réplicas; y **A7b al final**. `scripts/estimate_run.py` sigue el mismo
  orden. Los tests ejecutan una copia del script con `python` y `motion-sport` falsos y
  `HOME` redirigido, sin leer claves. El script anterior falla 4 de ellos.
- **Por qué.** D7 ponía A7b «justo después de sus celdas primarias», delante de las
  secundarias pre-registradas. Si la corrida se cortaba, lo que se perdía era
  pre-registrado y lo que quedaba, exploratorio. Y un fallo al escribir el plan no
  paraba el lanzador.

### D15 (2026-09-28). Informe exploratorio para A7b

- **Qué cambia.** `motion-sport report --exploratory`, o `"exploratory": true` en
  `plan.json` (`pipeline.is_exploratory`), no calcula contrastes primarios ni Holm. Todo
  sale en «exploratory contrasts (not pre-registered; raw p)», sin `p_holm` y sin
  asteriscos, y `order` conserva su nombre. `run_plan.py write --exploratory` pone la
  marca, que sobrevive a los relanzamientos. `final_run.sh` la pone en `runs/final-d8` y
  `primary_view.py` la pasa a la vista. La sección 5 añade `--exploratory` al comando de
  A7b.
- **Por qué.** D7 aceptaba que el informe de A7b sacara `order` como «primario» con un Holm
  propio, y confiaba en que se leyera bien. Así ya no hay nada que malinterpretar.

### D16 (2026-09-28). NFL: la fase aleatoria empieza a 1,0 s del snap

- **Qué cambia.** Los clips NFL se reingestan con `--nfl-min-after-snap 1.0` en vez de 0,5
  (`data/clips_nflrand10`, `runs/ingest_nflrand10.log`). Lo demás de D1 se mantiene.
- **Por qué.** D1 dejó como limitación que los clips seguían pegados al arranque de la
  jugada. Para medirlo: por clip, velocidad media de los jugadores en los 5 primeros pasos
  dividida por la de los 5 últimos, con los mismos controles. Menos de 1 = el clip
  acelera. Media geométrica [IC 95 % bootstrap por partido]:
  - NFL a 0,5 s, todos los clips del pool: **0,866** [0,850; 0,882], mediana 0,859. Por
    desfase: < 1,0 s 0,833; 1,0-2,0 s 0,948; ≥ 2,0 s 0,849.
  - Las otras fuentes, en el conjunto anterior: 0,93-1,04.

  A 0,5 s la NFL estaba por debajo de todas las demás fuentes: una firma de fase de
  jugada que ningún otro deporte tiene.
- **Regla fijada antes del ingest.** Adoptar 1,0 s si da **al menos 400 clips de al menos
  80 partidos**; si no, quedarse en 0,5 s y documentarlo como limitación. Resultado:
  **1.125 clips de 122 partidos**. Se adopta 1,0 s.

  | | 0,5 s (`clips_nflrand`) | 1,0 s (`clips_nflrand10`) |
  |---|---|---|
  | Clips / partidos | 1.686 / 122 | 1.125 / 122 |
  | Clips por partido (mín. / mediana / máx.) | 5 / 13 / 25 | 1 / 9 / 19 |
  | Desfase: mín., cuartiles, p90, máx. (s) | 0,5; 0,6/0,8/1,3; 2,0; 15,7 | 1,0; 1,1/1,3/1,8; 2,5; 15,7 |
  | Desfase medio (s) | 1,10 | 1,60 |
  | Clips a < 1,5 s / < 2,0 s | 80 % / 89 % | 62 % / 80 % |

  Por semana: 127, 141, 159, 149, 165, 131, 134 y 119 clips. Hay 24 jugadas sin snap.
- **Efecto.** Rampa NFL a 1,0 s: **0,938** [0,916; 0,961], mediana 0,937, en todo el
  pool. En los 400 de `runs/final`: 0,885 [0,851; 0,919] antes y 0,932 [0,893; 0,974]
  ahora. En el `runs/final` nuevo, las otras fuentes dan: SportVU 0,936 [0,876; 0,997];
  EIGD 1,026 [1,003; 1,048]; Metrica 0,965 [0,898; 1,037]; SkillCorner 0,969 [0,917;
  1,030]; TeamTrack 0,850 / 0,963 / 0,997 en baloncesto, balonmano y fútbol (un partido
  cada uno, sin IC).
- **Límite que queda.** A 1,0 s la rampa NFL se solapa con la de SportVU, pero sigue por
  debajo de la del balonmano y el fútbol. La explosión del snap ya no está en casi todos
  los clips, pero los clips siguen en la primera parte de la jugada (mediana 1,3 s tras
  el snap), porque en BDB 2023 las jugadas son cortas. Es una restricción de la fuente
  (A12) y el corte por desfase de la sección 7 lo sigue mostrando.
- **Coste conocido.** 561 clips NFL menos y menos clips por partido (mediana 9). Siguen
  sobrando para 400 clips de 122 partidos.

### D17 (2026-09-28). Conjuntos reconstruidos; especialistas; test de fuga por fuente

- **Qué cambia.**
  - `runs/final` pasa a `runs/final-v2` y `runs/final-d8` a `runs/final-d8-v1`. Ningún
    modelo respondió sobre ellos.
  - `data/pool_final3` (37.052 clips) es `pool_final2` sin sus 1.686 clips NFL, con los
    1.125 a 1,0 s (D16) y con los 390 clips de fútbol de TeamTrack apuntando a los
    reetiquetados (D10). `data/pool_d8_final3` es el pool de 8 s con el mismo
    reetiquetado.
  - `runs/final` y `runs/final-d8` se preparan de nuevo con los mismos parámetros. Las
    cifras de la sección 3 son las nuevas, y los dos preflights (con los valores de D13)
    pasan.
- **Todos los deportes cambian de muestra, no solo la NFL.** `balanced_sample` usa un
  solo generador aleatorio para todos los deportes y saca primero el fútbol americano.
  Al cambiar el pool NFL y fusionar el fútbol de TeamTrack, cambia el sorteo de todos.
  Solapamiento con `runs/final-v2`:

  | Deporte | Todos los clips | 400 primeros |
  |---|---|---|
  | Fútbol americano | 113/400 | 20/100 |
  | Baloncesto | 5/397 | 1/100 |
  | Balonmano | 86/397 | 21/100 |
  | Fútbol | 19/337 | 3/100 |

  En `runs/final-d8` el baloncesto y el balonmano son idénticos a los de v1 (el pool de 8
  s no tiene NFL). Del fútbol, 88 de los 100 de los 300 primeros son los mismos. Es otra
  muestra de la misma población, sorteada antes de cualquier dato de modelo: no se elige
  a la vista de nada.
- **Especialistas sobre `runs/final`** (azar 0,25; CV agrupada por partido;
  `runs/final/report_specialists.txt` y `specialists_breakdown.txt`):

  | Especialista | 1.531 clips | 400 primeros |
  |---|---|---|
  | MiniRocket `motion` | 0,84 [0,82; 0,86] | 0,85 [0,81; 0,88] |
  | MiniRocket `motion_shuffled` | 0,72 | 0,74 |
  | MiniRocket `kinematics` | 0,76 | 0,76 |
  | MiniRocket `kinematics_solo` | 0,70 | 0,69 |
  | DeepSets `motion` | 0,80 | 0,79 |
  | DeepSets `formation` | 0,54 | 0,51 |
  | Baseline cinemático | 0,67 | 0,67 |
  | Baseline `tempo` | 0,50 | 0,50 |
  | Baseline `nuisance` (degenerado, D11) | 0,25 [0,18; 0,31], kappa 0,00 | 0,24 |

  - Recall de MiniRocket `motion` por deporte: fútbol americano 0,92, baloncesto 0,82,
    balonmano 0,79 y fútbol 0,83. El baseline cinemático sigue acertando el fútbol
    americano (0,91).
  - Por fuente: EIGD 0,81, Metrica 0,82, NFL 0,92, SkillCorner 0,84, SportVU 0,82, y en
    TeamTrack 0,67 (baloncesto), 0,67 (balonmano) y 0,74 (fútbol).
  - Contrastes emparejados:

    | Contraste | Todos los clips | 400 primeros |
    |---|---|---|
    | MiniRocket `motion − motion_shuffled` | +0,12 [0,10; 0,14] | +0,11 [0,07; 0,15] |
    | MiniRocket `kinematics − kinematics_solo` | +0,07 [0,04; 0,09] | +0,065 [0,02; 0,11] |
    | DeepSets `motion − formation` | +0,25 [0,21; 0,31] | +0,28 [0,22; 0,36] |

  - `nuisance` contesta «baloncesto» en 901 de 1.531 clips.
- **Especialistas sobre `runs/final-d8`** (azar 0,33):
  - MiniRocket `motion`: 0,88 [0,85; 0,91] en todos los clips y 0,88 en los 300 primeros.
  - `motion_shuffled`: 0,79 en los dos.
  - `motion − motion_shuffled`: +0,08 [0,05; 0,11] en todos y +0,09 [0,04; 0,14] en los
    300 primeros.
  - `nuisance` 0,35 (degenerado, D11).
- **Entre fuentes** (fútbol y baloncesto, TeamTrack fuera del entrenamiento;
  `runs/final3-xs`):

  | | Dentro de las fuentes de entrenamiento | TeamTrack, nunca visto |
  |---|---|---|
  | MiniRocket | 0,97 | 0,95 |
  | Cinemático | 0,73 | 0,73 |

  El JSON es idéntico byte a byte al de `runs/final2-xs`. TeamTrack solo está en test,
  así que su id de partido no interviene.
- **Test de fuga nuevo: ¿se reconoce la fuente dentro de un deporte?** (sustituye como
  evidencia al `nuisance`, D11; `runs/final/source_id.json`).
  - **Método.** Para cada deporte, las fuentes con al menos 2 partidos. Con CV agrupada
    por partido, se predice la fuente del clip a partir de:
    - las 14 variables del baseline cinemático, con una logística balanceada;
    - las series de `motion`, con MiniRocket y una logística.

    Se mide la exactitud balanceada (azar = 1/nº de fuentes). El IC sale de un bootstrap
    de partidos estratificado por fuente. El p sale de una permutación exacta de las
    fuentes entre partidos.
  - **Solo el fútbol se puede probar**: Metrica (2 partidos, 62 clips) frente a
    SkillCorner (10 partidos, 248). Los demás deportes tienen una sola fuente con 2
    partidos o más, y TeamTrack es ahora un partido por deporte.

    | Variables | Exactitud balanceada | Recall Metrica / SkillCorner | p exacto (permutación) |
    |---|---|---|---|
    | Cinemáticas | 0,70 [0,66; 0,72] | 0,63 / 0,76 | 0,015 (nulo: media 0,49, p95 0,60) |
    | Series MiniRocket | 0,69 [0,66; 0,71] | 0,48 / 0,89 | 0,015 (nulo: media 0,47, p95 0,51) |

  - La permutación recorre las 66 formas de asignar 2 de los 12 partidos a Metrica. El
    valor observado supera a las otras 65, así que p = 1/66, el mínimo que admite el
    test. Los IC se apoyan en solo 2 partidos de Metrica, por debajo del mínimo del
    proyecto (5, `evaluate.MIN_MATCHES`): no son fiables.
  - **Qué muestra.** Dentro del fútbol, las dos familias de variables reconocen la fuente
    muy por encima del azar. Las variables de movimiento siguen llevando una firma de la
    fuente. Puede ser el tracker o la competición (ligas y estilos distintos), y este test
    no los separa.
  - **Qué no muestra.** No se puede correr donde una fuga importaría más. TeamTrack es la
    única fuente con varios deportes, pero tiene un partido por deporte, y la NFL tiene
    una sola fuente. Para TeamTrack, la evidencia relevante sigue siendo el test entre
    fuentes: 0,95 sobre una fuente nunca vista.
- **Análisis y scripts.** `ramp.py`, `breakdown.py`, `source_id.py` y `ttseam.py`, con
  `ramp.json`, están en `runs/analysis-2026-09-28/`. Desde D18 están versionados en
  `scripts/analysis/` (`faaac76`); `breakdown.py` reproduce byte a byte el
  `specialists_breakdown.txt` de este conjunto, y `contrasts.py` (`86806f4`) todos los
  contrastes de arriba.
- **Actualizado por D18 (2026-09-28).** Este conjunto es ahora `runs/final-v3` (y el de 8
  s, `runs/final-d8-v2`). Tenía 21 clips con jugadores que no se movían (19 de balonmano
  de TeamTrack y 2 de EIGD), 7 de ellos entre los 400 primeros. Los números de arriba son
  los de ese conjunto; los del reconstruido están en D18.

### D18 (2026-09-28). TeamTrack: (0, 0) es un jugador no detectado; las pistas congeladas no son jugadores

- **Qué encontró el verificador.** En los CSV de TeamTrack (`data/raw/teamtrack/*.csv`,
  de `scripts/teamtrack_to_long_csv.py`), un jugador no detectado viene como (0,0; 0,0)
  exacto. El pipeline lo leía como una posición real. Una pista pegada a (0, 0) toda la
  ventana contaba como observada del todo, pasaba el techo de 12 m/s (no se mueve) y se
  dibujaba como un punto quieto en una esquina (p. ej.
  `runs/final-v3/render/teamtrack-1st-train-1st_fisheye_480-510-00001/motion_sheet.png`).
- **Evidencia en los datos crudos.**
  - Nunca hay una sola coordenada a 0: siempre las dos. Es un código de «falta», no una
    posición.
  - Puntos (0, 0): baloncesto 52 de 90.000 filas (1 de 10 segmentos); balonmano 21.139
    de 799.200 (2,6 %; 25 de 62 segmentos); fútbol 7.665 de 1.073.688 (5 de 66). En 16
    segmentos de balonmano hay exactamente 900 o 1.800 ceros en 900 fotogramas: en cada
    fotograma falta uno (o dos) de los 15-16 huecos de jugador, repartido entre dos
    pistas que se turnan (p. ej. pistas 1 y 6).
  - En los clips ingeridos (`data/pool_final3`): 160 de los 894 clips de TeamTrack
    tienen algún (0, 0) (4.560 puntos), y 185 pistas están en (0, 0) todo el clip. En el
    pool de 8 s, 73 de 408 (4.100 puntos).
- **Afectados en los conjuntos de D17** (auditoría `scripts/analysis/frozen_tracks.py`:
  jugadores conservados cuya posición no cambia en ningún fotograma, comprobado sobre
  los clips controlados y repitiendo con la semilla de `prepare` qué pistas se
  eligieron; los dos métodos coinciden clip a clip).
  - `runs/final-v3`: **19 de los 63 clips de balonmano de TeamTrack**. En 17 hay un punto
    en (0, 0), en 1 hay dos y en 1 hay una pista retenida en un valor distinto de cero.
    **7 de los 16** clips de balonmano de TeamTrack de los 400 primeros. Además, **2 clips
    de EIGD con los 10 jugadores quietos** (fuera de los 400; ver abajo). Ninguno en
    otras fuentes.
  - `runs/final-d8-v2`: 9 clips (8 de balonmano de TeamTrack y 1 de fútbol de TeamTrack
    con 4 pistas en (0, 0)), 3 de ellos entre los 300 primeros. El verificador contó 4;
    aquí cuenta cualquier jugador conservado sin ningún movimiento.
  - `runs/final3-xs` (entre fuentes): 38 jugadores quietos de Metrica y 15 de TeamTrack.
  - Con los (0, 0) parciales pasaba otra cosa: un salto a (0, 0) y de vuelta supera los
    12 m/s y el clip se rechazaba como teletransporte. Por eso TeamTrack perdía 6 clips
    por teletransporte en D17; ahora pierde 2.
- **¿Hay pistas congeladas en otras fuentes?** Auditoría de todos los clips de
  `data/pool_final3` (37.052) y del pool de 8 s:

  | Fuente | Clips | Puntos (0, 0) | Pistas congeladas | Clips con alguna | Clips con todas |
  |---|---|---|---|---|---|
  | NFL | 1.125 | 0 | 0 | 0 | 0 |
  | SportVU | 23.306 | 0 | 0 | 0 | 0 |
  | SkillCorner | 7.456 | 0 | 0 | 0 | 0 |
  | Metrica | 2.861 | 0 | 79 | 23 | 0 |
  | EIGD | 1.410 | 0 | 72 | 5 | 5 |
  | TeamTrack | 894 | 4.560 | 191 (185 en (0, 0)) | 135 | 0 |

  - **EIGD: artefacto, no parada de juego.** Los 5 clips enteramente quietos son el
    arranque de 3 de las 25 secuencias de 5 min: los primeros 13,2 s de
    `48dcd3_00-06-00`, 4,7 s de `e0e547_00-00-00` y 4,6 s de `e8a35a_00-02-00`. En esos
    tramos están exactamente quietos a la vez todos los jugadores (14-16) y el balón, y
    empiezan en t = 0. Al acabar, los jugadores se mueven con un paso normal (0,04-0,06 m
    por fotograma a 30 Hz), sin salto: es el primer dato real copiado hacia atrás, no un
    tiempo muerto. En el resto de EIGD (5.535 ventanas de 4 s con al menos 10 jugadores
    observados) **ninguna** tiene una sola pista exactamente quieta. El tramo real más
    tranquilo (mediana 0,155 m/s, `ad969d_00-00-30` en el segundo 49, probablemente una
    parada) tiene a todos los jugadores moviéndose al menos 6,5 cm. Un tiempo muerto real
    tiene ese aspecto, no el de posiciones idénticas al milímetro. Estos 5 clips se
    quedan fuera (2 estaban en `runs/final-v3`).
  - **Metrica: artefacto.** Las 79 pistas congeladas son jugadores sueltos quietos entre
    otros que se mueven (mediana de los demás 0,5-1,4 m/s), a menudo en la misma posición
    en clips seguidos (hasta 24 s). Ninguna llegaba a `runs/final` ni a `runs/final-d8`;
    sí a `runs/final3-xs`.
  - **TeamTrack distinto de cero**: 6 pistas retenidas en 5 clips, también entre jugadores
    que se mueven. Una llegaba a `runs/final-v3`.
- **Qué cambia.**
  - `scripts/teamtrack_to_long_csv.py` escribe los (0, 0) como NaN (`mask_missing`), y con
    `--from-cache` reconstruye los CSV desde las ventanas ya descargadas, sin Kaggle. Los
    CSV reconstruidos (`data/raw/teamtrack_fix0/`) son idénticos a los viejos salvo esos
    puntos (y el `match_id`, que los viejos tenían de antes de D3 y D10).
  - Los clips ya ingeridos se corrigen **como copias** (`scripts/mask_teamtrack_zeros.py`):
    `data/clips_tt_fix0` (894 clips de 4 s) y `data/clips_d8_tt_fix0` (408 de 8 s), con los
    mismos `clip_id` y `match_id`, así que todas las semillas se mantienen. Reingerir
    habría cambiado los identificadores de clip (llevan el `match_id` viejo) y con ellos
    el sorteo. Los 1.302 clips corregidos coinciden exactamente, en valores y en NaN, con
    los CSV reconstruidos. `data/clips` no se toca.
  - Pools nuevos: `data/pool_final4`, `data/pool_d8_final4` y `data/pool_final4_sb`. Son
    los de D17 con las entradas de TeamTrack apuntando a las copias; el resto de enlaces
    es idéntico.
  - **Comprobación general en los controles** (`ControlConfig.drop_frozen`, activa en todos
    los presets): una pista observada entera cuya posición es exactamente constante en la
    ventana se descarta antes de elegir los 10 jugadores, en cualquier fuente. Un clip
    sin pistas congeladas sale igual que antes, con los mismos sorteos (test). `prepare`
    guarda `frozen_dropped_by_sport` y `frozen_dropped_by_source` en `config.json`.
  - El preflight y el de A7b exigen `drop_frozen = True` y rechazan un conjunto con algún
    jugador conservado que no se mueve. `runs/final-v3` ya no lo pasa.
  - `runs/final` pasa a `runs/final-v3` y `runs/final-d8` a `runs/final-d8-v2`; los dos se
    preparan de nuevo con los mismos parámetros que en D17 (`runs/prepare_final4.log`,
    `runs/final-d8.prepare4.log`).
- **Conjuntos nuevos.**
  - `runs/final`: de 1.600 clips de entrada se conservan **1.532** (D17: 1.531). Balonmano
    398 (EIGD 332 + TeamTrack 66; antes 334 + 63). EIGD pierde los 2 clips congelados;
    TeamTrack gana 3 que antes se rechazaban por teletransporte (sus saltos a (0, 0)).
    Pistas descartadas: 28 en 2 clips de EIGD (los dos rechazados) y 1 en un clip de
    TeamTrack, que se conserva con 10 jugadores reales. Los 400 primeros son **los mismos
    400 clips** que en D17, con la misma composición (151 partidos); cambian los
    jugadores de 8 clips de balonmano de TeamTrack (los 7 afectados y uno cuya pista en
    (0, 0), no elegida, cambiaba el sorteo). En todo el conjunto cambian 23 clips, todos
    de balonmano de TeamTrack; los otros 1.506 comunes son idénticos byte a byte.
  - `runs/final-d8`: se conservan **1.065** (D17: 1.063); balonmano 397, fútbol 281. De
    los 300 primeros, 299 son los de D17.
  - **Ningún jugador conservado está quieto ni en (0, 0)** en `runs/final`, `runs/final-d8`
    ni en el conjunto entre fuentes (`runs/final4-xs`): auditoría sobre los clips
    controlados y repitiendo la selección desde los pools corregidos
    (`runs/analysis-2026-09-28/d18/`). Los dos preflights pasan.
  - Renders revisados a ojo de 3 clips afectados que siguen en los 400
    (`teamtrack-1st-train-1st_fisheye_480-510-00001`, `…2nd_fisheye_810-840-00002`,
    `…2nd_fisheye_270-300-00005`): 10 jugadores, sin punto en la esquina. Los puntos
    aislados que quedan son porteros en su área (se mueven 0,9-2,0 m en la ventana).
- **Especialistas sobre `runs/final`** (azar 0,25; mismos comandos que en D17;
  `runs/final/report_specialists.txt`, `specialists_breakdown.txt`, `contrasts.json`):

  | Especialista | 1.532 clips | 400 primeros | D17 (1.531 / 400) |
  |---|---|---|---|
  | MiniRocket `motion` | 0,83 [0,81; 0,86] | 0,84 [0,80; 0,88] | 0,84 / 0,85 |
  | MiniRocket `motion_shuffled` | 0,72 | 0,71 | 0,72 / 0,74 |
  | MiniRocket `kinematics` | 0,75 | 0,76 | 0,76 / 0,76 |
  | MiniRocket `kinematics_solo` | 0,71 | 0,72 | 0,70 / 0,69 |
  | DeepSets `motion` | 0,80 | 0,79 | 0,80 / 0,79 |
  | DeepSets `formation` | 0,55 | 0,55 | 0,54 / 0,51 |
  | Baseline cinemático | 0,68 | 0,68 | 0,67 / 0,67 |
  | Baseline `tempo` | 0,50 | 0,50 | 0,50 / 0,50 |
  | Baseline `nuisance` (degenerado, D11) | 0,24 [0,17; 0,30], kappa −0,02 | 0,23 | 0,25 / 0,24 |

  - Recall de MiniRocket `motion` por deporte: fútbol americano 0,92, baloncesto 0,82,
    balonmano 0,78 y fútbol 0,82. El cinemático acierta el fútbol americano en 0,90.
  - Por fuente, todos los clips: EIGD 0,80, Metrica 0,87, NFL 0,92, SkillCorner 0,81,
    SportVU 0,82, y en TeamTrack 0,58 (baloncesto, 12 clips), 0,68 (balonmano, 66) y 0,74
    (fútbol, 27). En los 400 primeros: EIGD 0,77, Metrica 0,93, NFL 0,93, SkillCorner
    0,77, SportVU 0,88; TeamTrack 0,33 (3), 0,81 (16) y 0,75 (8).
  - En los 19 clips afectados, MiniRocket `motion` acierta 15 (antes 11; los errores eran
    sobre todo «fútbol»). El punto quieto **restaba** acierto: no inflaba el balonmano.
  - Contrastes emparejados:

    | Contraste | Todos los clips | 400 primeros |
    |---|---|---|
    | MiniRocket `motion − motion_shuffled` | +0,11 [0,09; 0,13] | +0,13 [0,08; 0,18] |
    | MiniRocket `kinematics − kinematics_solo` | +0,04 [0,02; 0,06] | +0,035 [−0,01; 0,08], p = 0,15 |
    | DeepSets `motion − formation` | +0,25 [0,21; 0,29] | +0,245 [0,19; 0,31] |

  - `nuisance` contesta «baloncesto» en 906 de 1.532 clips.
  - **Cuánto de esto es la corrección y cuánto el reentrenamiento.** Sobre los 1.529
    clips comunes, 1.506 idénticos byte a byte, las etiquetas de MiniRocket coinciden con
    las de D17 solo en el 85-90 % de los clips (DeepSets `formation` 79 %, cinemático 97
    %, `tempo` 99,6 %): MiniRocket ajusta sus sesgos con los datos de entrenamiento, y
    con 23 clips distintos cambian todas las predicciones. `kinematics_solo` pasa de
    0,695 a 0,714 sobre clips idénticos. Los cambios de 1-3 puntos de la tabla son de ese
    tamaño. En particular, `kinematics − kinematics_solo` en los 400 primeros pasa de
    +0,065 (p = 0,008) a +0,035 (p = 0,15), sobre todo por ese ruido (8 de los 400 clips
    cambian), y se informa así: con los especialistas, esa diferencia no es estable en
    400 clips. En todos los clips sigue por encima de cero (+0,04 [0,02; 0,06]).
- **Especialistas sobre `runs/final-d8`** (azar 0,33): MiniRocket `motion` 0,87 [0,84;
  0,90] en los 1.065 clips y 0,86 en los 300 primeros; `motion_shuffled` 0,78 y 0,77;
  `motion − motion_shuffled` +0,08 [0,05; 0,11] y +0,09 [0,04; 0,14]; `nuisance` 0,33
  (degenerado, D11).
- **Entre fuentes** (`runs/final4-xs`, desde `data/pool_final4_sb`; TeamTrack solo en
  test): MiniRocket 0,97 dentro de las fuentes de entrenamiento y **0,96** sobre
  TeamTrack nunca visto (D17: 0,95); cinemático 0,73 y 0,73. TeamTrack aporta 417 clips
  (antes 420: 3 se quedan sin 10 jugadores observados). `drop_frozen` descarta además
  79 pistas de Metrica en 23 clips, que siguen dentro con otros jugadores.
- **Fuente dentro del fútbol** (`runs/final/source_id.json`): idéntico a D17, porque los
  clips de fútbol no cambian. Cinemático 0,70 [0,66; 0,72], MiniRocket 0,69 [0,66; 0,71],
  p exacto 1/66 en los dos.
- **Coste.** `scripts/estimate_run.py` da lo mismo (575,95 USD, 12,3 h) y `DRY_RUN=1 bash
  scripts/final_run.sh` sale con 0 y 28.600 llamadas por pasada: no cambia ni una celda
  ni un `--limit`.
- **No había datos de modelo.** Al escribir esto, `runs/final-v3/predictions/` y
  `runs/final-d8-v2/predictions/` (y los de los conjuntos anteriores) contienen solo
  especialistas: ningún modelo del pre-registro había respondido sobre ningún conjunto
  final. La corrección se decide por lo que dicen los datos de seguimiento, no por
  resultados.

- **Notas de los verificadores (añadidas en D19, 2026-09-28).** Lo que los controles de
  D18 dejan dentro, medido sobre los jugadores conservados de los conjuntos de D19
  (`scripts/analysis/track_notes.py`, que repite la selección desde el crudo; salidas en
  `runs/analysis-2026-09-28/d19/notes_*.json`):
  - **Retenciones parciales.** Pistas exactamente quietas una parte de la ventana: al
    menos 5 fotogramas (1 s), pero no la ventana entera, que es `drop_frozen`, ni 4 s o
    más, que es `drop_linear` (D19). Se ven como un jugador que se para, y en casi todas
    lo es.
    - Recuento en `runs/final`: NFL 10, TeamTrack 4, Metrica 3 y SkillCorner 1 (400
      primeros: NFL 3 y Metrica 1). En `runs/final-d8`: TeamTrack 10 y Metrica 2 (300
      primeros: TeamTrack 1).
    - Ejemplos:
      - `nfl-2021090900-3406-rand-00000` (posición 8): tres jugadores quietos 5-7
        fotogramas que en la ventana recorren 2,0-4,3 m. La NFL redondea a 0,01 yardas (9
        mm), así que un jugador parado repite la posición exacta.
      - `metrica-Sample_Game_1-00707` (139): 7 fotogramas quieto y 3,7 m recorridos.
      - `teamtrack-P3-train-P3_fisheye_0-30-00002` (1021): 9 fotogramas y 1,9 m.
    - El caso dudoso es `metrica-Sample_Game_1-00910` (1307, fuera de los 400): 10
      fotogramas quieto y 1,5 cm en toda la ventana. Es de la fuente cuyas retenciones
      largas D18 ya clasificó como artefacto, pero está por debajo de los 4 s de la regla
      y se queda.
  - **Pistas casi quietas (< 5 cm en 4 s) en la NFL y EIGD: probablemente reales.**
    - Recuento: pistas conservadas que recorren menos de 5 cm en la ventana sin ser
      constantes. En `runs/final`: NFL 7, EIGD 6, SportVU 4 y Metrica 1 (400 primeros:
      NFL 2 y EIGD 1). En `runs/final-d8`: EIGD 2.
    - EIGD: pasos medianos de 2-3 mm cada 0,2 s, el ruido del sensor Kinexon sobre un
      jugador parado (p. ej. `eigd-e8a35a-00-02-00-00043`, posición 62, 3,2 cm).
    - NFL: pasos de 0 o 9,14 mm, exactamente el redondeo de 0,01 yardas (p. ej.
      `nfl-2021090900-3110-rand-00000`, 1432, 1,8 cm).
    - SportVU: pasos de 2-5 mm.
    - Ninguna es exactamente constante ni recta, y todas salen de fuentes que en D18 no
      tenían ni una pista congelada. Son jugadores parados, y se quedan.
  - **La pista de baloncesto de TeamTrack de la posición 381**
    (`teamtrack-P3-val-P3_fisheye_30-60-00004`). En el crudo es una interpolación lineal
    de 5,4 s, y en la ventana su segunda diferencia es 0,00 mm en los 18 pasos. Se decide
    en D19: un tramo exactamente recto de ≥ 4 s es faltante (`drop_linear`). El clip se
    queda con 9 jugadores y sale de los 400.
  - **`fill_short_gaps` en los clips de 8 s.** La regla es la misma a 4 y a 8 s: huecos
    interiores de ≤ 3 fotogramas (0,6 s), rellenos en línea recta, sobre el clip entero
    antes de elegir la ventana, nunca en los extremos.
    - Fotogramas rellenos entre los jugadores conservados:
      - `runs/final`: SkillCorner 38 de 49.600 (0,08 %) y 0 en el resto.
      - `runs/final-d8`: SkillCorner 66 de 78.000 (0,085 %), TeamTrack 3 y EIGD 1.
    - La proporción es la misma a las dos duraciones.
    - Un relleno da como mucho 5 fotogramas rectos, muy por debajo de los 20 de
      `drop_linear`.
  - **Entre fuentes.** El 0,96 de arriba se midió con copias y pistas rectas dentro. En el
    conjunto de D19 (`runs/final5-xs`) es **0,95** (0,954).
- **Actualizado por D19 (2026-09-28).** Este conjunto es ahora `runs/final-v4` (y el de 8
  s, `runs/final-d8-v3`). Tenía 3 clips con un jugador duplicado (1 entre los 400) y 9
  pistas de TeamTrack interpoladas toda la ventana (1 entre los 400). Los números de arriba
  son los de ese conjunto; los del reconstruido están en D19.

### D19 (2026-09-28). Pistas duplicadas (un jugador con dos identificadores) y tramos interpolados

- **Qué encontró el verificador.** Algunas pistas crudas son el **mismo jugador con dos
  identificadores**. En SportVU, partido 0021500368, segmento 12, los jugadores 2755 y 2749
  comparten coordenadas en los 100 momentos. Su clip, `sportvu-0021500368-s012-00000`,
  estaba en la posición 389 del orden `interleave` de `runs/final` (dentro de los 400). Hay
  más en las posiciones 593 y 1281, y el verificador sospechaba de la 1261. En el conjunto
  entre fuentes (`runs/final4-xs`) contó 107 clips, también de Metrica. El modelo ve 10
  puntos, de los que 2 van siempre juntos: 9 jugadores. El segundo hallazgo: la pista de
  baloncesto de TeamTrack del clip de la posición 381 es, en el crudo, una interpolación
  lineal de 5,4 s.
- **Qué es una copia: la regla y su umbral.** Dos pistas observadas enteras cuya distancia
  **máxima** en la ventana es ≤ 5 cm son un jugador con dos identificadores. Se usa la
  máxima, no la media: el par tiene que ir junto toda la ventana. Para fijar el umbral se
  midió, en todos los clips del pool (`scripts/analysis/duplicate_tracks.py pool`, después
  de `fill_short_gaps` y `drop_frozen`, como en los controles), la distancia máxima en la
  ventana de cada par de pistas. La pregunta: ¿cuánto se acercan durante 4 s dos jugadores
  que de verdad son distintos?

  | Fuente (pool de 4 s) | Pares | Copias (máx. ≤ 5 cm) | Par más cercano que nunca coincide (máx.; mediana) |
  |---|---|---|---|
  | NFL | 259.875 | 0 | 0,53 m (0,46 m) |
  | EIGD | 119.725 | 0 | 0,55 m (0,41 m) |
  | SkillCorner | 614.338 | 0 | 0,51 m (0,40 m) |
  | TeamTrack | 130.776 | 0 | 0,45 m (0,37 m) |
  | SportVU | 1.046.951 | 114 pistas en 112 clips (105 idénticas al float) | 0,47 m (0,42 m) |
  | Metrica | 659.240 | 32 pistas en 29 clips (32 idénticas) | 0,31 m (0,21 m) |

  - «Nunca coincide» quiere decir ningún fotograma a ≤ 5 cm. Las copias van de 0 a 4,4 cm
    de máxima y coinciden en los 20 fotogramas.
  - Ningún par de jugadores distintos se queda a menos de **0,31 m** durante 4 s en
    ninguna fuente. En la NFL (líneas en contacto) el par más cercano está a 0,53 m.
    El umbral de 5 cm queda 6 veces por debajo del par real más cercano y 10 veces por
    debajo en la NFL: no puede tocar a dos jugadores reales. En el pool de 8 s hay 21
    copias en SportVU y 2 en Metrica; el par real más cercano está a 0,42 m (Metrica), y en
    las demás fuentes a 0,79-1,45 m.
  - En las cuatro fuentes sin copias ningún par pasa más de 4 de los 20 fotogramas a ≤ 5
    cm (casi siempre 1: un cruce). Salidas en `runs/analysis-2026-09-28/d19/`
    (`dup_pool_*.json`, `close_pairs_pool_final4.json`).
- **Lo que la regla no quita: coincidencias parciales.** En SportVU y Metrica, entre 1 y 20
  fotogramas a ≤ 5 cm hay un continuo, sin hueco (SportVU: 1.200 pares con 1 fotograma,
  178 con 5, 34 con 10, 12 con 19). Son pares que coinciden la mayor parte de la ventana y
  se separan en un extremo: una copia que empieza o acaba dentro de la ventana, o dos
  pistas que el tracker funde mientras dos jugadores están en contacto. La máxima de esos
  pares llega a 0,6 m en el extremo, por encima del par real más cercano (0,31 m), así que
  **ningún umbral sobre la máxima los separa**. El de la posición 1261 es uno de ellos
  (`sportvu-0021500314-s035-00000`): 19 de 20 fotogramas a ≤ 5 cm, 21 cm en el último.
  - Pares conservados a ≤ 5 cm en al menos la mitad de la ventana (tras D19):
    - `runs/final`: 4 clips (posiciones 985, 1261, 1439 y 1460), ninguno entre los 400.
    - `runs/final-d8`: 3 clips (539, 803 y 851), ninguno entre los 300.
    - `runs/final5-xs`: 212 de 32.167 clips (SportVU 191, Metrica 21).
  - En `runs/final5-xs`, 2 de esos pares (`metrica-Sample_Game_1-00464` y
    `sportvu-0021500496-s016-00002`, 10,5 y 8,1 cm de máxima en crudo) quedan a ≤ 5 cm
    después del suavizado.
  - Esta desviación aplica la regla de la ventana entera. La parcial queda medida y
    abierta (hueco A16 en `gaps.md`).
- **Tramos interpolados: decisión con datos.** La posición 381 era
  `teamtrack-P3-val-P3_fisheye_30-60-00004`: una de sus pistas tiene segunda diferencia
  0,00 mm en los 18 pasos de la ventana. Es una línea exactamente recta a velocidad
  exactamente constante durante los 4 s. Medido en todo el pool
  (`scripts/analysis/linear_tracks.py pool`; tolerancia de 0,1 mm por paso a 5 Hz, cuando
  el redondeo de float32 a escala de campo es de ~0,01 mm y un jugador medido tiene de
  milímetros a centímetros):

  | Fuente (pool de 4 s) | Pistas | Rectas toda la ventana | Fotogramas en tramos rectos ≥ 1 s |
  |---|---|---|---|
  | TeamTrack (baloncesto / balonmano / fútbol) | 698 / 6.019 / 8.504 | 3 / 57 / 34 | 6,3 % / 9,4 % / 2,5 % |
  | NFL, SportVU, EIGD, SkillCorner, Metrica | 434.859 | **0** | 0,02-0,28 % |

  - TeamTrack anota posiciones clave e interpola entre ellas. En las otras cinco fuentes,
    ninguna de 434.859 pistas es recta los 4 s, y solo 22 llegan a 15-19 fotogramas rectos
    (quietas en Metrica y NFL, o SportVU).
  - En el pool de 8 s, las pistas con ≥ 4 s rectos son:
    - TeamTrack: 101 (98 en movimiento y 3 quietas);
    - EIGD: 44, todas quietas (los arranques congelados de D18, que una ventana de 8 s
      coge a medias);
    - Metrica: 18 (15 quietas y 3 que derivan a < 0,03 m/s);
    - SportVU y SkillCorner: 0.

    Todo lo que la regla quita en otras fuentes es el artefacto que D18 ya quitaba.
  - **Decisión:** un tramo exactamente recto de **4 s o más** dentro de la ventana es
    interpolado, no medido. Se trata como faltante y la pista no se puede conservar
    (`drop_linear`, `linear_min_s = 4.0`). En la ventana de 4 s es la ventana entera, que
    nunca pasa fuera de TeamTrack. En la de 8 s coge además las retenciones de ≥ 4 s.
  - Los tramos rectos más cortos siguen dentro, igual que las retenciones parciales
    (notas en D18). `drop_frozen` es el caso de velocidad cero en la ventana entera y
    sigue contándose aparte.
- **Qué cambia** (`f42e8ea`; scripts de análisis en `3a80332`; este texto en `20f421e`):
  - `ControlConfig.drop_linear` y `drop_duplicates`, activos en todos los presets, con
    `linear_min_s = 4.0` y `duplicate_tol_m = 0.05`. Después de `drop_frozen` y antes de
    elegir los 10 se quitan primero los tramos rectos y luego las copias (se conserva la de
    índice menor).
  - `config.json` guarda `linear_dropped_by_*` y `duplicate_dropped_by_*` (por deporte y
    por fuente: pistas, clips con alguna y clips rechazados después).
  - Un clip sin pistas rectas ni copias sale igual que antes, con los mismos sorteos
    (tests).
  - Los dos preflights exigen los cuatro valores y rechazan un conjunto con dos jugadores
    conservados a ≤ 5 cm en toda la ventana, medido en metros (unidades × `space_scale`).
    `runs/final-v4` ya no lo pasa.
  - Los tramos rectos se comprueban en la configuración y en la auditoría sobre el crudo,
    no en los clips preparados, porque el suavizado curva los extremos de una ventana
    recta.
  - Los paseos sintéticos de los tests van ahora por arcos amplios, porque una recta
    exacta es, por definición, una interpolación.
- **Afectados antes de D19** (reproduciendo la selección desde el crudo):
  - `runs/final` (ahora `runs/final-v4`):
    - 3 clips con una copia conservada: 389 (dentro de los 400), 593 y 1281.
    - 9 pistas de TeamTrack rectas toda la ventana: 3 de baloncesto, 4 de balonmano y 2 de
      fútbol. Una estaba entre los 400: la 381.
  - `runs/final-d8` (ahora `runs/final-d8-v3`): ninguna copia; 16 pistas de TeamTrack con
    ≥ 4 s rectos, 6 de ellas entre los 300.
  - `runs/final4-xs`: 119 pares copiados entre los conservados (SportVU 114, Metrica 5; el
    verificador contó 107).
- **Conjuntos nuevos** (mismos comandos que en D18: `runs/prepare_final5.log`,
  `runs/final-d8.prepare5.log`, `runs/final5-xs.log`).
  - `runs/final`: **1.526** clips (D18: 1.532), 173 partidos.
    - SportVU pierde los 3 clips copiados, que se quedan con 9 jugadores; TeamTrack pierde
      3 de baloncesto (de 12 a 9), que se quedan con 9 al quitar la pista recta.
    - En otros 7 clips de TeamTrack (5 de balonmano y 2 de fútbol) cambian los jugadores.
      Los otros 1.519 son idénticos a los de D18.
    - **Los 400 primeros:** salen 2 y entran 2, en la misma posición y de la misma fuente.
      Sale el 381 (`teamtrack-P3-val-P3_fisheye_30-60-00004`) y entra
      `teamtrack-P3-test-P3_fisheye_60-90-00000`. Sale el 389
      (`sportvu-0021500368-s012-00000`) y entra `sportvu-0021500368-s107-00014`. Los otros
      398 son idénticos, con la misma composición por fuente y los mismos 151 partidos.
  - `runs/final-d8`: **1.064** (D18: 1.065).
    - `drop_linear` quita 24 pistas de TeamTrack en 18 clips. Sale uno de baloncesto y
      cambian los jugadores de 16.
    - Ninguna copia entra en el sorteo.
    - **Los 300 primeros:** sale 1 (el 189, `teamtrack-P3-val-P3_fisheye_30-60-00002`) y
      entra `teamtrack-P3-train-P3_fisheye_270-300-00001`. Cambian los jugadores de otros 4
      de TeamTrack. La composición no cambia (51 partidos).
  - `runs/final5-xs` (entre fuentes, desde `data/pool_final4_sb`): 32.167 clips (D18:
    32.280).
    - SportVU pierde 112 clips por copias.
    - Metrica conserva sus 29 clips con copia, sin la copia.
    - TeamTrack aporta 416 (65 de baloncesto y 351 de fútbol; D18: 68 y 349). 3 clips
      salen por pistas rectas. Entran 2 de fútbol que antes se rechazaban por
      teletransporte: al quitar su pista recta cambia el sorteo de los 10, y el jugador del
      salto (36 y 6.797 m/s) ya no sale elegido.
  - Preflights en verde (`preflight` y `preflight-a7b`).
- **Auditoría** (recalculada sobre los clips preparados y repitiendo la selección desde el
  crudo, en metros):
  - En `runs/final`, `runs/final-d8` y `runs/final5-xs`, entre los jugadores conservados:
    - 0 pares a ≤ 5 cm en toda la ventana en crudo;
    - 0 pistas exactamente constantes;
    - 0 puntos en (0, 0);
    - 0 pistas con ≥ 4 s rectos.
  - Paso más rápido: 11,02, 11,40 y 11,98 m/s (techo 12).
  - Después del suavizado, 0 pares a ≤ 5 cm en `runs/final` y `runs/final-d8`, y los 2 de
    `runs/final5-xs` de arriba.
  - `frozen_tracks.py items` no marca ningún clip.
  - Salidas en `runs/analysis-2026-09-28/d19/` (`dup_items_*_after.json`,
    `frozen_items_*_after.json`, `notes_*.json`, `sample_diff_*.json`).
- **Especialistas sobre `runs/final`** (azar 0,25; mismos comandos que en D18;
  `runs/final_specialists5.log`, `report_specialists.txt`, `specialists_breakdown.txt`,
  `contrasts.json`):

  | Especialista | 1.526 clips | 400 primeros | D18 (1.532 / 400) |
  |---|---|---|---|
  | MiniRocket `motion` | 0,83 [0,81; 0,85] | 0,83 [0,80; 0,87] | 0,83 / 0,84 |
  | MiniRocket `motion_shuffled` | 0,72 | 0,72 | 0,72 / 0,71 |
  | MiniRocket `kinematics` | 0,75 | 0,74 | 0,75 / 0,76 |
  | MiniRocket `kinematics_solo` | 0,71 | 0,68 | 0,71 / 0,72 |
  | DeepSets `motion` | 0,80 | 0,80 | 0,80 / 0,79 |
  | DeepSets `formation` | 0,53 | 0,53 | 0,55 / 0,55 |
  | Baseline cinemático | 0,68 | 0,67 | 0,68 / 0,68 |
  | Baseline `tempo` | 0,51 | 0,50 | 0,50 / 0,50 |
  | Baseline `nuisance` (degenerado, D11) | 0,22 [0,15; 0,30], kappa −0,04 | 0,21 | 0,24 / 0,23 |

  - Recall de MiniRocket `motion` por deporte: fútbol americano 0,92, baloncesto 0,79,
    balonmano 0,79 y fútbol 0,83. El cinemático acierta el fútbol americano en 0,90.
  - Por fuente, todos los clips: EIGD 0,81, Metrica 0,85, NFL 0,92, SkillCorner 0,83,
    SportVU 0,79, y en TeamTrack 0,67 (baloncesto, 9 clips), 0,67 (balonmano, 66) y 0,78
    (fútbol, 27). En los 400 primeros: EIGD 0,81, Metrica 0,93, NFL 0,90, SkillCorner
    0,78, SportVU 0,82; TeamTrack 1,00 (3), 0,69 (16) y 0,88 (8).
  - Contrastes emparejados (`scripts/analysis/contrasts.py`):

    | Contraste | Todos los clips | 400 primeros |
    |---|---|---|
    | MiniRocket `motion − motion_shuffled` | +0,11 [0,09; 0,13] | +0,12 [0,07; 0,16] |
    | MiniRocket `kinematics − kinematics_solo` | +0,04 [0,02; 0,07] | +0,06 [0,01; 0,10], p = 0,011 |
    | DeepSets `motion − formation` | +0,27 [0,23; 0,31] | +0,27 [0,21; 0,33] |

  - `nuisance` contesta «baloncesto» en 670 de 1.526 clips.
  - `kinematics − kinematics_solo` en los 400 primeros vuelve a +0,06 (D17: +0,065; D18:
    +0,035, p = 0,15) con 398 clips idénticos a los de D18. Confirma lo que decía D18: con
    los especialistas esa diferencia no es estable en 400 clips, porque MiniRocket se
    reentrena y cambia sus predicciones. En todos los clips se mantiene en +0,04.
- **Especialistas sobre `runs/final-d8`** (azar 0,33; `runs/final-d8.spec5.log`):
  - MiniRocket `motion`: 0,87 [0,84; 0,90] en los 1.064 clips y 0,89 [0,85; 0,92] en los
    300 primeros (D18: 0,87 y 0,86).
  - `motion_shuffled`: 0,79 y 0,77.
  - `motion − motion_shuffled`: +0,08 [0,05; 0,11] y +0,11 [0,07; 0,16] (D18: +0,08 y
    +0,09).
  - `nuisance`: 0,34 (degenerado, D11).
- **Entre fuentes** (`runs/final5-xs`, TeamTrack solo en test;
  `cross_source_teamtrack_s0.json`):
  - MiniRocket: 0,97 dentro de las fuentes de entrenamiento y **0,95** sobre TeamTrack
    nunca visto (0,954; D18: 0,964).
  - Cinemático: 0,73 y 0,73.
  - Lo corrige TeamTrack sin sus pistas rectas: la tabla del test pierde 3 clips de
    baloncesto, y la confusión de MiniRocket queda en 7 de 65 baloncesto leídos como fútbol
    y 12 de 351 fútbol leídos como baloncesto.
- **Fuente dentro del fútbol** (`runs/final/source_id.json`): los clips de fútbol no
  cambian y el resultado es el mismo. Cinemático 0,70 [0,66; 0,72] y MiniRocket 0,69
  [0,66; 0,71], p exacto 1/66 en los dos. Solo cambian los recuentos de baloncesto del
  JSON.
- **Coste.** `scripts/estimate_run.py` da lo mismo: 575,95 USD y 12,3 h.
  `DRY_RUN=1 bash scripts/final_run.sh` sale con 0 y 28.600 llamadas por pasada.
- **No había datos de modelo.** Al escribir esto, los directorios `predictions/` de
  `runs/final`, `runs/final-d8`, `runs/final-v1` a `runs/final-v4` y `runs/final-d8-v1` a
  `runs/final-d8-v3` contienen solo especialistas (`baseline__*`, `minirocket__*`,
  `deepsets__*`), y ninguno tiene `logs/` del lanzador. Ningún modelo del pre-registro ha
  respondido sobre ningún conjunto final. La corrección se decide por lo que dicen los
  datos de seguimiento y los datos crudos, no por resultados.


### D20 (2026-09-28). Decisión sobre A16 y matices de D19, antes de lanzar

Sin datos de ningún modelo en `runs/final` ni en `runs/final-d8` (solo especialistas).

- **A16 (coincidencias parciales de dos pistas): no se aplica ninguna regla nueva.** Contando
  todos los pares del pool, no solo los conservados, el máximo de fotogramas a ≤ 5 cm es 6 de 20
  en los 400 primeros y 9 de 40 en los 300 de A7b. Una regla de «la mitad de la ventana o más»
  no cambiaría ningún clip que vean los modelos. Solo tocaría 6 clips de `runs/final` fuera de
  los 400 (985, 1023, 1135, 1261, 1439, 1460) y 3 de `runs/final-d8` fuera de los 300. Como los
  especialistas se comparan con los modelos sobre los mismos 400 clips (D4), no afecta a
  ninguna comparación pre-registrada.
- **Fusiones breves que quedan dentro de la muestra.** En SportVU y Metrica hay pares a 0,0 m
  durante 1 a 6 fotogramas (16 pares en 7 clips de los 400: posiciones 77, 97, 193, 265, 295,
  297 y 301; en A7b, 4 de los 300, hasta 9 de 40 fotogramas). En esos fotogramas el modelo ve
  dos jugadores en un solo punto. Es el comportamiento de fusión del tracker cuando dos
  jugadores están en contacto. Es breve, casi solo en baloncesto, y aparece igual en las dos
  condiciones de cada contraste emparejado.
- **Matiz a la evidencia de D19 sobre tramos rectos.** La tolerancia de 0,1 mm está por debajo
  del cuanto de coordenadas de NFL (9,14 mm), SkillCorner (10 mm), EIGD (1 mm) y Metrica (~1 mm).
  Por eso el «0 de 434.859 pistas rectas fuera de TeamTrack» es un artefacto de la prueba, no
  una evidencia. Con una tolerancia igual para todas (4 mm, en movimiento), el 0,55 % de las
  pistas de Metrica son rectas durante los 4 s, probablemente estimaciones fuera de cámara;
  SportVU y EIGD dan 0 %. En los 400 primeros ninguna pista conservada es recta a 4 mm toda la
  ventana. En A7b, la posición 26 (`metrica-Sample_Game_1-00452`) conserva 2 pistas rectas los
  8 s. Son jugadores reales con posición estimada, no fantasmas, y A7b es exploratoria.
- **Tiempo entre instantáneas.** El prompt de hoja y de texto dice «0,5 s». Con 8 instantáneas
  sobre 20 fotogramas a 5 Hz, el paso alterna entre 0,4 y 0,6 s (media 0,5). No se cambia el
  prompt: rehacer los ítems solo por esto no compensa, el paso es el mismo en todos los
  deportes y condiciones, y queda anotado aquí.
- **Errata:** `gaps.md` C4 citaba `runs/final2-xs` para el 0,95 del conjunto final; es
  `runs/final5-xs`.

### D21 (2026-09-28, POSTERIOR a los datos). Margen de razonamiento en la ruta Anthropic

Escrita **después** de ver todas las respuestas de la corrida final. Los números
pre-registrados no cambian: `runs/final` queda intacto y es la fuente del análisis
primario. Lo corregido se informa solo como sensibilidad.

- **Qué pasó.** El §4 pre-registra `--max-tokens 1024` «más el margen de razonamiento
  del backend, 16.384». Las rutas de OpenAI (Azure) y Gemini (Vertex) sumaban ese margen.
  La ruta Anthropic (`vertex-anthropic:`, que usan Opus 5.5 y Sonnet 5) **no lo sumaba**:
  enviaba `max_tokens = 1024`. Claude 5.x razona de forma adaptativa y ese razonamiento
  cuenta contra `max_tokens`. Cuando razona largo, se agota el presupuesto antes de la
  respuesta, y la respuesta sale vacía o cortada. Es un bug del harness en el sentido del
  §9: el ajuste pre-registrado no se aplicaba en una ruta.
- **Evidencia** (`scripts/analysis/token_ceiling.py`, sobre `usage.output_tokens` de cada
  fila; `runs/analysis-final/token_ceiling.{txt,json}`):
  - Sonnet 5 `motion/text`: en la pasada 1, 102 de 400 respuestas no se pudieron leer (91
    sin JSON y 11 con JSON mal formado). La pasada 2, con los mismos ajustes (§8), recuperó
    62. Quedan **40 errores** (10 %): 36 sin JSON y 4 con JSON mal formado. **Los 40
    acaban exactamente en 1.024 tokens de salida**, y ninguna respuesta válida llega a
    1.024. Otras 14 respuestas válidas llegan a 924 tokens o más, y 6 de ellas están entre 1.000
    y 1.019, todas equivocadas. Los errores se reparten así: 14 de fútbol americano, 11 de
    baloncesto, 9 de fútbol y 6 de balonmano.
  - En el resto de celdas de la ruta Anthropic no hay ninguna fila en el tope, tampoco en
    A7b. Opus 5.5 `motion/text` llega a 988 tokens (3 filas en 924 o más),
    `kinematics_solo/sheet` a 975 y `kinematics/sheet` a 967. Todas las demás celdas se
    quedan por debajo de 870. Ninguna otra celda de la ruta termina con errores. Sonnet
    `kinematics/sheet` tuvo 1 JSON mal formado («Extra data») en la pasada 1, que se
    recuperó en la pasada 2.
  - Lo que no se puede descartar: el razonamiento adaptativo podría ajustar su longitud al
    presupuesto que recibe. Entonces el tope habría acortado el razonamiento también en
    celdas donde ninguna respuesta se cortó. No se ha medido.
- **Arreglo** (commit `d6a3833`, 2026-09-28 23:28, después del final de la corrida):
  `_messages_payload` envía `max_tokens = 1024 + REASONING_HEADROOM`, como las otras
  rutas, con un test. Ninguna fila de `runs/final` ni de `runs/final-d8` se ha re-corrido
  con el arreglo.
- **Sensibilidad** (`runs/final-sens`, una copia con los ítems enlazados y las predicciones
  copiadas): solo se re-corrieron las **40 filas con error** de Sonnet 5 `motion/text`,
  con el arreglo. Se recuperan 39 y queda 1 JSON mal formado. De las 39, aciertan 18 (6 de
  fútbol americano, 6 de baloncesto y 6 de fútbol). Las recuperadas usan una mediana de
  1.122 tokens de salida, con un máximo de 8.693. Coste: 1,40 USD, unos 4 minutos.
  Cambian esas 40 filas y ninguna otra (comparado fichero a fichero con `runs/final`).
  - Exactitud de la celda: 0,28 → 0,33 [0,23; 0,43]. Exactitud corregida por prior: 0,32
    → 0,38.
  - `text_vs_image` de Sonnet: d +0,005 → +0,05 [−0,005; 0,12], p 0,90 → 0,085, p_holm
    1,00 → 1,00.
  - **No cambia ninguna decisión de Holm ni ninguna clasificación del §6.**
- **Los errores no eran al azar.** Las 39 recuperadas aciertan el 46 %, frente al 31 %
  (113 de 360) de las respuestas válidas de la celda original. Además se concentran en
  fútbol americano y baloncesto. Contar esos errores como fallo (§8) empujaba `motion/text`
  hacia abajo, en la dirección de H3. Aun así, con la sensibilidad H3 sigue sin evidencia.
- **Conflicto con el §9: no se cumple.** El §9 exige, ante un bug del harness visto
  después de los datos, repetir **entera** la celda afectada e informar las dos versiones.
  Aquí solo se re-corrieron las 40 filas con error, no las 400. Las 360 filas válidas se
  obtuvieron con el tope, y algunas podrían cambiar con el margen, sobre todo las 14 que
  llegan a 924 tokens o más. La sensibilidad parcial es, por tanto, una aproximación a la
  segunda versión, no la segunda versión. Falta re-correr las 400 filas de Sonnet 5
  `motion/text` con el arreglo, en una copia, como segunda versión informada al lado de
  la pre-registrada. El coste medido de la celda da unos 6 USD y minutos de reloj. Esta
  fase de análisis no tenía autorizado ningún gasto en API, así que queda abierta como
  hueco A17 en [`gaps.md`](gaps.md). Las demás celdas de la ruta Anthropic no muestran el
  efecto del bug (ninguna fila en el tope) y no se re-corren. Queda dicho arriba lo que eso
  no descarta.
- **Qué número manda.** Los pre-registrados, con los 40 errores contados como fallo y la
  celda marcada (`!`, 40/400), son los primarios. Los de `runs/final-sens` se citan
  siempre como sensibilidad, al lado de los pre-registrados
  ([`results-final.md`](results-final.md)).

## Anexo A. Dimensionado (del piloto 4, 400 clips, 55 partidos)

La semianchura del IC al 95 % de los contrastes primarios fue de 0,04 a 0,10 (EE ≈
0,02-0,05). El dataset final tiene más partidos por deporte, así que se esperan IC algo
más estrechos. Con esos EE, el primer escalón de Holm (α/20 = 0,0025, z ≈ 3,0)
detecta diferencias de unos 0,07-0,15 y el último (0,05) de unos 0,04-0,10. Las
réplicas (200 clips) solo sirven para estimar el ruido de muestreo del modelo, no para
el análisis primario.

## Anexo B. Coste y tiempo estimados

`scripts/estimate_run.py` (recalculado el 2026-09-28 después de D14, D17, D18 y D19, sin cambios:
575,95 USD y 12,3 h; el coste depende del número de llamadas y de los tokens del piloto, no
de qué clips salen), con los tokens medidos en `runs/pilot4-strict` (y en
`runs/pilot-strict` para Gemini 3.1) y precios de lista: unos **576 USD** en total, 508
de las celdas pre-registradas y 68 de A7b. Gemini 3.1 se lleva unos 355 (40 de A7b), por
sus ~5.000 tokens de razonamiento por ítem. Siguen gpt-5.6-sol ~109 (15 de A7b), Opus 5.5
~60 (7; estimado con los tokens de Sonnet 5), Sonnet 5 ~36 (4), terra ~16 (2) y Jev < 1.
Unas **12 h** de reloj, marcadas por Gemini (los modelos corren en paralelo). A7b se
estima con los tokens de la hoja a 4 s con 4 opciones: la hoja es la misma imagen de 8
fotogramas y con 3 opciones el prompt es algo más corto, así que la estimación queda un
poco alta. Los precios son supuestos, no facturas, y no incluyen los reintentos de las
llamadas con error (`PASSES=2`), que también se facturan. El orden de las celdas en el
estimador es el del lanzador (D14): por modelo, las 5 primarias, las secundarias
pre-registradas y A7b al final.

## Anexo C. Requisitos antes de lanzar

- `runs/final` construido (con `--reprs sheet,trails,text,video`, que necesita `.[video]`)
  y el preflight en verde, con los valores fijados de D13, `drop_frozen` (D18) y
  `drop_linear` / `drop_duplicates` (D19). Lo está desde el 2026-09-28 (reconstruido en D19
  desde `data/pool_final4`).
- `runs/final-d8` construido y `preflight-a7b` en verde. También lo está (reconstruido en
  D19 desde `data/pool_d8_final4`), y el lanzador marca su `plan.json` como exploratorio (D15).
- Opus 5.5 habilitado en el Model Garden de Vertex (hueco A6; lo hace Javier). Si no lo
  está, `report` lo saca de los contrastes y de la familia (D4).
- `gcloud` autenticado. Claves de Azure y OpenRouter en sus ficheros, que el
  lanzador lee sin imprimirlas.

```bash
DRY_RUN=1 bash scripts/final_run.sh     # comprobar los comandos
bash scripts/final_run.sh               # la corrida (reanudable)
```
