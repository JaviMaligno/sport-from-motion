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

- **Conjunto**: `runs/final`, reconstruido el 2026-09-28 desde `data/pool_final2`
  (37.613 clips; D8), antes de cualquier llamada a un modelo. Preset `strict_smooth`
  (suavizado gaussiano, σ = 2 fotogramas), ventanas de 4 s a 5 Hz (20 fotogramas),
  `--per-sport 400`, las 5 condiciones y las representaciones `sheet`, `trails`, `text`
  y `video`. Además del preset:
  - **Jugadores**: N = 10 **elegidos al azar** en cada clip (`--player-mode random
    --n-players 10`), no los 10 más centrales (hueco A14; D5 y D8).
  - **Techo de velocidad**: se rechaza el clip si algún jugador conservado da un paso de
    más de 12 m/s (`max_speed_ms`; D2).
  - **Fútbol americano**: un clip por jugada, que empieza en una fase **aleatoria**
    después del snap (D1).
  - **Balonmano**: 6 partidos. Las dos partes del partido de TeamTrack forman un solo
    grupo, `tt-handball` (D3).

  Los parámetros exactos quedan en `runs/final/config.json`: `controls`, clips de entrada
  y conservados por deporte y por fuente, y rechazos por motivo.
- **Fuentes** (clips conservados en `runs/final` y partidos):

  | Deporte | Fuentes (clips) | Clips | Partidos |
  |---|---|---|---|
  | Fútbol americano | NFL Big Data Bowl 2023, semanas 1-8, fase aleatoria (D1) | 400 | 122 |
  | Baloncesto | NBA SportVU 2015-16 (386) + TeamTrack (12) | 398 | 31 + 1 |
  | Balonmano | EIGD-H, 5 partidos de la HBL medidos con Kinexon (334) + TeamTrack (59) | 393 | 5 + 1 |
  | Fútbol | SkillCorner Open Data, solo posiciones detectadas (224) + Metrica (57) + TeamTrack (51) | 332 | 10 + 2 + 2 |

  De 1.600 clips de entrada (400 por deporte) se conservan 1.523. Rechazos: SkillCorner
  62 por número de jugadores (22 %, el único aviso de `prepare`), TeamTrack 12 por
  teletransporte (9 %), SportVU 1 por teletransporte y 1 por jugadores, Metrica 1 por
  teletransporte.
- **Preflight**. El lanzador se niega a arrancar (`scripts/run_plan.py preflight`) si el
  preset no es `strict_smooth`; si en `controls` no están `player_mode = random`,
  `n_players = 10` y un `max_speed_ms`; si algún clip de fútbol americano no lleva la
  etiqueta `random_phase`; si algún deporte tiene menos de 100 clips; si faltan los
  prompts informados, o si falta una representación planificada.
- **Opciones**: las de `config.json` (`candidates`), es decir, los 4 deportes sin
  distractores. Azar = 1/|candidates| = 0,25.
- **Muestra**: los **400 primeros clips** del orden `interleave` (reparto por
  deporte y, dentro de cada deporte, por partido; la clave depende solo del clip). Son
  100 por deporte y **los mismos clips en todas las celdas**. Cubren 152 partidos
  (fútbol americano 100, baloncesto 32, fútbol 14, balonmano 6). Por fuente: NFL 100,
  SportVU 97, EIGD 84, SkillCorner 71, Metrica 14 y TeamTrack 34 (3 de baloncesto, 16 de
  balonmano y 15 de fútbol). Las réplicas 2 y 3 usan los 200 primeros (50 por deporte),
  que son un subconjunto de los 400.
- **Celda A7b, clips de 8 s (secundaria y exploratoria; D7)**: `runs/final-d8`, con los
  mismos controles (`strict_smooth`, N = 10 al azar, 12 m/s) pero ventanas de **8 s**
  (`--n-frames 40`) y **3 deportes**: baloncesto, balonmano y fútbol (azar 1/3). No hay
  fútbol americano porque solo 42 clips NFL de mitad de jugada llegan a 8 s. Sale de
  `data/pool_d8_final2`, que son los clips de 8 s del barrido A7 con el balonmano de
  TeamTrack reagrupado como en D3. Se conservan 1.069 clips (baloncesto 387, balonmano
  394, fútbol 288) de 52 partidos. Muestra: los **300 primeros** del orden `interleave`
  (100 por deporte, 52 partidos). Celdas: `motion/sheet` y `motion_shuffled/sheet`, con
  prompt neutro. La hoja sigue teniendo 8 fotogramas: a 8 s quedan a unos 1,1 s uno de
  otro (a 4 s, a unos 0,55 s). Tiene su propio preflight: `scripts/run_plan.py
  preflight-a7b`.

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
luego A7b (`motion/sheet` y `motion_shuffled/sheet` a 8 s, todos menos Jev); luego
`kinematics/sheet`, `kinematics_solo/sheet`, `motion/trails`, vídeo (Gemini) y
réplicas. Si la corrida se corta, lo que falte es secundario.

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
.venv/bin/motion-sport report --items runs/final-d8-view --n-boot 10000 --json > runs/final-d8/report_a7b.json
```

`--clips-from-models` restringe los especialistas de la vista a los clips que se
preguntaron a los modelos (D4).

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
con más de un 50 % de errores no recuperados, también (sección 8). Las celdas
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
  como referencia de «la señal está». Se entrenan con CV agrupada por partido sobre
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
  (siguiente punto). Se fija antes de cualquier dato de modelo (sección 11).
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
final**. Cuando se escribieron, ningún modelo había respondido sobre `runs/final`, ni sobre
el conjunto anterior (`runs/final-v1`) ni sobre `runs/final-d8`: los tres directorios
`predictions/` contienen solo especialistas. Los especialistas (MiniRocket, DeepSets y los
baselines) sí se corrieron, sobre el conjunto viejo y sobre el nuevo. No son datos de
modelo en el sentido de este pre-registro (sección 7): sirvieron para encontrar los
problemas y para comprobar el conjunto reconstruido, y sus números están abajo. Las
desviaciones cambian cómo se construye el conjunto de ítems, añaden una celda exploratoria
(A7b) y corrigen texto. No cambian las hipótesis, las celdas primarias, los modelos ni la
familia de Holm.

Todos los cambios desde el commit del pre-registro (`745f699`, 2026-09-27 21:00):

| # | Qué cambia | Por qué | Evidencia | Commit |
|---|---|---|---|---|
| D1 | NFL: un clip por jugada, en fase aleatoria tras el snap | con el recorte fijo, casi todos los clips NFL empezaban en la misma fase de la jugada | verificador: 381 de 400 clips de `runs/final-v1` empezaban justo 1,0 s después del snap | `eda4e4a` |
| D2 | Techo de velocidad de 12 m/s | los teletransportes del tracker son una firma de la fuente, muy desigual | barrido de velocidades por fuente; 7 clips graves en `runs/final-v1` (hasta 6.797 m/s), 1 dentro de los 400 | `02e0d0c` |
| D3 | El balonmano de TeamTrack es un partido | sus dos partes no son grupos independientes | verificador: al fusionarlas, el recall de MiniRocket en ese corte baja de 0,614 a 0,544 | `cd52ef4` |
| D4 | `report` y vista primaria: modelos que no corren, filas que faltan, especialistas sobre los mismos clips | casos que el pre-registro no resolvía de forma mecánica | verificador del pre-registro | `5bbdd24` |
| D5 | El preflight exige el conjunto nuevo | que no dependa de la memoria | `runs/final-v1` no lo pasa | `b2dffa2` |
| D6 | `prepare --n-frames`, relleno de series de MiniRocket; ablaciones A7 y A8 | herramientas del barrido de duración | sin efecto sobre la corrida (20 fotogramas) | `a722feb`, `b1306f5` |
| D7 | Celda A7b: clips de 8 s, exploratoria | a 8 s los especialistas mejoran (C11) | especialistas sobre `runs/final-d8` | este commit |
| D8 | Conjunto reconstruido con N = 10 al azar; secciones 3, 4, 7, 8 y 10 al día | la selección central quitaba señal de forma desigual (A8) y la sección 3 no describía las fuentes reales | A8 (C10); especialistas sobre el conjunto nuevo | este commit |
| D9 | Corregida la frase sobre la resolución del bootstrap | era falsa | 2/2.001 ≈ 0,001 < 0,0025 | este commit |

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
  `runs/final-d8` (sección 3), justo después de sus celdas primarias. Son 3.000 llamadas
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
  corregir. En ese informe `order` sale con la etiqueta de contraste primario y con un
  Holm propio solo porque es la misma pareja de celdas; esa etiqueta no aplica aquí.
- **Límites conocidos.** El `nuisance` a 8 s acierta al azar (0,35 [0,20; 0,47], kappa
  0,00), pero porque contesta «balonmano» en 829 de 1.069 clips: es un clasificador
  degenerado, y eso es una prueba más débil de que no quedan atajos que un nuisance que
  reparte sus respuestas. En los especialistas, `motion − motion_shuffled` a 8 s es +0,07
  [0,03; 0,10]; en `abl-d8_noaf`, con selección central y sin techo de velocidad, era
  +0,12.

### D8 (2026-09-28). Conjunto de ítems reconstruido; secciones 3, 4, 7, 8 y 10 al día

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
  0,25): MiniRocket `motion` 0,82 [0,79; 0,85] (v1: 0,75), `motion_shuffled` 0,70,
  `kinematics` 0,75, `kinematics_solo` 0,72; DeepSets `motion` 0,81 y `formation` 0,54;
  baseline cinemático 0,69, `tempo` 0,53 y `nuisance` 0,29 [0,21; 0,39] (kappa 0,04).
  En los 400 primeros: MiniRocket `motion` 0,80 [0,75; 0,85] y `nuisance` 0,31 [0,22;
  0,41]. MiniRocket `motion − motion_shuffled` +0,12 [0,09; 0,15]. Entre fuentes
  (TeamTrack fuera del entrenamiento, `runs/final2-xs`), MiniRocket da 0,97 dentro de las
  fuentes y **0,95** en TeamTrack, frente a 0,94 → 0,86 en el conjunto viejo.
- **Límites conocidos.**
  - El `nuisance` nunca contesta «fútbol» y contesta «balonmano» en la mayoría de los
    clips de TeamTrack, sea cual sea su deporte (46 de 59 de balonmano, 9 de 12 de
    baloncesto, 13 de 51 de fútbol). Queda una firma de TeamTrack en las variables de
    nuisance aunque su balonmano sea un solo grupo. Un modelo que leyera esa firma
    acertaría más en el balonmano de TeamTrack y menos en su fútbol y su baloncesto. En
    los 400 primeros, TeamTrack son 34 clips (16 de balonmano). El corte por fuente
    (sección 7) lo deja ver, y se informa como limitación.
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

## Anexo A. Dimensionado (del piloto 4, 400 clips, 55 partidos)

La semianchura del IC al 95 % de los contrastes primarios fue de 0,04 a 0,10 (EE ≈
0,02-0,05). El dataset final tiene más partidos por deporte, así que se esperan IC algo
más estrechos. Con esos EE, el primer escalón de Holm (α/20 = 0,0025, z ≈ 3,0)
detecta diferencias de unos 0,07-0,15 y el último (0,05) de unos 0,04-0,10. Las
réplicas (200 clips) solo sirven para estimar el ruido de muestreo del modelo, no para
el análisis primario.

## Anexo B. Coste y tiempo estimados

`scripts/estimate_run.py` con los tokens medidos en `runs/pilot4-strict` (y en
`runs/pilot-strict` para Gemini 3.1) y precios de lista: unos **576 USD** en total, 508
de las celdas pre-registradas y 68 de A7b. Gemini 3.1 se lleva unos 355 (40 de A7b), por
sus ~5.000 tokens de razonamiento por ítem. Siguen gpt-5.6-sol ~109 (15 de A7b), Opus 5.5
~60 (7; estimado con los tokens de Sonnet 5), Sonnet 5 ~36 (4), terra ~16 (2) y Jev < 1.
Unas **12 h** de reloj, marcadas por Gemini (los modelos corren en paralelo). A7b se
estima con los tokens de la hoja a 4 s con 4 opciones: la hoja es la misma imagen de 8
fotogramas y con 3 opciones el prompt es algo más corto, así que la estimación queda un
poco alta. Los precios son supuestos, no facturas, y no incluyen los reintentos de las
llamadas con error (`PASSES=2`), que también se facturan.

## Anexo C. Requisitos antes de lanzar

- `runs/final` construido (con `--reprs sheet,trails,text,video`, que necesita `.[video]`)
  y el preflight en verde. Lo está desde el 2026-09-28 (D8).
- `runs/final-d8` construido y `preflight-a7b` en verde. También lo está.
- Opus 5.5 habilitado en el Model Garden de Vertex (hueco A6; lo hace Javier). Si no lo
  está, `report` lo saca de los contrastes y de la familia (D4).
- `gcloud` autenticado. Claves de Azure y OpenRouter en sus ficheros, que el
  lanzador lee sin imprimirlas.

```bash
DRY_RUN=1 bash scripts/final_run.sh     # comprobar los comandos
bash scripts/final_run.sh               # la corrida (reanudable)
```
