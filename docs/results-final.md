# Resultados de la corrida final (pre-registrada)

> Corrida del 2026-09-28, de 16:08 a 23:26. El análisis sigue el
> [pre-registro](preregistration.md) (§5 a §10). Todo lo que no es el análisis primario
> se marca como secundario o exploratorio, con p sin corregir. La única desviación
> posterior a los datos es **D21** (§11 del pre-registro). No cambia ningún número
> primario: lo corregido se informa solo como sensibilidad, al lado del original
> (sección 11 de este documento).

## 0. En pocas líneas

- Según la regla del §6, **Claude Opus 5.5 es el único de los cinco modelos de chat que
  «lee el orden temporal»**. Si se barajan los instantes, pierde 10 puntos (IC 95 %
  [5; 16], p_holm = 0,004). Su exactitud corregida por prior en `motion/sheet` es 0,47
  [0,39; 0,57], y el azar está en 0,25.
- Ese efecto del orden se concentra en fútbol americano y baloncesto. En balonmano y
  fútbol está en torno a cero. Con clips de 8 s y sin fútbol americano (A7b,
  exploratorio) también aparece: +0,08 [0,02; 0,14].
- En gpt-5.6-sol, gpt-5.6-terra, Claude Sonnet 5 y Gemini 3.1 Pro **no hay evidencia**
  de que lean el orden ni de que les sirvan varios instantes. Por el IC, el orden les
  aporta como mucho entre 3 y 6 puntos. Eso no prueba que el efecto sea cero, y hay
  señales exploratorias, sin corregir, que apuntan a algo: sol da `order` +0,08 [0,01;
  0,16] (p = 0,032) con clips de 8 s, y sol y Gemini dan un `order` positivo dentro del
  fútbol americano (sección 13).
- Ningún modelo se acerca a los especialistas entrenados sobre los mismos 400 clips:
  MiniRocket acierta 0,83 y DeepSets 0,80, frente a 0,47 del mejor modelo.
- H3 (el texto empeora frente a la imagen) y H4 (decir qué mirar ayuda) quedan **sin
  evidencia** en todos los modelos.

## 1. Pregunta y diseño en breve

¿Puede un modelo reconocer un deporte de equipo **solo por cómo se mueven los
jugadores**? Se le quitan el campo, las líneas, el balón, los colores y el equipamiento,
y se le deja un vídeo de puntos visto desde arriba. Diseño completo en
[`design.md`](design.md) y análisis fijado en [`preregistration.md`](preregistration.md).

- **Tarea**: elegir entre 4 deportes (fútbol americano, baloncesto, balonmano y fútbol),
  con probabilidades por opción. Azar = 0,25.
- **Condiciones**, cada una con los mismos clips:
  - `motion`: 8 instantes en orden.
  - `motion_shuffled`: los mismos 8 instantes, desordenados.
  - `formation`: un solo instante.
  - `kinematics`: la trayectoria de cada jugador llevada a su propio punto de una
    rejilla, lo que destruye la forma del equipo.
  - `kinematics_solo`: lo mismo, y además cada trayectoria girada al azar, lo que
    destruye también la dirección compartida.
- **Hipótesis** (§2), con contrastes bilaterales:
  - H1: el orden importa (`order` = `motion` − `motion_shuffled`).
  - H2: el movimiento aporta sobre la forma (`motion_over_shape` = `motion` −
    `formation`).
  - H3: el texto empeora frente a la imagen (`text_vs_image` = `motion/text` −
    `motion/sheet`).
  - H4: el prompt informado ayuda (`prompt`).
- **Análisis primario** (§5):
  - Estadístico: diferencia emparejada por clip, con los errores contados como fallo.
  - IC y p bilateral: bootstrap agrupado por partido, B = 10.000, semilla 0.
  - Familia de Holm: 20 contrastes (4 por modelo × 5 modelos de chat), α = 0,05.
- **Clasificación por modelo** (§6):
  - «Lee el orden temporal»: `order` > 0 con p_holm < 0,05 **y** el límite inferior
    del IC de la exactitud corregida por prior de `motion/sheet` por encima de 0,25.
  - «Le sirven varios instantes, pero no su orden»: `motion_over_shape` > 0 con p_holm <
    0,05, la misma condición de exactitud y `order` no significativo.
  - «Sin evidencia de que vea movimiento»: en otro caso.

## 2. Datos

Conjunto `runs/final` (preset `strict_smooth`: suavizado gaussiano σ= 2 fotogramas,
ventanas de 4 s a 5 Hz, N = 10 jugadores al azar, techo de 12 m/s; sin pistas
congeladas, interpoladas ni duplicadas; D1-D19). La muestra de los modelos son los
**400 primeros clips** del orden `interleave`, 100 por deporte, los mismos en todas las
celdas. Cubren **151 partidos**.

| Deporte | Fuentes (clips en los 400) | Partidos |
|---|---|---|
| Fútbol americano | NFL Big Data Bowl 2023, un clip por jugada en fase aleatoria desde 1,0 s tras el snap (100) | 100 |
| Baloncesto | NBA SportVU 2015-16 (97) + TeamTrack (3) | 32 |
| Balonmano | EIGD-H, HBL con Kinexon (84) + TeamTrack (16) | 6 |
| Fútbol | SkillCorner, solo posiciones detectadas (77) + Metrica (15) + TeamTrack (8) | 13 |

- **Réplicas 2 y 3**: los 200 primeros clips (50 por deporte), un subconjunto de los 400.
- **A7b**: `runs/final-d8`, 300 clips de 8 s (100 de baloncesto, 100 de balonmano y 100
  de fútbol) de 51 partidos, con azar 1/3. No hay fútbol americano porque la NFL de BDB
  2023 casi no tiene jugadas de 8 s después del snap (D7).
- **Comprobaciones antes de analizar**:
  - Todos los ficheros de modelo de 400 filas comparten el mismo conjunto de `clip_id`, y
    todos los de 200 filas, otro, contenido en el primero.
  - No hay filas que falten: 25.600 filas en `runs/final` y 3.000 en A7b, 28.600 en
    total, que son las llamadas planificadas.
  - No hay ningún modelo ni ninguna celda excluidos.

## 3. Qué recibieron los modelos

- **Modelos**:
  - gpt-5.6-sol y gpt-5.6-terra, por Azure OpenAI.
  - Claude Opus 5.5 y Claude Sonnet 5, por Vertex (ruta `vertex-anthropic:`).
  - Gemini 3.1 Pro (`gemini-3.1-pro-preview`), por Vertex.
  - Jev (OpenRouter), solo en texto, con el estado en texto plano y en JSON.
- **Ajustes**: `--max-tokens 1024`, más un margen de razonamiento de 16.384 en las rutas
  OpenAI y Gemini. La ruta Anthropic **no** lo aplicaba (D21, sección 11). Se pidió
  `--temperature 0`, y los modelos que no lo aceptan usan la suya.
- **Prompt** (`src/motion_sport/prompts.py`):
  - Dice qué se ha quitado: balón, campo, colores y equipamiento. También que la
    posición, la escala y la orientación son aleatorias y que el número de puntos no es
    el número de jugadores.
  - Describe la vista, por ejemplo: «8 snapshots of the same play, 0.5 s apart, in
    chronological order». El paso real alterna entre 0,4 y 0,6 s (D20).
  - Pide JSON con una probabilidad por opción, la respuesta y una justificación breve.
    Las opciones se barajan en cada ítem.
  - El prompt **informado** añade una descripción general, simétrica y sin números, de
    cómo se mueve cada deporte.
- **Representaciones**:
  - `sheet`: hoja de contacto de 8 paneles, con puntos grises iguales sobre blanco y sin
    identidad de jugador.
  - `trails`: una imagen con estelas que se desvanecen.
  - `text`: coordenadas enteras por instante. En `motion` lleva el tiempo de cada
    instante y la nota «Players are listed in the same order in every snapshot».
  - `video` (solo Gemini): mp4 de los 20 fotogramas a 5 fps.
- **Una diferencia entre texto e imagen que importa para H3**: el texto da de forma
  explícita la correspondencia de cada jugador entre instantes, y la hoja no. `motion/text`
  y `motion/sheet` no llevan exactamente la misma información en dos formatos.

## 4. Errores por celda (§8)

Los errores que quedan al final cuentan como fallo. Nunca se quitan filas.

| Modelo | Celda | Errores no recuperados | Detalle | Tratamiento |
|---|---|---|---|---|
| Claude Sonnet 5 | `motion/text` | **40 / 400 (10 %)** | 36 sin JSON, 4 con JSON mal formado; los 40 en el tope de 1.024 tokens (D21) | celda y contraste `text_vs_image` **marcados** (`!`) |
| Gemini 3.1 Pro | `motion/text` | 6 / 400 (1,5 %) | sin JSON | por debajo del 2 %, sin marca |
| Gemini 3.1 Pro | `motion/video` | 1 / 400 | JSON mal formado | sin marca |
| Todas las demás (réplicas y A7b incluidas) | | 0 | | |

Reintentos: el lanzador hace dos pasadas por celda (§8). En la pasada 1, Sonnet 5
`motion/text` falló 102 de 400 ítems (91 sin JSON y 11 con JSON mal formado), y la
pasada 2 recuperó 62. Además, Sonnet `kinematics/sheet` tuvo 1 fallo, que la pasada 2
recuperó. Los logs cuentan también 14 intentos fallidos de Gemini, 2 de Opus y 1 de
terra, todos recuperados.

## 5. Los 20 contrastes primarios

Familia de Holm = 20, α = 0,05, n = 400 clips y 151 partidos en todos. d es la diferencia
de exactitud emparejada (a − b).

| Modelo | Contraste | d | IC 95 % | p | p_holm | ¿Significativo? |
|---|---|---|---|---|---|---|
| gpt-5.6-sol | `order` | +0,01 | [−0,04; 0,06] | 0,725 | 1,000 | no |
| gpt-5.6-sol | `motion_over_shape` | +0,08 | [0,01; 0,16] | 0,024 | 0,415 | no |
| gpt-5.6-sol | `text_vs_image` | −0,01 | [−0,07; 0,04] | 0,737 | 1,000 | no |
| gpt-5.6-sol | `prompt` | +0,03 | [−0,04; 0,10] | 0,468 | 1,000 | no |
| gpt-5.6-terra | `order` | +0,005 | [−0,05; 0,06] | 0,892 | 1,000 | no |
| gpt-5.6-terra | `motion_over_shape` | +0,07 | [−0,00; 0,15] | 0,064 | 1,000 | no |
| gpt-5.6-terra | `text_vs_image` | −0,02 | [−0,08; 0,04] | 0,550 | 1,000 | no |
| gpt-5.6-terra | `prompt` | +0,005 | [−0,06; 0,07] | 0,938 | 1,000 | no |
| **Claude Opus 5.5** | **`order`** | **+0,10** | **[0,05; 0,16]** | 0,0002 | **0,004** | **sí** |
| **Claude Opus 5.5** | **`motion_over_shape`** | **+0,20** | **[0,13; 0,29]** | 0,0002 | **0,004** | **sí** |
| Claude Opus 5.5 | `text_vs_image` | +0,05 | [−0,01; 0,09] | 0,075 | 1,000 | no |
| Claude Opus 5.5 | `prompt` | +0,03 | [−0,01; 0,07] | 0,172 | 1,000 | no |
| Claude Sonnet 5 | `order` | −0,02 | [−0,06; 0,03] | 0,536 | 1,000 | no |
| Claude Sonnet 5 | `motion_over_shape` | +0,02 | [−0,03; 0,06] | 0,421 | 1,000 | no |
| Claude Sonnet 5 | `text_vs_image` ! | +0,005 | [−0,05; 0,07] | 0,904 | 1,000 | no |
| Claude Sonnet 5 | `prompt` | +0,015 | [−0,03; 0,06] | 0,525 | 1,000 | no |
| Gemini 3.1 Pro | `order` | −0,01 | [−0,06; 0,05] | 0,860 | 1,000 | no |
| Gemini 3.1 Pro | `motion_over_shape` | +0,02 | [−0,07; 0,11] | 0,691 | 1,000 | no |
| Gemini 3.1 Pro | `text_vs_image` | +0,10 | [0,02; 0,19] | 0,008 | 0,151 | no |
| Gemini 3.1 Pro | `prompt` | +0,03 | [−0,03; 0,09] | 0,331 | 1,000 | no |

`!` = 40/400 errores en `motion/text` (sección 4). p = 0,0002 es el mínimo que admite el
bootstrap (2/10.001). Ningún contraste negativo es significativo. Solo sobreviven a Holm
los dos de Opus 5.5.

- **H1 (orden)**: solo en Opus 5.5.
- **H2 (movimiento sobre forma)**: solo en Opus 5.5. En sol (+0,08, p = 0,024) no
  sobrevive a Holm.
- **H3 (el texto empeora)**: sin evidencia en ningún modelo. En sol y terra el IC admite
  un empeoramiento de hasta −0,07 y −0,08. En Gemini la estimación va en la dirección
  contraria (+0,10), sin sobrevivir a Holm.
- **H4 (el prompt informado ayuda)**: sin evidencia. Los efectos van de +0,005 (terra) a
  +0,03, y el IC admite hasta +0,10 en sol.

## 6. Clasificación del §6

| Modelo | pc-acc `motion/sheet` [IC 95 %] | ¿Límite inferior > 0,25? | `order`, p_holm | Clasificación | Límite superior del IC de `order` |
|---|---|---|---|---|---|
| gpt-5.6-sol | 0,44 [0,38; 0,52] | sí | +0,01, 1,000 | sin evidencia de que vea movimiento | +0,06 |
| gpt-5.6-terra | 0,40 [0,32; 0,47] | sí | +0,005, 1,000 | sin evidencia de que vea movimiento | +0,06 |
| **Claude Opus 5.5** | 0,47 [0,39; 0,57] | sí | **+0,10, 0,004** | **lee el orden temporal** | — |
| Claude Sonnet 5 | 0,26 [0,20; 0,30] | **no** | −0,02, 1,000 | sin evidencia de que vea movimiento | +0,03 |
| Gemini 3.1 Pro | 0,32 [0,26; 0,39] | sí | −0,01, 1,000 | sin evidencia de que vea movimiento | +0,05 |

pc-acc es la exactitud corregida por prior (calibración contextual con el prior de los
*otros* partidos, IC por bootstrap de partidos). Ningún modelo queda en «le sirven varios
instantes, pero no su orden»: el único `motion_over_shape` significativo es el de Opus,
que ya está en la clase de arriba.

**Comprobación de robustez del §6** (réplicas; `order` sobre la corrección media de las
tres réplicas, 200 clips):

| Modelo | `order` en la réplica 1 (400) | `order` con la media de las 3 réplicas (200) [IC], p | ¿Mismo signo? |
|---|---|---|---|
| gpt-5.6-sol | +0,01 | +0,01 [−0,04; 0,06], 0,71 | sí |
| gpt-5.6-terra | +0,005 | +0,04 [−0,02; 0,09], 0,15 | sí |
| Claude Opus 5.5 | +0,10 | +0,105 [0,05; 0,16], 0,0002 | sí |
| Claude Sonnet 5 | −0,0175 | +0,005 [−0,05; 0,06], 0,85 | **no** |
| Gemini 3.1 Pro | −0,0075 | −0,01 [−0,06; 0,06], 0,82 | sí |

El §6 pide decirlo cuando el signo cambia. En Sonnet 5 los dos valores están en torno a
cero. La clasificación no cambia.

## 7. Resultados secundarios (§7; exploratorios, p sin corregir)

### 7.1 Todas las celdas

Cada casilla da exactitud / pc-acc. El κ de Cohen solo se da en `motion/sheet`. Las
tablas completas, con exactitud balanceada, macro-F1, log-loss e IC, están en
`runs/analysis-final/tables.md` §3.

| Celda | sol | terra | Opus 5.5 | Sonnet 5 | Gemini 3.1 |
|---|---|---|---|---|---|
| `motion/sheet` | 0,42 / 0,44 (κ 0,23) | 0,38 / 0,40 (κ 0,17) | 0,47 / 0,47 (κ 0,29) | 0,28 / 0,26 (κ 0,04) | 0,32 / 0,32 (κ 0,09) |
| `motion_shuffled/sheet` | 0,41 / 0,42 | 0,38 / 0,37 | 0,37 / 0,38 | 0,29 / 0,25 | 0,32 / 0,32 |
| `formation/sheet` | 0,34 / 0,37 | 0,31 / 0,26 | 0,27 / 0,32 | 0,26 / 0,27 | 0,30 / 0,29 |
| `motion/text` | 0,41 / 0,46 | 0,36 / 0,40 | 0,52 / 0,55 | 0,28 / 0,32 ! | 0,42 / 0,42 |
| `motion/sheet` informado | 0,45 / 0,47 | 0,39 / 0,42 | 0,50 / 0,51 | 0,29 / 0,29 | 0,34 / 0,35 |
| `kinematics/sheet` | 0,27 / 0,30 | 0,27 / 0,28 | 0,30 / 0,28 | 0,27 / 0,26 | 0,28 / 0,27 |
| `kinematics_solo/sheet` | 0,27 / 0,28 | 0,25 / 0,27 | 0,28 / 0,28 | 0,28 / 0,27 | 0,26 / 0,26 |
| `motion/trails` | 0,35 / 0,38 | 0,35 / 0,34 | 0,43 / 0,46 | 0,26 / 0,26 | 0,38 / 0,38 |
| `motion/video` | | | | | 0,39 / 0,39 |
| `motion_shuffled/video` | | | | | 0,32 / 0,32 |

La exactitud corregida por prior cuenta algo que la exactitud bruta no cuenta:

- sol, terra y Gemini clasifican por encima del azar incluso con los instantes
  desordenados. En `motion_shuffled/sheet` dan 0,42 [0,36; 0,50], 0,37 [0,30; 0,46] y 0,32
  [0,27; 0,39]. Además, sol lo hace con un solo instante: `formation/sheet` 0,37 [0,30;
  0,43].
- Es decir, sacan información de las formas que ven, pero no hay evidencia de que les
  sirva el orden ni de que les sirvan varios instantes frente a uno.

### 7.2 Recall por clase y sesgo de respuesta (RQ5)

Recall en `motion/sheet`:

| Modelo | Fútbol americano | Baloncesto | Balonmano | Fútbol | Respuesta favorita |
|---|---|---|---|---|---|
| sol | 0,92 | 0,21 | 0,07 | 0,49 | fútbol americano, 62 % |
| terra | 0,53 | 0,35 | 0,06 | 0,58 | fútbol, 42 % |
| Opus 5.5 | 0,85 | 0,19 | 0,04 | 0,79 | fútbol americano 44 % y fútbol 44 % |
| Sonnet 5 | 0,01 | 0,22 | 0,15 | 0,73 | fútbol, 64 % |
| Gemini 3.1 | 0,88 | 0,08 | 0,11 | 0,19 | fútbol americano, 66 % |
| MiniRocket (referencia) | 0,90 | 0,83 | 0,79 | 0,81 | — |

- **El balonmano no lo reconoce ningún modelo**: su recall va de 0,04 a 0,15.
- **El baloncesto y el balonmano se leen sobre todo como fútbol americano o como fútbol.**
  - Baloncesto: como fútbol americano en sol (62 de 100), Opus (51) y Gemini (62); como
    fútbol en terra (43) y Sonnet (71).
  - Balonmano: como fútbol en Opus (67), Sonnet (57) y terra (43); como fútbol americano
    en sol (50) y Gemini (62).
- **Sonnet 5 casi nunca responde fútbol americano**: recall 0,01. Queda en el azar una vez
  corregido su sesgo (pc-acc 0,26 [0,20; 0,30]). Es el único de los cinco en esa
  situación.
- **Opus 5.5 en `formation/sheet` responde «fútbol» el 94 % de las veces** (κ 0,03). Su
  `motion_over_shape` de +0,20 compara, por tanto, contra una celda casi degenerada. Lo
  que dice es, en buena parte, que con un solo instante Opus deja de distinguir y
  contesta «fútbol».

### 7.3 Cinemática (RQ2)

| Contraste | sol | terra | Opus 5.5 | Sonnet 5 | Gemini 3.1 | MiniRocket |
|---|---|---|---|---|---|---|
| `motion − kinematics` | +0,15 [0,10; 0,21] | +0,11 [0,05; 0,18] | +0,16 [0,11; 0,22] | +0,01 | +0,04 (p 0,09) | +0,10 [0,06; 0,13] |
| `kinematics − kinematics_solo` | +0,01 | +0,02 | +0,02 | −0,01 | +0,02 [0,00; 0,04] (p 0,085) | +0,06 [0,01; 0,10] (p 0,013) |

- **Exactitud bruta**: con la cinemática sola, todos los modelos quedan en el azar (0,25 a
  0,30, κ ≤ 0,07). Casi siempre responden fútbol americano (sol, Gemini y Opus) o fútbol
  (Sonnet).
- **Corregida por prior**: sol queda algo por encima del azar en `kinematics/sheet`, con
  0,30 [0,26; 0,36]. Los demás no.
- **Grupo frente a individuo**: en los modelos no hay diferencia entre ver al grupo y ver
  a cada jugador solo. En MiniRocket sí la hay, +0,06.

### 7.4 Representación: estelas y vídeo

| Contraste | d [IC 95 %] | p |
|---|---|---|
| sol, `motion/trails − motion/sheet` | −0,07 [−0,12; −0,03] | 0,001 |
| terra, ídem | −0,03 [−0,08; 0,03] | 0,32 |
| Opus 5.5, ídem | −0,03 [−0,07; 0,01] | 0,15 |
| Sonnet 5, ídem | −0,02 [−0,07; 0,04] | 0,57 |
| Gemini 3.1, ídem | +0,065 [−0,01; 0,14] | 0,086 |
| Gemini 3.1, `motion/video − motion_shuffled/video` | +0,07 [−0,00; 0,15] | 0,062 |
| Gemini 3.1, `motion/video − motion/sheet` | +0,075 [0,004; 0,15] | 0,040 |
| Gemini 3.1, `motion_shuffled/video − motion_shuffled/sheet` | −0,00 [−0,06; 0,05] | 0,99 |

- En Gemini, el vídeo apunta a algo: frente a la hoja da +0,075 con p = 0,04 sin
  corregir, y frente al vídeo desordenado, +0,07 con p = 0,06. Es exploratorio y no
  entra en la clasificación.
- Las estelas no ayudan a ningún modelo, y a sol le quitan 7 puntos.

### 7.5 Réplicas (informe sobre `runs/final` completo, 200 clips)

| Modelo | Celda | Exactitud media [IC] | DE entre réplicas | Acuerdo por ítem |
|---|---|---|---|---|
| sol | `motion/sheet` | 0,43 [0,35; 0,53] | 0,014 | 0,55 |
| sol | `motion_shuffled/sheet` | 0,42 | 0,013 | 0,49 |
| terra | `motion/sheet` | 0,40 | 0,008 | 0,38 |
| terra | `motion_shuffled/sheet` | 0,36 | 0,028 | 0,33 |
| Opus 5.5 | `motion/sheet` | 0,48 [0,39; 0,60] | 0,025 | 0,71 |
| Opus 5.5 | `motion_shuffled/sheet` | 0,38 | 0,018 | 0,80 |
| Sonnet 5 | `motion/sheet` | 0,30 | 0,008 | 0,55 |
| Sonnet 5 | `motion_shuffled/sheet` | 0,29 | 0,013 | 0,67 |
| Gemini 3.1 | `motion/sheet` | 0,30 | 0,021 | 0,68 |
| Gemini 3.1 | `motion_shuffled/sheet` | 0,31 | 0,013 | 0,65 |

- La DE entre réplicas va de 0,008 a 0,028, muy por debajo de los efectos que se
  discuten.
- En ese mismo informe, Gemini `text_vs_image` sale +0,14 [0,06; 0,23] con p_holm 0,004.
  **No es el análisis primario**: se calcula sobre 200 clips y con la media de tres
  réplicas en la hoja (§7). Va en la misma dirección que el primario (+0,10, p_holm
  0,15), que es el que cuenta.

### 7.6 Cortes por deporte, por fuente, fase de la jugada NFL y clips estáticos

**Por deporte** (`motion/sheet`; contrastes dentro del corte, p sin corregir):

| Modelo | `order` en fútbol americano | en baloncesto | en balonmano | en fútbol |
|---|---|---|---|---|
| sol | **+0,13 [0,04; 0,22], p 0,008** | +0,04 | −0,05 | −0,08 (p 0,13) |
| terra | +0,02 | +0,01 | +0,01 | −0,02 |
| Opus 5.5 | **+0,27 [0,18; 0,37], p < 0,001** | **+0,15 [0,07; 0,23], p < 0,001** | +0,01 | −0,03 |
| Sonnet 5 | +0,01 | +0,06 (p 0,13) | **−0,12 [−0,20; −0,03], p 0,017** | −0,02 |
| Gemini 3.1 | **+0,17 [0,07; 0,27], p < 0,001** | +0,01 | −0,09 (p 0,27) | **−0,12 [−0,21; −0,03], p 0,006** |

- **El efecto del orden en fútbol americano no es solo de Opus.** Sol (+0,13) y Gemini
  (+0,17) también lo tienen. En su total se compensa con efectos negativos en otros
  deportes: Gemini en fútbol (−0,12) y en EIGD, el balonmano de la HBL (−0,15 [−0,27;
  −0,06], p 0,001), y Sonnet en balonmano (−0,12). Lo propio de Opus es que también lo
  tiene en baloncesto (+0,15, casi todo SportVU) y que no tiene ningún corte negativo que
  lo compense.
- **Relación con la NFL**: el fútbol americano es de una sola fuente (A12), y sus clips
  siguen en la primera parte de la jugada. La rampa de velocidad es 0,938, y la mediana
  del desfase, 1,3 s tras el snap (D16, C12). Un efecto del orden en fútbol americano
  puede ser, en parte, leer la aceleración del arranque de la jugada. En el corte por
  fase, el recall de fútbol americano de Opus es 0,84 en orden frente a 0,57 desordenado
  con desfases de 1,0 a 1,5 s (n = 68), y 0,94 frente a 0,67 entre 1,5 y 2 s (n = 18).
  La diferencia no se limita a los clips más cercanos al snap. Aun así, todos están en
  los primeros segundos de la jugada (máximo 12,2 s; `tables.md` §13).
- **`motion_over_shape` es negativo en fútbol en todos los modelos**, de −0,12 a −0,47:
  un solo instante ya se lee como «fútbol».
- **Tamaño de los cortes**: el balonmano tiene solo 6 partidos, y todos estos cortes son
  exploratorios.

**Por fuente**: Metrica tiene 2 partidos, y cada deporte de TeamTrack, uno. No hay IC
(`too_few_matches`). SportVU, EIGD, SkillCorner y NFL reproducen el patrón por deporte
de arriba.

**`--tag static`** (mediana de velocidad < 1 m/s): 89 clips de 28 partidos (fútbol
americano 3, baloncesto 19, balonmano 60, fútbol 7). Ningún contraste es significativo,
y todas las exactitudes están entre 0,03 y 0,24. La muestra es sobre todo balonmano, que
ningún modelo reconoce.

## 8. Especialistas sobre los mismos 400 clips

CV agrupada por partido sobre los 1.526 clips, evaluada en los 400 que vieron los modelos
(D4).

| Especialista | Exactitud [IC 95 %] |
|---|---|
| MiniRocket `motion` | **0,83 [0,80; 0,87]** |
| MiniRocket `motion_shuffled` | 0,71 [0,67; 0,76] |
| MiniRocket `kinematics` | 0,74 |
| MiniRocket `kinematics_solo` | 0,68 |
| DeepSets `motion` | 0,80 [0,76; 0,85] |
| DeepSets `formation` | 0,54 [0,49; 0,59] |
| Baseline cinemático | 0,67 [0,62; 0,71] |
| Baseline `tempo` | 0,51 |
| Baseline `nuisance` (degenerado, D11) | 0,20 |

- **Contrastes de los especialistas**: MiniRocket `order` da +0,12 [0,07; 0,16], y
  DeepSets `motion − formation`, +0,27 [0,21; 0,32].
- **La señal está, y el orden importa**: MiniRocket pierde 12 puntos al barajar, y
  Opus 5.5 pierde 10. La diferencia está en el nivel. Opus acierta 0,47, frente a 0,83 de
  MiniRocket, y MiniRocket reconoce todos los deportes por igual (recall de 0,79 a 0,90).

## 9. A7b: clips de 8 s (exploratoria)

Tres deportes (baloncesto, balonmano y fútbol, sin fútbol americano), azar 0,33, 300
clips de 51 partidos, p sin corregir y fuera de la familia de Holm (D7, D15).

| Modelo | Exactitud `motion` [IC] | pc-acc [IC] | `order` a 8 s [IC], p | `order` a 4 s (primario) |
|---|---|---|---|---|
| sol | 0,53 [0,43; 0,65] | 0,54 [0,44; 0,65] | **+0,08 [0,01; 0,16], 0,032** | +0,01 |
| terra | 0,44 [0,36; 0,53] | 0,43 [0,36; 0,53] | +0,03 [−0,02; 0,09], 0,27 | +0,005 |
| Opus 5.5 | 0,44 [0,34; 0,57] | 0,52 [0,42; 0,63] | **+0,08 [0,02; 0,14], 0,012** | +0,10 |
| Sonnet 5 | 0,37 [0,29; 0,46] | 0,36 [0,30; 0,44] | +0,01 [−0,04; 0,07], 0,65 | −0,02 |
| Gemini 3.1 | 0,33 [0,23; 0,42] | 0,34 [0,23; 0,42] | +0,01 [−0,06; 0,09], 0,75 | −0,01 |
| MiniRocket | 0,89 [0,85; 0,92] | — | +0,11 [0,07; 0,16] | +0,12 |

- **Dónde es distinto de cero `order` a 8 s**: en sol y en Opus 5.5. Es mayor que a 4 s
  en sol y no en Opus. Es una comparación descriptiva: los clips y el número de opciones
  son distintos (§7).
- **Opus 5.5 a 8 s**: A7b no tiene fútbol americano, así que es el mejor indicio de que la
  lectura del orden de Opus no se reduce a la firma de arranque de la NFL. Sigue siendo
  exploratorio.
- **sol a 8 s**: aparece un efecto del orden que a 4 s no se ve. También es exploratorio,
  sin corregir, y no cambia su clasificación.

## 10. Jev

Jev es un modelo de decisión tipada, solo texto, y no tiene hipótesis confirmatoria (§2).

- **Exactitud**: entre 0,24 y 0,26 en todas las celdas, con κ ≈ 0. Responde «fútbol» en
  el 90-100 % de los clips.
- **Exactitud corregida por prior**: esta sí se mueve.
  - `motion/text`: 0,34 [0,29; 0,39] en texto plano y 0,34 [0,29; 0,40] en JSON.
  - `motion_shuffled` y `formation`: 0,21-0,26.
  - Prompt informado: 0,41.
  - Sus probabilidades llevan algo de información aunque la etiqueta sea siempre la
    misma.
- **Contrastes**: todos en torno a cero (|d| ≤ 0,01).
- **Texto plano frente a JSON**: la misma etiqueta en el 91-100 % de los clips.
- **Truncado**: `max_input_tokens` llega como mucho a 1.574, así que no hay señal de que
  se truncara la entrada.

## 11. Sensibilidad D21, al lado de los números pre-registrados

**Qué pasó** (detalle en el pre-registro, §11 D21):

- La ruta Anthropic no sumaba el margen de razonamiento que el §4 pre-registra, y Sonnet
  5 agotó los 1.024 tokens pensando en 40 ítems de `motion/text`.
- `runs/final-sens` es una copia en la que solo se re-corrieron **esas 40 filas**, con el
  arreglo (commit `d6a3833`). `runs/final` no se ha tocado.

| | Pre-registrado (primario) | Sensibilidad D21 |
|---|---|---|
| Errores en Sonnet 5 `motion/text` | 40 / 400 | 1 / 400 |
| Exactitud de la celda | 0,28 [0,20; 0,37] | 0,33 [0,23; 0,43] |
| pc-acc de la celda | 0,32 [0,27; 0,38] | 0,38 |
| Sonnet `text_vs_image`, d [IC] | +0,005 [−0,05; 0,07] | +0,05 [−0,005; 0,12] |
| p | 0,904 | 0,085 |
| p_holm | 1,000 | 1,000 |
| Decisiones de Holm que cambian | — | 0 |
| Clasificaciones del §6 que cambian | — | 0 |

- **Los errores no eran al azar.** Se recuperan 39, y de esas aciertan 18, el 46 %. Las
  respuestas válidas de la celda original aciertan 113 de 360, el 31 %.
- **Contar los errores como fallo** (§8) sesgaba `motion/text` hacia abajo, en la
  dirección de H3. Con la sensibilidad, H3 sigue sin evidencia.
- **Qué no cumple (§9)**: el §9 pide repetir **entera** la celda afectada, las 400 filas,
  e informar las dos versiones. Aquí solo se re-corrieron las 40 con error. Las 360
  restantes se obtuvieron con el tope, y alguna podría cambiar. Lo que falta está en el
  hueco A17 de [`gaps.md`](gaps.md): unos 6 USD de API, que esta fase no tenía
  autorizados.
- **Otras celdas de la ruta Anthropic**: ninguna tiene filas en el tope, y el máximo es
  988 tokens (Opus `motion/text`). No se puede descartar que el razonamiento adaptativo se
  acortara por el tope sin llegar a cortarse.


### Segunda versión entera de la celda (2026-09-29, cierra A17)

 Se re-corrieron las 400 filas de Sonnet 5
`motion/text` con el arreglo (`d6a3833`) en `runs/final-sens2`. El recurso fue otro: el mismo
modelo (`claude-sonnet-5`) en un Foundry de Azure distinto (`azure-anthropic:`, Sweden Central),
porque Vertex no se quiso volver a usar por coste y el Foundry de la sandbox sigue bloqueado por
Anthropic. Resultados:
- 0 errores; **129 de 400 respuestas superan los 1.024 tokens** de salida (máx. 12.368). Con el
  tope, Sonnet recortaba su razonamiento en texto, no solo fallaba en 40 filas.
- Acierto 0,35 (pre-registrado 0,28). `text_vs_image` = **+0,07 [0,01; 0,14], p = 0,019**
  (pre-registrado +0,005 [−0,05; 0,07], p = 0,90), frente a la misma `motion/sheet`
  pre-registrada. Va en la dirección contraria a H3 (el texto no empeora; mejora algo). Con el
  mismo rango en la familia de Holm de 20 no sería significativo. **No cambia la clasificación
  §6 de Sonnet** (depende de `motion/sheet` y su pc-acc) **ni la conclusión sobre H3**
  (sin evidencia de que el texto empeore).
- **El tope no afectó a las celdas de imagen.** Diagnóstico sobre los 100 primeros clips de
  Sonnet `motion/sheet` con margen: mediana de salida 151 → 144 tokens, ninguna por encima de
  1.024, acierto 0,29 → 0,29, misma respuesta en el 68 % (dentro de la variación entre réplicas,
  0,55-0,67). En la corrida pre-registrada ninguna celda de imagen de Opus ni de Sonnet llega a
  900 tokens (medianas 137-459). Los contrastes `order` y `motion_over_shape` no dependen del
  tope. Opus `motion/text` (1 % a ≥ 900 tokens) no se pudo repetir: Opus no está desplegado en
  ese Foundry. Su `text_vs_image` ya era +0,05, no significativo, y el margen solo podría
  subir la celda de texto.
- Coste: 8,38 USD (celda) + ~1,5 USD (diagnóstico), facturados en ese Foundry, no en Vertex.

## 12. Coste medido y tiempo

Precios de lista de `scripts/estimate_run.py` (anexo B), aplicados a los tokens de
`usage` de cada fila (`scripts/analysis/measured_cost.py`). Son supuestos de precio, no
facturas.

| Modelo | Estimado (USD) | Medido: final + A7b (USD) |
|---|---|---|
| gpt-5.6-sol | 109,40 | 87,37 + 12,96 = 100,33 |
| gpt-5.6-terra | 15,67 | 12,95 + 1,81 = 14,76 |
| Claude Opus 5.5 | 59,63 | 57,49 + 7,95 = 65,44 |
| Claude Sonnet 5 | 35,78 | 30,41 + 4,09 = 34,50 |
| Gemini 3.1 Pro | 355,24 | 381,21 + 39,74 = 420,95 |
| Jev | 0,24 | 0,24 |
| **Total de la corrida** | **575,95** | **636,20** |

- **Diferencia con la estimación**: +60,25 USD (+10,5 %). El total sale de los tokens sin
  redondear; la suma de las filas redondeadas de la tabla da 636,22. Casi todo es Gemini (+65,7):
  gastó 30,1 millones de tokens de razonamiento en `final` y 3,1 en A7b. sol costó 9 USD
  menos de lo estimado.
- **Fuera de la corrida**:
  - La sensibilidad D21: 1,40 USD (con ella, 637,60).
  - Los primeros intentos que se reintentaron, cuya fila se sobrescribió: ≈ 3,4 USD.
    Contados desde los logs, es una cota aproximada.
- **Tiempo de reloj**: de 16:08 a 23:26, 7,3 h, marcado por Gemini. La estimación era
  12,3 h. Por modelo: sol 5,3 h, terra 2,3 h, Opus 1,4 h, Sonnet 1,2 h y Jev 0,5 h. La
  re-corrida de sensibilidad tardó unos 4 minutos.

## 13. Qué se puede afirmar y qué no

**Se puede afirmar** (con los límites de este diseño: 4 deportes, clips de 4 s a 5 Hz, 8
instantes, un prompt, 400 clips de 151 partidos):

1. Con la regla pre-registrada, Claude Opus 5.5 usa el orden de los instantes:
   - desordenarlos le quita 10 puntos [5; 16], con p_holm = 0,004;
   - corregido su sesgo, clasifica por encima del azar;
   - el efecto se repite en las réplicas (+0,105) y a 8 s sin fútbol americano (+0,08,
     exploratorio).
2. En los otros cuatro modelos de chat no hay evidencia pre-registrada de que usen el
   orden ni de que les sirvan varios instantes. Si el orden les aporta algo, por el IC es
   como mucho de 3 a 6 puntos.
3. sol, terra y Gemini clasifican por encima del azar una vez corregido su sesgo, también
   con los instantes desordenados, y sol incluso con un solo instante. Algo sacan de las
   formas que ven. Sonnet 5 no se distingue del azar una vez corregido su sesgo.
4. Todos los modelos quedan muy por debajo de especialistas entrenados sobre los mismos
   clips (0,47 como máximo, frente a 0,80-0,83). El balonmano no lo reconoce ninguno.
5. No hay evidencia de que el texto empeore frente a la imagen (H3) ni de que el prompt
   informado ayude (H4).

**No se puede afirmar:**

1. Que los otros cuatro modelos «no ven el movimiento». Es ausencia de evidencia, no
   evidencia de ausencia:
   - los IC admiten hasta 6 puntos de efecto del orden;
   - sol da +0,08 a 8 s (p = 0,032);
   - sol y Gemini dan un `order` positivo en el corte de fútbol americano (+0,13 y +0,17),
     compensado en el total por cortes negativos;
   - Gemini con vídeo apunta a +0,07.

   Todo esto es exploratorio y sin corregir.
2. Que Opus 5.5 lea el movimiento en general:
   - su efecto está en fútbol americano (+0,27) y baloncesto (+0,15), y es nulo en
     balonmano y fútbol;
   - el fútbol americano es de una sola fuente, y sus clips están cerca del arranque de
     la jugada (A12, C12, D16);
   - A7b apoya que no es solo eso, pero es exploratoria.
3. Que su `motion_over_shape` de +0,20 mida cuánto le aporta el movimiento sobre la
   forma: con un solo instante contesta «fútbol» en el 94 % de los clips.
4. Que el texto y la imagen se comparen a igualdad de información. El texto da de forma
   explícita qué jugador es cuál en cada instante, y la hoja no (sección 3). Tampoco que
   H3 «no se cumple»: el IC de sol y de terra admite −0,07 y −0,08.
5. Nada sobre personas (A13), ni sobre si los resultados valen para otros deportes, otras
   duraciones, otras representaciones u otros prompts.
6. Que la celda de Sonnet 5 `motion/text` pre-registrada mida a Sonnet en igualdad con los
   demás: corrió con un tope de razonamiento que no era el pre-registrado (D21). La segunda
   versión entera, con margen (A17, sección 11), sube su exactitud de 0,28 a 0,35 y su
   `text_vs_image` a +0,07, sin cambiar ninguna clasificación ni la conclusión sobre H3.

7. Que la firma de cada fuente no influya. Dentro del fútbol, la fuente se reconoce por el
   movimiento (A15). En la NFL no se puede medir.

## 14. Reproducir

Todo sale de ficheros locales, sin llamadas a ninguna API. `runs/` está en `.gitignore`.

```bash
.venv/bin/python scripts/primary_view.py runs/final runs/final-primary --clips-from-models
.venv/bin/motion-sport report --items runs/final-primary --n-boot 10000 --json > runs/analysis-final/report_primary.json
.venv/bin/python scripts/primary_view.py runs/final-d8 runs/final-d8-view --clips-from-models
.venv/bin/motion-sport report --items runs/final-d8-view --n-boot 10000 --exploratory --json > runs/analysis-final/report_a7b.json
.venv/bin/motion-sport report --items runs/final --n-boot 10000 --json > runs/analysis-final/report_full_replicates.json
.venv/bin/motion-sport report --items runs/final-primary --tag static --n-boot 10000 --json > runs/analysis-final/report_static.json
.venv/bin/python scripts/primary_view.py runs/final-sens runs/final-sens-primary --clips-from-models
.venv/bin/motion-sport report --items runs/final-sens-primary --n-boot 10000 --json > runs/analysis-final/report_sens.json
PYTHONPATH=src .venv/bin/python scripts/analysis/final_extras.py runs/final-primary --n-boot 10000 --out runs/analysis-final/extras.json
PYTHONPATH=src .venv/bin/python scripts/analysis/measured_cost.py --dir final=runs/final --dir a7b=runs/final-d8 \
    --dir sens=runs/final-sens --only-changed sens=runs/final --logs runs/final/logs --json runs/analysis-final/cost.json
.venv/bin/python scripts/analysis/token_ceiling.py runs/final runs/final-d8 --sens runs/final-sens \
    --json runs/analysis-final/token_ceiling.json
PYTHONPATH=src .venv/bin/python scripts/analysis/final_tables.py runs/analysis-final
```

- **Salidas**: `runs/analysis-final/`, con `tables.md` (todas las tablas),
  `classification.json`, los `report_*.{json,txt}`, `extras.json`, `cost.{json,txt}` y
  `token_ceiling.{json,txt}`.
- **Copias en las rutas que nombra el pre-registro**: `runs/final/report_primary.json` y
  `runs/final-d8/report_a7b.json`. Son idénticas byte a byte a las de
  `runs/analysis-final/`.
