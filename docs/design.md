# ¿Qué deporte es? Diseño del experimento

> Experimento hermano de *Where's the ball?*. Estado: **diseño + harness listo, sin
> resultados**. Nada de lo que produce `loaders/synthetic.py` es un resultado.

## 1. La pregunta

Viendo un vídeo en el que no se distinguía bien el campo, supe que era rugby y no
fútbol por **cómo se movían los jugadores**. La pregunta es si un modelo puede hacer
lo mismo:

> **¿Se puede identificar un deporte de equipo solo por el movimiento de sus
> jugadores, sin campo, líneas, superficie, balón ni equipamiento?**

Encaja con la serie en los dos sentidos. En la Parte 2, "conocer el juego" resultó ser
"saber qué hacer con el movimiento". Esta pregunta le da la vuelta: si el
movimiento es lo que define el juego, debería bastar para reconocerlo.

Subpreguntas:

- **RQ1.** ¿Qué parte de la señal es **formación** (la forma del grupo en un instante)
  y qué parte es **movimiento** (cómo cambia)?
- **RQ2.** Dentro del movimiento, ¿qué parte es **colectiva** (todos corren igual,
  compresión y expansión) y qué parte es **individual** (ritmo de sprints, giros,
  arrancadas)?
- **RQ3.** ¿Los VLM frontera **ven** esa señal, o solo la ve un especialista? ¿Y un
  encoder de vídeo congelado (V-JEPA 2) con un probe lineal?
- **RQ4.** ¿Cambia algo si la información llega como **imagen o como texto**? (En la
  Parte 2, el texto hizo peores a los VLM.)
- **RQ5.** ¿Qué deportes se confunden entre sí, y se ordenan por tamaño de campo,
  por densidad de contacto o por otra cosa?

## 2. Representación: point-light displays

Cada jugador es un punto gris sobre fondo blanco, vista cenital, lienzo cuadrado, sin
ejes. Es la versión de equipo de los *point-light displays* de Johansson (1973), en
los que los humanos reconocen acciones a partir de unos pocos puntos de luz.

- **Nivel A (implementado):** trayectorias de puntos.
- **Nivel B (futuro):** esqueletos 2D. Añade placajes, rucks y el gesto del pase,
  pero exige estimación de pose sobre vídeo y reabre fugas (el equipamiento cambia la
  silueta), así que va después y con sus propios controles.

## 3. Fugas y controles

Es la parte de la que depende que el resultado sea creíble. Cada atajo tiene su
control y **cada control tiene un test que demuestra que funciona**: un clasificador
entrenado solo con las variables del atajo debe caer al azar después del control.

| Atajo | Control (`controls.py`) | Verificación |
|---|---|---|
| Número de jugadores (30 en rugby, 22 en fútbol, 10 en baloncesto) | `fix_player_count`: N fijo (12 por defecto), los más centrales, orden aleatorio | baseline `nuisance` al azar |
| Tamaño del campo y dispersión absoluta | `normalize_space("spread")` sobre la ventana final | baseline `nuisance` al azar |
| Orientación del campo, dirección de ataque, proporciones del campo | `random_rigid`: rotación arbitraria + reflejo; lienzo cuadrado | por construcción |
| Velocidad absoluta (en unidades de campo codifica el tamaño) | preset `strict_tempo`: remuestreo temporal a velocidad mediana común | baseline `tempo` al azar |
| Frecuencia de muestreo y duración | se diezma todo a 5 Hz y ventanas de longitud fija | baseline `nuisance` al azar |
| Estructura de equipos dada gratis | se eliminan las etiquetas de equipo | por construcción |
| Firma del pipeline de captura | mismo pipeline para los deportes comparados (ver `datasets.md`) | fútbol extraído vs. GSR |
| Cámara de retransmisión | solo coordenadas de campo (Nivel A) o estabilizadas simétricamente | — |
| Deriva del prompt (posición de las opciones) | orden de opciones barajado por ítem, con semilla | — |
| Desgaste diferencial (un control descarta más clips de un deporte) | aviso en `prepare` y recuento por deporte en `config.json` | — |

**Lo que el smoke test ya cazó** (y es material para el artículo, en el espíritu de
la serie): con datos de juguete diseñados para tener atajos, dos versiones del
propio harness tenían fugas.

1. Normalizar la escala sobre el clip entero y *después* recortar la ventana dejaba
   que la dispersión de la ventana final siguiera delatando el deporte (el baseline
   de sesgos daba 100%). Ahora se elige la ventana primero y se normaliza sobre ella.
2. Igualar la velocidad a la *mediana* del dataset obligaba a acelerar todos los
   clips del deporte rápido. Sin metraje suficiente, se descartaba **el deporte
   entero**, y la desigualdad de descartes es una fuga en sí misma. Ahora el
   objetivo es un percentil bajo (casi todo se ralentiza) y `prepare` avisa si un
   deporte pierde más de la mitad de sus clips.

Tras las correcciones, con datos de juguete: `nuisance` ≈ 0.50 en `strict`,
`tempo` ≈ 0.46 en `strict_tempo`, `kinematic` ≈ 1.0 en ambos. Es decir, los controles
quitan los atajos y dejan la señal de movimiento.

**Presets** (ablaciones con nombre):

| Preset | Quita | Deja |
|---|---|---|
| `raw` | solo el balón | todo, para medir cuánto se puede hacer "haciendo trampa" |
| `strict` (principal) | nº de jugadores, escala, orientación, equipos | forma y movimiento, incluida la velocidad relativa |
| `strict_tempo` | lo anterior + ritmo global | la *forma* del movimiento |
| `field_scaled` | lo de `strict`, pero la escala se divide por la longitud del campo | la dispersión relativa, para la comparación entre tamaños de campo |

## 4. Condiciones: qué parte de la señal ve el modelo

| Condición | Qué ve | Qué aísla |
|---|---|---|
| `formation` | un instante | solo la forma del grupo |
| `motion` | K = 8 instantáneas en orden (4 s a 5 Hz → cada 0.5 s) | forma + cambio |
| `motion_shuffled` | las mismas 8, desordenadas | si iguala a `motion`, el modelo lee formas, no movimiento (el control que decidió la Parte 1) |
| `kinematics` | cada trayectoria recolocada en una rejilla neutra | movimiento sin formación (conserva la coordinación) |
| `kinematics_solo` | lo anterior + cada trayectoria rotada por separado | solo el ritmo individual |

Contrastes emparejados que calcula `report` (por clip, IC agrupado por partido):

- `motion − motion_shuffled` → **¿importa el orden?** (¿hay lectura temporal?)
- `motion − formation` → **¿aporta algo el movimiento sobre la forma?**
- `kinematics − kinematics_solo` → **¿cuánto vale lo colectivo?**

### La formación como subcaso propio

Además de la condición `formation` sobre todos los clips, se evalúa **sobre
subconjuntos etiquetados** (`report --tag ...`):

- `pre_snap` (NFL): la formación en su forma más pura, antes de que nadie se mueva.
- `static`: clips con velocidad mediana < 1 m/s (se calcula en metros, antes de los
  controles). Conecta con la Parte 4 ("cuando nada se mueve").
- Jugadas a balón parado (futuro): córneres, faltas y saques en fútbol a partir de
  los eventos de Metrica/PFF; melés y touches en rugby, anotadas en la extracción.

La hipótesis es que en jugadas a balón parado la formación sola basta (una melé o
una línea de scrimmage son inconfundibles) y que en juego abierto hace falta el
movimiento. Si es así, el titular del artículo no es "reconoce el deporte" sino "sabe
**cuándo** le basta con la foto".

## 5. Deportes y escalas

- **Pareja principal:** fútbol frente a rugby union. Campos casi iguales (105×68 y
  100×70), así que el tamaño no ayuda ni siquiera antes de los controles. Es la
  pregunta original.
- **Escalera de tamaños** para RQ5: baloncesto (28 m), balonmano y futsal (40 m),
  hockey hierba (91 m), fútbol (105 m), rugby (100 m), fútbol americano (110 m).
  Con `strict` y `field_scaled` se ve si la confusión sigue al tamaño del campo o a
  la dinámica del juego.
- El conjunto de opciones que ve el modelo se fija por experimento
  (`prepare --candidates`), y puede incluir distractores sin datos (por ejemplo,
  ofrecer "rugby league" aunque solo haya union).

## 6. Modelos

Todos sobre **exactamente los mismos ítems**.

1. **VLM frontera vía Azure** (`backends/chat.py`), con los mismos nombres de
   variables de entorno que `experiments/judge-bias`:
   - `azure-anthropic:<deployment>`: Claude en Microsoft Foundry (Messages API nativa).
     Por ejemplo `claude-opus-5-5` o `claude-sonnet-5`.
   - `azure-openai:<deployment>`: GPT (y Grok) por la ruta `/openai/v1` del recurso.
   - `azure-foundry:<deployment>`: otros modelos del catálogo (Llama, Mistral,
     Qwen...) por la ruta de Model Inference, la misma que usó
     `wheres-the-ball/scripts/fase1_run_foundry.py`.
   - En imagen (`sheet`, `trails`) y en texto (`text`) para RQ4.
2. **Encoder de vídeo congelado + probe lineal** (`backends/probe.py`). Por defecto
   V-JEPA 2 (Meta, auto-supervisado prediciendo movimiento en espacio latente); como
   contraste, VideoMAE (reconstrucción de píxeles). Solo se ajusta una regresión
   logística sobre el embedding, con validación cruzada por partido. Si esto basta,
   **no hace falta diseñar ni entrenar un especialista propio**.
3. **Baseline cinemático** (`baselines.py`): 14 estadísticas de movimiento hechas a
   mano más una regresión logística. Es el especialista barato y transparente: si
   acierta, la señal está en el movimiento, y la pregunta pasa a ser si los modelos
   la *ven*.
4. **Humanos:** los GIF (`--reprs gif`) sirven para una prueba con personas (tú y
   algunas más). Es el origen de la pregunta y el contraste más interesante del
   artículo.

## 7. Métricas y estadística

- Exactitud, exactitud balanceada, log-loss (el modelo da una probabilidad por
  opción) y matriz de confusión.
- **IC por bootstrap agrupado por partido** desde el primer día (la lección de los
  14 ítems de la Parte 1).
- Validación cruzada del probe y de los baselines con `GroupKFold` por partido.
- Muestra balanceada por deporte y repartida entre partidos (`balanced_sample`).

## 8. Fases

1. **Fase 0 (hecha):** harness, controles verificados con datos de juguete, tests.
2. **Fase 1, piloto sin rugby:** Metrica + SkillCorner (fútbol), SportVU o TeamTrack
   (baloncesto), TeamTrack (balonmano), NFL 2025. Unos 200 clips por deporte,
   `strict` + `raw`, las 5 condiciones, imagen y texto. Baselines, probe V-JEPA 2 y
   dos o tres VLM de Azure. Coste orientativo por modelo: 4 deportes × 200 clips ×
   ~8 ítems ≈ 6.400 llamadas; se puede empezar con `--limit`.
3. **Fase 2, rugby:** pipeline de extracción simétrico (ver `datasets.md`), rugby frente
   a fútbol extraído, y validación del ruido contra SoccerNet-GSR.
4. **Fase 3 (opcional):** esqueletos (Nivel B) y subcaso de balón parado con eventos.

## 9. Riesgos y preguntas abiertas

- **La perspectiva en el rugby extraído (Opción 1)** no se corrige del todo. Se
  mitiga comparando solo con fútbol extraído igual.
- **Los clips de fútbol americano son jugadas, no juego continuo.** Tienen una
  estructura temporal (parado → explosión) que es parte del deporte, pero que también
  podría funcionar como atajo. Se puede controlar muestreando solo post-snap.
- **Duración del clip.** 4 s puede ser poco para algunos patrones (una fase de
  rucks). Conviene hacer un barrido de duración en el piloto.
- **Solo 3 partidos en Metrica.** Con TeamTrack/PFF se amplía.
