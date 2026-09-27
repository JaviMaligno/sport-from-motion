# Huecos antes del artículo

Criterio: nada que en el artículo tenga que aparecer como «limitación que podríamos haber
superado» o «decisión de diseño que no hicimos bien». Cada hueco se cierra con datos o
análisis, no con prosa. Estado a 2026-09-28.

## Cerrados

| # | Hueco | Cómo se cerró | Evidencia |
|---|---|---|---|
| C1 | El nº de jugadores delataba el deporte / el baloncesto desaparecía (N=12) | `--n-players 10` | 0 → 300 clips de baloncesto |
| C2 | `--limit` cogía un solo deporte | `interleave` por deporte y partido, mismos clips en todas las condiciones | 50/50 en cualquier prefijo; test |
| C3 | CV agrupada con clases desplazadas (balonmano con 2 grupos) | `StratifiedGroupKFold` + `class_weight="balanced"` | nuisance 0,00 → 0,26-0,32 (azar 0,25) |
| C4 | Firma del tracker (TeamTrack ojo de pez) | preset `strict_smooth` (σ = 2 fotogramas) | entre fuentes: 0,17 → 0,89; fuente de fútbol indistinguible (0,33 = azar) |
| C5 | Balonmano de un solo partido | EIGD-H: 5 partidos HBL (Kinexon) + TeamTrack | 6 partidos, 2 sistemas de captura |
| C6 | NFL alineada con el snap | clips de mitad de jugada (`--trim-start-seconds 1.5`) | NFL 0,96 → 0,92: el snap no es lo que la identifica |
| C7 | Truncado de razonamiento, `temperature`, token caducado | margen de razonamiento, fallback, refresco en 401 | 0 errores en 14.400 llamadas |
| C8 (A2) | Sesgo de respuesta de los VLM | `report` da recall por clase, macro-F1, kappa de Cohen y exactitud corregida por prior (calibración contextual con prior de los *otros* partidos), todo con IC agrupado | piloto 4, gpt-5.6-sol `kinematics`: exactitud 0,26, kappa 0,02, corregida 0,45 [0,34; 0,59]; recall fútbol americano 1,00 y balonmano 0,01 |
| C9 (A10) | Comparaciones múltiples | `PRIMARY_CONTRASTS` fijados en `pipeline.py` (orden, movimiento sobre forma, texto frente a imagen, prompt informado frente a neutro); Holm sobre todos los primarios del informe como una familia; el resto sale aparte como exploratorio con p sin corregir | piloto 4: de 9 primarios, sobreviven 2 (Sonnet 5: `motion − formation` p_holm 0,02; texto − imagen p_holm 0,01) |
| C10 (A8) | Asimetría de N: el baloncesto veía su plantilla entera y los demás a los 10 centrales | Especialistas sobre `runs/final` con `--player-mode random` y N=10 (`runs/abl-n10r`) y N=6 (`runs/abl-n6r`, nadie ve su plantilla entera). Diferencias emparejadas por clip, IC agrupado por partido | Ver la plantilla entera **no explica** el recall del baloncesto. Con N=10 aleatorio los clips de baloncesto son idénticos (399/399, mismas distancias entre jugadores) y aun así su recall MiniRocket `motion` **sube** 0,70 → 0,80 (+0,10 [0,06; 0,14]); exactitud 0,75 → 0,82: los 10 centrales de fútbol/balonmano se parecían al baloncesto, no al revés. Con N=6 baja todo (0,71 [0,68; 0,74]) y el baloncesto sigue en 0,62 [0,58; 0,65] (azar 0,25), por encima del balonmano (0,59); frente a `final`, baloncesto −0,08 [−0,13; −0,04], resto −0,02 a −0,05. `motion − motion_shuffled` se mantiene (+0,12 [0,09; 0,15]). Solo el baseline cinemático depende de N: baloncesto 0,51 → 0,28 con N=6 |
| C11 (A7) | Duración del clip fija (4 s) | `prepare --n-frames` (10/20/40 = 2/4/8 s a 5 Hz) sobre re-ingestas de todas las fuentes de `pool_final` a 2 y 8 s (`data/clips_d2`, `data/clips_d8`, paso = duración). A 8 s solo hay 42 clips de NFL de mitad de jugada (37 jugadas, 31 partidos): el barrido a 8 s va **sin fútbol americano**, y se compara con 2 y 4 s también sin él (`runs/abl-d{2,4,8}_noaf`) | 3 deportes (azar 0,33), MiniRocket `motion`: 2 s 0,68 · 4 s 0,72 · **8 s 0,82** [0,79; 0,85]; 8 − 4 s +0,09 [0,06; 0,13] (bootstrap por partido), con subida en los tres deportes (+0,08 a +0,10) y en cada fuente grande (SportVU 0,76 → 0,85, EIGD 0,70 → 0,80). Cinemático 0,62 · 0,63 · 0,67; `nuisance` al azar en las tres (0,32 · 0,33 · 0,28). El orden vale lo mismo a cualquier duración (`motion − motion_shuffled` +0,13 · +0,11 · +0,12). 4 deportes a 2 s (`runs/abl-d2`): MiniRocket 0,69 frente a 0,75 (−0,06 [−0,09; −0,03]), NFL 0,88. Coste: a 8 s el fútbol conserva 296 de 400 clips (SkillCorner detectado pierde más por la regla de 10 jugadores) |

Harness listo, falta la corrida (rama `sfm/harness-gaps`):

- **A3** `run --replicate K`: cada réplica en su fichero (`__rK`), sin reutilizar respuestas;
  `report` da media, DE entre réplicas, acuerdo por ítem y contrastes sobre la corrección media.
- **A4** `run --prompt-style informed`: describe cómo se mueve cada deporte (textos simétricos
  en `prompts.MOVEMENT`, sin números ni datos nuestros). Necesita un `prepare` nuevo
  (`prompt_informed` en los ítems).
- **A5** `prepare --reprs video`: mp4 H.264 de todos los fotogramas a 5 fps; solo la ruta
  `vertex:` (Gemini) lo acepta y se envía con `videoMetadata.fps` (Gemini muestrea a 1 fps
  por defecto). Necesita `pip install -e '.[video]'`.

## Abiertos

| # | Hueco | Por qué importa | Cómo cerrarlo | Coste |
|---|---|---|---|---|
| A1 | **Las corridas VLM usan el dataset viejo** (`strict`, NFL alineada, balonmano solo TeamTrack) | Los números del artículo tienen que salir del dataset final | Repetir sobre el dataset final (`strict_smooth`, NFL mitad de jugada, EIGD) | ≈ lo de la corrida de 4 deportes por modelo |
| A3 | **Una sola tirada por ítem** | Los modelos de razonamiento no son deterministas; sin réplicas no se separa ruido de señal | 3 réplicas en un subconjunto (p. ej. 100 clips × `motion`/`motion_shuffled`) | ~15-20 % de una corrida |
| A4 | **Solo prompt neutro** | «No le dijisteis qué mirar» es la primera objeción | Variante informada: describir cómo se mueve cada deporte (el equivalente al prompt informado de la Parte 1) | 1 celda más por modelo |
| A5 | **Representación**: solo hoja de 8 fotogramas y texto | «Con vídeo lo vería» | Añadir `trails` (ya preparado) y vídeo/GIF para el modelo que acepta vídeo (Gemini) | 1-2 celdas |
| A6 | **Modelos frontera incompletos** | Falta el mejor de Anthropic (Opus 5.5 no está habilitado en el Model Garden) y Gemini 3.1 se quedó fuera por coste | Habilitar Opus 5.5 en Vertex (lo hace Javier en la consola) y decidir Gemini 3.1 | Opus caro; Gemini 3.1 ~150 $ en 4 deportes |
| A7b | **VLM con clips de 8 s** | En los especialistas 8 s gana a 4 s (+0,09, C11); falta ver si los VLM también lo notan. Solo con 3 deportes: no hay NFL de mitad de jugada a 8 s | Una celda `motion`/`sheet` sobre `runs/abl-d8_noaf` (8 fotogramas, uno por segundo) con el mejor modelo del piloto; si no se mueve, cerrar | 1 celda por modelo, 3 deportes |
| A9 | **Pocos partidos de baloncesto** (7) | IC agrupados anchos | Bajar más partidos de SportVU (hay cientos) | local, descarga |
| A11 | **Posiciones extrapoladas de SkillCorner** | Firma de pipeline propia | Variante `--detected-only` o comprobar que no cambia el resultado | local |
| A12 | **NFL de una sola fuente** | No hay otra fuente pública de tracking NFL (BDB 2024/2025 retirados) | No se puede eliminar; acotado por C4 (la fuente deja de distinguirse tras el suavizado) y C6 | — (restricción de datos, no de diseño) |
| A13 | **Sin línea base humana** | Es el origen de la pregunta | Estudio pequeño con GIF (`--reprs gif`) | tiempo de personas |
| A14 | **Selección de jugadores del dataset final: central o aleatoria** | Con N=10 aleatorio los especialistas suben 0,75 → 0,82 (C10): la selección central quita señal a fútbol y balonmano | Decidir antes de A1 si el dataset final pasa a `--player-mode random` (los VLM aún no lo han visto) o se queda en central y se documenta la diferencia | 1 `prepare` (~25 min) + especialistas |
