# Huecos antes del artículo

Criterio: nada que en el artículo tenga que aparecer como «limitación que podríamos haber
superado» o «decisión de diseño que no hicimos bien». Cada hueco se cierra con datos o
análisis, no con prosa. Estado a 2026-09-27.

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

Harness listo, falta la corrida (rama `sfm/harness-gaps`):

- **A3** `run --replicate K`: cada réplica en su fichero (`__rK`), sin reutilizar respuestas;
  `report` da media, DE entre réplicas, acuerdo por ítem y contrastes sobre la corrección media.
- **A4** `run --prompt-style informed`: describe cómo se mueve cada deporte (textos simétricos
  en `prompts.MOVEMENT`, sin números ni datos nuestros). Necesita un `prepare` nuevo
  (`prompt_informed` en los ítems).
- **A5** `prepare --reprs video`: mp4 H.264 de todos los fotogramas a 5 fps; solo la ruta
  `vertex:` (Gemini) lo acepta y se envía con `videoMetadata.fps` (Gemini muestrea a 1 fps
  por defecto). Necesita `pip install -e '.[video]'`.
- **A8** `prepare --player-mode random`: subconjunto aleatorio de jugadores.

## Abiertos

| # | Hueco | Por qué importa | Cómo cerrarlo | Coste |
|---|---|---|---|---|
| A1 | **Las corridas VLM usan el dataset viejo** (`strict`, NFL alineada, balonmano solo TeamTrack) | Los números del artículo tienen que salir del dataset final | Repetir sobre el dataset final (`strict_smooth`, NFL mitad de jugada, EIGD) | ≈ lo de la corrida de 4 deportes por modelo |
| A3 | **Una sola tirada por ítem** | Los modelos de razonamiento no son deterministas; sin réplicas no se separa ruido de señal | 3 réplicas en un subconjunto (p. ej. 100 clips × `motion`/`motion_shuffled`) | ~15-20 % de una corrida |
| A4 | **Solo prompt neutro** | «No le dijisteis qué mirar» es la primera objeción | Variante informada: describir cómo se mueve cada deporte (el equivalente al prompt informado de la Parte 1) | 1 celda más por modelo |
| A5 | **Representación**: solo hoja de 8 fotogramas y texto | «Con vídeo lo vería» | Añadir `trails` (ya preparado) y vídeo/GIF para el modelo que acepta vídeo (Gemini) | 1-2 celdas |
| A6 | **Modelos frontera incompletos** | Falta el mejor de Anthropic (Opus 5.5 no está habilitado en el Model Garden) y Gemini 3.1 se quedó fuera por coste | Habilitar Opus 5.5 en Vertex (lo hace Javier en la consola) y decidir Gemini 3.1 | Opus caro; Gemini 3.1 ~150 $ en 4 deportes |
| A7 | **Duración del clip fija (4 s)** | El diseño pedía barrido; algunas pautas (rucks, jugadas) pueden necesitar más | Barrido 2 / 4 / 8 s en especialistas; VLM solo si cambia algo | local |
| A8 | **Asimetría de N**: baloncesto ve a todos sus jugadores, los demás a los 10 centrales | Ver el equipo completo es información que los otros no dan | Repetir especialistas con N=6 y selección aleatoria en todos | local |
| A9 | **Pocos partidos de baloncesto** (7) | IC agrupados anchos | Bajar más partidos de SportVU (hay cientos) | local, descarga |
| A11 | **Posiciones extrapoladas de SkillCorner** | Firma de pipeline propia | Variante `--detected-only` o comprobar que no cambia el resultado | local |
| A12 | **NFL de una sola fuente** | No hay otra fuente pública de tracking NFL (BDB 2024/2025 retirados) | No se puede eliminar; acotado por C4 (la fuente deja de distinguirse tras el suavizado) y C6 | — (restricción de datos, no de diseño) |
| A13 | **Sin línea base humana** | Es el origen de la pregunta | Estudio pequeño con GIF (`--reprs gif`) | tiempo de personas |
