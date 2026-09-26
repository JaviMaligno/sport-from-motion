# Datasets: qué hay, qué sirve y qué hay que construir

> Revisado en septiembre de 2026. Las celdas marcadas **(verificar)** son datos que
> no he podido confirmar desde el entorno donde se escribió esto (licencia exacta,
> frecuencia de muestreo); hay que comprobarlos antes de citarlos en un artículo.

## El criterio

Para este experimento no vale cualquier dataset de tracking. Tiene que cumplir tres
condiciones, en este orden de importancia:

1. **Coordenadas de campo (vista cenital), no de imagen.** Las cajas en coordenadas de
   imagen de una retransmisión llevan dentro la cámara: el encuadre, el zoom, el
   paneo. Y la cámara se mueve distinto en cada deporte. Es exactamente la trampa
   de la Parte 1 (la cámara que sigue al balón), en otra versión.
2. **Todos los deportes pasan por el mismo pipeline, o por pipelines equivalentes.**
   Si el fútbol viene de GPS a 25 Hz y el rugby de un detector sobre vídeo, el
   modelo puede aprender a reconocer el *ruido del pipeline*. Por eso se
   diezma todo a la misma frecuencia (5 Hz por defecto) y se prefieren fuentes que
   compartan método de captura.
3. **Varios partidos por deporte.** La validación cruzada se agrupa por partido; con
   un solo partido por deporte no hay forma honesta de separar "reconoce el deporte"
   de "reconoce este partido".

## Nivel A: tracking en coordenadas de campo, listo para usar

| Deporte | Fuente | Qué trae | Loader | Notas |
|---|---|---|---|---|
| Fútbol | [Metrica Sports sample-data](https://github.com/metrica-sports/sample-data) | 3 partidos, 25 Hz, 22 jugadores + balón, coords normalizadas | `metrica` (juegos 1-2, CSV) | **Ya lo usaste** en la Parte 2. El juego 3 viene en formato EPTS/FIFA: vía kloppy → CSV largo. |
| Fútbol | [SkillCorner Open Data](https://github.com/SkillCorner/opendata) | 10 partidos de la A-League 2024/25, tracking *de retransmisión* | kloppy → `long-csv` | Solo jugadores visibles en cámara: parcial, como lo que obtendremos para rugby. **Ya lo usaste** en la Parte 2. |
| Fútbol | [PFF FC World Cup 2022](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data) | 64 partidos, tracking de retransmisión + eventos | kloppy → `long-csv` | Se pide por formulario. El mayor volumen abierto de fútbol. |
| Fútbol | [SoccerNet-GSR](https://arxiv.org/abs/2404.11335) | 200 clips de 30 s con posiciones en el minimapa (m) | `long-csv` | Versión en coordenadas de campo de SoccerNet-Tracking (ver abajo). |
| Fútbol, baloncesto, balonmano | [TeamTrack](https://github.com/AtomScott/TeamTrack) ([Kaggle](https://www.kaggle.com/datasets/atomscott/teamtrack)) | ~150 min, >4 M cajas, dron cenital + ojo de pez, trayectorias proyectadas a coords de campo | `long-csv` | **La joya para el control 2**: tres deportes con el *mismo* montaje de captura. Nivel de juego y licencia **(verificar)**. |
| Baloncesto | NBA SportVU 2015-16 (volcados JSON) | Cientos de partidos, 25 Hz, 10 jugadores | `sportvu` | **Ya lo usaste**. Licencia no explícita (se usa en investigación; no redistribuir). |
| Fútbol americano | [NFL Big Data Bowl](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/data) | 10 Hz, yardas, todos los jugadores + balón | `nfl` | Mejor la edición **2025** (jugadas completas, pre-snap incluido). La 2026 solo trae parte de los jugadores con el balón en el aire. Uso limitado por las reglas de Kaggle **(verificar)**. El pre-snap es el subcaso de *formación* más limpio de todo el estudio. |

**Sugerencia práctica:** usar [kloppy](https://kloppy.pysport.org/) (PySport) como
conversor universal de fútbol. Lee Metrica, SkillCorner, PFF y otros proveedores con
un único modelo de datos, y de ahí se exporta el CSV largo
(`match_id,frame,track_id,x,y`) que come el loader genérico. No lo hago dependencia
del proyecto: es un paso previo de conversión.

## Revisión de lo que ya usaste para fútbol

- **SoccerNet-Tracking (Parte 1): no, tal cual.** Son cajas en coordenadas de imagen
  de retransmisión, justo lo que el criterio 1 excluye. Sirve su hermano
  **SoccerNet-GSR**, que da los mismos clips en coordenadas de campo. Además, esos
  mismos vídeos son el banco de pruebas perfecto para el pipeline de extracción de
  rugby: se pasa nuestro pipeline por los vídeos de SoccerNet y se compara con el
  ground truth de GSR, lo que *mide* cuánto ruido mete el pipeline.
- **Metrica (Parte 2): sí.** Limpio, completo y ya con loader. Pero son solo 3
  partidos, lo que para validación agrupada por partido es el mínimo.
- **SkillCorner (Parte 2): sí, y es especialmente útil** porque es tracking desde
  retransmisión con jugadores parciales, el mismo "sabor" de dato que tendremos en
  rugby.

## El hueco: rugby

No he encontrado **ningún dataset público de tracking de rugby union** comparable a
los de fútbol. Los clubes y World Rugby tienen GPS y tracking óptico, pero no es
abierto. Lo más cercano que existe:

| Fuente | Qué es | Utilidad |
|---|---|---|
| [CEA Rugby Sevens tracking](https://kalisteo.cea.fr/index.php/free-resources/) | 3 clips de 40 s del World Rugby Sevens (Dubái 2021), cámara PTZ de retransmisión, tracking anotado | Validar el tracker en rugby (oclusiones en rucks). Demasiado pequeño para el experimento, y además es *sevens*, no union. |
| [Rugby League Player Tracking (Roboflow)](https://universe.roboflow.com/max-hornigold-muhgz/rugby-league-player-tracking) | Dataset de detección, CC BY 4.0 | Afinar un detector de jugadores de rugby. |
| [SportsMOT](https://github.com/MCG-NJU/SportsMOT) | 240 clips de baloncesto, fútbol y voleibol, en coordenadas de imagen | Sin rugby. Útil solo para entrenar o evaluar el tracker. |

**Conclusión: el rugby hay que extraerlo de vídeo.** Con dos niveles de ambición:

### Opción 1 (piloto, la más sencilla): plano de imagen estabilizado

1. Planos generales de retransmisión (se descartan repeticiones y primeros planos
   con un filtro de tamaño medio de jugador).
2. Detección y tracking: detector de personas (familia YOLO/RT-DETR afinado con el
   dataset de Roboflow) más ByteTrack/BoT-SORT; o bien **SAM 3**, que detecta y sigue
   instancias de un concepto dado por texto ("rugby player") en un solo modelo, sin
   entrenar detector. Hay que evaluar SAM 3 en rucks y melés, donde las oclusiones
   son extremas.
3. Compensación del movimiento de cámara: homografía global entre fotogramas
   consecutivos (flujo óptico/ECC sobre el fondo) para llevar todo al plano del
   primer fotograma.
4. Exportar a `long-csv` y **pasar exactamente el mismo pipeline por vídeos de
   fútbol** (SoccerNet). La comparación rugby vs fútbol se hace solo entre datos
   extraídos así, y el fútbol extraído se contrasta con GSR para medir el ruido.

La perspectiva no se corrige del todo, pero afecta igual a los dos deportes, y los
controles (escala, rotación) absorben buena parte.

### Opción 2 (definitiva): coordenadas de campo reales

Lo anterior más la calibración del campo: un detector de puntos clave de las líneas
del campo y una homografía a metros, como hace el pipeline de
[SoccerNet-GSR / sn-gamestate](https://arxiv.org/abs/2404.11335) para fútbol. En rugby
las líneas son otras (22, 10 m, 5 m, 15 m), así que haría falta anotar puntos clave o
reaprovechar un modelo genérico de líneas. Es más trabajo, pero los clips de rugby
quedarían en el Nivel A junto a todo lo demás.

**Recomendación:** hacer la fase 1 solo con Nivel A (fútbol, baloncesto, balonmano,
fútbol americano), que ya permite validar todo el aparato y la comparación entre
tamaños de campo sin tocar el rugby. La fase 2 añade rugby frente a fútbol con la
Opción 1, simétrica, y la Opción 2 solo si el resultado lo merece.

Sobre las imágenes de rugby: se usan para investigación, no se redistribuyen ni los
vídeos ni los fotogramas. Lo que se publica son trayectorias anónimas y puntos grises.
