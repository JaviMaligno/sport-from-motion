# Resultados: brazo Laya afinado

Pre-registro: [`preregistration-laya.md`](preregistration-laya.md) (desviaciones DL1 y DL2).
Experimento principal: [`results-final.md`](results-final.md).

**En una frase.** Laya multilingüe (322M), afinado con los mismos ~1.100 clips por fold que
los especialistas, **aprende** a reconocer el deporte leyendo las coordenadas como texto y
**lee el orden de los fotogramas** (+0,08, p_holm 0,007), pero solo cuando la optimización
despega, cosa que depende de la semilla; queda a ≈0,48 de MiniRocket.

## Configuraciones

| Configuración | Épocas | Planificador | Resultado |
|---|---|---|---|
| Sección 4 (notebook oficial) | 4 | coseno | colapso al azar (solo `motion`, semilla 0) |
| DL1 | 8 (fijadas con un control positivo) | constante | colapso al azar (5 condiciones × 3 semillas) |
| **DL2** | hasta 32, época elegida en calibración | constante | aprende en 9 de 15 ajustes de `motion` |

Todo en Kaggle (2×T4, cuota gratuita, 0 USD). Cada configuración se informa con su propia
familia de Holm (4 contrastes, α = 0,05) sobre los 400 clips de evaluación.

## 1. Configuración del notebook (sección 4)

`motion`, semilla 0, 5 folds: acierto 0,235–0,270 por fold. La cabeza acaba dando el mismo
logit a las cuatro opciones. `ft_gain` −0,01 [−0,09; 0,09], p 0,85.

**Control positivo** (etiqueta arbitraria por deporte al principio del estado, `tag: Q7`…):
el modelo sin afinar acierta 0,24; con esta configuración, 0,72. El código entrena, pero 72
actualizaciones no bastan ni para una pista perfecta.

## 2. DL1: 8 épocas

E = 8 es la primera época en que el control llega a ≥ 0,95 en la porción de calibración de
los folds 0 y 1 (fold 0: 1,00 en la época 8; fold 1: 1,00 en la época 2).

| Contraste | d | IC 95 % | p | p_holm |
|---|---|---|---|---|
| `ft_gain` | +0,04 | [−0,03; 0,11] | 0,26 | 0,54 |
| `order` | +0,06 | [−0,03; 0,14] | 0,18 | 0,54 |
| `motion_over_shape` | +0,03 | [−0,03; 0,09] | 0,30 | 0,54 |
| `vs_minirocket` | −0,55 | [−0,62; −0,48] | 0,0002 | 0,0008 |

Acierto medio por condición: `motion` 0,28, `motion_shuffled` 0,22, `formation` 0,25,
`kinematics` 0,25, `kinematics_solo` 0,27. La entropía cruzada **de entrenamiento** se queda
en ≈ ln 4 en casi todos los ajustes: el modelo no ajusta ni los clips que ve. Las tres semillas
coinciden en la respuesta en el 1 % de los clips de `motion`: cada una colapsa hacia una clase
distinta.

## 3. DL2: parada temprana sobre calibración (resultado principal)

Hasta 32 épocas; por fold y semilla se toma la época con mejor acierto en la porción de
calibración (10 % de entrenamiento), sin mirar el test. `motion`, `motion_shuffled`,
`formation` × 5 folds × 3 semillas = 45 ajustes (≈70 min los de `motion`).

| Contraste | d | IC 95 % | p | p_holm |
|---|---|---|---|---|
| `ft_gain` | **+0,12** | [0,04; 0,20] | 0,0018 | **0,005** |
| `order` | **+0,08** | [0,03; 0,14] | 0,0036 | **0,007** |
| `motion_over_shape` | **+0,08** | [0,02; 0,14] | 0,0074 | **0,007** |
| `vs_minirocket` | −0,48 | [−0,54; −0,42] | 0,0002 | 0,0008 |

Por semilla (d): `order` +0,14 / +0,10 / +0,01; `ft_gain` +0,16 / +0,16 / +0,03;
`motion_over_shape` +0,11 / +0,12 / +0,02. El signo se mantiene en las tres; la semilla 2
casi no aprende.

| Condición | Acierto por semilla | Media [IC 95 %] |
|---|---|---|
| `motion` | 0,40 · 0,40 · 0,27 | 0,36 [0,31; 0,41] |
| `motion_shuffled` | 0,26 · 0,30 · 0,26 | 0,27 [0,22; 0,33] |
| `formation` | 0,29 · 0,28 · 0,25 | 0,27 [0,21; 0,34] |
| Laya sin afinar, `motion` | 0,24 | |

Recall medio por deporte en `motion`: fútbol americano 0,51, baloncesto 0,50, balonmano 0,32,
fútbol 0,09.

**Acierto en test por fold** en `motion` (época elegida entre paréntesis):

| Semilla | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| 0 | 0,50 (26) | 0,42 (25) | 0,53 (29) | 0,46 (26) | 0,22 (24) |
| 1 | 0,47 (19) | 0,38 (17) | 0,53 (26) | 0,41 (31) | 0,22 (2) |
| 2 | 0,25 (8) | 0,20 (3) | 0,24 (9) | 0,23 (6) | 0,47 (32) |

**Inestabilidad.** Criterio descriptivo, fijado después de ver las curvas: un ajuste
«despega» si su entropía cruzada de entrenamiento baja de 1,30 en alguna época (ln 4 ≈ 1,386).

| Condición | DL1 (8 épocas) | DL2 (hasta 32) |
|---|---|---|
| `motion` | 3 / 15 | 9 / 15 |
| `motion_shuffled` | 0 / 15 | 6 / 15 |
| `formation` | 0 / 15 | 6 / 15 |
| `kinematics` | 2 / 15 | — |
| `kinematics_solo` | 1 / 15 | — |

Con los mismos datos y la misma configuración, que el modelo empiece a aprender depende de la
semilla. Cuando no despega, la entropía de entrenamiento se queda en ln 4 las 32 épocas (p. ej.
semilla 2, `motion`, folds 0–3).

## 4. Secundarios (`motion`, DL2, p crudos)

| Frente a | d | IC 95 % | p |
|---|---|---|---|
| Jev (texto) | +0,11 | [−0,04; 0,25] | 0,15 |
| Claude Opus 5.5 (texto) | −0,16 | [−0,30; −0,03] | 0,011 |
| Gemini 3.1 Pro (texto) | −0,06 | [−0,19; 0,05] | 0,30 |
| Claude Sonnet 5 (texto, A17) | +0,01 | [−0,11; 0,12] | 0,91 |
| gpt-5.6-sol (texto) | −0,06 | [−0,14; 0,02] | 0,14 |
| gpt-5.6-terra (texto) | −0,01 | [−0,11; 0,09] | 0,92 |
| DeepSets | −0,45 | [−0,51; −0,38] | 0,0002 |
| MiniRocket, `motion_shuffled` | −0,44 | [−0,52; −0,37] | 0,0002 |

## 5. Lectura

- Un modelo de 322M afinado con ~1.100 clips por fold extrae del texto de coordenadas algo que
  depende del orden temporal: el efecto `order` (+0,08) es del mismo orden que el de Opus 5.5
  en imagen (+0,10), el único modelo de frontera que lo mostró.
- En acierto (0,36) queda a la altura de los modelos de frontera que leyeron el mismo texto
  (GPT-5.6 Sol 0,41, Terra 0,36, Sonnet 5 0,35 en la versión A17, Gemini 3.1 Pro 0,42; ninguna
  diferencia significativa), por debajo de Opus 5.5 en texto (0,52; d −0,16 [−0,30; −0,03],
  p 0,011 sin corregir) y de los especialistas (MiniRocket 0,83, DeepSets 0,80, en los mismos
  400 clips). Frente a Jev, +0,11 [−0,04; 0,25], p 0,15.
- En el efecto del orden queda con MiniRocket (+0,12) y Opus 5.5 en imagen (+0,10); los otros
  cuatro modelos de frontera (imagen) y Jev (texto) están entre −0,02 y +0,01.
- El presupuesto del notebook (1.200 casos, ≈6.000 decisiones, ≈375 actualizaciones) no basta con ~1.100 ejemplos (72 actualizaciones): con 4 u 8
  épocas el resultado es el azar y no dice nada del modelo. Incluso con 32 épocas, un tercio
  de los ajustes de `motion` no despega.
- El balonmano y el fútbol siguen siendo los difíciles, como en el experimento principal.

## Reproducir

```bash
PYTHONPATH=src .venv/bin/python scripts/laya/export.py runs/final runs/laya/export
# Kaggle: datasets privados con export/items.jsonl + scripts/laya/train.py, y wheels + checkpoint
# (ver scripts/laya/kaggle_run.py); un kernel por lote, FT_EXTRA según la configuración:
#   DL1: --epochs 8 --schedule constant --tag e8
#   DL2: --epochs 32 --schedule constant --early-stop --tag es
PYTHONPATH=src .venv/bin/python scripts/laya/report.py runs/laya/all runs/final laya-ft-es
```
