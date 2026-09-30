# Pre-registro: brazo Laya afinado

Fecha: 2026-09-30. Rama `laya-finetune`. Escrito **antes de entrenar ningún modelo**.
Continúa el experimento principal ([`preregistration.md`](preregistration.md),
resultados en [`results-final.md`](results-final.md)); todo lo que aquí no se dice se
hereda de allí (ítems, preset `strict_smooth`, estadístico, bootstrap).

## 0. Qué se ha visto antes

- Resultados del experimento principal: ningún modelo de frontera pasa de 0,47 de
  acierto; MiniRocket llega a 0,83 en `motion`. Jev (texto) está en
  `results-final.md`.
- Laya multilingüe **sin afinar**, en 20 clips `motion`/texto (los 20 primeros del
  orden intercalado): 6/20 aciertos, 12 de las 20 respuestas «soccer». Es una prueba de
  humo del código, no un resultado; no entra en ningún análisis.
- Longitud de las secuencias completas (cabecera + opciones + estado) con el tokenizador
  de Laya multilingüe, sobre los 7.630 ítems: máximo 250 (`formation`), 886 (`motion`),
  825 (`motion_shuffled`), 958 (`kinematics`), 976 (`kinematics_solo`). **Ninguna se
  trunca** con el contexto por defecto de 1.024.
- Una prueba de humo del entrenamiento en CPU (2 actualizaciones, 32 clips, fold 0 de
  `formation`), solo para comprobar que el código corre. Su acierto no se usa.

## 1. Pregunta

Los modelos de frontera no reconocen el deporte a partir de los puntos y un especialista
entrenado sí. ¿Lo consigue un modelo de lenguaje pequeño **entrenado con los mismos
datos que el especialista**, leyendo las coordenadas como texto? Si lo consigue, ¿lo
hace leyendo el movimiento (el orden de los fotogramas) o solo la forma?

## 2. Hipótesis (antes de los datos)

- **HL1 (afinar sirve).** Laya afinado acierta más que Laya sin afinar en `motion`.
- **HL2 (orden).** Laya afinado acierta más con los fotogramas en orden (`motion`) que
  barajados (`motion_shuffled`).
- **HL3 (movimiento sobre forma).** Laya afinado acierta más con `motion` que con
  `formation`.
- **HL4 (especialista).** Laya afinado acierta **distinto** que MiniRocket en `motion`
  (contraste bilateral). La expectativa es que MiniRocket gane, pero el test no lo supone.

## 3. Modelo y datos

- **Checkpoint.** `convaiinnovations/laya`, subcarpeta `multilingual` (mmBERT-base,
  322M parámetros, contexto 1.024), revisión fijada
  `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`. Se elige el multilingüe porque su
  contexto de 1.024 cubre todas las secuencias sin truncar; el inglés (512) truncaría
  todas las condiciones menos `formation`.
- **Ítems.** Las filas `repr = text` de `runs/final/items.jsonl`: 1.526 clips × 5
  condiciones (`motion`, `motion_shuffled`, `formation`, `kinematics`,
  `kinematics_solo`) = 7.630. El texto de estado, las instrucciones y los criterios
  son exactamente los que recibieron Jev y los modelos de chat en la celda texto
  (`state_text`, `decision_instructions`, `decision_criteria`, prompt neutro). Export:
  `scripts/laya/export.py`.
- **Folds.** Los de los especialistas: `StratifiedGroupKFold(n_splits=5, shuffle=True,
  random_state=0)` agrupado por partido y estratificado por deporte, sobre el orden de
  `load_clips`, el mismo que usa `learners.grouped_oof` para MiniRocket y DeepSets. Cada
  clip cae en el mismo fold en las cinco condiciones. Laya entrena con cuatro folds y
  predice el quinto: al final cada uno de los 1.526 clips tiene una predicción de un
  modelo que no lo vio, igual que en los especialistas.
- **Conjunto de evaluación.** Los 400 clips que vieron los modelos (los 400 primeros de
  `interleave` en `motion`; 100 por deporte). Se ha comprobado que coincide clip a clip
  con el conjunto de Jev en `motion`/texto.

## 4. Entrenamiento

Script: `scripts/laya/train.py`. Réplica del notebook oficial de ajuste fino en 2×T4
(`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`, repo NandhaKishorM/laya),
con sus hiperparámetros **sin tocar**:

| Parámetro | Valor |
|---|---|
| Pérdida | gradiente de política con perturbación gaussiana de los logits (G = 4) sobre recompensa propia (log + 0,75 · esférica) + entropía cruzada suave con peso 1 |
| Ruido σ | 0,4 → 0,1 lineal por época |
| Optimizador | AdamW, lr codificador 2,5e-5, lr cabeza 1e-4, weight decay 0,01, coseno hasta 1e-6 |
| Épocas | 4, fijas (sin parada temprana ni selección por validación) |
| Lote efectivo | 64 secuencias (micro-lote 8 × acumulación 8) |
| Precisión | fp16 con autocast y GradScaler, recorte de gradiente 1,0, gradient checkpointing |
| Objetivo | one-hot sobre el deporte correcto |

La única diferencia con el notebook es de disposición: cada trabajo corre en **una**
GPU con acumulación 8 en vez de repartirse entre dos con acumulación 4; el lote
efectivo es el mismo. Así las dos T4 entrenan dos folds a la vez.

**Calibración.** Como en el notebook: un 10 % de los clips de entrenamiento (tope 400),
elegido con semilla fija, se aparta **antes** de entrenar y sirve para ajustar una
temperatura escalar (LBFGS sobre log-loss, recortada a [0,1; 10]). No toca el fold de
test. La temperatura no cambia el argmax, así que el acierto no depende de ella; sí las
probabilidades (acierto corregido por prior, ECE).

**Semillas.** 0, 1 y 2 para las cinco condiciones: 5 condiciones × 5 folds × 3
semillas = **75 ajustes finos**. Cada semilla es una réplica (`replicate` = semilla + 1).

**Sin afinar.** Laya multilingüe sin tocar sobre los mismos ítems, por el mismo camino de
puntuación (logits de la cabeza, temperatura del checkpoint = 1,0): la referencia de HL1.

**Cómputo.** Kaggle, 2×T4, cuota gratuita (0 USD). Nada en GCP (prohibido) ni en Azure.
Los ítems suben a Kaggle como **dataset privado**; no se redistribuyen.

## 5. Análisis primario

Estadístico, bootstrap y p: los de la sección 5 del pre-registro principal
(`evaluate.paired_difference`, bootstrap agrupado por partido, B = 10.000, semilla 0,
p bilateral). Sobre los 400 clips de evaluación. La corrección por ítem de Laya afinado
es la **media de las tres semillas** (la fusión de réplicas de `report`); los clips
entran si responden las tres.

| Nombre | a | b | Hipótesis |
|---|---|---|---|
| `ft_gain` | Laya afinado, `motion` | Laya sin afinar, `motion` | HL1 |
| `order` | Laya afinado, `motion` | Laya afinado, `motion_shuffled` | HL2 |
| `motion_over_shape` | Laya afinado, `motion` | Laya afinado, `formation` | HL3 |
| `vs_minirocket` | Laya afinado, `motion` | MiniRocket, `motion` | HL4 |

**Familia propia** de 4 contrastes, Holm, α = 0,05. No se mezcla con la familia de 20
del experimento principal: es una pregunta nueva y posterior, y así se dice.

## 6. Secundario y exploratorio (sin Holm, p crudos)

- Acierto, acierto balanceado, acierto corregido por prior, log-loss, ECE (antes y
  después de la temperatura) y recall por deporte, por condición.
- `kinematics` frente a `kinematics_solo` y frente a `motion`.
- Las 5 condiciones sobre los 1.526 clips (todas las predicciones fuera de fold), junto a
  MiniRocket y DeepSets en las condiciones en que existen.
- Laya afinado frente a Jev y a los modelos de chat en `motion`/texto.
- Dispersión entre semillas: acierto por semilla y si el signo de cada contraste
  primario se mantiene en las tres.

## 7. Errores, fallos y lo que no se toca

- Un trabajo que se cae (sesión de Kaggle cortada, OOM) se relanza **igual**, con la misma
  semilla; se anota. Un entrenamiento que diverge (pérdida NaN) se informa y no se
  relanza con otros hiperparámetros sin una desviación escrita.
- No se ajusta ningún hiperparámetro, ni el número de épocas, a la vista de los
  resultados. Si se quisiera otra configuración sería un brazo nuevo, con su propio
  pre-registro.
- No se toca el conjunto de ítems, los folds ni el conjunto de evaluación.

## 8. Qué se informa sea cual sea el resultado

Los cuatro contrastes primarios con IC y p_holm, la tabla por condición, la dispersión
entre semillas y el tiempo de GPU usado. Si Laya afinado no supera al azar, se dice así.

## 9. Desviaciones

### DL1 (2026-09-30, POSTERIOR a los datos de un lote). Presupuesto de entrenamiento

- **Qué se vio.** Lote b0 (Kaggle 2×T4): `motion`, semilla 0, 5 folds, con la
  configuración de la sección 4. Acierto 0,235–0,270 por fold; la pérdida no baja de
  ≈1,3 y la cabeza acaba dando el mismo logit a las cuatro opciones (rango medio dentro
  de cada fila 0,01, frente a 0,62 sin afinar). Tiempo: ≈9,5 min por ajuste en una T4.
  Laya sin afinar, las 5 condiciones × 5 folds: 0,21–0,30.
- **Control positivo** (`train.py --control`): el mismo fold 0 de `motion` con una
  etiqueta arbitraria por deporte al principio del estado (`tag: Q7`, …; sin significado,
  el modelo sin afinar acierta 0,24). Con la configuración de la sección 4 la
  entropía cruzada va 1,44 → 1,38 → 0,74 → 0,54 por época y el acierto final es 0,72.
  El código entrena; el presupuesto de 4 épocas × ≈1.100 clips = 72 actualizaciones
  (pensado para ≈30.000 ejemplos) no basta ni para aprender una pista perfecta. El
  colapso de b0 no se puede leer como «Laya no ve el movimiento».
- **Qué cambia.**
  1. **Planificador.** Tasa de aprendizaje constante en los valores iniciales del
     notebook (2,5e-5 codificador, 1e-4 cabeza) en vez de coseno; σ baja 0,4 → 0,1 en
     las 4 primeras épocas, como en el notebook, y se queda en 0,1. Así el modelo tras
     la época E es el mismo que si se hubiera entrenado E épocas, y el barrido cabe en
     una sola corrida.
  2. **Épocas, elegidas con el control y sin tocar el test.** Una corrida de 32 épocas
     del control en los folds 0 y 1 de `motion` (semilla 0). Al final de cada época se
     mide el acierto en la porción de calibración (el 10 % de entrenamiento apartado),
     nunca en el fold de test. **E = la primera época en que el control llega a ≥ 0,95
     en los dos folds.** Si ninguna lo consigue en 32, E = 32 y se dice.
  3. Con E fijado, se corre el diseño completo de la sección 4 (5 condiciones × 5 folds
     × 3 semillas) con el planificador constante y E épocas. Análisis de la sección 5
     sin cambios.
- **Qué no cambia.** Ítems, folds, conjunto de evaluación, pérdida, optimizador, lote
  efectivo, calibración, contrastes, familia y α.
- **Lo que se informa.** La configuración original (sección 4) queda como resultado
  pre-registrado de b0 (`motion`, semilla 0: colapso al azar) junto al control. No se
  completan sus otros 70 ajustes: sabemos que está infraentrenada y costarían ≈6 h de
  cuota para confirmarlo.
- **Por qué no se encadenan folds.** Arrancar un fold desde el modelo de otro filtraría
  test: el modelo del fold k se entrenó con clips que son test en los demás. Cada ajuste
  parte del checkpoint original.
