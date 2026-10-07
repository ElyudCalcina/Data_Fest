# Comparación de modelos, mes e interacción

216 evaluaciones: seis modelos × cuatro variantes × nueve meses. Se mantienen los parámetros iniciales y la semilla 42. Cada ventana usa todas las filas anteriores al mes evaluado, desde enero; las transformaciones se ajustan solo con entrenamiento.

## Gini promedio marzo–noviembre

| Modelo | Sin mes / sin interacción | Sin mes / con interacción | Con mes / sin interacción | Con mes / con interacción |
| --- | --- | --- | --- | --- |
| CatBoost | 0.258212 | 0.257017 | 0.257906 | 0.255869 |
| HistGradientBoosting | 0.257388 | 0.256053 | 0.256676 | 0.257288 |
| LightGBM | 0.250561 | 0.251112 | 0.253094 | 0.253021 |
| XGBoost | 0.251874 | 0.249045 | 0.252127 | 0.252802 |
| Random Forest | 0.251352 | 0.251480 | 0.250649 | 0.251455 |
| Regresión logística | 0.234578 | 0.233683 | 0.234557 | 0.233688 |

Ganador por promedio mensual: **CatBoost**, mes=False, interacción=False. Gini promedio **0.258212**; noviembre **0.239219**; últimos tres meses **0.250012**.

El ganador se reentrenó con el 100 % de train (enero–noviembre). submission.csv contiene las 9 900 probabilidades de diciembre. Su Gini real no se conoce. Las 24 configuraciones se seleccionan usando estas validaciones: no constituyen una prueba independiente ni una demostración de superioridad estadística.

## Columnas y porcentajes de entrenamiento

Siempre se excluyen id_cliente y objetivo de los predictores. mes y dias_ultima_interaccion se incluyen según la variante. dias_ultima_transaccion se conserva en todas. mes se usa siempre para la separación temporal. Las variantes tienen 21, 22, 22 y 23 predictores, respectivamente.

| mes_validacion | ultimo_mes_train | filas_train | filas_validacion | porcentaje_train_completo |
| --- | --- | --- | --- | --- |
| 202603 | 202602 | 20200 | 10300 | 18.35 |
| 202604 | 202603 | 30500 | 9600 | 27.70 |
| 202605 | 202604 | 40100 | 10100 | 36.42 |
| 202606 | 202605 | 50200 | 10350 | 45.59 |
| 202607 | 202606 | 60550 | 9700 | 55.00 |
| 202608 | 202607 | 70250 | 10050 | 63.81 |
| 202609 | 202608 | 80300 | 9900 | 72.93 |
| 202610 | 202609 | 90200 | 10400 | 81.93 |
| 202611 | 202610 | 100600 | 9500 | 91.37 |

## ¿Qué mes tuvo más unos?

| Mes | Filas | Unos | Tasa (%) |
| --- | --- | --- | --- |
| Enero | 10400 | 1521 | 14.62 |
| Febrero | 9800 | 1543 | 15.74 |
| Marzo | 10300 | 1527 | 14.83 |
| Abril | 9600 | 1473 | 15.34 |
| Mayo | 10100 | 1584 | 15.68 |
| Junio | 10350 | 1600 | 15.46 |
| Julio | 9700 | 1410 | 14.54 |
| Agosto | 10050 | 1449 | 14.42 |
| Septiembre | 9900 | 1393 | 14.07 |
| Octubre | 10400 | 1628 | 15.65 |
| Noviembre | 9500 | 1439 | 15.15 |

Octubre tuvo más unos (1 628), mientras que febrero tuvo la tasa más alta (15,7449 %). El número de filas varía: comparar solo cantidades puede confundir tamaño de la muestra con propensión a convertir.

## Mes, festividades y diciembre

La hipótesis de que las festividades influyen es razonable, pero estos datos no permiten confirmar que Navidad aumente esta conversión. No hay diciembre etiquetado y solo hay un año de observaciones. Las diferencias mensuales también pueden reflejar cambios en clientes o campañas; no prueban causalidad ni estacionalidad repetida.

Incluir mes proporciona la posición temporal (AAAAMM); no informa al modelo de qué festividades ocurrieron. Una bandera es_navidad sería cero en todo el entrenamiento: sin ejemplos positivos de esa bandera no se puede aprender su efecto. Por ello no se agregó una variable navideña sin evidencia. Serían útiles diciembres de años previos y datos de campañas disponibles al momento de predecir.

Dentro del test todos los clientes tienen el mismo mes. Subir por igual las puntuaciones con una transformación estrictamente creciente conserva su orden y no mejora el Gini. El mes puede ayudar si cambia las relaciones entre características, algo que evaluamos empíricamente. Los árboles no extrapolan automáticamente una tendencia creciente hacia diciembre.

Las tasas mensuales de objetivo son análisis descriptivo; no se incorporan como predictor. Usar la tasa real del mes evaluado revelaría sus respuestas.

## Reproducción

```bash
python ejecutar.py --mes ambos --interaccion ambos --salida nueva_comparacion
python analizar_variantes.py --salida nueva_comparacion
```

Fuentes: data/train.csv, resultados_completa/comparacion_modelos.csv, resumen_modelos.csv y configuracion_ejecucion.json. efecto_variables.csv compara pares sobre los mismos nueve meses (delta = con variable menos sin variable).
