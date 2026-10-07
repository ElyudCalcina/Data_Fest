NOTA DE ESTA EDICIÓN COMPACTA: resultados_base y las predicciones individuales de validación no se incluyen. Las referencias a ellos más abajo describen la edición histórica. Los resultados actuales están en resultados_completa. Consulta LEEME_PRIMERO.md.

# Actualización: mes e interacción

La ejecución ampliada está en `resultados_completa`: 216 evaluaciones, las cuatro combinaciones y el nuevo ganador entrenado con enero–noviembre. Consulta `resultados_completa/ANALISIS_VARIANTES.md` para las tablas y el análisis mensual. El notebook apunta a esta ejecución.

Para repetirla:

```bash
python -m pip install -r requirements.txt
python ejecutar.py --mes ambos --interaccion ambos --salida mi_comparacion
python analizar_variantes.py --salida mi_comparacion
```

Se conserva `resultados_base` como referencia histórica de la comparación de 54 pruebas. Los apartados siguientes describen esa base; las variantes con mes y sin interacción ahora sí se ejecutaron en `resultados_completa`.

---

# DataFest: comparación de seis modelos

Proyecto ejecutable con datos originales, código, pruebas, resultados y modelo final. Python 3.11 recomendado. Las versiones utilizadas están fijadas en requirements.txt. CPU; no requiere GPU.

## Empezar

Desde esta carpeta, crea y activa un entorno virtual si lo deseas y ejecuta:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python ejecutar.py --salida mi_ejecucion
```

La carpeta `resultados_base` incluye la ejecución entregada. La salida nueva debe estar vacía: se evita mezclar métricas de ejecuciones distintas. En Windows se pueden usar los mismos comandos dentro de una terminal con Python instalado. En Colab, descomprime el ZIP, abre `Comparacion_DataFest.ipynb` y ajusta la carpeta indicada.

## Qué se compara

1. Regresión logística.
2. HistGradientBoosting (los árboles del proyecto anterior).
3. CatBoost, con categorías nativas.
4. XGBoost.
5. LightGBM.
6. Random Forest.

Los seis reciben las mismas variables originales. Por defecto se incluye `dias_ultima_interaccion` y se excluye `mes` de los predictores. `id_cliente` y `objetivo` siempre se excluyen de las entradas. No se eliminan filas. Las variables categóricas se convierten en one-hot dentro de cada entrenamiento, excepto en CatBoost, que las procesa internamente. Solo logística estandariza las numéricas. El tratamiento de categorías es parte de cada pipeline; no se fuerza una representación perjudicial para CatBoost.

Esta base NO presupone que incluir interacción o excluir mes sea mejor: permite comparar primero algoritmos con una lista común de variables. La variante anterior ganadora sin interacción no se mezcla con los resultados nuevos.

## Protocolo fijado antes de ejecutar

- Entrenar enero–febrero, evaluar marzo; enero–marzo, evaluar abril; continuar hasta evaluar noviembre.
- Nueve ventanas × seis modelos = 54 ajustes de validación; más un ajuste final del ganador.
- Una configuración inicial por algoritmo, visible en `config_modelos.json`. No hay búsqueda de hiperparámetros ni selección de la mejor semilla.
- No se usa el mes evaluado para ajustar parámetros, transformaciones o early stopping. No se pasa `eval_set` al entrenamiento.
- Semilla 42 y CPU con cuatro hilos por defecto. Versiones, parámetros, columnas y SHA-256 de los CSV quedan registrados.
- Cada candidato se evalúa sobre exactamente los mismos IDs, mes y objetivos; se verifica mediante una firma.
- Criterio: mayor promedio aritmético de Gini de los nueve meses, igual peso por mes. Desempate por Gini mínimo y nombre del candidato.
- Se reportan también desviación, mínimo, máximo, últimos tres meses, último mes y duración. La desviación no es incertidumbre estadística ni intervalo de confianza.
- Se guardan todas las probabilidades de validación para análisis posterior.
- Se reentrena la configuración seleccionada con enero–noviembre y se predice diciembre, respetando la plantilla.

**No es una evaluación definitiva de los algoritmos:** cambiar hiperparámetros puede cambiar el ranking. Un mismo número de árboles no implica la misma complejidad entre implementaciones. Una configuración por algoritmo es un presupuesto común inicial, no una optimización equivalente. Los resultados del CatBoost de tu compañero pueden diferir porque sus parámetros pueden ser distintos.

## Variantes opcionales (no ejecutadas en la entrega base)

Incluir `mes` en los seis modelos:

```bash
python ejecutar.py --mes con --salida resultados_con_mes
```

Comparar con y sin `dias_ultima_interaccion` (108 ajustes):

```bash
python ejecutar.py --interaccion ambos --salida resultados_interaccion
```

Cruzar ambas decisiones para todos los algoritmos (216 ajustes):

```bash
python ejecutar.py --mes ambos --interaccion ambos --salida resultados_completa
```

Probar solo los finalistas con ambas decisiones (ejemplo, no selección anticipada):

```bash
python ejecutar.py --modelos catboost hist_gradient_boosting --mes ambos --interaccion ambos --salida resultados_finalistas
```

En este dataset de un solo año, `mes` AAAAMM es numérico y conserva el orden cronológico. Su inclusión no prueba estacionalidad: no tenemos varios diciembres y los árboles no extrapolan tendencias como un modelo lineal. La variable se sigue usando para separar las ventanas aunque se excluya como predictor.

Para comparar específicamente noviembre con el compañero:

```bash
python ejecutar.py --meses 202611 --salida resultados_noviembre
```

Con una sola ventana, la desviación es indefinida (vacía en CSV) y “últimos 3 meses” resume los meses disponibles; no debe confundirse con el promedio de nueve meses.

## Archivos

- `ejecutar.py`: orquestación, configuración, entrenamiento final y validación de entrega.
- `config_modelos.json`: hiperparámetros iniciales de los seis candidatos.
- `src/datos.py`: controles de datos y ventanas temporales.
- `src/modelos.py`: construcción de cada estimador y sus transformaciones.
- `src/evaluacion.py`: comparación y selección.
- `src/reporte.py`: informe y gráfico.
- `tests/test_integridad.py`: pruebas de fuga temporal, categorías futuras, ranking y entrega.
- `resultados_base/comparacion_modelos.csv`: 54 filas, una por modelo y mes.
- `resultados_base/resumen_modelos.csv`: clasificación de candidatos.
- `resultados_base/predicciones_validacion/*.csv.gz`: probabilidades por cliente y mes evaluado.
- `resultados_base/RESULTADOS.md` y `gini_por_mes.png`: resumen para discutir con el equipo.
- `resultados_base/configuracion_ejecucion.json`: configuración, versiones y trazabilidad.
- `resultados_base/modelo_final.joblib` y `submission.csv`: modelo reentrenado y 9.900 probabilidades.

Si una ejecución falla, queda el manifiesto con `estado=en_ejecucion` y `resultados_parciales.csv`. No se elige un ganador con datos incompletos. Reinicia en otra carpeta después de corregir el problema. No hay reanudación automática.

Reutilizar el modelo guardado sin entrenar:

```bash
python predecir.py --modelo resultados_base/modelo_final.joblib --test data/test.csv --salida nueva_prediccion.csv
```

## Cómo interpretar el resultado

Gini = 2 × ROC AUC − 1. Evalúa ordenamiento, no porcentaje de aciertos ni calibración de probabilidades. Los nueve meses se usan para seleccionar candidatos: el promedio es una medida de selección, no una prueba independiente. Noviembre ya fue consultado anteriormente; no es un holdout intacto. Diciembre sigue sin etiquetas, y su puntaje no está garantizado.

Clientes repetidos entre meses son compatibles con el caso de uso; nunca se permite información de un mes futuro en el entrenamiento de una ventana. Toda ingeniería de históricos futura deberá respetar esa misma regla.

Si el equipo ajusta hiperparámetros, debe usar validaciones temporales internas anteriores al mes evaluado y un presupuesto acordado, o declarar los puntajes como resultados de desarrollo. No optimizar configuraciones con noviembre y luego presentarlo como evaluación independiente.

Para la reunión: mostrar el promedio, la curva mensual y noviembre por separado. Una diferencia pequeña puede deberse al muestreo; no demuestra superioridad estadística. Los tiempos dependen del equipo y no son un benchmark dedicado de velocidad.

## Documentación de las implementaciones

- https://scikit-learn.org/stable/modules/ensemble.html
- https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier
- https://xgboost.readthedocs.io/en/stable/python/python_api.html
- https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html
