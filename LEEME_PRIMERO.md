# DataFest: 24 variantes

Esta edición compacta contiene código, notebook, datos originales, las 216 métricas de evaluación, análisis, gráficos, modelo final y submission.csv.

Se omiten únicamente los resultados de la ejecución anterior (resultados_base) y las predicciones individuales de validación. Se pueden regenerar ejecutando el código. Las 216 evaluaciones están completas en resultados_completa/comparacion_modelos.csv.

Abre resultados_completa/ANALISIS_VARIANTES.md para la explicación y resultados_completa/comparacion_cuatro_variantes.csv para la tabla resumida.

La entrega nueva está en resultados_completa/submission.csv.

Para repetir todo desde esta carpeta:

python -m pip install -r requirements.txt
python ejecutar.py --mes ambos --interaccion ambos --salida mi_comparacion
python analizar_variantes.py --salida mi_comparacion

También puedes consultar los resultados ya incluidos con Comparacion_DataFest.ipynb.
