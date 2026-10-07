# Verificación del proyecto

- Se ejecutaron seis pruebas automatizadas con los datos reales y casos controlados: todas pasaron.
- Se comprobó la separación temporal y que las transformaciones no aprendan una categoría ni una escala que solo aparezcan en el mes futuro.
- Se probó el rechazo de probabilidades inválidas, IDs reordenados, comparaciones incompletas y candidatos evaluados sobre filas distintas.
- Se comprobó que los interruptores de mes/interacción nunca incluyan objetivo ni identificador como predictores.
- Se revisó la sintaxis de todos los scripts y de las celdas de código del notebook. El notebook es una interfaz al script; no se ejecutó completo como notebook.
- La ejecución completa y su estado se documentan en `resultados_base/configuracion_ejecucion.json`. El programa solo marca `completo` después de validar el CSV y reproducir todas las predicciones al recargar el modelo.
- Las variantes opcionales con mes y/o sin interacción no forman parte de la ejecución base incluida.
