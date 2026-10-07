# Comparación DataFest

216 evaluaciones terminadas. Meses: 202603, 202604, 202605, 202606, 202607, 202608, 202609, 202610, 202611.

| Candidato | Gini promedio | Desv. entre meses | Últimos 3 | Último mes |
|---|---:|---:|---:|---:|
| catboost__mes0__interaccion0 | 0.258212 | 0.011689 | 0.250012 | 0.239219 |
| catboost__mes1__interaccion0 | 0.257906 | 0.012871 | 0.249511 | 0.238266 |
| hist_gradient_boosting__mes0__interaccion0 | 0.257388 | 0.014508 | 0.245599 | 0.229575 |
| hist_gradient_boosting__mes1__interaccion1 | 0.257288 | 0.016274 | 0.248410 | 0.233891 |
| catboost__mes0__interaccion1 | 0.257017 | 0.013062 | 0.249910 | 0.238452 |
| hist_gradient_boosting__mes1__interaccion0 | 0.256676 | 0.013790 | 0.247529 | 0.232308 |
| hist_gradient_boosting__mes0__interaccion1 | 0.256053 | 0.014887 | 0.246745 | 0.232080 |
| catboost__mes1__interaccion1 | 0.255869 | 0.011724 | 0.249958 | 0.236664 |
| lightgbm__mes1__interaccion0 | 0.253094 | 0.017357 | 0.241301 | 0.220313 |
| lightgbm__mes1__interaccion1 | 0.253021 | 0.019418 | 0.246161 | 0.225375 |
| xgboost__mes1__interaccion1 | 0.252802 | 0.016218 | 0.248206 | 0.235162 |
| xgboost__mes1__interaccion0 | 0.252127 | 0.015751 | 0.244231 | 0.227033 |
| xgboost__mes0__interaccion0 | 0.251874 | 0.013642 | 0.243936 | 0.232470 |
| random_forest__mes0__interaccion1 | 0.251480 | 0.013163 | 0.243625 | 0.227738 |
| random_forest__mes1__interaccion1 | 0.251455 | 0.015143 | 0.240841 | 0.225168 |
| random_forest__mes0__interaccion0 | 0.251352 | 0.013840 | 0.241064 | 0.224939 |
| lightgbm__mes0__interaccion1 | 0.251112 | 0.016736 | 0.242901 | 0.228704 |
| random_forest__mes1__interaccion0 | 0.250649 | 0.013957 | 0.239178 | 0.225014 |
| lightgbm__mes0__interaccion0 | 0.250561 | 0.016129 | 0.239234 | 0.224930 |
| xgboost__mes0__interaccion1 | 0.249045 | 0.015222 | 0.244284 | 0.231066 |
| logistica__mes0__interaccion0 | 0.234578 | 0.016038 | 0.225270 | 0.202240 |
| logistica__mes1__interaccion0 | 0.234557 | 0.016029 | 0.225268 | 0.202258 |
| logistica__mes1__interaccion1 | 0.233688 | 0.016734 | 0.225413 | 0.202157 |
| logistica__mes0__interaccion1 | 0.233683 | 0.016760 | 0.225393 | 0.202171 |

Selección: mayor promedio de Gini mensual, con el mismo peso por mes. Desempate por mayor Gini mínimo y nombre del candidato. La desviación entre meses no es un intervalo de confianza.

Cada candidato usa una configuración fijada antes de ejecutar; esto no demuestra cuál algoritmo sería mejor tras una optimización exhaustiva. CatBoost procesa categorías internamente; los demás usan one-hot aprendido dentro de cada entrenamiento. Solo logística estandariza las variables numéricas.

Las ventanas se utilizan para seleccionar; su promedio puede ser optimista. Noviembre ya fue consultado previamente. No conocemos el Gini de diciembre. Pequeñas diferencias no demuestran superioridad estadística.

El modelo final se reentrenó con enero–noviembre. submission.csv respeta los IDs y el orden del test. Se verificó que recargar modelo_final.joblib reproduce sus probabilidades.

Las predicciones de cada ventana están en predicciones_validacion/, para comparar equipos o hacer análisis posterior. Los parámetros, versiones, variables y hashes de los datos constan en configuracion_ejecucion.json.
