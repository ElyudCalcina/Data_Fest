"""Resultados legibles y figura; no extrapola el Gini a diciembre."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generar_reporte(resultados, resumen, meses, salida):
    fig, ax = plt.subplots(figsize=(12,6))
    for candidato, g in resultados.groupby('candidato',sort=False):
        g = g.sort_values('mes_validacion')
        ax.plot(g.mes_validacion.astype(str),g.gini,marker='o',label=candidato,linewidth=1.5)
    ax.set(xlabel='Mes de validación (entrenamiento: todos los meses anteriores)',ylabel='Gini = 2 × AUC − 1',title='Validación temporal: comparación sobre las mismas filas')
    ax.grid(alpha=.2)
    ax.legend(fontsize=7,bbox_to_anchor=(1.02,1),loc='upper left')
    fig.tight_layout()
    fig.savefig(salida/'gini_por_mes.png',dpi=160)
    plt.close(fig)
    tabla = ['| Candidato | Gini promedio | Desv. entre meses | Últimos 3 | Último mes |',
             '|---|---:|---:|---:|---:|']
    for r in resumen.itertuples():
        tabla.append(f'| {r.candidato} | {r.gini_promedio:.6f} | {r.gini_desviacion:.6f} | {r.gini_ultimos_3_meses:.6f} | {r.gini_ultimo_mes:.6f} |')
    texto = '# Comparación DataFest\n\n'
    texto += f'{len(resultados)} evaluaciones terminadas. Meses: {", ".join(map(str,meses))}.\n\n'
    texto += '\n'.join(tabla)
    texto += '\n\nSelección: mayor promedio de Gini mensual, con el mismo peso por mes. '
    texto += 'Desempate por mayor Gini mínimo y nombre del candidato. La desviación entre meses no es un intervalo de confianza.\n\n'
    texto += 'Cada candidato usa una configuración fijada antes de ejecutar; esto no demuestra cuál algoritmo sería mejor tras una optimización exhaustiva. '
    texto += 'CatBoost procesa categorías internamente; los demás usan one-hot aprendido dentro de cada entrenamiento. '
    texto += 'Solo logística estandariza las variables numéricas.\n\n'
    texto += 'Las ventanas se utilizan para seleccionar; su promedio puede ser optimista. Noviembre ya fue consultado previamente. '
    texto += 'No conocemos el Gini de diciembre. Pequeñas diferencias no demuestran superioridad estadística.\n\n'
    texto += 'El modelo final se reentrenó con enero–noviembre. submission.csv respeta los IDs y el orden del test. '
    texto += 'Se verificó que recargar modelo_final.joblib reproduce sus probabilidades.\n\n'
    texto += 'Las predicciones de cada ventana están en predicciones_validacion/, para comparar equipos o hacer análisis posterior. '
    texto += 'Los parámetros, versiones, variables y hashes de los datos constan en configuracion_ejecucion.json.\n'
    (salida/'RESULTADOS.md').write_text(texto,encoding='utf-8')
