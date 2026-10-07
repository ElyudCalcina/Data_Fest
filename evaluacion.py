"""Walk-forward expansivo. El mes evaluado nunca se pasa a fit/eval_set."""
import hashlib
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from .datos import separar_mes, seleccionar_columnas, verificar_probabilidades
from .modelos import crear_modelo, probabilidad_positiva

def metricas(y, p):
    p = verificar_probabilidades(p, len(y))
    auc = float(roc_auc_score(y, p))
    return {'auc': auc, 'gini': 2 * auc - 1}

def comparar(train, configs, variantes, meses, salida, semilla=42, hilos=4):
    filas = []
    for mes in meses:
        aprendizaje, validacion = separar_mes(train, mes)
        firma = hashlib.sha256(validacion[['id_cliente', 'mes', 'objetivo']].to_csv(index=False).encode()).hexdigest()
        for usar_mes, usar_interaccion in variantes:
            columnas = seleccionar_columnas(train, usar_mes, usar_interaccion)
            for nombre, parametros in configs.items():
                candidato = f'{nombre}__mes{int(usar_mes)}__interaccion{int(usar_interaccion)}'
                modelo = crear_modelo(nombre, aprendizaje[columnas], parametros, semilla, hilos)
                inicio = time.perf_counter()
                with warnings.catch_warnings(record=True) as avisos, threadpool_limits(limits=hilos):
                    warnings.simplefilter('always')
                    modelo.fit(aprendizaje[columnas], aprendizaje.objetivo)
                    p = probabilidad_positiva(modelo, validacion[columnas])
                resultado = dict(candidato=candidato, modelo=nombre, usar_mes=usar_mes,
                    usar_interaccion=usar_interaccion, mes_validacion=int(mes),
                    ultimo_mes_train=int(aprendizaje.mes.max()), filas_train=len(aprendizaje),
                    filas_validacion=len(validacion), tasa_conversion=float(validacion.objetivo.mean()),
                    firma_validacion=firma, segundos=time.perf_counter()-inicio,
                    advertencias=' | '.join(sorted(set(str(w.message) for w in avisos))),
                    **metricas(validacion.objetivo, p))
                filas.append(resultado)
                oof = validacion[['id_cliente', 'mes', 'objetivo']].copy()
                oof['prediccion'] = p
                oof.to_csv(salida / 'predicciones_validacion' / f'{candidato}_{mes}.csv.gz', index=False)
                pd.DataFrame(filas).to_csv(salida / 'resultados_parciales.csv', index=False)
                print(f'{len(filas):3d} | {mes} | {candidato} | Gini={resultado["gini"]:.6f} | {resultado["segundos"]:.1f}s', flush=True)
    resultados = pd.DataFrame(filas)
    validar_resultados(resultados, list(configs), variantes, meses)
    return resultados

def validar_resultados(r, nombres, variantes, meses):
    esperado = {(f'{n}__mes{int(m)}__interaccion{int(i)}', mes)
                for n in nombres for m, i in variantes for mes in meses}
    claves = ['candidato', 'mes_validacion']
    if r.duplicated(claves).any() or set(r[claves].itertuples(index=False, name=None)) != esperado:
        raise ValueError('Comparación incompleta o duplicada: no se seleccionará un ganador.')
    if not np.isfinite(r[['auc', 'gini']].to_numpy()).all():
        raise ValueError('Métricas no finitas.')
    if (r.groupby('mes_validacion').firma_validacion.nunique() != 1).any():
        raise ValueError('Se evaluaron filas diferentes entre candidatos.')

def resumir(r, meses):
    resumen = r.groupby(['candidato', 'modelo', 'usar_mes', 'usar_interaccion'], as_index=False).agg(
        gini_promedio=('gini','mean'), gini_desviacion=('gini','std'),
        gini_minimo=('gini','min'), gini_maximo=('gini','max'),
        meses_evaluados=('mes_validacion','nunique'), segundos_totales=('segundos','sum'))
    recientes = r[r.mes_validacion.isin(sorted(meses)[-3:])].groupby('candidato').gini.mean()
    ultimo = r[r.mes_validacion == max(meses)].set_index('candidato').gini
    resumen['gini_ultimos_3_meses'] = resumen.candidato.map(recientes)
    resumen['gini_ultimo_mes'] = resumen.candidato.map(ultimo)
    return resumen.sort_values(['gini_promedio','gini_minimo','candidato'], ascending=[False,False,True]).reset_index(drop=True)
