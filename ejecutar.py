#!/usr/bin/env python3
"""Ejecutar desde cualquier carpeta: python ejecutar.py --help."""
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.datos import cargar_datos, seleccionar_columnas, verificar_submission, MESES
from src.evaluacion import comparar, resumir
from src.modelos import crear_modelo, probabilidad_positiva

ROOT = Path(__file__).resolve().parent

def opciones():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datos', type=Path, default=ROOT / 'data')
    p.add_argument('--config', type=Path, default=ROOT / 'config_modelos.json')
    p.add_argument('--salida', type=Path, default=ROOT / 'resultados_base')
    p.add_argument('--mes', choices=['sin','con','ambos'], default='sin')
    p.add_argument('--interaccion', choices=['sin','con','ambos'], default='con')
    p.add_argument('--modelos', nargs='+', help='Nombres del archivo de configuración; omitir para los seis.')
    p.add_argument('--meses', nargs='+', type=int, default=list(MESES), help='Meses AAAAMM; por defecto marzo–noviembre.')
    p.add_argument('--semilla', type=int, default=42)
    p.add_argument('--hilos', type=int, default=4)
    return p.parse_args()

def valores(opcion):
    return [False, True] if opcion == 'ambos' else [opcion == 'con']

def main():
    args = opciones()
    if args.hilos < 1 or len(args.meses) != len(set(args.meses)) or not set(args.meses).issubset(MESES):
        raise ValueError('Hilos positivos y meses únicos entre 202603 y 202611.')
    train, test, sample = cargar_datos(args.datos)
    configs = json.loads(args.config.read_text(encoding='utf-8'))
    if args.modelos:
        if len(set(args.modelos)) != len(args.modelos):
            raise ValueError('No repetir nombres de modelos.')
        configs = {n: configs[n] for n in args.modelos}
    if not configs:
        raise ValueError('No hay modelos configurados.')
    # Las dependencias se comprueban antes de empezar; nunca omitir un modelo en silencio.
    for nombre in configs:
        if nombre in ('catboost', 'lightgbm', 'xgboost'):
            __import__(nombre)
    variantes = list(itertools.product(valores(args.mes), valores(args.interaccion)))
    meses = sorted(args.meses)
    if args.salida.exists() and any(args.salida.iterdir()):
        raise FileExistsError(f'{args.salida} ya contiene archivos. Elige otra --salida para conservar la trazabilidad.')
    args.salida.mkdir(parents=True, exist_ok=True)
    (args.salida / 'predicciones_validacion').mkdir()
    paquetes = ['numpy','pandas','scipy','scikit-learn','joblib','threadpoolctl','catboost','xgboost','lightgbm','matplotlib']
    versiones = {n: importlib.metadata.version(n) for n in paquetes}
    manifest = dict(estado='en_ejecucion', fecha_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version, plataforma=platform.platform(), versiones=versiones,
        semilla=args.semilla, hilos=args.hilos, meses=meses, parametros=configs,
        variantes=[dict(usar_mes=m, usar_interaccion=i,
                       columnas=seleccionar_columnas(train,m,i)) for m,i in variantes],
        archivos_sha256={n:hashlib.sha256((args.datos/n).read_bytes()).hexdigest()
                        for n in ('train.csv','test.csv','sample_submission.csv')},
        criterio='Mayor promedio mensual de Gini; desempate por Gini mínimo y nombre. Igual peso por mes.',
        nota='Configuraciones iniciales, sin búsqueda de hiperparámetros ni early stopping. Validación usada para selección; no es test independiente.')
    ruta_manifest = args.salida / 'configuracion_ejecucion.json'
    ruta_manifest.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    resultados = comparar(train,configs,variantes,meses,args.salida,args.semilla,args.hilos)
    resultados.to_csv(args.salida/'comparacion_modelos.csv',index=False)
    resumen = resumir(resultados,meses)
    resumen.to_csv(args.salida/'resumen_modelos.csv',index=False)
    ganador = resumen.iloc[0]
    columnas = seleccionar_columnas(train,bool(ganador.usar_mes),bool(ganador.usar_interaccion))
    modelo = crear_modelo(ganador.modelo,train[columnas],configs[ganador.modelo],args.semilla,args.hilos)
    with threadpool_limits(limits=args.hilos):
        modelo.fit(train[columnas],train.objetivo)
        submission = sample.copy()
        submission['prediccion'] = probabilidad_positiva(modelo,test[columnas])
    verificar_submission(submission,test)
    submission.to_csv(args.salida/'submission.csv',index=False)
    verificar_submission(pd.read_csv(args.salida/'submission.csv'),test)
    ruta_modelo = args.salida/'modelo_final.joblib'
    joblib.dump(dict(modelo=modelo,columnas=columnas,candidato=ganador.candidato),ruta_modelo,compress=3)
    guardado = joblib.load(ruta_modelo)
    with threadpool_limits(limits=args.hilos):
        np.testing.assert_allclose(submission.prediccion,
            probabilidad_positiva(guardado['modelo'],test[guardado['columnas']]),rtol=1e-12,atol=1e-12)
    from src.reporte import generar_reporte
    generar_reporte(resultados,resumen,meses,args.salida)
    manifest.update(estado='completo', ganador=ganador.candidato,
                    gini_promedio=float(ganador.gini_promedio), filas_submission=len(submission),
                    comprobacion_modelo_recargado=True)
    ruta_manifest.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (args.salida/'resultados_parciales.csv').unlink()
    print('\nComparación COMPLETA. Ganador de estas configuraciones: '+ganador.candidato)
    print(resumen[['modelo','usar_mes','usar_interaccion','gini_promedio','gini_ultimo_mes']].to_string(index=False))

if __name__ == '__main__':
    main()
