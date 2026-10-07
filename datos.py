"""Validación estricta: nunca eliminar filas silenciosamente."""
from pathlib import Path
import numpy as np
import pandas as pd

MESES = tuple(range(202603, 202612))

def cargar_datos(carpeta):
    carpeta = Path(carpeta)
    train, test, sample = [pd.read_csv(carpeta / f'{n}.csv') for n in ('train', 'test', 'sample_submission')]
    if not {'id_cliente', 'mes', 'objetivo'}.issubset(train.columns):
        raise ValueError('Faltan identificadores, mes u objetivo en train.')
    if set(train.columns) - {'objetivo'} != set(test.columns):
        raise ValueError('Las columnas de train y test no coinciden.')
    if set(train.objetivo.unique()) != {0, 1}:
        raise ValueError('objetivo debe contener exactamente las clases 0 y 1.')
    for nombre, frame in [('train', train), ('test', test)]:
        if frame.isna().any().any():
            raise ValueError(f'{nombre}: hay faltantes. Revisar antes de comparar.')
        if frame.duplicated(['id_cliente', 'mes']).any():
            raise ValueError(f'{nombre}: claves cliente-mes duplicadas.')
        if not np.isfinite(frame.select_dtypes(include='number').to_numpy()).all():
            raise ValueError(f'{nombre}: valores numéricos no finitos.')
    if set(train.mes) != set(range(202601, 202612)) or set(test.mes) != {202612}:
        raise ValueError('Se espera train enero–noviembre 2026 y test diciembre 2026.')
    if sample.columns.tolist() != ['id_cliente', 'prediccion']:
        raise ValueError('Plantilla inesperada: se requiere id_cliente,prediccion.')
    if not sample.id_cliente.equals(test.id_cliente) or test.id_cliente.duplicated().any():
        raise ValueError('Plantilla y test no coinciden o hay IDs duplicados en test.')
    return train, test, sample

def seleccionar_columnas(train, usar_mes=False, usar_interaccion=True):
    excluir = {'id_cliente', 'objetivo'}
    if not usar_mes:
        excluir.add('mes')
    if not usar_interaccion:
        excluir.add('dias_ultima_interaccion')
    return [c for c in train.columns if c not in excluir]

def separar_mes(train, mes):
    aprendizaje = train.loc[train.mes < mes].copy()
    validacion = train.loc[train.mes == mes].copy()
    if aprendizaje.empty or validacion.empty:
        raise ValueError(f'Ventana vacía: {mes}.')
    if aprendizaje.objetivo.nunique() != 2 or validacion.objetivo.nunique() != 2:
        raise ValueError(f'Se necesitan ambas clases en entrenamiento y validación: {mes}.')
    if aprendizaje.mes.max() >= validacion.mes.min():
        raise ValueError('Hay solapamiento temporal.')
    return aprendizaje, validacion

def verificar_probabilidades(p, n):
    p = np.asarray(p, dtype=float)
    if p.shape != (n,) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Probabilidades inválidas o número de filas incorrecto.')
    return p

def verificar_submission(submission, test):
    if submission.columns.tolist() != ['id_cliente', 'prediccion']:
        raise ValueError('Columnas de entrega incorrectas.')
    if not submission.id_cliente.equals(test.id_cliente):
        raise ValueError('Los IDs o su orden no coinciden con test.')
    verificar_probabilidades(submission.prediccion, len(test))
