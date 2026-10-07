"""Reutiliza exclusivamente un modelo propio/de confianza, sin reentrenar."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits
from src.datos import verificar_submission
from src.modelos import probabilidad_positiva

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--modelo',type=Path,required=True)
    p.add_argument('--test',type=Path,required=True)
    p.add_argument('--salida',type=Path,required=True)
    args=p.parse_args()
    if args.salida.exists():
        raise FileExistsError('La salida ya existe; elige otro nombre.')
    bundle=joblib.load(args.modelo)
    test=pd.read_csv(args.test)
    with threadpool_limits(limits=4):
        pred=probabilidad_positiva(bundle['modelo'],test[bundle['columnas']])
    submission=pd.DataFrame({'id_cliente':test.id_cliente,'prediccion':pred})
    verificar_submission(submission,test)
    submission.to_csv(args.salida,index=False)
    print(f'{len(submission)} probabilidades guardadas en {args.salida}')
