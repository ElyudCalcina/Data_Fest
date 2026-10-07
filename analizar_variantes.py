"""Resumen reproducible de las cuatro variantes y conversiones mensuales."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
NOMBRES = {'catboost':'CatBoost','hist_gradient_boosting':'HistGradientBoosting',
           'random_forest':'Random Forest','lightgbm':'LightGBM','xgboost':'XGBoost',
           'logistica':'Regresión logística'}
MESES = ['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto',
         'Septiembre','Octubre','Noviembre']

def tabla(df, decimales=6):
    def valor(v):
        return f'{v:.{decimales}f}' if isinstance(v,float) else str(v)
    return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |',
                     '| '+' | '.join(['---']*len(df.columns))+' |']+
                    ['| '+' | '.join(valor(v) for v in fila)+' |'
                     for fila in df.itertuples(index=False,name=None)])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--salida',type=Path,default=ROOT/'resultados_completa')
    args=p.parse_args(); out=args.salida
    train=pd.read_csv(ROOT/'data/train.csv')
    r=pd.read_csv(out/'comparacion_modelos.csv')
    s=pd.read_csv(out/'resumen_modelos.csv')
    assert len(r)==216 and len(s)==24
    mensual=train.groupby('mes').objetivo.agg(filas='size',unos='sum',tasa_conversion='mean').reset_index()
    mensual.insert(1,'nombre_mes',MESES)
    mensual['ceros']=mensual.filas-mensual.unos
    mensual.to_csv(out/'conversiones_por_mes.csv',index=False)
    orden=[(False,False),(False,True),(True,False),(True,True)]
    etiquetas=['Sin mes / sin interacción','Sin mes / con interacción',
               'Con mes / sin interacción','Con mes / con interacción']
    resumen=s.pivot(index='modelo',columns=['usar_mes','usar_interaccion'],values='gini_promedio').reindex(columns=orden)
    resumen.columns=etiquetas
    resumen=resumen.loc[s.drop_duplicates('modelo').modelo]
    resumen.index=resumen.index.map(NOMBRES); resumen.index.name='Modelo'
    resumen.to_csv(out/'comparacion_cuatro_variantes.csv')
    efectos=[]
    for modelo,g in r.groupby('modelo'):
        a=g.pivot(index='mes_validacion',columns=['usar_mes','usar_interaccion'],values='gini')
        for variable in ['mes','interaccion']:
            for fija in [False,True]:
                diff=a[(True,fija)]-a[(False,fija)] if variable=='mes' else a[(fija,True)]-a[(fija,False)]
                efectos.append(dict(modelo=modelo,variable_agregada=variable,
                    otra_variable_incluida=fija,delta_gini_promedio=diff.mean(),
                    meses_mejora=int((diff>0).sum()),meses_empeora=int((diff<0).sum()),
                    delta_noviembre=diff.loc[202611]))
    pd.DataFrame(efectos).to_csv(out/'efecto_variables.csv',index=False)
    ventanas=r[['mes_validacion','ultimo_mes_train','filas_train','filas_validacion']].drop_duplicates().sort_values('mes_validacion')
    ventanas['porcentaje_train_completo']=ventanas.filas_train/len(train)*100
    ventanas.to_csv(out/'ventanas_entrenamiento.csv',index=False)
    fig,axes=plt.subplots(3,2,figsize=(13,11),sharex=True,sharey=True)
    for ax,(modelo,g) in zip(axes.flat,r.groupby('modelo')):
        for (m,i),label in zip(orden,etiquetas):
            d=g[(g.usar_mes==m)&(g.usar_interaccion==i)].sort_values('mes_validacion')
            ax.plot(range(9),d.gini,marker='o',ms=3,label=label)
        ax.set_title(NOMBRES[modelo]); ax.grid(alpha=.2)
        ax.set_xticks(range(9),['Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov'])
        ax.set_ylabel('Gini de validación')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2)
    fig.suptitle('Cuatro variantes por modelo · mismas ventanas temporales')
    fig.tight_layout(rect=(0,.06,1,.96));fig.savefig(out/'gini_por_mes.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    cortos=[m[:3] for m in MESES]
    axes[0].bar(cortos,mensual.unos,color='#2368a0');axes[0].set_title('Cantidad de conversiones (objetivo = 1)')
    axes[1].bar(cortos,mensual.tasa_conversion*100,color='#21877f');axes[1].set_title('Tasa de conversión (%)')
    axes[1].set_ylim(0,18)
    for ax in axes:ax.tick_params(axis='x',rotation=45);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Datos observados de enero a noviembre de 2026')
    fig.tight_layout();fig.savefig(out/'conversiones_por_mes.png',dpi=150);plt.close(fig)
    ganador=s.iloc[0]
    detalle=mensual[['nombre_mes','filas','unos','tasa_conversion']].copy()
    detalle['tasa_conversion']=detalle.tasa_conversion*100
    detalle.columns=['Mes','Filas','Unos','Tasa (%)']
    contenido='# Comparación de modelos, mes e interacción\n\n'
    contenido+='216 evaluaciones: seis modelos × cuatro variantes × nueve meses. Se mantienen los parámetros iniciales y la semilla 42. Cada ventana usa todas las filas anteriores al mes evaluado, desde enero; las transformaciones se ajustan solo con entrenamiento.\n\n'
    contenido+='## Gini promedio marzo–noviembre\n\n'+tabla(resumen.reset_index())+'\n\n'
    contenido+=f'Ganador por promedio mensual: **{NOMBRES[ganador.modelo]}**, mes={bool(ganador.usar_mes)}, interacción={bool(ganador.usar_interaccion)}. Gini promedio **{ganador.gini_promedio:.6f}**; noviembre **{ganador.gini_ultimo_mes:.6f}**; últimos tres meses **{ganador.gini_ultimos_3_meses:.6f}**.\n\n'
    contenido+='El ganador se reentrenó con el 100 % de train (enero–noviembre). submission.csv contiene las 9 900 probabilidades de diciembre. Su Gini real no se conoce. Las 24 configuraciones se seleccionan usando estas validaciones: no constituyen una prueba independiente ni una demostración de superioridad estadística.\n\n'
    contenido+='## Columnas y porcentajes de entrenamiento\n\n'
    contenido+='Siempre se excluyen id_cliente y objetivo de los predictores. mes y dias_ultima_interaccion se incluyen según la variante. dias_ultima_transaccion se conserva en todas. mes se usa siempre para la separación temporal. Las variantes tienen 21, 22, 22 y 23 predictores, respectivamente.\n\n'
    contenido+=tabla(ventanas,2)+'\n\n'
    contenido+='## ¿Qué mes tuvo más unos?\n\n'+tabla(detalle,2)+'\n\n'
    contenido+='Octubre tuvo más unos (1 628), mientras que febrero tuvo la tasa más alta (15,7449 %). El número de filas varía: comparar solo cantidades puede confundir tamaño de la muestra con propensión a convertir.\n\n'
    contenido+='## Mes, festividades y diciembre\n\n'
    contenido+='La hipótesis de que las festividades influyen es razonable, pero estos datos no permiten confirmar que Navidad aumente esta conversión. No hay diciembre etiquetado y solo hay un año de observaciones. Las diferencias mensuales también pueden reflejar cambios en clientes o campañas; no prueban causalidad ni estacionalidad repetida.\n\n'
    contenido+='Incluir mes proporciona la posición temporal (AAAAMM); no informa al modelo de qué festividades ocurrieron. Una bandera es_navidad sería cero en todo el entrenamiento: sin ejemplos positivos de esa bandera no se puede aprender su efecto. Por ello no se agregó una variable navideña sin evidencia. Serían útiles diciembres de años previos y datos de campañas disponibles al momento de predecir.\n\n'
    contenido+='Dentro del test todos los clientes tienen el mismo mes. Subir por igual las puntuaciones con una transformación estrictamente creciente conserva su orden y no mejora el Gini. El mes puede ayudar si cambia las relaciones entre características, algo que evaluamos empíricamente. Los árboles no extrapolan automáticamente una tendencia creciente hacia diciembre.\n\n'
    contenido+='Las tasas mensuales de objetivo son análisis descriptivo; no se incorporan como predictor. Usar la tasa real del mes evaluado revelaría sus respuestas.\n\n'
    contenido+='## Reproducción\n\n```bash\npython ejecutar.py --mes ambos --interaccion ambos --salida nueva_comparacion\npython analizar_variantes.py --salida nueva_comparacion\n```\n\n'
    contenido+='Fuentes: data/train.csv, resultados_completa/comparacion_modelos.csv, resumen_modelos.csv y configuracion_ejecucion.json. efecto_variables.csv compara pares sobre los mismos nueve meses (delta = con variable menos sin variable).\n'
    (out/'ANALISIS_VARIANTES.md').write_text(contenido,encoding='utf-8')
    print(resumen.to_string());print('\n',detalle.to_string(index=False))

if __name__=='__main__':main()
