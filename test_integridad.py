"""Pruebas de riesgos: fuga temporal, categoría futura, entrega y ranking incompleto."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.datos import separar_mes, seleccionar_columnas, verificar_submission, cargar_datos
from src.evaluacion import metricas, validar_resultados
from src.modelos import crear_modelo

class Integridad(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({'id_cliente':[1,2,1,2,1,2],
            'mes':[202601,202601,202602,202602,202603,202603],
            'objetivo':[0,1,0,1,0,1], 'ingresos':[1.,2.,3.,4.,1e9,2e9],
            'region':['A','B','A','B','FUTURA','FUTURA'],
            'dias_ultima_interaccion':[1,2,3,4,5,6]})

    def test_futuro_no_entra_en_entrenamiento(self):
        pasado,futuro = separar_mes(self.train,202603)
        self.assertEqual(len(pasado),4)
        self.assertLess(pasado.mes.max(),futuro.mes.min())
        cols=seleccionar_columnas(self.train)
        model=crear_modelo('logistica',pasado[cols],{'max_iter':200},42,1)
        model.fit(pasado[cols],pasado.objetivo)
        prep=model.named_steps['preparacion']
        self.assertNotIn('FUTURA',prep.named_transformers_['categorias'].categories_[0])
        self.assertEqual(prep.named_transformers_['numericas'].mean_[0],2.5)
        self.assertTrue(np.isfinite(model.predict_proba(futuro[cols])).all())

    def test_flags_no_filtran_objetivo_o_id(self):
        for mes in (True,False):
            for interaccion in (True,False):
                cols=seleccionar_columnas(self.train,mes,interaccion)
                self.assertNotIn('objetivo',cols)
                self.assertNotIn('id_cliente',cols)
                self.assertEqual('mes' in cols,mes)
                self.assertEqual('dias_ultima_interaccion' in cols,interaccion)

    def test_gini_y_probabilidades_invalidas(self):
        self.assertEqual(metricas([0,1],[.1,.9])['gini'],1)
        self.assertEqual(metricas([0,1],[.5,.5])['gini'],0)
        for p in ([float('nan'),.5],[-.1,.5],[1.2,.5]):
            with self.assertRaises(ValueError): metricas([0,1],p)

    def test_submission_reordenado_se_rechaza(self):
        test=pd.DataFrame({'id_cliente':[12,15]})
        bien=pd.DataFrame({'id_cliente':[12,15],'prediccion':[.2,.8]})
        verificar_submission(bien,test)
        with self.assertRaises(ValueError):
            verificar_submission(bien.iloc[::-1].reset_index(drop=True),test)

    def test_ranking_incompleto_o_filas_distintas_se_rechaza(self):
        r=pd.DataFrame([dict(candidato='a__mes0__interaccion1',mes_validacion=202603,auc=.6,gini=.2,firma_validacion='x')])
        with self.assertRaises(ValueError):
            validar_resultados(r,['a','b'],[(False,True)],[202603])
        r2=pd.concat([r,r.assign(candidato='b__mes0__interaccion1',firma_validacion='y')])
        with self.assertRaises(ValueError):
            validar_resultados(r2,['a','b'],[(False,True)],[202603])

    def test_datos_reales_y_ventanas(self):
        root=Path(__file__).resolve().parents[1]
        train,test,sample=cargar_datos(root/'data')
        self.assertEqual(len(train),110100)
        self.assertEqual(len(test),9900)
        for mes in range(202603,202612):
            a,b=separar_mes(train,mes)
            self.assertTrue((a.mes < mes).all())
            self.assertTrue((b.mes == mes).all())

if __name__=='__main__': unittest.main()
