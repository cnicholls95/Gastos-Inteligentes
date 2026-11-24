"""Pruebas locales. Ejecutar desde raíz: python -m unittest discover -s tests -v"""
import sys
from pathlib import Path
import unittest
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analisis import leer_excel,datos_demostracion,detalle_por_concepto,detalle_por_categoria,indicadores_mes,consejos_financieros,calcular_estado
from modelos import entrenar_modelos


class PruebasPresupuesto(unittest.TestCase):
    def test_plantilla_vacia_se_abre(self):
        root=Path(__file__).resolve().parents[1]
        mov,pre,conf,_=leer_excel(root/'plantilla'/'Plantilla_Control_Gastos_IA.xlsx')
        self.assertTrue(mov.empty)
        self.assertTrue(pre.empty)
        self.assertEqual(conf['umbral'],0.8)

    def test_alertas(self):
        self.assertEqual(calcular_estado(200000,230000),'EXCEDIDO')
        self.assertEqual(calcular_estado(200000,180000),'ADVERTENCIA')
        self.assertEqual(calcular_estado(200000,200000),'EN EL LIMITE')
        self.assertEqual(calcular_estado(200000,100000),'NORMAL')
        self.assertEqual(calcular_estado(0,500),'SIN PRESUPUESTO')

    def test_ingresos_no_contados_como_gastos(self):
        mov,pre,conf,_=datos_demostracion()
        ind=indicadores_mes(mov,pre,'2026-10',conf['saldo_inicial'])
        gastos=mov.loc[(mov['Mes']=='2026-10')&(mov['Tipo']=='Gasto'),'Valor_COP'].sum()
        ingresos=mov.loc[(mov['Mes']=='2026-10')&(mov['Tipo']=='Ingreso'),'Valor_COP'].sum()
        self.assertAlmostEqual(ind['Gastos_COP'],gastos)
        self.assertAlmostEqual(ind['Ingresos_COP'],ingresos)
        self.assertFalse((ind['conceptos']['Concepto']=='SUELDO').any())

    def test_gasto_sin_presupuesto_pasa_a_categoria(self):
        mov,pre,conf,_=datos_demostracion()
        nuevo=pd.DataFrame([{'Fecha':pd.Timestamp('2026-10-11'),'Tipo':'Gasto','Categoria':'TRANSPORTE','Concepto':'TAXI','Descripcion':'Ficticio','Valor_COP':99999,'Mes':'2026-10'}])
        mov=pd.concat([mov,nuevo],ignore_index=True)
        ind=indicadores_mes(mov,pre,'2026-10',conf['saldo_inicial'])
        taxi=ind['conceptos'].loc[ind['conceptos']['Concepto']=='TAXI'].iloc[0]
        self.assertEqual(taxi['Alerta'],'SIN PRESUPUESTO')
        transport=ind['categorias'].loc[ind['categorias']['Categoria']=='TRANSPORTE'].iloc[0]
        self.assertAlmostEqual(transport['Real_COP'],float(ind['conceptos'].loc[ind['conceptos']['Categoria']=='TRANSPORTE','Real_COP'].sum()))

    def test_saldo_negativo_generar_consejo(self):
        mov,pre,conf,_=datos_demostracion()
        ind=indicadores_mes(mov,pre,'2026-01',saldo_inicial=-10000000)
        self.assertTrue(any('negativo' in texto.lower() for _,texto in consejos_financieros(ind)))

    def test_entrenamiento_temporal(self):
        mov,pre,conf,_=datos_demostracion()
        res=entrenar_modelos(mov,pre)
        self.assertEqual(len(res['metricas']),3)
        self.assertEqual(res['meses_test'],['2026-09','2026-10'])
        self.assertGreater(len(res['pronostico']),0)
        self.assertEqual(res['pronostico'].iloc[0]['Mes'],'2026-11')

if __name__=='__main__':unittest.main()
