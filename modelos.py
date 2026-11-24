"""Modelos para estimar gasto del mes siguiente con validación temporal.

Requisito mínimo operativo: 8 meses con registros y al menos 20 filas
para entrenamiento; 12+ meses completos son preferibles.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from analisis import detalle_por_concepto, key, calcular_estado

FEATURES = ['Categoria','Concepto','Presupuesto_COP','Lag1_COP','Lag2_COP','Mes_sin','Mes_cos']
NUMS = ['Presupuesto_COP','Lag1_COP','Lag2_COP','Mes_sin','Mes_cos']
CATS = ['Categoria','Concepto']


def _crear_panel(mov,pre):
    if mov.empty:
        raise ValueError('No hay movimientos. Registra los gastos reales antes de entrenar.')
    meses_observados=sorted(mov['Mes'].unique())
    if len(meses_observados)<8:
        raise ValueError(f'Solo hay {len(meses_observados)} meses con movimientos. Se requieren al menos 8 y se recomiendan 12 o más meses completos.')
    # No tratar presupuestos para meses futuros como gastos observados iguales a 0.
    ultimo=meses_observados[-1]
    conceptos=detalle_por_concepto(mov,pre)
    conceptos=conceptos[conceptos['Mes']<=ultimo].copy()
    if conceptos.empty:raise ValueError('No hay gastos ni presupuestos por concepto.')
    conceptos['CatKey']=conceptos['Categoria'].map(key)
    conceptos['ConKey']=conceptos['Concepto'].map(key)
    claves=conceptos[['CatKey','ConKey']].drop_duplicates()
    inicio=min(meses_observados)
    meses=[str(p) for p in pd.period_range(inicio,ultimo,freq='M')]
    if len(meses_observados)<len(meses):
        raise ValueError('Hay meses intermedios sin movimientos. Completa o confirma esos meses antes de entrenar; no asumimos silenciosamente gasto cero.')
    grid=pd.MultiIndex.from_product([meses,range(len(claves))],names=['Mes','Index']).to_frame(index=False)
    keys=claves.reset_index(drop=True)
    grid=grid.merge(keys,left_on='Index',right_index=True).drop(columns='Index')
    panel=grid.merge(conceptos,on=['Mes','CatKey','ConKey'],how='left')
    panel['Activo']=panel['Categoria'].notna()
    nombres=conceptos.sort_values('Mes').drop_duplicates(['CatKey','ConKey'],keep='last')
    catmap={(r.CatKey,r.ConKey):r.Categoria for r in nombres.itertuples(index=False)}
    conmap={(r.CatKey,r.ConKey):r.Concepto for r in nombres.itertuples(index=False)}
    panel['Categoria']=[catmap[(a,b)] for a,b in zip(panel['CatKey'],panel['ConKey'])]
    panel['Concepto']=[conmap[(a,b)] for a,b in zip(panel['CatKey'],panel['ConKey'])]
    panel['Presupuesto_COP']=panel['Presupuesto_COP'].fillna(0.0)
    panel['Real_COP']=panel['Real_COP'].fillna(0.0)
    panel=panel.sort_values(['CatKey','ConKey','Mes']).reset_index(drop=True)
    agrupado=panel.groupby(['CatKey','ConKey'],sort=False)['Real_COP']
    panel['Lag1_COP']=agrupado.shift(1).fillna(0.0)
    panel['Lag2_COP']=agrupado.shift(2).fillna(0.0)
    panel['Mes_num']=pd.to_datetime(panel['Mes']+'-01').dt.month
    panel['Mes_sin']=np.sin(2*np.pi*panel['Mes_num']/12)
    panel['Mes_cos']=np.cos(2*np.pi*panel['Mes_num']/12)
    panel['Ordinal']=pd.Categorical(panel['Mes'],categories=meses,ordered=True).codes
    entrenamiento=panel[(panel['Ordinal']>=2)&(panel['Activo'])].copy()
    if entrenamiento.empty:raise ValueError('No hay suficientes conceptos activos después de calcular los dos meses previos.')
    return panel,entrenamiento,meses


def _modelo(base):
    transform=ColumnTransformer([
        ('numericos',Pipeline([('completar',SimpleImputer(strategy='constant',fill_value=0)),('estandarizar',StandardScaler())]),NUMS),
        ('categoricos',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CATS)
    ])
    pipeline=Pipeline([('preprocesar',transform),('modelo',base)])
    return TransformedTargetRegressor(regressor=pipeline,transformer=StandardScaler())


def _estimadores():
    return {
        'Regresión Lineal':_modelo(LinearRegression()),
        'Random Forest':_modelo(RandomForestRegressor(n_estimators=120,min_samples_leaf=2,random_state=42,n_jobs=-1)),
        'MLP':_modelo(MLPRegressor(hidden_layer_sizes=(32,16),alpha=0.05,max_iter=800,random_state=42))
    }


def _forecast(panel,pre,mejor,ultimo_mes,umbral):
    siguiente=str(pd.Period(ultimo_mes,freq='M')+1)
    ult=panel[panel['Mes']==ultimo_mes].copy()
    if ult.empty:return pd.DataFrame()
    pre_fut=pre[pre['Mes']==siguiente].copy()
    pre_fut['CatKey']=pre_fut['Categoria'].map(key)
    pre_fut['ConKey']=pre_fut['Concepto'].map(key)
    # Crear nuevas combinaciones si ya se definieron presupuestos para el futuro.
    nuevas=pre_fut[['CatKey','ConKey','Categoria','Concepto','Presupuesto_COP']].copy()
    ult_claves=set(zip(ult['CatKey'],ult['ConKey']))
    nuevas=nuevas[[ (a,b) not in ult_claves for a,b in zip(nuevas['CatKey'],nuevas['ConKey']) ]]
    if not nuevas.empty:
        nuevas['Real_COP']=0.0;nuevas['Lag1_COP']=0.0
        ult=pd.concat([ult,nuevas],ignore_index=True)
    # Columnas de último mes: Real_COP = lag1 futuro; Lag1_COP = lag2 futuro.
    ult['Lag2_COP']=ult['Lag1_COP'].fillna(0.0)
    ult['Lag1_COP']=ult['Real_COP'].fillna(0.0)
    presup_map={(r.CatKey,r.ConKey):float(r.Presupuesto_COP) for r in pre_fut.itertuples(index=False)}
    ult['Presupuesto_provisional']=[(a,b) not in presup_map for a,b in zip(ult['CatKey'],ult['ConKey'])]
    ult['Presupuesto_COP']=[presup_map.get((a,b),float(old)) for a,b,old in zip(ult['CatKey'],ult['ConKey'],ult['Presupuesto_COP'])]
    mes_num=pd.Period(siguiente,freq='M').month
    ult['Mes_sin']=math.sin(2*math.pi*mes_num/12)
    ult['Mes_cos']=math.cos(2*math.pi*mes_num/12)
    ult['Mes']=siguiente
    # Solo conceptos con actividad/budget reciente. No inventar futuros gastos de conceptos inactivos.
    ult=ult[(ult['Lag1_COP']>0)|(ult['Presupuesto_COP']>0)].copy()
    if ult.empty:return pd.DataFrame()
    ult['Prediccion_COP']=np.maximum(0,mejor.predict(ult[FEATURES])).round(0)
    ult['Riesgo_predicho']=[calcular_estado(b,r,umbral) for b,r in zip(ult['Presupuesto_COP'],ult['Prediccion_COP'])]
    return ult[['Mes','Categoria','Concepto','Presupuesto_COP','Presupuesto_provisional','Prediccion_COP','Riesgo_predicho']].sort_values(['Categoria','Concepto']).reset_index(drop=True)


def entrenar_modelos(mov:pd.DataFrame,pre:pd.DataFrame,umbral=0.8):
    panel,entrenamiento,meses=_crear_panel(mov,pre)
    meses_modelo=sorted(entrenamiento['Mes'].unique())
    if len(meses_modelo)<4:raise ValueError('Se necesitan al menos 4 meses evaluables después de los rezagos.')
    meses_test=meses_modelo[-2:]
    tr=entrenamiento[~entrenamiento['Mes'].isin(meses_test)]
    te=entrenamiento[entrenamiento['Mes'].isin(meses_test)]
    if len(tr)<20 or len(te)<4:
        raise ValueError(f'Muestra demasiado pequeña: {len(tr)} registros de entrenamiento y {len(te)} de prueba. Añade más meses o conceptos reales.')
    resultados=[];modelos={}
    Xtr=tr[FEATURES];ytr=tr['Real_COP'];Xte=te[FEATURES];yte=te['Real_COP']
    for nombre,model in _estimadores().items():
        model.fit(Xtr,ytr)
        pron=np.maximum(0,model.predict(Xte))
        mae=float(mean_absolute_error(yte,pron));rmse=float(np.sqrt(mean_squared_error(yte,pron)))
        r2=float(r2_score(yte,pron)) if len(yte)>1 and yte.nunique()>1 else float('nan')
        resultados.append({'Modelo':nombre,'MAE_COP':mae,'RMSE_COP':rmse,'R2':r2,'Filas_prueba':len(te),'Meses_prueba':', '.join(meses_test)})
        modelos[nombre]=model
    metrics=pd.DataFrame(resultados).sort_values('MAE_COP').reset_index(drop=True)
    ganador=metrics.iloc[0]['Modelo']
    # Reajustar el modelo elegido con todos los meses observados, solo DESPUÉS de medir el error
    # del conjunto de prueba. Así el pronóstico futuro aprovecha el historial reciente sin
    # contaminar las métricas ya calculadas.
    modelo_futuro=_estimadores()[ganador]
    modelo_futuro.fit(entrenamiento[FEATURES],entrenamiento['Real_COP'])
    modelos[ganador]=modelo_futuro
    pred=_forecast(panel,pre,modelo_futuro,meses[-1],umbral)
    return {'metricas':metrics,'mejor_modelo':ganador,'modelos':modelos,'pronostico':pred,
            'meses_test':meses_test,'meses_modelados':len(meses),
            'filas_train':len(tr),'filas_test':len(te),
            'advertencia':'Resultados exploratorios: pocos meses y conceptos no independientes. No se garantiza precisión futura. Los presupuestos no definidos para el siguiente mes se consideran provisionales (último mes conocido).'}
