"""Lectura, validación y reglas deterministas del control presupuestal.

Entradas: hojas MOVIMIENTOS y PRESUPUESTOS de la plantilla descargable.
Los consejos y las alertas se calculan con reglas; no se presentan como IA.
"""
from __future__ import annotations
import io
import re
import unicodedata
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

COL_MOV = ['Fecha','Tipo','Categoria','Concepto','Descripcion','Valor_COP']
COL_BUD = ['Mes','Categoria','Concepto','Presupuesto_COP']
CAT_INCOME = 'INGRESOS'


def key(value: Any) -> str:
    """Llave tolerante a tildes, espacios y mayúsculas."""
    if pd.isna(value): return ''
    x = ' '.join(str(value).strip().upper().split())
    return ''.join(c for c in unicodedata.normalize('NFD',x) if unicodedata.category(c)!='Mn')


def display(value: Any) -> str:
    return '' if pd.isna(value) else ' '.join(str(value).strip().upper().split())


def mes_valido(value: Any) -> str:
    if isinstance(value,(pd.Timestamp,)):
        return value.strftime('%Y-%m') if not pd.isna(value) else ''
    if hasattr(value,'strftime') and not isinstance(value,str):
        return value.strftime('%Y-%m')
    txt='' if pd.isna(value) else str(value).strip()
    return txt if re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])',txt) else ''


def _read_sheet(book:pd.ExcelFile,name:str,columns:list[str]):
    df=pd.read_excel(book,sheet_name=name,header=3,engine='openpyxl',dtype=object)
    needed=set(columns)-set(df.columns)
    if needed:raise ValueError(f'La hoja {name} no tiene columnas obligatorias: {sorted(needed)}. Utiliza la plantilla oficial.')
    df=df[columns].copy()
    df['Fila_excel']=df.index+5
    return df[df[columns].notna().any(axis=1)].reset_index(drop=True)


def leer_excel(archivo: Any):
    """Retorna movimientos, presupuestos, config, advertencias.
    Rechaza filas incompletas en vez de dar resultados engañosos.
    """
    libro=pd.ExcelFile(archivo, engine='openpyxl')
    requeridas={'MOVIMIENTOS','PRESUPUESTOS'}
    if not requeridas.issubset(libro.sheet_names):
        raise ValueError('Faltan hojas MOVIMIENTOS o PRESUPUESTOS. Descarga y utiliza la plantilla oficial.')
    mov=_read_sheet(libro,'MOVIMIENTOS',COL_MOV)
    pre=_read_sheet(libro,'PRESUPUESTOS',COL_BUD)
    conf={'mes':'','saldo_inicial':0.0,'umbral':0.8}
    if 'CONFIG' in libro.sheet_names:
        cfg=pd.read_excel(libro,sheet_name='CONFIG',header=None,engine='openpyxl')
        def celda(fila,col):
            return cfg.iat[fila-1,col-1] if fila<=cfg.shape[0] and col<=cfg.shape[1] else None
        conf['mes']=mes_valido(celda(4,2))
        raw=celda(5,2)
        if pd.notna(raw):
            try:conf['saldo_inicial']=float(raw)
            except (ValueError,TypeError):raise ValueError('CONFIG!B5: el saldo inicial debe ser numérico.')
        raw=celda(6,2)
        if pd.notna(raw):
            try:conf['umbral']=float(raw)
            except (ValueError,TypeError):raise ValueError('CONFIG!B6: el umbral de aviso debe ser numérico.')
        if not (0<conf['umbral']<1):raise ValueError('CONFIG!B6 debe estar entre 0 y 1 (por ejemplo, 0,80).')
    advertencias=[]
    if not mov.empty:
        mov['Fecha']=pd.to_datetime(mov['Fecha'],errors='coerce')
        mov['Tipo']=mov['Tipo'].map(lambda x:display(x).capitalize())
        mov['Categoria']=mov['Categoria'].map(display)
        mov['Concepto']=mov['Concepto'].map(display)
        mov['Descripcion']=mov['Descripcion'].fillna('').map(str)
        mov['Valor_COP']=pd.to_numeric(mov['Valor_COP'],errors='coerce')
        error=mov['Fecha'].isna() | ~mov['Tipo'].isin(['Gasto','Ingreso']) | mov['Categoria'].eq('') | mov['Concepto'].eq('') | mov['Valor_COP'].isna() | mov['Valor_COP'].le(0)
        if error.any():
            indices=mov.loc[error,'Fila_excel'].tolist()
            raise ValueError(f'MOVIMIENTOS: revisa las filas {indices[:15]}; fecha, tipo, categoría, concepto y valor positivo son obligatorios.')
        ingresos_as_gasto=mov['Tipo'].eq('Gasto') & mov['Categoria'].map(key).eq('INGRESOS')
        if ingresos_as_gasto.any():
            indices=mov.loc[ingresos_as_gasto,'Fila_excel'].tolist()
            raise ValueError(f'MOVIMIENTOS: INGRESOS aparece como Gasto en filas {indices}. El sueldo es Ingreso.')
        mov['Mes']=mov['Fecha'].dt.strftime('%Y-%m')
    else:
        mov['Mes']=pd.Series(dtype=str)
    if not pre.empty:
        pre['Mes']=pre['Mes'].map(mes_valido)
        pre['Categoria']=pre['Categoria'].map(display)
        pre['Concepto']=pre['Concepto'].map(display)
        pre['Presupuesto_COP']=pd.to_numeric(pre['Presupuesto_COP'],errors='coerce')
        error=pre['Mes'].eq('') | pre['Categoria'].eq('') | pre['Concepto'].eq('') | pre['Presupuesto_COP'].isna() | pre['Presupuesto_COP'].lt(0)
        if error.any():
            indices=pre.loc[error,'Fila_excel'].tolist()
            raise ValueError(f'PRESUPUESTOS: revisa filas {indices[:15]}; usa Mes AAAA-MM, categoría, concepto y presupuesto numérico no negativo.')
        if pre['Categoria'].map(key).eq('INGRESOS').any():
            raise ValueError('PRESUPUESTOS: INGRESOS no se presupuestan como gasto. Define únicamente presupuestos de gastos.')
        claves=pre['Mes']+'|'+pre['Categoria'].map(key)+'|'+pre['Concepto'].map(key)
        if claves.duplicated().any():
            indices=pre.loc[claves.duplicated(keep=False),'Fila_excel'].tolist()
            raise ValueError(f'PRESUPUESTOS: hay conceptos duplicados para el mismo mes y categoría en filas {indices[:20]}. Consolida el presupuesto en una sola fila.')
    if not mov.empty:
        ingreso_sueldo=mov['Concepto'].map(key).eq('SUELDO') & mov['Tipo'].eq('Gasto')
        if ingreso_sueldo.any():
            raise ValueError('El concepto SUELDO fue registrado como Gasto. Regístralo como Ingreso.')
    return mov,pre,conf,advertencias


def calcular_estado(presupuesto:float,real:float,umbral:float=0.8) -> str:
    presupuesto=float(presupuesto);real=float(real)
    if presupuesto<=0:return 'SIN PRESUPUESTO' if real>0 else 'SIN DEFINIR'
    if real>presupuesto:return 'EXCEDIDO'
    if np.isclose(real,presupuesto,atol=0.01):return 'EN EL LIMITE'
    if real/presupuesto>=umbral:return 'ADVERTENCIA'
    return 'NORMAL'


def _items(df,amount):
    if df.empty:
        return pd.DataFrame(columns=['Mes','CatKey','ConKey','Categoria','Concepto',amount])
    d=df.copy()
    d['CatKey']=d['Categoria'].map(key)
    d['ConKey']=d['Concepto'].map(key)
    return d.groupby(['Mes','CatKey','ConKey'],as_index=False).agg(
        Categoria=('Categoria','first'),Concepto=('Concepto','first'),**{amount:(amount,'sum')}
    )


def detalle_por_concepto(mov:pd.DataFrame,pre:pd.DataFrame,umbral=0.8):
    gastos=mov[mov['Tipo']=='Gasto'] if not mov.empty else mov
    a=_items(pre,'Presupuesto_COP');b=_items(gastos,'Valor_COP')
    merged=a.merge(b,on=['Mes','CatKey','ConKey'],how='outer',suffixes=('_pre','_mov'))
    if merged.empty:
        return pd.DataFrame(columns=['Mes','Categoria','Concepto','Presupuesto_COP','Real_COP','Disponible_COP','Uso_%','Alerta'])
    merged['Categoria']=merged['Categoria_pre'].combine_first(merged['Categoria_mov'])
    merged['Concepto']=merged['Concepto_pre'].combine_first(merged['Concepto_mov'])
    merged['Presupuesto_COP']=merged['Presupuesto_COP'].fillna(0.0)
    merged['Real_COP']=merged['Valor_COP'].fillna(0.0)
    merged['Disponible_COP']=merged['Presupuesto_COP']-merged['Real_COP']
    merged['Uso_%']=np.where(merged['Presupuesto_COP']>0,merged['Real_COP']/merged['Presupuesto_COP'],np.where(merged['Real_COP']>0,1.0,0.0))
    merged['Alerta']=[calcular_estado(b,r,umbral) for b,r in zip(merged['Presupuesto_COP'],merged['Real_COP'])]
    return merged[['Mes','Categoria','Concepto','Presupuesto_COP','Real_COP','Disponible_COP','Uso_%','Alerta']].sort_values(['Mes','Categoria','Concepto']).reset_index(drop=True)


def detalle_por_categoria(conceptos:pd.DataFrame,umbral=0.8):
    if conceptos.empty:return pd.DataFrame(columns=['Mes','Categoria','Presupuesto_COP','Real_COP','Disponible_COP','Uso_%','Alerta'])
    d=conceptos.copy();d['CatKey']=d['Categoria'].map(key)
    grupos=d.groupby(['Mes','CatKey'],as_index=False).agg(Categoria=('Categoria','first'),Presupuesto_COP=('Presupuesto_COP','sum'),Real_COP=('Real_COP','sum'))
    grupos['Disponible_COP']=grupos['Presupuesto_COP']-grupos['Real_COP']
    grupos['Uso_%']=np.where(grupos['Presupuesto_COP']>0,grupos['Real_COP']/grupos['Presupuesto_COP'],np.where(grupos['Real_COP']>0,1.0,0.0))
    grupos['Alerta']=[calcular_estado(b,r,umbral) for b,r in zip(grupos['Presupuesto_COP'],grupos['Real_COP'])]
    return grupos[['Mes','Categoria','Presupuesto_COP','Real_COP','Disponible_COP','Uso_%','Alerta']].sort_values(['Mes','Categoria']).reset_index(drop=True)


def saldos_por_mes(mov:pd.DataFrame,saldo_inicial=0.0):
    if mov.empty:return pd.DataFrame(columns=['Mes','Ingresos_COP','Gastos_COP','Balance_COP','Saldo_COP'])
    d=mov.pivot_table(index='Mes',columns='Tipo',values='Valor_COP',aggfunc='sum',fill_value=0)
    d=d.reindex(columns=['Ingreso','Gasto'],fill_value=0).reset_index()
    d=d.rename(columns={'Ingreso':'Ingresos_COP','Gasto':'Gastos_COP'}).sort_values('Mes')
    d['Balance_COP']=d['Ingresos_COP']-d['Gastos_COP']
    d['Saldo_COP']=float(saldo_inicial)+d['Balance_COP'].cumsum()
    return d


def indicadores_mes(mov,pre,mes,saldo_inicial=0.0,umbral=0.8):
    conceptos=detalle_por_concepto(mov,pre,umbral)
    cats=detalle_por_categoria(conceptos,umbral)
    c=conceptos[conceptos['Mes']==mes].copy();ct=cats[cats['Mes']==mes].copy()
    ingresos=float(mov.loc[(mov['Mes']==mes)&(mov['Tipo']=='Ingreso'),'Valor_COP'].sum()) if not mov.empty else 0.0
    gastos=float(mov.loc[(mov['Mes']==mes)&(mov['Tipo']=='Gasto'),'Valor_COP'].sum()) if not mov.empty else 0.0
    presup=float(pre.loc[pre['Mes']==mes,'Presupuesto_COP'].sum()) if not pre.empty else 0.0
    # El saldo inicial representa el dinero disponible antes del primer movimiento del libro.
    anteriores=mov[mov['Mes']<=mes] if not mov.empty else mov
    acumulado=float(saldo_inicial)+float(anteriores.loc[anteriores['Tipo']=='Ingreso','Valor_COP'].sum())-float(anteriores.loc[anteriores['Tipo']=='Gasto','Valor_COP'].sum()) if not anteriores.empty else float(saldo_inicial)
    return {'Mes':mes,'Ingresos_COP':ingresos,'Gastos_COP':gastos,'Balance_COP':ingresos-gastos,
            'Saldo_COP':acumulado,'Presupuesto_COP':presup,'Disponible_Presupuesto_COP':presup-gastos,
            'conceptos':c,'categorias':ct,'detalle_conceptos_completo':conceptos,'detalle_categorias_completo':cats}


def consejos_financieros(indicadores:dict):
    """Consejos de organización del presupuesto, no recomendaciones de inversión."""
    consejos=[];saldo=indicadores['Saldo_COP'];balance=indicadores['Balance_COP'];ingresos=indicadores['Ingresos_COP'];gastos=indicadores['Gastos_COP'];presup=indicadores['Presupuesto_COP']
    f=lambda x:f'${x:,.0f} COP'
    if saldo<0:
        consejos.append(('🔴 Prioridad alta',f'Tu saldo acumulado es negativo ({f(saldo)}). Revisa los compromisos próximos y prioriza alimentación, vivienda y otros gastos esenciales.'))
    elif saldo < gastos*0.5 and gastos>0:
        consejos.append(('🟡 Margen ajustado',f'Tu saldo acumulado es {f(saldo)}, inferior a la mitad de lo gastado este mes. Antes de nuevos compromisos, conserva margen para imprevistos.'))
    elif gastos>0:
        consejos.append(('🟢 Saldo positivo',f'Tu saldo acumulado es {f(saldo)}. Mantén un colchón para imprevistos y supervisa los conceptos próximos al límite.'))
    if ingresos<=0 and gastos>0:
        consejos.append(('🟡 Ingresos no registrados','Tienes gastos este mes, pero no hay ingresos registrados. Verifica si faltan entradas de dinero antes de interpretar el balance.'))
    elif balance<0:
        consejos.append(('🔴 Déficit mensual',f'Tus gastos superaron los ingresos en {f(-balance)} este mes. Busca primero los conceptos no esenciales con mayor exceso.'))
    elif ingresos>0 and gastos>0:
        porcentaje=100*balance/ingresos
        if porcentaje>0:
            consejos.append(('🟢 Balance del mes',f'Tu balance mensual es positivo ({f(balance)}, aproximadamente {porcentaje:.1f}% de tus ingresos). Considera reservar parte como margen de seguridad.'))
    if gastos>presup:
        consejos.append(('🔴 Presupuesto global excedido',f'Superaste el presupuesto total del mes en {f(gastos-presup)}. Revisa gastos sin presupuesto y los conceptos excedidos.'))
    elif presup>0 and gastos/presup>=0.8:
        consejos.append(('🟡 Presupuesto casi agotado',f'Ya utilizaste {100*gastos/presup:.1f}% del presupuesto total. Prioriza lo pendiente y evita gastos no planificados.'))
    conceptos=indicadores['conceptos']; categorias=indicadores['categorias']
    if not conceptos.empty:
        crit=conceptos[conceptos['Disponible_COP']<0].sort_values('Disponible_COP')
        if not crit.empty:
            fila=crit.iloc[0]
            consejos.append(('🔴 Concepto con mayor exceso',f'{fila["Concepto"]} ({fila["Categoria"]}) supera lo previsto en {f(-fila["Disponible_COP"])}. Revisa si ese gasto puede reducirse o ajusta un presupuesto futuro realista.'))
        sin=conceptos[conceptos['Alerta']=='SIN PRESUPUESTO']
        if not sin.empty:
            consejos.append(('🟡 Gastos sin presupuesto',f'Hay {len(sin)} concepto(s) con gastos y sin presupuesto registrado. Asigna límites para monitorearlos.'))
    if not consejos:
        consejos.append(('ℹ️ Sin datos suficientes','Registra ingresos, gastos y presupuestos del mes seleccionado para recibir consejos basados en tus saldos.'))
    return consejos


def datos_demostracion():
    """Genera DEMO FICTICIA, nunca presentada como datos del usuario."""
    import calendar
    rng=np.random.default_rng(102)
    meses=pd.period_range('2026-01','2026-10',freq='M')
    items=[('VIVIENDA','ARRIENDO',700000),('ALIMENTACIÓN','MERCADO',430000),('TRANSPORTE','GASOLINA',210000),('DIOS','DIEZMO',150000),('SUKI','ALIMENTO MASCOTA',120000),('DEUDAS','TARJETA',200000),('CUIDADO PERSONAL','SHAMPOO',35000)]
    mov=[];pre=[]
    for periodo in meses:
        mes=str(periodo)
        mov.append({'Fecha':pd.Timestamp(f'{mes}-01'),'Tipo':'Ingreso','Categoria':'INGRESOS','Concepto':'SUELDO','Descripcion':'Ingreso ficticio','Valor_COP':3000000+int(rng.integers(-150000,150000)),'Mes':mes})
        for j,(cat,conc,bud) in enumerate(items):
            pre.append({'Mes':mes,'Categoria':cat,'Concepto':conc,'Presupuesto_COP':bud})
            real=max(1000,round(bud*float(rng.uniform(0.72,1.25)),0))
            dia=3+j
            mov.append({'Fecha':pd.Timestamp(f'{mes}-{dia:02d}'),'Tipo':'Gasto','Categoria':cat,'Concepto':conc,'Descripcion':'Gasto ficticio','Valor_COP':float(real),'Mes':mes})
    return pd.DataFrame(mov),pd.DataFrame(pre),{'mes':'2026-10','saldo_inicial':500000.0,'umbral':0.8},['ATENCIÓN: valores inventados exclusivamente para probar la aplicación; no son datos financieros reales.']
