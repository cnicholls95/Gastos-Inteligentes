"""Aplicación de gastos personales: plantillas, saldos, alertas y ML.
Ejecutar: streamlit run app.py
"""
from __future__ import annotations
import io
import hashlib
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from analisis import (leer_excel,datos_demostracion,detalle_por_concepto,
                      indicadores_mes,consejos_financieros,saldos_por_mes)
from modelos import entrenar_modelos

ROOT=Path(__file__).resolve().parent
PLANTILLA=ROOT/'plantilla'/'Plantilla_Control_Gastos_IA.xlsx'
COP=lambda valor:('$'+f'{valor:,.0f}'.replace(',','.')+' COP')

st.set_page_config(page_title='Control inteligente de gastos',page_icon='💰',layout='wide')
st.title('💰 Control inteligente de gastos personales')
st.caption('Presupuestos por concepto y categoría · Alertas en tiempo real sobre datos cargados · Consejos basados en saldo · Predicción exploratoria con ML')

with st.sidebar:
    st.header('📁 Empezar')
    st.download_button('⬇️ Descargar plantilla de Excel',data=PLANTILLA.read_bytes(),file_name=PLANTILLA.name,mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',use_container_width=True)
    uploaded=st.file_uploader('Sube tu Excel ya diligenciado',type=['xlsx'])
    demo=st.checkbox('Probar con datos ficticios de demostración',value=False,disabled=uploaded is not None)
    st.info('La plantilla y la aplicación funcionan desde cero. El entrenamiento ML requiere al menos 8 meses con registros (preferible 12+).')

if uploaded is None and not demo:
    st.subheader('Primer paso')
    st.markdown('''1. Descarga la **plantilla de Excel** desde el panel lateral.
2. En **CONFIG** indica tu saldo inicial y mes a analizar.
3. Llena **PRESUPUESTOS** con el monto planeado por **mes + categoría + concepto**.
4. Llena **MOVIMIENTOS** con ingresos y gastos individuales (siempre valores positivos).
5. Guarda tu archivo y súbelo aquí para ver alertas, resúmenes, consejos y predicciones cuando haya historial suficiente.''')
    st.warning('No pongas el sueldo como gasto. Usa Tipo = Ingreso, Categoría = INGRESOS, Concepto = SUELDO.')
    st.stop()

try:
    mov,pre,conf,avisos=datos_demostracion() if demo else leer_excel(uploaded)
except Exception as exc:
    st.error(f'No se puede procesar el archivo: {exc}')
    st.stop()

firma_archivo = 'demo' if demo else hashlib.sha256(uploaded.getvalue()).hexdigest()

if demo:
    st.warning('**MODO DEMOSTRACIÓN:** todos los importes son ficticios. No los utilices como resultados de tu investigación.')
for aviso in avisos:st.info(aviso)

meses=sorted(set(mov['Mes'].tolist())|set(pre['Mes'].tolist()))
if not meses:
    st.info('El archivo es válido, pero aún está vacío. Registra movimientos y presupuestos en Excel y vuelve a cargarlo.')
    st.stop()

with st.sidebar:
    defecto=conf['mes'] if conf['mes'] in meses else meses[-1]
    mes=st.selectbox('Mes para analizar',meses,index=meses.index(defecto))
    saldo_inicial=st.number_input('Saldo inicial antes del primer movimiento (COP)',value=float(conf['saldo_inicial']),step=50000.0,format='%.0f')
    umbral=st.slider('Advertencia al usar el presupuesto (%)',min_value=50,max_value=99,value=int(round(conf['umbral']*100)),step=1)/100.0

ind=indicadores_mes(mov,pre,mes,saldo_inicial,umbral)
conceptos=ind['conceptos'];categorias=ind['categorias'];historico=saldos_por_mes(mov,saldo_inicial)

m1,m2,m3,m4=st.columns(4)
m1.metric('Ingresos del mes',COP(ind['Ingresos_COP']))
m2.metric('Gastos del mes',COP(ind['Gastos_COP']))
m3.metric('Balance del mes',COP(ind['Balance_COP']))
m4.metric('Saldo acumulado',COP(ind['Saldo_COP']))

st.caption(f'Presupuesto total: {COP(ind["Presupuesto_COP"])} · Diferencia frente al gasto: {COP(ind["Disponible_Presupuesto_COP"])} · Mes: {mes}')
t1,t2,t3,t4=st.tabs(['🚦 Alertas y conceptos','📊 Categorías y evolución','💡 Consejos según tu saldo','🤖 Predicción con ML'])

with t1:
    st.subheader('Control por concepto')
    if conceptos.empty:st.info('No hay conceptos con presupuesto ni gasto en este mes.')
    else:
        resumen=conceptos['Alerta'].value_counts()
        colA,colB,colC=st.columns(3)
        colA.metric('Conceptos excedidos',int(resumen.get('EXCEDIDO',0)))
        colB.metric('Sin presupuesto',int(resumen.get('SIN PRESUPUESTO',0)))
        colC.metric('En advertencia',int(resumen.get('ADVERTENCIA',0)))
        peligros=conceptos[conceptos['Alerta'].isin(['EXCEDIDO','SIN PRESUPUESTO'])]
        if not peligros.empty:
            st.error(f'⚠️ {len(peligros)} concepto(s) con gasto fuera del presupuesto o sin límite asignado.')
        st.dataframe(conceptos.sort_values('Disponible_COP').style.format({'Presupuesto_COP':'{:,.0f}','Real_COP':'{:,.0f}','Disponible_COP':'{:,.0f}','Uso_%':'{:.1%}'}),use_container_width=True,hide_index=True)
        st.download_button('Descargar control por conceptos (CSV)',conceptos.to_csv(index=False).encode('utf-8-sig'),file_name=f'conceptos_{mes}.csv',mime='text/csv')

with t2:
    st.subheader('Control por categoría')
    if not categorias.empty:
        st.dataframe(categorias.style.format({'Presupuesto_COP':'{:,.0f}','Real_COP':'{:,.0f}','Disponible_COP':'{:,.0f}','Uso_%':'{:.1%}'}),use_container_width=True,hide_index=True)
        fig=px.bar(categorias.sort_values('Real_COP',ascending=False),x='Categoria',y=['Presupuesto_COP','Real_COP'],barmode='group',labels={'value':'COP','variable':'Tipo'},title='Presupuesto frente a gasto real')
        st.plotly_chart(fig,use_container_width=True)
    if not historico.empty:
        fig2=px.line(historico,x='Mes',y=['Ingresos_COP','Gastos_COP','Saldo_COP'],markers=True,title='Evolución de ingresos, gastos y saldo acumulado')
        st.plotly_chart(fig2,use_container_width=True)

with t3:
    st.subheader('Consejos basados en tus saldos y presupuestos')
    st.caption('Estas recomendaciones siguen reglas transparentes; no implican entrenamiento de inteligencia artificial ni constituyen asesoramiento financiero individual profesional.')
    for nivel,consejo in consejos_financieros(ind):
        if '🔴' in nivel:st.error(f'**{nivel}** · {consejo}')
        elif '🟡' in nivel:st.warning(f'**{nivel}** · {consejo}')
        else:st.info(f'**{nivel}** · {consejo}')

with t4:
    st.subheader('Comparación de Regresión Lineal, Random Forest y MLP')
    st.write('El entrenamiento utiliza **gasto histórico de cada concepto**, los dos meses previos y el presupuesto conocido. La evaluación reserva cronológicamente los últimos dos meses evaluables como prueba.')
    st.warning('Los modelos no sustituyen las alertas: estas comparan cifras reales. Los pronósticos solo estiman posibles excesos futuros.')
    if st.button('Entrenar y evaluar modelos',type='primary'):
        try:
            with st.spinner('Entrenando modelos con tus datos...'):
                st.session_state['modelos_resultados']=entrenar_modelos(mov,pre,umbral)
            st.session_state['firma_datos']=(firma_archivo,mes,len(mov),len(pre),umbral)
        except Exception as exc:
            st.error(f'No es posible entrenar todavía: {exc}')
            st.session_state.pop('modelos_resultados',None)
    firma=(firma_archivo,mes,len(mov),len(pre),umbral)
    result=st.session_state.get('modelos_resultados') if st.session_state.get('firma_datos')==firma else None
    if result is not None:
        st.success(f'Modelo con menor MAE en prueba: **{result["mejor_modelo"]}**')
        st.dataframe(result['metricas'].style.format({'MAE_COP':'{:,.0f}','RMSE_COP':'{:,.0f}','R2':'{:.3f}'}),use_container_width=True,hide_index=True)
        st.caption(f'Entrenamiento: {result["filas_train"]} registros · Prueba: {result["filas_test"]} registros · Meses de prueba: {", ".join(result["meses_test"])}')
        st.warning(result['advertencia'])
        pron=result['pronostico']
        if not pron.empty:
            st.subheader('Estimación para el siguiente mes')
            st.dataframe(pron.style.format({'Presupuesto_COP':'{:,.0f}','Prediccion_COP':'{:,.0f}'}),use_container_width=True,hide_index=True)
            proy_cat=pron.groupby(['Mes','Categoria'],as_index=False)[['Presupuesto_COP','Prediccion_COP']].sum()
            proy_cat['Exceso_estimado_COP']=(proy_cat['Prediccion_COP']-proy_cat['Presupuesto_COP']).clip(lower=0)
            st.markdown('**Estimación agregada por categoría**')
            st.dataframe(proy_cat.style.format({'Presupuesto_COP':'{:,.0f}','Prediccion_COP':'{:,.0f}','Exceso_estimado_COP':'{:,.0f}'}),use_container_width=True,hide_index=True)
            st.download_button('Descargar pronóstico (CSV)',pron.to_csv(index=False).encode('utf-8-sig'),file_name='prediccion_siguiente_mes.csv',mime='text/csv')
        st.caption('La señal de riesgo se calcula sobre el gasto estimado y el presupuesto siguiente. Cuando no existe presupuesto futuro, se usa de manera provisional el último conocido.')

st.divider()
st.caption('Privacidad: esta versión ejecuta el análisis localmente en Streamlit; no necesita enviar registros financieros a un servicio externo. Si despliegas la app en la nube, revisa la configuración de alojamiento, acceso y retención de datos.')
