# Proyecto desde cero · Control inteligente y predicción de gastos personales

**Propósito universitario:** registrar y analizar gastos personales con presupuestos por concepto y categoría, calcular saldos, generar alertas y consejos basados en reglas claras, y comparar **Regresión Lineal, Random Forest Regressor y MLP Regressor** para pronosticar el gasto del siguiente mes.

**No necesitas archivos previos.** Este paquete incluye la plantilla de Excel, la aplicación web, el cuaderno Colab y las funciones de análisis y entrenamiento. Los datos de demostración son inventados, solo para pruebas.

## 1. Estructura del proyecto

```
Proyecto_Control_Gastos_IA/
├── plantilla/
│   └── Plantilla_Control_Gastos_IA.xlsx
├── colab/
│   └── 01_Inicio_Proyecto_Control_Gastos_IA.ipynb
├── app.py                 # Sitio Streamlit: descargar plantilla, cargar datos, alertas, gráficos, consejos, ML
├── analisis.py            # Limpieza, verificación, conceptos, categorías, saldos, consejos
├── modelos.py             # Regresión Lineal, Random Forest, MLP y validación temporal
├── requirements.txt       # Dependencias del proyecto
├── tests/test_basicos.py  # Pruebas básicas de lectura, alertas, saldos y ML
└── README.md
```

## 2. Completar el Excel de forma correcta

Abre `plantilla/Plantilla_Control_Gastos_IA.xlsx`. El archivo principal está **vacío** (la hoja EJEMPLOS contiene números ficticios que **no se suman**).

1. **CONFIG:** B4 = mes de interés (`2026-11`), B5 = saldo inicial antes del primer movimiento que registrarás, B6 = umbral de alerta (`0.80`).
2. **PRESUPUESTOS:** escribe **una fila por mes + categoría + concepto** en columnas A–D. Ejemplo: `2026-11 | TRANSPORTE | GASOLINA | 200000`.
3. **MOVIMIENTOS:** columnas A–F, `Fecha | Tipo | Categoria | Concepto | Descripcion | Valor_COP`. Ejemplo: `2026-11-03 | Gasto | TRANSPORTE | GASOLINA | Recarga | 230000`. Los importes son positivos. Si realizas tres compras del mismo concepto, registra tres movimientos independientes.
4. Para **SUELDO**, utiliza `Tipo = Ingreso`, `Categoria = INGRESOS`, `Concepto = SUELDO`. **Nunca sumes el sueldo a los gastos.**
5. No agregues los totales o subtotales de tus presupuestos como movimientos: las sumas se calculan automáticamente.
6. Consulta **PRESUPUESTOS** para las alertas por concepto, **CATEGORIAS** para el resumen por categoría y **RESUMEN** para balances y consejos.
7. Las columnas calculadas del Excel cubren 500 movimientos y 200 presupuestos. Si necesitas más, la aplicación procesa cualquier número de filas con la misma estructura y encabezados; puedes extender las fórmulas Excel copiándolas hacia abajo.
8. Usa una **misma denominación de categoría y concepto** en presupuestos y movimientos. La aplicación tolera diferencias de acentos y mayúsculas, pero el Excel funciona mejor con texto idéntico.

### Interpretación de las alertas

- **NORMAL:** menos del 80 % del presupuesto utilizado.
- **ADVERTENCIA:** entre 80 % (incluido) y menos del 100 %.
- **EN EL LÍMITE:** 100 % exacto.
- **EXCEDIDO:** gasto superior al monto presupuestado.
- **SIN PRESUPUESTO:** hay gastos para un concepto que no tiene monto planeado.
- **SIN DEFINIR:** hay una fila de presupuesto con valor cero y ningún gasto real.

**Importante:** La hoja PRESUPUESTOS solo puede mostrar conceptos que hayas definido allí. La aplicación también detecta **conceptos con gastos pero sin presupuesto**, y la hoja CATEGORIAS suma los gastos directamente de MOVIMIENTOS para no ocultarlos.

## 3. Abrir en Google Colab desde cero

1. Entra a [Google Colab](https://colab.research.google.com/).
2. Extrae este ZIP en tu computador y abre en Colab `colab/01_Inicio_Proyecto_Control_Gastos_IA.ipynb` mediante **Archivo → Subir cuaderno**.
3. Ejecuta la primera celda para instalar dependencias.
4. En la etapa 2, sube el **archivo ZIP original completo (`Proyecto_Control_Gastos_IA_Completo.zip`)**; el cuaderno lo extrae para disponer de `analisis.py` y `modelos.py`.
5. En la etapa 4, sube tu **Excel diligenciado**. Si todavía no tienes datos, activa `usar_ejemplo=True` solamente para una prueba ficticia.
6. Continúa ejecutando las celdas por orden: validación → saldos → alertas → gráficos → entrenamiento → pronósticos → descargas.
7. Recibirás un ZIP con los resultados, el modelo mejor evaluado (si se pudo entrenar) y los CSV necesarios para documentar el proyecto.

**Por qué ya no buscamos títulos de categoría en celdas variables de tu Excel anterior:** la nueva plantilla utiliza dos tablas uniformes. Esto evita confundir encabezados con conceptos como CARO y reduce errores por totales duplicados. No utilizamos un supuesto total mensual variable ni reproducimos el error de tratar SUELDO como gasto.

## 4. Ejecutar la aplicación Streamlit en Windows

Necesitas **Python 3.11 o posterior** y conexión a Internet para instalar las bibliotecas la primera vez.

Abre la carpeta extraída en VS Code y una terminal (PowerShell o Git Bash). Ejecuta:

```bash
python -m venv .venv
```

**PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

**Git Bash:**

```bash
source .venv/Scripts/activate
pip install -r requirements.txt
streamlit run app.py
```

La consola mostrará una dirección local similar a `http://localhost:8501`. Ábrela en tu navegador.

**En la aplicación podrás:**

- Descargar la plantilla vacía tantas veces como quieras.
- Subir el Excel con tus registros y ver indicadores por mes.
- Revisar alertas reales por concepto y categoría, incluso gastos sin presupuesto.
- Consultar ingresos, gastos, balance mensual y saldo acumulado.
- Obtener consejos presupuestarios basados en el saldo real, balance, excesos y conceptos críticos.
- Graficar la evolución de ingresos, gastos y saldos.
- Entrenar los tres modelos y descargar pronósticos cuando haya suficientes datos.
- Probar el flujo con **datos ficticios de demostración** sin introducir información real.

## 5. Diseño del Machine Learning

- **Problema:** regresión para estimar el **gasto real de un concepto en el mes siguiente**.
- **Variables de entrada:** categoría, concepto, presupuesto conocido del mes objetivo, gasto del concepto en el mes previo, gasto de dos meses atrás, componente estacional del calendario.
- **Objetivo:** valor real gastado en COP.
- **Evaluación:** los **últimos dos meses evaluables** se reservan como prueba. No se mezclan filas aleatoriamente entre entrenamiento y prueba.
- **Modelos:** Regresión Lineal (referencia), Random Forest y MLP.
- **Métricas:** MAE, RMSE y R², sobre la muestra de prueba.
- **Umbral mínimo técnico:** 8 meses consecutivos con movimientos y 20 registros de entrenamiento, con al menos 4 registros de prueba; **se recomiendan 12 o más meses completos**. Cumplir el mínimo no garantiza modelos confiables.
- **Pronóstico futuro:** para el siguiente mes, utiliza el presupuesto real si ya lo definiste; de lo contrario se usa un monto provisional del último mes, etiquetado claramente.
- **Limitación importante:** con solo 8–15 meses, las métricas son exploratorias y pueden variar mucho. Las categorías y conceptos pueden no ser estadísticamente independientes. Si faltan registros, los modelos pierden confiabilidad.

### Alertas vs Machine Learning

Las **alertas actuales** por exceso de presupuesto se calculan exactamente: `Real > Presupuestado`. **No necesitan ML**. Las **alertas predictivas** se basan en estimaciones: `Pronóstico > Presupuesto del mes siguiente`; son advertencias, no certezas.

### Consejos financieros

Se generan mediante **reglas transparentes** relacionadas con saldo acumulado, balance del mes, margen disponible, ingresos faltantes, gasto por encima del presupuesto y conceptos de mayor exceso. Son sugerencias educativas sobre organización del dinero; no son una recomendación de inversión, crédito o contratación de productos financieros.

## 6. Pruebas y privacidad

Para ejecutar pruebas automáticas:

```bash
python -m unittest discover -s tests -v
```

Esta versión analiza archivos **en tu computador** al ejecutarla localmente. Si la publicas en un servidor, revisa la seguridad del alojamiento, las políticas de retención y el acceso a los archivos cargados. Evita publicar datos financieros personales en repositorios públicos. **Google Colab procesa los archivos en la infraestructura en la nube de Google**; utiliza datos anónimos o ficticios si no deseas subir información sensible.

## 7. Secuencia de trabajo académico

1. Planteamiento del problema y objetivos.
2. Diseño de plantilla normalizada por concepto y categoría.
3. Ingesta y validación de datos desde Colab.
4. Análisis exploratorio de ingresos, gastos, balances y categorías.
5. Alertas actuales y consejos deterministas.
6. Construcción de variables y validación temporal.
7. Entrenamiento y comparación de tres regresores.
8. Predicción del mes siguiente y alertas predictivas.
9. Implementación en Streamlit.
10. Pruebas, resultados, discusión de limitaciones y sustentación.

**Fecha de creación:** noviembre de 2026. **Moneda:** pesos colombianos (COP).
