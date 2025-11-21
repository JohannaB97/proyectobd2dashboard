# Proyecto: Sistema Predictivo de Precios de Casas
## MongoDB en Azure + Machine Learning + Dashboard Interactivo

---

## RESUMEN EJECUTIVO

Este proyecto implementa un sistema completo de predicción de precios de viviendas utilizando:
- Base de datos en la nube: Azure Cosmos DB (MongoDB)
- Machine Learning: 3 modelos comparativos (Regresión Lineal, Random Forest, Red Neuronal)
- Visualización: Dashboard interactivo con Streamlit
- Datos: 20,640 casas de California

---

## OBJETIVOS DEL PROYECTO

✅ Desplegar base de datos NoSQL en Azure

✅ Entrenar 3 modelos predictivos

✅ Crear dashboard interactivo conectado a la nube
---

##  PASO A PASO

### Instalación

```bash
# Abrir tu carpeta
cd ProyectoCasas

# Instalar dependencias (versiones para Python 3.13.3)
pip install -r requirements.txt
```

### Configurar Azure

1. Abrir cuenta de Azure for Students: https://azure.microsoft.com/free/students
2. Crear Cosmos DB 
3. Copiar Connection String
4. Pegar en los 3 archivos .py donde dice: `MONGODB_CONNECTION_STRING = "XXXX"`

### Ejecutar Proyecto

```bash
# Paso 1: Subir datos a Azure
python 1_preparar_datos.py

# Paso 2: Entrenar modelos
python 2_modelos_predictivos.py

# Paso 3: Lanzar dashboard
streamlit run 3_dashboard.py
```

### Abrir Dashboard

El navegador debe abrir automáticamente en: http://localhost:8501

Si no abre, copia esa URL manualmente en tu navegador.

---

## CARACTERÍSTICAS DEL DASHBOARD

### Página 1: 🏡 Explorador de Datos
- Filtros interactivos (precio, habitaciones, edad)
- 4 visualizaciones dinámicas
- Tabla de datos filtrados

### Página 2: 🤖 Predictor de Precios
- Formulario con 8 características
- Predicciones de 3 modelos simultáneos
- Comparación visual

### Página 3: 📊 Comparación de Modelos
- Tabla de métricas (RMSE, R², MAE)
- Gráficos comparativos
- Recomendación del mejor modelo

### Página 4: 📈 Análisis Geográfico
- Mapa de calor de California
- Análisis por región
- Estadísticas descriptivas

---

## 🤖 MODELOS IMPLEMENTADOS

### Modelo 1: Regresión Lineal
- Tipo: Baseline simple
- Ventajas: Rápido, interpretable
- R² esperado: ~0.59

### Modelo 2: Random Forest
- Tipo: Ensemble de árboles
- Ventajas: Balance precisión/velocidad
- R² esperado: ~0.80

### Modelo 3: Red Neuronal 
- Arquitectura:
  - Entrada: 8 neuronas
  - Oculta 1: 64 neuronas + Dropout(20%) + ReLU
  - Oculta 2: 32 neuronas + Dropout(20%) + ReLU
  - Oculta 3: 16 neuronas + ReLU
  - Salida: 1 neurona
- Optimizador: Adam (lr=0.001)
- Loss: MSE
- Técnicas anti-overfitting: Dropout, Early Stopping
- R² esperado: ~0.79

---

## VARIABLES DEL DATASET

| Variable | Descripción | Unidad |
|----------|-------------|--------|
| MedInc | Ingreso medio del área | $10,000 |
| HouseAge | Edad de la casa | Años |
| AveRooms | Habitaciones promedio | Número |
| AveBedrms | Dormitorios promedio | Número |
| Population | Población del área | Personas |
| AveOccup | Ocupantes por hogar | Número |
| Latitude | Latitud | Grados |
| Longitude | Longitud | Grados |
| Precio | **Variable objetivo** | $100,000 |

*Total de registros:* 20,640 casas
---

## STACK TECNOLÓGICO

### Cloud
- Azure Cosmos DB: Base de datos NoSQL (MongoDB API)

### Machine Learning
- scikit-learn 1.5.2: Regresión Lineal, Random Forest
- TensorFlow 2.18+: Red Neuronal (actualizado para Python 3.13.3)
- Keras 3.x: API standalone (nuevo formato)
- Preprocesamiento: pandas, numpy, StandardScaler

### Dashboard
- Streamlit 1.40.1: Framework web
- Plotly 5.24.1: Visualizaciones interactivas
- PyMongo: Conexión a MongoDB

### Lenguaje
- Python 3.13.3
