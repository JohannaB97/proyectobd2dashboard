# Dashboard interactivo con Streamlit

"""
- Datos de casas con filtros
- Gráficos interactivos
- Predictor de precios en tiempo real
- Comparación de modelos

"""

import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import pickle
import keras

# CONFIGURACIÓN DE LA PÁGINA

st.set_page_config(
    page_title="Predictor de Precios de Casas",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CONFIGURACIÓN

MONGODB_CONNECTION_STRING = "mongodb+srv://proyecto2bdjoha97:Felicidad2025*@proyecto2bdjoha97.global.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"
DATABASE_NAME = "casas_california"
COLLECTION_NAME = "viviendas"

# FUNCIONES DE CARGA DE DATOS

@st.cache_data(ttl=600)  # Cache por 10 minutos
def cargar_datos_mongodb():
    """
    Carga datos desde MongoDB (con caché para mejor rendimiento).
    """
    client = MongoClient(MONGODB_CONNECTION_STRING)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    
    datos = list(collection.find())
    df = pd.DataFrame(datos)
    
    if '_id' in df.columns:
        df = df.drop('_id', axis=1)
    
    client.close()
    return df

@st.cache_resource
def cargar_modelos():
    """
    Carga los modelos entrenados (se carga una sola vez).
    """
    with open('modelo_lineal.pkl', 'rb') as f:
        modelo_lineal = pickle.load(f)
    
    with open('modelo_random_forest.pkl', 'rb') as f:
        modelo_rf = pickle.load(f)
    
    modelo_nn = keras.models.load_model('modelo_red_neuronal.keras')
    
    # Cargar scaler
    df_temp = cargar_datos_mongodb()
    columnas = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                'Population', 'AveOccup', 'Latitude', 'Longitude']
    scaler = StandardScaler()
    scaler.fit(df_temp[columnas])
    
    return modelo_lineal, modelo_rf, modelo_nn, scaler


# SIDEBAR - NAVEGACIÓN

st.sidebar.title("🏠 Predictor de Precios")
st.sidebar.markdown("---")

pagina = st.sidebar.radio(
    "Navegación:",
    ["🏡 Explorador de Datos",
     "🤖 Predictor de Precios",
     "📊 Comparación de Modelos",
     "📈 Análisis Geográfico"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Proyecto:** Predicción de Precios de Casas en California
    
    **Datos:** 20,640 viviendas
    
    **Modelos:** 
    - Regresión Lineal
    - Random Forest
    - Red Neuronal
    """
)


# CARGAR DATOS

try:
    df = cargar_datos_mongodb()
    modelos_cargados = True
    try:
        modelo_lineal, modelo_rf, modelo_nn, scaler = cargar_modelos()
    except:
        modelos_cargados = False
        st.warning("⚠️ No se pudieron cargar los modelos. Ejecuta primero 2_modelos_predictivos.py")
except Exception as e:
    st.error(f"❌ Error al conectar con MongoDB: {e}")
    st.info("Asegúrate de haber configurado el CONNECTION STRING correctamente")
    st.stop()

# PÁGINA 1: EXPLORADOR DE DATOS

if pagina == "🏡 Explorador de Datos":
    st.title("🏡 Explorador de Datos de Casas")
    st.markdown("Explora las características de las viviendas en California")
    
    # Métricas generales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Casas", f"{len(df):,}")
    
    with col2:
        st.metric("Precio Promedio", f"${df['Precio'].mean():.2f}")
    
    with col3:
        st.metric("Precio Mínimo", f"${df['Precio'].min():.2f}")
    
    with col4:
        st.metric("Precio Máximo", f"${df['Precio'].max():.2f}")
    
    st.markdown("---")
    
    # Filtros
    st.subheader("🔍 Filtros")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rango_precio = st.slider(
            "Rango de Precio",
            float(df['Precio'].min()),
            float(df['Precio'].max()),
            (float(df['Precio'].min()), float(df['Precio'].max()))
        )
    
    with col2:
        rango_habitaciones = st.slider(
            "Habitaciones Promedio",
            float(df['AveRooms'].min()),
            min(float(df['AveRooms'].max()), 15.0),
            (float(df['AveRooms'].min()), min(float(df['AveRooms'].max()), 8.0))
        )
    
    with col3:
        rango_edad = st.slider(
            "Edad de la Casa (años)",
            float(df['HouseAge'].min()),
            float(df['HouseAge'].max()),
            (float(df['HouseAge'].min()), float(df['HouseAge'].max()))
        )
    
    # Aplicar filtros
    df_filtrado = df[
        (df['Precio'] >= rango_precio[0]) &
        (df['Precio'] <= rango_precio[1]) &
        (df['AveRooms'] >= rango_habitaciones[0]) &
        (df['AveRooms'] <= rango_habitaciones[1]) &
        (df['HouseAge'] >= rango_edad[0]) &
        (df['HouseAge'] <= rango_edad[1])
    ]
    
    st.info(f"📊 Mostrando {len(df_filtrado):,} casas de {len(df):,} totales")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de precios
        fig1 = px.histogram(
            df_filtrado,
            x='Precio',
            nbins=50,
            title='Distribución de Precios',
            labels={'Precio': 'Precio (en $100k)'},
            color_discrete_sequence=['#3498db']
        )
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Relación Precio vs Ingreso Medio
        fig2 = px.scatter(
            df_filtrado.sample(min(1000, len(df_filtrado))),
            x='MedInc',
            y='Precio',
            title='Precio vs Ingreso Medio del Área',
            labels={'MedInc': 'Ingreso Medio ($10k)', 'Precio': 'Precio ($100k)'},
            color='Precio',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Precio vs Habitaciones
        df_temp = df_filtrado[df_filtrado['AveRooms'] < 10].copy()
        df_temp['Rooms_Category'] = pd.cut(df_temp['AveRooms'], bins=5).astype(str)
        
        fig3 = px.box(
            df_temp,
            x='Rooms_Category',
            y='Precio',
            title='Precio según Número de Habitaciones',
            labels={'Rooms_Category': 'Habitaciones Promedio', 'Precio': 'Precio ($100k)'}
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Precio vs Edad
        fig4 = px.scatter(
            df_filtrado.sample(min(1000, len(df_filtrado))),
            x='HouseAge',
            y='Precio',
            title='Precio vs Edad de la Casa',
            labels={'HouseAge': 'Edad (años)', 'Precio': 'Precio ($100k)'},
            color='Precio',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    # Tabla de datos
    st.subheader("📋 Datos Filtrados")
    st.dataframe(
        df_filtrado[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'Precio']].head(100),
        use_container_width=True
    )

# PÁGINA 2: PREDICTOR DE PRECIOS

elif pagina == "🤖 Predictor de Precios":
    st.title("🤖 Predictor de Precios de Casas")
    st.markdown("Ingresa las características de una casa y obtén su precio estimado")
    
    if not modelos_cargados:
        st.error("❌ Modelos no disponibles. Ejecuta primero 2_modelos_predictivos.py")
        st.stop()
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Características de la Propiedad")
        
        med_inc = st.number_input(
            "Ingreso Medio del Área ($10,000)",
            min_value=0.5,
            max_value=15.0,
            value=3.5,
            step=0.1,
            help="Ingreso medio en unidades de $10,000"
        )
        
        house_age = st.slider(
            "Edad de la Casa (años)",
            min_value=1,
            max_value=52,
            value=20
        )
        
        ave_rooms = st.number_input(
            "Habitaciones Promedio",
            min_value=1.0,
            max_value=15.0,
            value=5.0,
            step=0.1
        )
        
        ave_bedrms = st.number_input(
            "Dormitorios Promedio",
            min_value=0.5,
            max_value=10.0,
            value=1.0,
            step=0.1
        )
    
    with col2:
        st.subheader("📍 Información Geográfica")
        
        population = st.number_input(
            "Población del Área",
            min_value=10,
            max_value=35000,
            value=1500,
            step=100
        )
        
        ave_occup = st.number_input(
            "Ocupantes Promedio por Hogar",
            min_value=0.5,
            max_value=15.0,
            value=3.0,
            step=0.1
        )
        
        latitude = st.number_input(
            "Latitud",
            min_value=32.0,
            max_value=42.0,
            value=34.0,
            step=0.1
        )
        
        longitude = st.number_input(
            "Longitud",
            min_value=-125.0,
            max_value=-114.0,
            value=-118.0,
            step=0.1
        )
    
    # Botón de predicción
    st.markdown("---")
    
    if st.button("🔮 PREDECIR PRECIO", use_container_width=True):
        # Preparar datos
        datos_casa = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, 
                                population, ave_occup, latitude, longitude]])
        
        datos_casa_scaled = scaler.transform(datos_casa)
        
        # Hacer predicciones con los 3 modelos
        pred_lineal = modelo_lineal.predict(datos_casa)[0]
        pred_rf = modelo_rf.predict(datos_casa)[0]
        pred_nn = modelo_nn.predict(datos_casa_scaled, verbose=0)[0][0]
        
        # Mostrar resultados
        st.success("✅ Predicción completada")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Regresión Lineal",
                f"${pred_lineal:.2f}",
                delta=f"${pred_lineal - pred_rf:.2f}"
            )
        
        with col2:
            st.metric(
                "Random Forest",
                f"${pred_rf:.2f}",
                delta="Referencia"
            )
        
        with col3:
            st.metric(
                "Red Neuronal",
                f"${pred_nn:.2f}",
                delta=f"${pred_nn - pred_rf:.2f}"
            )
        
        # Promedio
        promedio = (pred_lineal + pred_rf + pred_nn) / 3
        st.markdown("---")
        st.info(f"💡 **Precio Estimado Promedio:** ${promedio:.2f} (en unidades de $100,000)")
        st.info(f"💰 **Equivalente a:** ${promedio * 100000:,.0f}")
        
        # Gráfico de comparación
        fig = go.Figure(data=[
            go.Bar(
                x=['Regresión Lineal', 'Random Forest', 'Red Neuronal'],
                y=[pred_lineal, pred_rf, pred_nn],
                marker_color=['#3498db', '#2ecc71', '#e74c3c'],
                text=[f'${pred_lineal:.2f}', f'${pred_rf:.2f}', f'${pred_nn:.2f}'],
                textposition='auto'
            )
        ])
        fig.update_layout(
            title='Comparación de Predicciones',
            yaxis_title='Precio Predicho ($100k)',
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# PÁGINA 3: COMPARACIÓN DE MODELOS

elif pagina == "📊 Comparación de Modelos":
    st.title("📊 Comparación de Modelos")
    st.markdown("Análisis comparativo de los tres modelos predictivos")
    
    # Cargar resultados
    try:
        df_comparacion = pd.read_csv('comparacion_modelos.csv')
        
        st.subheader("📋 Tabla de Resultados")
        st.dataframe(df_comparacion, use_container_width=True)
        
        # Métricas destacadas
        col1, col2, col3 = st.columns(3)
        
        mejor_r2 = df_comparacion.loc[df_comparacion['R2_test'].idxmax()]
        mejor_rmse = df_comparacion.loc[df_comparacion['RMSE_test'].idxmin()]
        mejor_mae = df_comparacion.loc[df_comparacion['MAE_test'].idxmin()]
        
        with col1:
            st.metric(
                "Mejor R² (Test)",
                mejor_r2['Modelo'],
                f"{mejor_r2['R2_test']:.4f}"
            )
        
        with col2:
            st.metric(
                "Menor RMSE (Test)",
                mejor_rmse['Modelo'],
                f"{mejor_rmse['RMSE_test']:.4f}"
            )
        
        with col3:
            st.metric(
                "Menor MAE (Test)",
                mejor_mae['Modelo'],
                f"{mejor_mae['MAE_test']:.4f}"
            )
        
        st.markdown("---")
        
        # Gráficos comparativos
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(
                df_comparacion,
                x='Modelo',
                y='R2_test',
                title='R² Score (Test) - Mayor es Mejor',
                color='Modelo',
                color_discrete_sequence=['#3498db', '#2ecc71', '#e74c3c']
            )
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(
                df_comparacion,
                x='Modelo',
                y='RMSE_test',
                title='RMSE (Test) - Menor es Mejor',
                color='Modelo',
                color_discrete_sequence=['#3498db', '#2ecc71', '#e74c3c']
            )
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Análisis
        st.subheader("📝 Análisis")
        st.markdown(f"""
        **Modelo Recomendado:** {mejor_r2['Modelo']}
        
        **Justificación:**
        - Mejor R² en conjunto de test: {mejor_r2['R2_test']:.4f}
        - RMSE competitivo: {mejor_r2['RMSE_test']:.4f}
        - Balance entre precisión y capacidad de generalización
        
        **Interpretación:**
        - R² cercano a 1 indica excelente ajuste
        - RMSE bajo indica errores pequeños en predicción
        - El modelo explica más del {mejor_r2['R2_test']*100:.1f}% de la variación en precios
        """)
        
    except FileNotFoundError:
        st.warning("⚠️ No se encontró el archivo de comparación. Ejecuta primero 2_modelos_predictivos.py")

# PÁGINA 4: ANÁLISIS GEOGRÁFICO

elif pagina == "📈 Análisis Geográfico":
    st.title("📈 Análisis Geográfico de Precios")
    st.markdown("Visualización de precios según ubicación en California")
    
    # Mapa de calor
    st.subheader("🗺️ Mapa de Calor de Precios")
    
    # Muestrear datos para mejor rendimiento
    df_muestra = df.sample(min(5000, len(df)))
    
    fig = px.scatter_mapbox(
        df_muestra,
        lat='Latitude',
        lon='Longitude',
        color='Precio',
        size='Population',
        hover_data=['Precio', 'MedInc', 'AveRooms', 'HouseAge'],
        color_continuous_scale='Viridis',
        zoom=4,
        height=600,
        title='Distribución Geográfica de Precios'
    )
    
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis por región
    st.subheader("📊 Análisis por Región")
    
    # Dividir California en regiones
    def clasificar_region(lat, lon):
        if lat > 37.5:
            return "Norte"
        elif lat > 34:
            if lon < -120:
                return "Costa Central"
            else:
                return "Centro"
        else:
            if lon < -118:
                return "Los Ángeles"
            else:
                return "Sur"
    
    df['Region'] = df.apply(lambda x: clasificar_region(x['Latitude'], x['Longitude']), axis=1)
    
    # Estadísticas por región
    stats_region = df.groupby('Region').agg({
        'Precio': ['mean', 'median', 'std', 'count']
    }).round(2)
    
    st.dataframe(stats_region, use_container_width=True)
    
    # Gráfico de cajas por región
    fig = px.box(
        df,
        x='Region',
        y='Precio',
        color='Region',
        title='Distribución de Precios por Región'
    )
    st.plotly_chart(fig, use_container_width=True)

# FOOTER

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🏠 Dashboard de Predicción de Precios de Casas | Proyecto MongoDB + Azure + ML</p>
        <p>Datos: California Housing Dataset | Modelos: Regresión Lineal, Random Forest, Red Neuronal</p>
    </div>
    """,
    unsafe_allow_html=True
)
