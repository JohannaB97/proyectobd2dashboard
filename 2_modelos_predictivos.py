
# Crear y comparar 3 modelos de predicción de precios

"""
3 modelos diferentes que predicen precios:
1. Regresión Lineal (simple)
2. Random Forest (medio)
3. Red Neuronal (avanzado)
"""

import pandas as pd
import numpy as np
from pymongo import MongoClient
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
import keras
from keras import layers
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# CONFIGURACIÓN

MONGODB_CONNECTION_STRING = "mongodb+srv://proyecto2bdjoha97:Felicidad2025*@proyecto2bdjoha97.global.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"
DATABASE_NAME = "casas_california"
COLLECTION_NAME = "viviendas"

# CARGAR DATOS DESDE MONGODB

def cargar_datos_mongodb():
    """
    Carga los datos desde MongoDB en Azure.
    """
    print(" Conectando a MongoDB...")
    
    client = MongoClient(MONGODB_CONNECTION_STRING)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    
    # Obtener todos los documentos
    datos = list(collection.find())
    
    # Convertir a DataFrame
    df = pd.DataFrame(datos)
    
    # Eliminar la columna _id de MongoDB
    if '_id' in df.columns:
        df = df.drop('_id', axis=1)
    
    print(f" Datos cargados: {len(df)} registros")
    
    client.close()
    return df

# PREPARAR DATOS PARA MODELADO

def preparar_datos(df):
    """
    Prepara los datos para machine learning:
    - Selecciona variables numéricas
    - Divide en entrenamiento y prueba
    - Normaliza los datos
    """
    print("\n🔧 Preparando datos para modelado...")
    
    # Seleccionar solo columnas numéricas
    columnas_numericas = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                          'Population', 'AveOccup', 'Latitude', 'Longitude']
    
    # X = variables predictoras, y = variable objetivo (precio)
    X = df[columnas_numericas]
    y = df['Precio']
    
    print(f" Variables predictoras: {columnas_numericas}")
    print(f" Variable objetivo: Precio")
    
    # Dividir datos: 70% entrenamiento, 15% validación, 15% prueba
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42  # 0.176 de 0.85 ≈ 0.15 del total
    )
    
    print(f"\n División de datos:")
    print(f"   - Entrenamiento: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   - Validación: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   - Prueba: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Normalizar datos (importante para redes neuronales)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(" Datos normalizados")
    
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_val_scaled': X_val_scaled,
        'X_test_scaled': X_test_scaled,
        'scaler': scaler,
        'feature_names': columnas_numericas
    }

# MODELO 1: REGRESIÓN LINEAL

def modelo_regresion_lineal(datos):
    """
    Modelo más simple: Regresión Lineal.
    """
    print(" MODELO 1: REGRESIÓN LINEAL")
    
    # Crear modelo
    modelo = LinearRegression()
    
    # Entrenar
    print(" Entrenando modelo...")
    modelo.fit(datos['X_train'], datos['y_train'])
    
    # Predicciones
    y_pred_train = modelo.predict(datos['X_train'])
    y_pred_val = modelo.predict(datos['X_val'])
    y_pred_test = modelo.predict(datos['X_test'])
    
    # Evaluar
    resultados = evaluar_modelo(
        datos['y_train'], y_pred_train,
        datos['y_val'], y_pred_val,
        datos['y_test'], y_pred_test,
        "Regresión Lineal"
    )
    
    # Guardar modelo
    with open('modelo_lineal.pkl', 'wb') as f:
        pickle.dump(modelo, f)
    
    return modelo, resultados

# MODELO 2: RANDOM FOREST

def modelo_random_forest(datos):
    """
    Modelo intermedio: Random Forest.
    """
    print(" MODELO 2: RANDOM FOREST")
    
    # Crear modelo
    modelo = RandomForestRegressor(
        n_estimators=100,      # 100 árboles
        max_depth=20,          # Profundidad máxima
        min_samples_split=5,
        random_state=42,
        n_jobs=-1              # Usar todos los cores
    )
    
    # Entrenar
    print(" Entrenando modelo...")
    modelo.fit(datos['X_train'], datos['y_train'])
    
    # Predicciones
    y_pred_train = modelo.predict(datos['X_train'])
    y_pred_val = modelo.predict(datos['X_val'])
    y_pred_test = modelo.predict(datos['X_test'])
    
    # Evaluar
    resultados = evaluar_modelo(
        datos['y_train'], y_pred_train,
        datos['y_val'], y_pred_val,
        datos['y_test'], y_pred_test,
        "Random Forest"
    )
    
    # Importancia de variables
    importancias = pd.DataFrame({
        'Variable': datos['feature_names'],
        'Importancia': modelo.feature_importances_
    }).sort_values('Importancia', ascending=False)
    
    print("\n Variables más importantes:")
    print(importancias.to_string(index=False))
    
    # Guardar modelo
    with open('modelo_random_forest.pkl', 'wb') as f:
        pickle.dump(modelo, f)
    
    return modelo, resultados

# MODELO 3: RED NEURONAL 

def modelo_red_neuronal(datos):
    """
    Modelo avanzado: Red Neuronal con Keras/TensorFlow.
    """
    print(" MODELO 3: RED NEURONAL")
    
    # Construir arquitectura
    modelo = keras.Sequential([
        # Capa de entrada
        layers.Dense(64, activation='relu', input_shape=(8,)),
        layers.Dropout(0.2),  # Prevenir overfitting
        
        # Capas ocultas
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(16, activation='relu'),
        
        # Capa de salida
        layers.Dense(1)  # 1 neurona para regresión
    ])
    
    print("\n  Arquitectura de la Red Neuronal:")
    modelo.summary()
    
    # Compilar modelo
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',           # Mean Squared Error
        metrics=['mae']       # Mean Absolute Error
    )
    
    # Early stopping (detener si no mejora)
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    # Entrenar
    print("\n Entrenando red neuronal...")
    history = modelo.fit(
        datos['X_train_scaled'], datos['y_train'],
        validation_data=(datos['X_val_scaled'], datos['y_val']),
        epochs=200,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0  # No mostrar cada época
    )
    
    print(f" Entrenamiento completado en {len(history.history['loss'])} épocas")
    
    # Predicciones
    y_pred_train = modelo.predict(datos['X_train_scaled'], verbose=0).flatten()
    y_pred_val = modelo.predict(datos['X_val_scaled'], verbose=0).flatten()
    y_pred_test = modelo.predict(datos['X_test_scaled'], verbose=0).flatten()
    
    # Evaluar
    resultados = evaluar_modelo(
        datos['y_train'], y_pred_train,
        datos['y_val'], y_pred_val,
        datos['y_test'], y_pred_test,
        "Red Neuronal"
    )
    
    # Visualizar entrenamiento
    visualizar_entrenamiento(history)
    
    # Guardar modelo (formato .keras para Keras 3.x)
    modelo.save('modelo_red_neuronal.keras')
    
    return modelo, resultados, history

# FUNCIÓN DE EVALUACIÓN

def evaluar_modelo(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, nombre):
    """
    Evalúa el modelo con varias métricas.
    """
    # Calcular métricas para cada conjunto
    metricas = {
        'Modelo': nombre,
        'RMSE_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'RMSE_val': np.sqrt(mean_squared_error(y_val, y_pred_val)),
        'RMSE_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'R2_train': r2_score(y_train, y_pred_train),
        'R2_val': r2_score(y_val, y_pred_val),
        'R2_test': r2_score(y_test, y_pred_test),
        'MAE_train': mean_absolute_error(y_train, y_pred_train),
        'MAE_val': mean_absolute_error(y_val, y_pred_val),
        'MAE_test': mean_absolute_error(y_test, y_pred_test)
    }
    
    print(f"\n Resultados {nombre}:")
    print(f"   RMSE - Train: {metricas['RMSE_train']:.4f}, Val: {metricas['RMSE_val']:.4f}, Test: {metricas['RMSE_test']:.4f}")
    print(f"   R² - Train: {metricas['R2_train']:.4f}, Val: {metricas['R2_val']:.4f}, Test: {metricas['R2_test']:.4f}")
    print(f"   MAE - Train: {metricas['MAE_train']:.4f}, Val: {metricas['MAE_val']:.4f}, Test: {metricas['MAE_test']:.4f}")
    
    return metricas

# VISUALIZACIÓN DE ENTRENAMIENTO RED NEURONAL

def visualizar_entrenamiento(history):
    """
    Visualiza el proceso de entrenamiento de la red neuronal.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Pérdida (Loss)
    ax1.plot(history.history['loss'], label='Entrenamiento')
    ax1.plot(history.history['val_loss'], label='Validación')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Pérdida durante Entrenamiento')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Error Absoluto Medio
    ax2.plot(history.history['mae'], label='Entrenamiento')
    ax2.plot(history.history['val_mae'], label='Validación')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('MAE')
    ax2.set_title('Error Absoluto Medio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('entrenamiento_red_neuronal.png', dpi=300, bbox_inches='tight')
    print(" Gráfica guardada: entrenamiento_red_neuronal.png")
    plt.close()

# COMPARACIÓN DE MODELOS

def comparar_modelos(resultados_todos):
    """
    Compara todos los modelos y genera tabla comparativa.
    """
    print(" COMPARACIÓN DE TODOS LOS MODELOS")

    df_comparacion = pd.DataFrame(resultados_todos)
    
    # Mostrar tabla
    print("\n Tabla de Resultados:")
    print(df_comparacion.to_string(index=False))
    
    # Identificar mejor modelo
    mejor_modelo = df_comparacion.loc[df_comparacion['R2_test'].idxmax(), 'Modelo']
    mejor_r2 = df_comparacion['R2_test'].max()
    
    print(f"\n MEJOR MODELO: {mejor_modelo}")
    print(f"   R² en Test: {mejor_r2:.4f}")
    
    # Guardar resultados
    df_comparacion.to_csv('comparacion_modelos.csv', index=False)
    print("\n Resultados guardados en: comparacion_modelos.csv")
    
    # Visualización comparativa
    visualizar_comparacion(df_comparacion)
    
    return df_comparacion

def visualizar_comparacion(df):
    """
    Crea gráficas comparativas de los modelos.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metricas = [('RMSE_test', 'RMSE (menor es mejor)'),
                ('R2_test', 'R² (mayor es mejor)'),
                ('MAE_test', 'MAE (menor es mejor)')]
    
    for ax, (metrica, titulo) in zip(axes, metricas):
        ax.bar(df['Modelo'], df[metrica], color=['#3498db', '#2ecc71', '#e74c3c'])
        ax.set_title(titulo)
        ax.set_ylabel(metrica.replace('_test', ''))
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('comparacion_modelos.png', dpi=300, bbox_inches='tight')
    print("📊 Gráfica guardada: comparacion_modelos.png")
    plt.close()

# FUNCIÓN PRINCIPAL

def main():
    """
    Ejecuta todo el proceso de modelado.
    """
    print(" MODELADO PREDICTIVO - PRECIOS DE CASAS")
    
    # Verificar configuración
    if "TU_CONNECTION_STRING_AQUI" in MONGODB_CONNECTION_STRING:
        print("\n  ERROR: Debes configurar tu CONNECTION STRING primero")
        return
    
    # Paso 1: Cargar datos
    df = cargar_datos_mongodb()
    
    # Paso 2: Preparar datos
    datos = preparar_datos(df)
    
    # Paso 3: Crear modelos
    resultados_todos = []
    
    # Modelo 1: Regresión Lineal
    _, resultado1 = modelo_regresion_lineal(datos)
    resultados_todos.append(resultado1)
    
    # Modelo 2: Random Forest
    _, resultado2 = modelo_random_forest(datos)
    resultados_todos.append(resultado2)
    
    # Modelo 3: Red Neuronal
    _, resultado3, _ = modelo_red_neuronal(datos)
    resultados_todos.append(resultado3)
    
    # Paso 4: Comparar modelos
    comparacion = comparar_modelos(resultados_todos)
    
    print(" ¡MODELADO COMPLETADO!")
    print("\n Archivos generados:")
    print("   - modelo_lineal.pkl")
    print("   - modelo_random_forest.pkl")
    print("   - modelo_red_neuronal.keras")
    print("   - comparacion_modelos.csv")
    print("   - comparacion_modelos.png")
    print("   - entrenamiento_red_neuronal.png")

if __name__ == "__main__":
    main()
