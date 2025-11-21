# Cargar datos de casas y subirlos a MongoDB en Azure
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from pymongo import MongoClient
import json


# CONNECTION STRING de Azure
MONGODB_CONNECTION_STRING = "mongodb+srv://proyecto2bdjoha97:Felicidad2025*@proyecto2bdjoha97.global.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"

# Nombre base de datos
DATABASE_NAME = "casas_california"

# Nombre de la colección 
COLLECTION_NAME = "viviendas"

# CARGAR DATOS DE CALIFORNIA

def cargar_datos_california():
    """
    Carga el dataset de casas de California.
    El dataset ya viene incluido en sklearn
    """
    print("Cargando datos de California Housing...")
    
    # Cargar datos
    california = fetch_california_housing()
    
    # Convertir a DataFrame
    df = pd.DataFrame(
        california.data,
        columns=california.feature_names
    )
    
    # Agregar la variable objetivo (precio)
    df['Precio'] = california.target
    
    # Información del dataset
    print(f"✅ Datos cargados: {len(df)} casas")
    print(f"📊 Columnas: {list(df.columns)}")
    print(f"\n📈 Primeras 5 filas:")
    print(df.head())
    
    return df

# PREPARAR DATOS PARA MONGODB

def preparar_para_mongodb(df):
    """
    Convierte el DataFrame a formato que MongoDB entiende (documentos JSON).
    """
    print("\n Preparando datos para MongoDB...")
    
    # Crear categorías de precio (bajo, medio, alto)
    df['Categoria_Precio'] = pd.cut(
        df['Precio'],
        bins=[0, 1.5, 3, 5],
        labels=['Económica', 'Media', 'Premium']
    )
    
    # Crear categoría por edad de la casa
    df['Categoria_Edad'] = pd.cut(
        df['HouseAge'],
        bins=[0, 15, 35, 100],
        labels=['Nueva', 'Establecida', 'Antigua']
    )
    
    # Calcular habitaciones promedio por vivienda
    df['Habitaciones_Promedio'] = df['AveRooms']
    
    # Convertir a lista de documentos (diccionarios)
    documentos = df.to_dict('records')
    
    print(f" {len(documentos)} documentos preparados")
    
    return documentos

# SUBIR A MONGODB EN AZURE

def subir_a_mongodb(documentos, connection_string, db_name, collection_name):
    """
    Conecta a MongoDB en Azure y sube los datos.
    """
    try:
        print(f"\n Conectando a MongoDB en Azure...")
        
        # Conectar a MongoDB
        client = MongoClient(connection_string)
        
        # Acceder a la base de datos
        db = client[db_name]
        
        # Acceder a la colección
        collection = db[collection_name]
        
        # Limpiar colección si ya existe
        collection.delete_many({})
        print("🗑️  Colección limpiada")
        
        # Insertar documentos
        print(f" Subiendo {len(documentos)} documentos...")
        result = collection.insert_many(documentos)
        
        print(f" ¡Éxito! {len(result.inserted_ids)} documentos insertados")
        
        # Verificar
        count = collection.count_documents({})
        print(f"✅ Verificación: {count} documentos en la base de datos")
        
        # Mostrar ejemplo
        print("\n📄 Ejemplo de documento insertado:")
        ejemplo = collection.find_one()
        print(json.dumps(ejemplo, indent=2, default=str))
        
        # Cerrar conexión
        client.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


# FUNCIÓN PRINCIPAL

def main():
    """
    Función principal que ejecuta todo el proceso.
    """
    print(" PREPARACIÓN DE DATOS - PROYECTO CASAS CALIFORNIA")
    
    # Verificar que configuraste el connection string
    if "TU_CONNECTION_STRING_AQUI" in MONGODB_CONNECTION_STRING:
        print("\n  ERROR: Debes configurar tu CONNECTION STRING primero")
        print(" Edita este archivo y pega tu connection string de Azure")
        return
    
    # Paso 1: Cargar datos
    df = cargar_datos_california()
    
    # Paso 2: Preparar para MongoDB
    documentos = preparar_para_mongodb(df)
    
    # Paso 3: Subir a Azure
    exito = subir_a_mongodb(
        documentos,
        MONGODB_CONNECTION_STRING,
        DATABASE_NAME,
        COLLECTION_NAME
    )
    
    if exito:
        print("\n" + "=" * 60)
        print(" ¡PROCESO COMPLETADO CON ÉXITO!")
        print("=" * 60)
        print(f" Base de datos: {DATABASE_NAME}")
        print(f" Colección: {COLLECTION_NAME}")
        print(f" Total documentos: {len(documentos)}")
        print("\n Ahora puedes continuar con los modelos predictivos")
    else:
        print("\n Hubo un error. Revisa tu connection string.")

# EJECUTAR

if __name__ == "__main__":
    main()
