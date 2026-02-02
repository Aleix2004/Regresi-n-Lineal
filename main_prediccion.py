import pandas as pd
import numpy as np
import glob
import os
from sklearn.linear_model import LinearRegression

def procesar_sistema_defensa(directorio_raiz):
    all_data = []
    
    # Buscamos todos los CSV en las subcarpetas
    rutas = glob.glob(os.path.join(directorio_raiz, "**", "*.csv"), recursive=True)
    
    if not rutas:
        print("No se encontraron archivos CSV. Revisa la ruta.")
        return

    print(f"Analizando {len(rutas)} trayectorias...")

    for ruta in rutas:
        df = pd.read_csv(ruta)
        nombre_ataque = os.path.basename(os.path.dirname(ruta))
        nombre_misil = os.path.basename(ruta)
        
        # 1. Variables (X: tiempo, y: posX)
        X = df[['tiempo']].values
        y = df['posX'].values
        
        # 2. Regresión Lineal
        model = LinearRegression()
        model.fit(X, y)
        
        # 3. Métricas (R2 para fiabilidad, MSE para error)
        r2 = model.score(X, y)
        
        # 4. Predicción a T + 0.1
        ultimo_t = X[-1][0]
        t_target = ultimo_t + 0.1
        x_target = model.predict([[t_target]])[0]
        
        all_data.append({
            'Ataque': nombre_ataque,
            'Misil': nombre_misil,
            'Fiabilidad_R2': r2,
            'Media_X': np.mean(y),
            'T_Impacto': round(t_target, 4),
            'X_Impacto': round(x_target, 4)
        })

    # Convertir a DataFrame y seleccionar los 50 mejores por ataque
    df_resultados = pd.DataFrame(all_data)
    
    final_targets = df_resultados.sort_values(['Ataque', 'Fiabilidad_R2'], ascending=[True, False])
    final_targets = final_targets.groupby('Ataque').head(50)
    
    # Guardar resultado
    final_targets.to_csv('seleccion_objetivos.csv', index=False)
    print("Archivo 'seleccion_objetivos.csv' generado con éxito.")

if __name__ == "__main__":
    # "." significa que busque en la carpeta actual
    procesar_sistema_defensa(".")