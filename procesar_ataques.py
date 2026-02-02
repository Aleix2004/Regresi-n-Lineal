import pandas as pd
import numpy as np
import glob
import os
from sklearn.linear_model import LinearRegression

def ejecutar_defensa_balistica():
    resultados_totales = []
    
    # Buscamos en las carpetas Attack000 hasta Attack024
    for i in range(25):
        nombre_carpeta = f"Attack{i:03d}"
        ruta_ataque = os.path.join(".", nombre_carpeta)
        
        if not os.path.exists(ruta_ataque):
            continue
            
        print(f"Procesando {nombre_carpeta}...")
        archivos = glob.glob(os.path.join(ruta_ataque, "*.csv"))
        
        datos_ataque = []
        for archivo in archivos:
            df = pd.read_csv(archivo)
            
            # 1. Características (30% nota)
            t = df[['tiempo']].values
            x = df['posX'].values
            
            # 2. Regresión Lineal (35% nota)
            model = LinearRegression()
            model.fit(t, x)
            r2 = model.score(t, x) # Coeficiente de determinación (fiabilidad)
            
            # 3. Predicción (25% nota)
            # Calculamos t_final + 0.1
            t_objetivo = t[-1][0] + 0.1
            x_objetivo = model.predict([[t_objetivo]])[0]
            
            datos_ataque.append({
                'Ataque': nombre_carpeta,
                'Misil': os.path.basename(archivo),
                'Fiabilidad': r2,
                'T_Destino': t_objetivo,
                'X_Destino': x_objetivo
            })
        
        # 4. Selección de candidatos (10% nota)
        # Ordenamos por fiabilidad y nos quedamos con los 50 mejores
        top_50 = sorted(datos_ataque, key=lambda k: k['Fiabilidad'], reverse=True)[:50]
        resultados_totales.extend(top_50)

    # Guardar resultados
    df_final = pd.DataFrame(resultados_totales)
    df_final.to_csv('BN-ACTN-Resultados.csv', index=False)
    print("\nAnálisis completado. Archivo 'BN-ACTN-Resultados.csv' generado.")

if __name__ == "__main__":
    ejecutar_defensa_balistica()