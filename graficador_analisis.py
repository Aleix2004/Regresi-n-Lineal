import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# Cargar el archivo que me pasaste
# Si el misil está dentro de la carpeta Attack000
df = pd.read_csv('Attack000/missile091.csv')
X = df[['tiempo']].values
y = df['posX'].values

# Regresión
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Graficar
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Detecciones Radar (Ruido)', alpha=0.5)
plt.plot(X, y_pred, color='blue', linewidth=2, label='Trayectoria Predictiva (Regresión)')
plt.title('Análisis de Trayectoria Misil 091')
plt.xlabel('Tiempo (s)')
plt.ylabel('Posición X')
plt.legend()
plt.grid(True)
plt.show()