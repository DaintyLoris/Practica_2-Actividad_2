import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Cargar el conjunto de datos desde un archivo CSV
file_path = "auto_insurance_sweden.csv"
df = pd.read_csv(file_path)

# Utilizar las columnas 'Total payment' como características (X) y 'Number of claims' como etiquetas (Y)
X = df[["X"]]
y = df["Y"]

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar y entrenar un modelo de Regresión Lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Error cuadrático medio: {mse}")

plt.scatter(X_test, y_test, color='black', label='Datos reales')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Regresión Lineal')
plt.xlabel('Total payment')
plt.ylabel('Number of claims')
plt.legend()
plt.show()
