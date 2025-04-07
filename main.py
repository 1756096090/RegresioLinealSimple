# Importar las bibliotecas necesarias
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  

# Importar el dataset
dataSet = pd.read_csv('FuelConsumption (1).csv')

# Seleccionar las columnas de interés (Consumo de combustible y Emisiones de CO2)
X = dataSet.iloc[:, 8:9].values  # FUEL CONSUMPTION (índice 8)
y = dataSet.iloc[:, -1].values  # COEMISSIONS (última columna)

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Entrenar el modelo de Regresión Lineal Simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  
regressor.fit(X_train, y_train)  

# Predecir los resultados del conjunto de prueba
y_pred = regressor.predict(X_test)  

# Visualizar los resultados del conjunto de entrenamiento
plt.scatter(X_train, y_train, color='red')  
plt.plot(X_train, regressor.predict(X_train), color='blue')  
plt.title('Consumo de combustible vs Emisiones de CO2 (Entrenamiento)')  
plt.xlabel('Consumo de combustible (L/100km)')  
plt.ylabel('Emisiones de CO2 (g/km)')  
plt.show()  

# Visualizar los resultados del conjunto de prueba
plt.scatter(X_test, y_test, color='red')  
plt.plot(X_train, regressor.predict(X_train), color='blue')  
plt.title('Consumo de combustible vs Emisiones de CO2 (Prueba)')  
plt.xlabel('Consumo de combustible (L/100km)')  
plt.ylabel('Emisiones de CO2 (g/km)')  
plt.show()  

# Análisis estadístico con statsmodels
import statsmodels.api as sm

X_test_sm = sm.add_constant(X_test)

model = sm.OLS(y_test, X_test_sm).fit()

print(model.summary())
