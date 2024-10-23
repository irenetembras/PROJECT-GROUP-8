import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Paso 1: Cargar los datos desde el archivo CSV
df = pd.read_csv("C:/Users/User/Downloads/housing.csv")

# En lugar de usar inplace=True, reasignamos directamente
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].mean())


# Paso 3: Convertir la columna categórica 'ocean_proximity' en variables dummy
df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

# Paso 4: Definir las variables independientes (X) y la variable dependiente (y)
X = df.drop('median_house_value', axis=1)  # Variables predictoras
y = df['median_house_value']  # Variable objetivo

# Paso 5: Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 6: Entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Paso 7: Hacer predicciones y evaluar el modelo
y_train_pred = model.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

print("Error Cuadrático Medio (MSE):", mse_train)
print("Coeficiente de Determinación (R²):", r2_train)
