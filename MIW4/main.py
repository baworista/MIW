import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Wczytanie danych
a = np.loadtxt('dane16.txt')
X = a[:, [0]]
y = a[:, [1]]

# Podział danych na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Model 1: Liniowy model parametryczny
c1 = np.hstack([X_train, np.ones_like(X_train)])
v1 = np.linalg.pinv(c1) @ y_train

# Model 1 RMSE na danych treningowych
y_pred1tren = c1 @ v1
rmse1tren = np.sqrt(np.mean((y_pred1tren - y_train) ** 2))
# Model 1 RMSE na danych testowych
y_pred1_test = np.hstack([X_test, np.ones_like(X_test)]) @ v1
rmse1_test = np.sqrt(np.mean((y_pred1_test - y_test) ** 2))



# Model 2: Wielomianowy model parametryczny stopnia 3
c2 = np.hstack([X_train**3, X_train**2, X_train, np.ones_like(X_train)])
v2 = np.linalg.pinv(c2) @ y_train

# Model 2 RMSE na danych treningowych
y_pred2tren = c2 @ v2
rmse2tren = np.sqrt(np.mean((y_pred2tren - y_train) ** 2))
# Model 2 RMSE na danych testowych
c2_test = np.hstack([X_test**3, X_test**2, X_test, np.ones_like(X_test)])
y_pred2_test = c2_test @ v2
rmse2_test = np.sqrt(np.mean((y_pred2_test - y_test) ** 2))



# Wykresy dla danych treningowych
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(X_train, y_train, 'ro', label='Dane treningowe')
plt.plot(X_train, c1 @ v1, 'g-', label='Model 1')
plt.title(f'Model 1 (RMSE={rmse1tren:.2f})')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(X_train, y_train, 'ro', label='Dane treningowe')
plt.plot(X_train, c2 @ v2, 'bo', label='Model 2')
plt.title(f'Model 2 (RMSE={rmse2tren:.2f})')
plt.legend()

# Wykresy dla danych testowych
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(X_test, y_test, 'ro', label='Dane testowe')
plt.plot(X_test, y_pred1_test, 'g-', label='Model 1')
plt.title(f'Model 1 (RMSE={rmse1_test:.2f})')
plt.legend()
plt.xlim(X_train.min(), X_train.max())
plt.ylim(0, 100)  # Ustaw zakres dla osi Y
plt.xlim(0, 1.4)

plt.subplot(1, 2, 2)
plt.plot(X_test, y_test, 'ro', label='Dane testowe')
plt.plot(X_test, y_pred2_test, 'bo', label='Model 2')
plt.title(f'Model 2 (RMSE={rmse2_test:.2f})')
plt.legend()
plt.xlim(X_train.min(), X_train.max())
plt.ylim(0, 100)
plt.xlim(0, 1.4)

plt.tight_layout()
plt.show()



# Wyświetlanie wyników
print("Model 1 RMSE (training):", rmse1tren)
print("Model 1 RMSE (test):", rmse1_test)
print("Model 2 RMSE (training):", rmse2tren)
print("Model 2 RMSE (test):", rmse2_test)