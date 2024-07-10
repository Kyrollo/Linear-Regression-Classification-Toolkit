import numpy as np

def lin_equation(w, x, c):
    return w * x + c

X_values = np.array([1, 2, 3, -4, -5])
X_values = np.reshape(X_values, (-1, 1))
y_values = lin_equation(3, X_values, -5)
y_values = np.reshape(y_values, (-1, 1))

def matrix_inverse(X):
    return np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), X.transpose())

def MSSE(X, Y):
    X_inv = matrix_inverse(X)
    W = np.matmul(X_inv, Y)
    cost = (1 / 2) * (np.matmul(Y.transpose(), Y) + np.matmul(np.matmul(X, W).transpose(), np.matmul(X, W)) - 2 * np.matmul(np.matmul(X, W).transpose(), Y))
    print("cost=", cost)
    return np.array(W)

def classification(Y_hat):
    return np.where(Y_hat >= 0, "class 1", "class 2")

def add_noise(Y):
    noise = np.random.normal(loc=0, scale=1, size=Y.shape)
    return Y + noise

W = np.zeros((X_values.shape[1], 1))

W = MSSE(X_values, y_values)
Y_hat = np.matmul(X_values, W)
classes = classification(Y_hat)
print("Classification:", classes)
print("weights:", W)

for i in range(10):
    print("\niteration ", i + 1)
    y_values = add_noise(y_values)
    W = MSSE(X_values, y_values)
    Y_hat = np.matmul(X_values, W)
    classes = classification(Y_hat)
    print("Classification:", classes)
    print("Updated weights:", W)
