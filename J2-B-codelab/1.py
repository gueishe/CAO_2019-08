import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


x = []
for i in range(10):
    for j in range(10 - i):
        x.append([i, j])

y = []
for j in range(10):
    for i in range(10 - j, 10):
        y.append([i, j])

x1 = [[3, 5, 7, 8], [3, 8, 15, 2], [-1, 15, 3, 7]]

x2 = [[7, -3, 2, 30], [2, 4, 3, 8], [4, -15, 15, 45], [15, -10, -10, 17]]


x = x * 50
y = y * 50

X = x + y
y_ = [0 for a in x] + [1 for b in y]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_, test_size=0.3, random_state=42
)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

l_reg = LinearRegression()
l_reg.fit(X_train, y_train)

pred = l_reg.predict(X_test)

h = 0.03
x_start = y_start = -1
x_end = y_end = 10
x_grid, y_grid = np.meshgrid(np.arange(x_start, x_end, h), np.arange(y_start, y_end, h))
Z = l_reg.predict(np.c_[x_grid.ravel(), y_grid.ravel()]).reshape(x_grid.shape)

# plt.pcolormesh(x_grid, y_grid, Z, cmap=plt.cm.Paired)
plt.contour(x_grid, y_grid, Z, levels=[0.5])
plt.scatter(X_train[0:, 0], X_train[0:, 1], c=y_train)
plt.scatter(X_test[0:, 0], X_test[0:, 1], c=y_test)

plt.xticks(())
plt.yticks(())

plt.show()
