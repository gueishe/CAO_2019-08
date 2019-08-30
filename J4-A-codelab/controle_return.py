import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

# Class 1
x1 = [[3, 5, 7, 8], [3, 8, 15, 2], [-1, 15, 3, 7]]

# Class 0
x2 = [[7, -3, 2, 30], [2, 4, 3, 8], [4, -15, 15, 45], [15, -10, -10, 17]]

# We generage response data
y1 = [1 for a in x1]
y2 = [0 for a in x2]

X = x1 + x2
Y = y1 + y2

clf = LogisticRegression()
clf.fit(X, Y)

to_pred = [
    (0, 0, 0, 0),
    (4, 3, 2, 4),
    (4, 3, 3, 4),
    (4, 3, 4, 4),
    (4, 3, 5, 4),
    (4, 3, 6, 4),
    (9, 9, 9, 9),
]

for x in to_pred:
    print(f"Input : {x} => Class {clf.predict([x])[0]}")
