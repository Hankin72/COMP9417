### Question 1
# (a)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
from prettytable import PrettyTable
from tabulate import tabulate

# Question2
X = [[-0.8, 1],
     [3.9, 0.4],
     [1.4, 1],
     [0.1, -3.3],
     [1.2, 2.7],
     [-2.45, 0.1],
     [-1.5, -0.5],
     [1.2, -1.5]]
y = [1, -1, 1, -1, -1, -1, 1, 1]
X = np.array(X)
y = np.array(y)

column = np.array([0, 0, 0, 0, 0, 0, 0, 0])
X = np.column_stack((X, column))

X_temp = X

x1 = X[:, 0]
x2 = X[:, 1]

X[:, 2] = pow(2, 0.5) * x1 * x2
X[:, 0] = x1 * x1
X[:, 1] = x2 * x2

w = np.array([1, 1, 1, 1], dtype='float64')
# learning rate = 0.2
lr = 0.2






converged = 0
m = 1
iteration = []
iteration.append(0)


iteration_total = []
n = 1
iteration_total.append(0)
weights = np.array([[1, 1, 1, 1]], dtype='float64')

weight_total = np.array([[1, 1, 1, 1]], dtype='float64')

while converged == 0:
    converged = 1
    for i in range(8):
        w0 = w[0]
        w1 = w[1]
        w2 = w[2]
        w3 = w[3]
        y_ = w0 + w1 * X[:, 0][i] + w2 * X[:, 1][i] + w3 * X[:, 2][i]
        signal = y_ * y[i]
        if signal <= 0:
            w[0] = w0 + (y[i] * 0.2 * 1)
            w[1] = w1 + (y[i] * 0.2 * X[:, 0][i])
            w[2] = w2 + (y[i] * 0.2 * X[:, 1][i])
            w[3] = w3 + (y[i] * 0.2 * X[:, 2][i])
            converged = 0
            iteration.append(m)
            weight_total = np.concatenate((weight_total, [w]))
            m = m + 1
        iteration_total.append(n)
        weights = np.concatenate((weights, [w]))
        n = n+1

print(f"The final weight vector is {w}.\n")

t = PrettyTable(['Iter_No. of each update', 'w0', 'w1', 'w2', 'w3'])
for i in range(len(iteration)):
    t.add_row([iteration[i], weight_total[i][0], weight_total[i][1], weight_total[i][2], weight_total[i][3] ])

print(t)

t_2 = PrettyTable(['Total iteration', 'w0', 'w1', 'w2', 'w3'])
for i in range(len(iteration_total)):
    t_2.add_row([iteration_total[i], weights[i][0], weights[i][1], weights[i][2], weights[i][3] ])

print(t_2)



