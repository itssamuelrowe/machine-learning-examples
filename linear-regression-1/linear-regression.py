import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

df = pd.read_csv('data.csv', header = None, names = [ 'x', 'y' ])
x = np.array(df.x)
y = np.array(df.y)
theta = np.zeros((2, 1))

def scatterPlot(x, y, yp, save, name):
    plt.xlabel('Population of city in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.scatter(x, y, marker='x')
    if yp is not None:
        plt.plot(x, yp)

    plt.show()

    if save:
        plt.savefig(name + '.png')

scatterPlot(x, y, None, True, 'data')

(m, b) = np.polyfit(x, y, 1)
print('Slope is ' + str(m))
print('Y intercept is ' + str(b))
yp = np.polyval([m, b], x)

scatterPlot(x, y, yp, True, 'curve')