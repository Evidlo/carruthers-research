#!/usr/bin/env python3

# Explorations of B-Spline in 1D

import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

# sample points
x = np.logspace(1, 10, 100)
# true curve
y = np.e**(-x)

# find Bspline rep
t, c, k = interpolate.splrep(x, y, s=0, k=4)


N = 100
xmin, xmax = x.min(), x.max()
xx = np.linspace(xmin, xmax, N)
spline = interpolate.BSpline(t, c, k, extrapolate=False)

plt.plot(x, y, 'bo', label='Original points')
plt.plot(xx, spline(xx), 'r', label='BSpline')
plt.grid()
plt.legend(loc='best')
plt.show()