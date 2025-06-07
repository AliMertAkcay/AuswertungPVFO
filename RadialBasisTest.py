#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 12:42:43 2025

@author: ali-mert-akcay
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from numpy.linalg import lstsq

# 1. Definition der Affensattel-Funktion und ihrer Ableitungen
def monkey_saddle(x, y):
    return x**3 - 3*x*y**2

def dfdx(x, y):
    return 3*x**2 - 3*y**2

def dfdy(x, y):
    return -6*x*y

# 2. Erzeuge Gitterpunkte
nx, ny = 100, 100
x_vec = np.linspace(-2, 2, nx)
y_vec = np.linspace(-2, 2, ny)
X, Y = np.meshgrid(x_vec, y_vec)
points = np.column_stack((X.ravel(), Y.ravel()))

# 3. Werte und Gradienten an den Gitterpunkten
Z_true = monkey_saddle(X, Y).ravel()
gx = dfdx(X, Y).ravel()
gy = dfdy(X, Y).ravel()

# 4. RBF-Definition (Gaussian)
def gaussian_rbf(r, epsilon=1.0):
    return np.exp(-(epsilon * r)**2)

# 5. Erzeuge das System für Gradienten-Matching
N = points.shape[0]
epsilon = 1.0  # RBF-Shape-Parameter

# Berechne Abstände zwischen allen Punkten
D = cdist(points, points)

# RBF-Werte
Phi = gaussian_rbf(D, epsilon)

# Ableitungen der RBFs nach x und y
def gaussian_rbf_dx(xi, xj, epsilon=1.0):
    r = np.linalg.norm(xi - xj)
    if r == 0:
        return 0.0
    return -2 * epsilon**2 * (xi[0] - xj[0]) * np.exp(-(epsilon * r)**2)

def gaussian_rbf_dy(xi, xj, epsilon=1.0):
    r = np.linalg.norm(xi - xj)
    if r == 0:
        return 0.0
    return -2 * epsilon**2 * (xi[1] - xj[1]) * np.exp(-(epsilon * r)**2)

# Erzeuge die Gradienten-Matching-Matrizen
Phi_dx = np.zeros((N, N))
Phi_dy = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        Phi_dx[i, j] = gaussian_rbf_dx(points[i], points[j], epsilon)
        Phi_dy[i, j] = gaussian_rbf_dy(points[i], points[j], epsilon)

# Kombiniere Gradientenbedingungen (Stacken)
A = np.vstack([Phi_dx, Phi_dy])
b = np.hstack([gx, gy])

# Löse das Least-Squares-Problem für die Koeffizienten
coeffs, _, _, _ = lstsq(A, b, rcond=None)

# Rekonstruiere die Oberfläche aus den Koeffizienten
Z_rbf = Phi @ coeffs

# 6. Visualisierung
Z_rbf_grid = Z_rbf.reshape((ny, nx))
Z_true_grid = Z_true.reshape((ny, nx))
diff = Z_true_grid - Z_rbf_grid

fig = plt.figure(figsize=(18, 5))
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z_true_grid, cmap='viridis')
ax1.set_title('Original: Affensattel')

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, Z_rbf_grid, cmap='viridis')
ax2.set_title('RBF-Rekonstruktion (Least Squares)')

ax3 = fig.add_subplot(133, projection='3d')
surf = ax3.plot_surface(X, Y, diff, cmap='seismic')
ax3.set_title('Differenz (Original - RBF)')
plt.tight_layout()
plt.show()
