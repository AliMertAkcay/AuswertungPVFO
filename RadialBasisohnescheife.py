#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 12:45:45 2025

@author: ali-mert-akcay
"""

#Hier ist die Differenz sehr Klein was gut ist mit RBF Funktionen ist es gut Realisiert 

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from numpy.linalg import lstsq

# Affensattel-Funktion und Ableitungen
def monkey_saddle(x, y):
    return x**3 - 3*x*y**2

def dfdx(x, y):
    return 3*x**2 - 3*y**2

def dfdy(x, y):
    return -6*x*y

# Gitterpunkte
nx, ny = 50, 50
x_vec = np.linspace(-1, 1, nx)
y_vec = np.linspace(-1, 1, ny)
X, Y = np.meshgrid(x_vec, y_vec)
points = np.column_stack((X.ravel(), Y.ravel()))

# Funktionswerte und Gradienten
Z_true = monkey_saddle(X, Y).ravel()
gx = dfdx(X, Y).ravel()
gy = dfdy(X, Y).ravel()

N = points.shape[0]
epsilon = 1.0

# Paarweise Differenzen für alle Punktpaare (vektorisiert)
diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (N, N, 2)
r = np.linalg.norm(diff, axis=2)  # (N, N)

# Gaußsche RBF und Ableitungen (vektorisiert)
Phi = np.exp(-(epsilon * r)**2)  # (N, N)
# Ableitungen nach x und y (Kettenregel, vektorisiert)
Phi_dx = -2 * epsilon**2 * diff[:, :, 0] * Phi
Phi_dy = -2 * epsilon**2 * diff[:, :, 1] * Phi

# Least-Squares-System aufstellen
A = np.vstack([Phi_dx, Phi_dy])
b = np.hstack([gx, gy])

# Lösung bestimmen
coeffs, _, _, _ = lstsq(A, b, rcond=None)

# Oberfläche rekonstruieren
Z_rbf = Phi @ coeffs

# Visualisierung
Z_rbf_grid = Z_rbf.reshape((ny, nx))
Z_true_grid = Z_true.reshape((ny, nx))
diff_grid = Z_true_grid - Z_rbf_grid

fig = plt.figure(figsize=(18, 5))
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z_true_grid, cmap='viridis')
ax1.set_title('Original: Affensattel')

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, Z_rbf_grid, cmap='viridis')
ax2.set_title('RBF-Rekonstruktion (Least Squares)')

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, diff_grid, cmap='seismic')
ax3.set_title('Differenz (Original - RBF)')
plt.tight_layout()
plt.show()
