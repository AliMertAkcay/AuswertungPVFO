#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU-beschleunigte RBF-Oberflächenrekonstruktion (Affensattel) mit CuPy
@author: ali-mert-akcay
"""

import cupy as cp
import matplotlib.pyplot as plt

# Affensattel-Funktion und Ableitungen (jetzt für CuPy)
def monkey_saddle(x, y):
    return x**3 - 3*x*y**2

def dfdx(x, y):
    return 3*x**2 - 3*y**2

def dfdy(x, y):
    return -6*x*y

# Gitterpunkte (z.B. 2000 x 2000)
nx, ny = 200, 200  # Für den Anfang, erhöhe ggf. auf 2000, wenn genug GPU-RAM vorhanden!
x_vec = cp.linspace(-1, 1, nx)
y_vec = cp.linspace(-1, 1, ny)
X, Y = cp.meshgrid(x_vec, y_vec)
points = cp.column_stack((X.ravel(), Y.ravel()))

# Funktionswerte und Gradienten
Z_true = monkey_saddle(X, Y).ravel()
gx = dfdx(X, Y).ravel()
gy = dfdy(X, Y).ravel()

N = points.shape[0]
epsilon = 1.0

# Paarweise Differenzen für alle Punktpaare (vektorisiert, auf GPU)
diff = points[:, cp.newaxis, :] - points[cp.newaxis, :, :]  # (N, N, 2)
r = cp.linalg.norm(diff, axis=2)  # (N, N)

# Gaußsche RBF und Ableitungen (vektorisiert)
Phi = cp.exp(-(epsilon * r)**2)  # (N, N)
Phi_dx = -2 * epsilon**2 * diff[:, :, 0] * Phi
Phi_dy = -2 * epsilon**2 * diff[:, :, 1] * Phi


# Least-Squares-System aufstellen
A = cp.vstack([Phi_dx, Phi_dy])
b = cp.hstack([gx, gy])

# Lösung bestimmen (GPU)
coeffs, _, _, _ = cp.linalg.lstsq(A, b, rcond=None)

# Oberfläche rekonstruieren
Z_rbf = Phi @ coeffs

# Für die Visualisierung zurück nach NumPy kopieren
Z_rbf_grid = cp.asnumpy(Z_rbf.reshape((ny, nx)))
Z_true_grid = cp.asnumpy(Z_true.reshape((ny, nx)))
X_plot = cp.asnumpy(X)
Y_plot = cp.asnumpy(Y)
diff_grid = Z_true_grid - Z_rbf_grid

fig = plt.figure(figsize=(18, 5))
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X_plot, Y_plot, Z_true_grid, cmap='viridis')
ax1.set_title('Original: Affensattel')

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X_plot, Y_plot, Z_rbf_grid, cmap='viridis')
ax2.set_title('RBF-Rekonstruktion (Least Squares, GPU)')

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X_plot, Y_plot, diff_grid, cmap='seismic')
ax3.set_title('Differenz (Original - RBF)')
plt.tight_layout()
plt.show()
