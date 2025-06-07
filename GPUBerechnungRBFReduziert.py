#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU-beschleunigte RBF-Oberflächenrekonstruktion (Affensattel) mit CuPy
@author: ali-mert-akcay
"""

import cupy as cp
import matplotlib.pyplot as plt

def monkey_saddle(x, y):
    return x**3 - 3*x*y**2

def dfdx(x, y):
    return 3*x**2 - 3*y**2

def dfdy(x, y):
    return -6*x*y

# Use float32 to halve memory usage
nx, ny = 50, 50
x_vec = cp.linspace(-1, 1, nx, dtype=cp.float32)
y_vec = cp.linspace(-1, 1, ny, dtype=cp.float32)
X, Y = cp.meshgrid(x_vec, y_vec)
points = cp.column_stack((X.ravel(), Y.ravel()))

Z_true = monkey_saddle(X, Y).ravel()
gx = dfdx(X, Y).ravel()
gy = dfdy(X, Y).ravel()

N = points.shape[0]
epsilon = cp.float32(1.0)

# --- BATCHED PAIRWISE RBF CALCULATION ---
def compute_Phi_batched(points, epsilon, batch_size=2000):
    N = points.shape[0]
    Phi = cp.empty((N, N), dtype=cp.float32)
    Phi_dx = cp.empty((N, N), dtype=cp.float32)
    Phi_dy = cp.empty((N, N), dtype=cp.float32)
    for i in range(0, N, batch_size):
        i_end = min(i + batch_size, N)
        diff = points[i:i_end, cp.newaxis, :] - points[cp.newaxis, :, :]  # (batch, N, 2)
        r = cp.linalg.norm(diff, axis=2)
        phi = cp.exp(-(epsilon * r) ** 2)
        Phi[i:i_end, :] = phi
        Phi_dx[i:i_end, :] = -2 * epsilon**2 * diff[:, :, 0] * phi
        Phi_dy[i:i_end, :] = -2 * epsilon**2 * diff[:, :, 1] * phi
        del diff, r, phi  # Free memory
        cp.get_default_memory_pool().free_bytes()  # Try to free memory
    return Phi, Phi_dx, Phi_dy

Phi, Phi_dx, Phi_dy = compute_Phi_batched(points, epsilon, batch_size=2000)

A = cp.vstack([Phi_dx, Phi_dy])
b = cp.hstack([gx, gy])

coeffs, _, _, _ = cp.linalg.lstsq(A, b, rcond=None)
Z_rbf = Phi @ coeffs

# Visualization
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
