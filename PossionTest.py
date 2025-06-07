#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 12:40:47 2025

@author: ali-mert-akcay
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2

# Affensattel-Funktion und Ableitungen
def monkey_saddle(x, y):
    return x**3 - 3*x*y**2

def dfdx(x, y):
    return 3*x**2 - 3*y**2

def dfdy(x, y):
    return -6*x*y

nx, ny = 100, 100
x_vec = np.linspace(-2, 2, nx)
y_vec = np.linspace(-2, 2, ny)
X, Y = np.meshgrid(x_vec, y_vec)
Z_true = monkey_saddle(X, Y)
grad_x = dfdx(X, Y)
grad_y = dfdy(X, Y)

# Berechne Divergenz (rechte Seite der Poisson-Gleichung)
dx = x_vec[1] - x_vec[0]
dy = y_vec[1] - y_vec[0]
f = np.zeros_like(Z_true)
f[:, :-1] += grad_x[:, :-1] / dx
f[:, 1:] -= grad_x[:, :-1] / dx
f[:-1, :] += grad_y[:-1, :] / dy
f[1:, :] -= grad_y[:-1, :] / dy

# Poisson-Gleichung per FFT lösen
def poisson_solver_fft(f):
    ny, nx = f.shape
    f_hat = fft2(f)
    y = np.fft.fftfreq(ny).reshape(-1, 1)
    x = np.fft.fftfreq(nx).reshape(1, -1)
    denom = (2 * np.cos(2 * np.pi * x) - 2) / dx**2 + (2 * np.cos(2 * np.pi * y) - 2) / dy**2
    denom[0, 0] = 1  # Verhindere Division durch Null
    u_hat = f_hat / denom
    u_hat[0, 0] = 0  # Setze Mittelwert auf 0 (Integrationskonstante)
    u = np.real(ifft2(u_hat))
    return u

Z_recon = poisson_solver_fft(f)

# Differenz berechnen
diff = Z_true - Z_recon

# Plot
fig = plt.figure(figsize=(18, 5))
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z_true, cmap='viridis')
ax1.set_title('Original: Affensattel')

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, Z_recon, cmap='viridis')
ax2.set_title('Poisson-Rekonstruktion')

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, diff, cmap='seismic')
ax3.set_title('Differenz (Original - Rekonstruktion)')

plt.tight_layout()
plt.show()
