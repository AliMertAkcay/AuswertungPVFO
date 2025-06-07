#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 09:15:52 2025

@author: ali-mert-akcay
"""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.close("all")

def poisson_surface_reconstruction_large(p, q, tol=1e-6, maxiter=1000):
    """

    Parameters
    ----------
    p : float
        df/dx
    q : float
        df/dy
    tol : TYPE, optional
        DESCRIPTION. The default is 1e-6.
    maxiter : TYPE, optional
        DESCRIPTION. The default is 1000.

    Returns
    -------
    f : TYPE
        DESCRIPTION.

    """
    
    ny, nx = p.shape
    N = nx * ny

    # Divergenz berechnen
    dpdx = np.zeros_like(p)
    dqdy = np.zeros_like(q)
    dpdx[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / 2
    dpdx[:, 0] = p[:, 1] - p[:, 0]
    dpdx[:, -1] = p[:, -1] - p[:, -2]
    dqdy[1:-1, :] = (q[2:, :] - q[:-2, :]) / 2
    dqdy[0, :] = q[1, :] - q[0, :]
    dqdy[-1, :] = q[-1, :] - q[-2, :]
    div = dpdx + dqdy
    b = -div.flatten()

    # Laplace-Operator als Sparse-Matrix (5-Punkt-Stern)
    Ix = eye(nx, format='csr')
    Iy = eye(ny, format='csr')
    ex = np.ones(nx)
    ey = np.ones(ny)
    Dx = diags([ex, -2*ex, ex], [-1, 0, 1], shape=(nx, nx), format='csr')
    Dy = diags([ey, -2*ey, ey], [-1, 0, 1], shape=(ny, ny), format='csr')
    L = kron(Iy, Dx) + kron(Dy, Ix)

    # Dirichlet-Randbedingungen: f=0 am Rand
    mask = np.ones((ny, nx), dtype=bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    mask_flat = mask.flatten()

    # System auf innere Punkte beschränken
    L_interior = L[mask_flat][:, mask_flat]
    b_interior = b[mask_flat]

    # Iterativer Löser (Conjugate Gradient)
    f_interior, info = cg(L_interior, b_interior, tol=tol, maxiter=maxiter)
    if info != 0:
        print("Warnung: CG-Löser hat nicht konvergiert (info =", info, ")")

    # Ergebnis in Gesamtgitter einsetzen
    f = np.zeros(N)
    f[mask_flat] = f_interior
    f = f.reshape((ny, nx))
    return f

# Beispiel für große Matrizen
nx, ny = 10, 10  # Für sehr große Matrizen ggf. kleiner wählen, sonst dauert es lange!
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)
F_true = np.sin(np.pi * X) * np.sin(np.pi * Y)
p = np.gradient(F_true, axis=1)  # df/dx
q = np.gradient(F_true, axis=0)  # df/dy

F_reconstructed = poisson_surface_reconstruction_large(p, q)

# Fehler berechnen (Differenz)
error = F_true - F_reconstructed
abs_error = np.abs(error)
max_error = np.max(abs_error)
mean_error = np.mean(abs_error)

print(f"Maximaler Fehler: {max_error:.3e}")
print(f"Mittlerer Fehler: {mean_error:.3e}")

# Visualisierung
plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(F_true, origin='lower', cmap='viridis')
plt.colorbar()
plt.subplot(1,3,2)
plt.title("Rekonstruiert")
plt.imshow(F_reconstructed, origin='lower', cmap='viridis')
plt.colorbar()
plt.subplot(1,3,3)
plt.title("Absoluter Fehler")
plt.imshow(abs_error, origin='lower', cmap='inferno')
plt.colorbar()
plt.tight_layout()
plt.show()

#%% Test des Algorihtmuss auf den Affensatell

nx,ny = 1000,1000
nx, ny = 1000, 1000  # Für sehr große Matrizen ggf. kleiner wählen, sonst dauert es lange!
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X,Y = np.meshgrid(x,y)

F_truemonkey = X**3-3*X*Y**2
p = np.gradient(F_truemonkey, axis=1)  # df/dx
q = np.gradient(F_truemonkey, axis=0)  # df/dy

F_reconstructedmonkey = poisson_surface_reconstruction_large(p, q)


error = F_truemonkey - F_reconstructedmonkey
abs_error = np.abs(error)
max_error = np.max(abs_error)
mean_error = np.mean(abs_error)

# Anpassungs Vorschlag von Perplexity:
F_reconstructedmonkey -= np.mean(F_reconstructedmonkey) 

print(f"Maximaler Fehler: {max_error:.3e}")
print(f"Mittlerer Fehler: {mean_error:.3e}")


fig = plt.figure("3D Dimnesionaler vergleich Zwischen dem Originalen und dem Reconstruierten")
ax = fig.add_subplot(111,projection='3d')

# Plots OHNE label-Argument!
surf1 = ax.plot_surface(X, Y, F_reconstructedmonkey, cmap='viridis', alpha=0.7)
#surf2 = ax.plot_surface(X, Y, F_truemonkey, cmap='plasma', alpha=0.7)

# Dummy-Handles für die Legende
legend_elements = [
    Patch(facecolor='mediumseagreen', edgecolor='k', label='Rekonstruierte Oberfläche'),
    #Patch(facecolor='orange', edgecolor='k', label='Originale Funktionsoberfläche')
]

ax.legend(handles=legend_elements)
ax.set_xlabel('X-Achse')
ax.set_ylabel('Y-Achse')
ax.set_zlabel('Z-Achse')
ax.set_title('3D-Oberflächenvergleich')


fig2 = plt.figure("OriginaleFunktion")
ax2 = fig2.add_subplot(111,projection='3d')

# Plots OHNE label-Argument!
#surf1 = ax.plot_surface(X, Y, F_reconstructedmonkey, cmap='viridis', alpha=0.7)
surf2 = ax2.plot_surface(X, Y, F_truemonkey, cmap='plasma', alpha=0.7)

# Dummy-Handles für die Legende
legend_elements = [
    #Patch(facecolor='mediumseagreen', edgecolor='k', label='Rekonstruierte Oberfläche'),
    Patch(facecolor='orange', edgecolor='k', label='Originale Funktionsoberfläche')
]

ax2.legend(handles=legend_elements)
ax2.set_xlabel('X-Achse')
ax2.set_ylabel('Y-Achse')
ax2.set_zlabel('Z-Achse')
ax2.set_title('3D-Oberflächenvergleich')
plt.show()



