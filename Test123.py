import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

# Affensattel-Funktion und Ableitungen
def monkey_saddle(x, y):
    return x**3 - 3*x*y**2

def dfdx(x, y):
    return 3*x**2 - 3*y**2

def dfdy(x, y):
    return -6*x*y

# Gitter erstellen
nx, ny = 100, 100
x_vec = np.linspace(-2, 2, nx)
y_vec = np.linspace(-2, 2, ny)
X, Y = np.meshgrid(x_vec, y_vec)

# Originalfunktion und Gradienten berechnen
Z_true = monkey_saddle(X, Y)
grad_x = dfdx(X, Y)
grad_y = dfdy(X, Y)

# Rekonstruktion aus Gradienten
def integrate_gradients_with_coords(grad_x, grad_y, x_vec, y_vec):
    ny, nx = grad_x.shape
    Zx = np.zeros((ny, nx))
    Zy = np.zeros((ny, nx))

    # Integration entlang x (jede Zeile)
    for i in range(ny):
        Zx[i, :] = cumtrapz(grad_x[i, :], x_vec, initial=0)

    # Integration entlang y (jede Spalte)
    for j in range(nx):
        Zy[:, j] = cumtrapz(grad_y[:, j], y_vec, initial=0)

    # Mittelung der beiden Ergebnisse
    Z = (Zx + Zy) / 2.0
    return Z

Z_recon = integrate_gradients_with_coords(grad_x, grad_y, x_vec, y_vec)

# Differenz berechnen
diff = Z_true - Z_recon

# Plot: Original, Rekonstruktion, Differenz
fig = plt.figure(figsize=(18, 5))

ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z_true, cmap='viridis')
ax1.set_title('Original: Affensattel')

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, Z_recon, cmap='viridis')
ax2.set_title('Rekonstruktion aus Gradienten')

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, diff, cmap='seismic')
ax3.set_title('Differenz (Original - Rekonstruktion)')

plt.tight_layout()
plt.show()
