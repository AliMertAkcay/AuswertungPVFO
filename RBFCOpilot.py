import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

def dfdx(x, y): return 3*x**2 - 3*y**2
def dfdy(x, y): return -6*x*y

nx, ny = 200, 200
x = np.linspace(-10, 10, nx)
y = np.linspace(-10, 10, ny)
X, Y = np.meshgrid(x, y, indexing='ij')
gx = dfdx(X, Y)
gy = dfdy(X, Y)

dx = x[1] - x[0]
dy = y[1] - y[0]
div = np.zeros_like(gx)
div[1:-1,1:-1] = ((gx[2:,1:-1] - gx[:-2,1:-1])/(2*dx) +
                  (gy[1:-1,2:] - gy[1:-1,:-2])/(2*dy))

N = nx * ny
def laplacian_2d(nx, ny, dx, dy):
    main = -2/(dx**2) -2/(dy**2)
    diag_x = 1/(dx**2)
    diag_y = 1/(dy**2)
    # Use LIL for efficient row modifications
    A = scipy.sparse.diags(
        [main*np.ones(N),
         diag_x*np.ones(N-1), diag_x*np.ones(N-1),
         diag_y*np.ones(N-ny), diag_y*np.ones(N-ny)],
        [0, -1, 1, -ny, ny], shape=(N, N), format='lil'
    )
    for i in range(nx):
        for j in range(ny):
            idx = i*ny + j
            if i==0 or i==nx-1 or j==0 or j==ny-1:
                A[idx, :] = 0
                A[idx, idx] = 1
    return A.tocsr()  # Convert to CSR for solving

A = laplacian_2d(nx, ny, dx, dy)
b = div.reshape(-1)
for i in range(nx):
    for j in range(ny):
        idx = i*ny + j
        if i==0 or i==nx-1 or j==0 or j==ny-1:
            b[idx] = 0

u = scipy.sparse.linalg.spsolve(A, b)
f = u.reshape((nx, ny))

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X, Y, f, cmap='viridis')
ax.set_title('Surface from gradient field')
ax2 = fig.add_subplot(122)
ax2.imshow(f, cmap='viridis')
ax2.set_title("Heightmap")
plt.tight_layout()
plt.show()