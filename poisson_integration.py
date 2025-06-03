import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def poisson_integration(fx, fy, dx, dy):
    """
    Reconstruct f from its gradient fields fx and fy via Poisson integration.
    
    Parameters:
      fx : 2D numpy array containing the gradient in x direction
      fy : 2D numpy array containing the gradient in y direction
      dx : grid spacing in the x-direction
      dy : grid spacing in the y-direction
    
    Returns:
      f  : 2D numpy array of the reconstructed surface
    """
    ny, nx = fx.shape

    # 1. Compute the divergence of the gradient field:
    #    div = ∂f_x/∂x + ∂f_y/∂y.
    dfx_dx = np.gradient(fx, dx, axis=1)
    dfy_dy = np.gradient(fy, dy, axis=0)
    div = dfx_dx + dfy_dy

    # 2. Set up the Poisson equation:
    # For interior points, the finite difference approximation for the Laplacian is:
    #
    #     [f(i+1,j) + f(i-1,j)]/dx^2 + [f(i,j+1) + f(i,j-1)]/dy^2 
    #     - f(i,j)*(2/dx^2 + 2/dy^2) = -divergence(i,j)
    #
    # For boundary points we enforce f = 0 (Dirichlet condition).
    N = nx * ny
    A = sp.lil_matrix((N, N))
    b = np.zeros(N)
    
    # Helper to convert 2D grid coordinates (i, j) to a linear index.
    def idx(i, j):
        return j * nx + i
    
    for j in range(ny):
        for i in range(nx):
            index = idx(i, j)
            if i == 0 or j == 0 or i == nx - 1 or j == ny - 1:
                # For boundary points: enforce f = 0.
                A[index, index] = 1.0
                b[index] = 0.0
            else:
                A[index, idx(i, j)]     = 2.0/dx**2 + 2.0/dy**2
                A[index, idx(i+1, j)]   = -1.0/dx**2
                A[index, idx(i-1, j)]   = -1.0/dx**2
                A[index, idx(i, j+1)]   = -1.0/dy**2
                A[index, idx(i, j-1)]   = -1.0/dy**2
                # Set the right-hand side to -divergence (from the rearranged Poisson eq.)
                b[index] = -div[j, i]
    
    # Convert the matrix to CSR format for efficient solving.
    A = A.tocsr()
    # Solve the linear system A f_flat = b.
    f_flat = spla.spsolve(A, b)
    # Reshape the solution back into a 2D array.
    f = f_flat.reshape(ny, nx)
    return f

# ========================
# Example Usage and Testing
# ========================

if __name__ == "__main__":
    # Define grid parameters.
    nx, ny = 50, 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)
    
    # Define an exact surface function, for instance:
    #    f(x,y) = sin(pi*x)*sin(pi*y)
    f_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    # Compute its gradients analytically.
    fx = np.pi * np.cos(np.pi * X) * np.sin(np.pi * Y)
    fy = np.pi * np.sin(np.pi * X) * np.cos(np.pi * Y)
    
    # Reconstruct f via Poisson integration from the gradients.
    f_reconstructed = poisson_integration(fx, fy, dx, dy)
    
    # ========================
    # Visualization of the results.
    # ========================
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    
    im0 = axs[0].imshow(f_exact, extent=[0,1,0,1], origin='lower')
    axs[0].set_title('Exact f')
    fig.colorbar(im0, ax=axs[0])
    
    im1 = axs[1].imshow(f_reconstructed, extent=[0,1,0,1], origin='lower')
    axs[1].set_title('Reconstructed f')
    fig.colorbar(im1, ax=axs[1])
    
    im2 = axs[2].imshow(f_exact - f_reconstructed, extent=[0,1,0,1], origin='lower')
    axs[2].set_title('Error')
    fig.colorbar(im2, ax=axs[2])
    
    plt.tight_layout()
    plt.show()
