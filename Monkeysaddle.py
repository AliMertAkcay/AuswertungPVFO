#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 15:11:55 2025

@author: ali-mert-akcay
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

def monkey_saddle(x, y):
    """
    Monkey saddle function: f(x, y) = x^3 - 3xy^2
    """
    return x**3 - 3 * x * y**2

def dfdx(x, y):
    """
    Partial derivative with respect to x: 3x^2 - 3y^2
    """
    return 3 * x**2 - 3 * y**2

def dfdy(x, y):
    """
    Partial derivative with respect to y: -6xy
    """
    return -6 * x * y


def path_integration_from_derivatives(dfdx, dfdy, X, Y):
    """
    Integrate a gradient field on a grid (X, Y) using path integration.
    
    Parameters:
        dfdx: array, shape (nx, ny)
            Partial derivative of f with respect to x at each grid point.
        dfdy: array, shape (nx, ny)
            Partial derivative of f with respect to y at each grid point.
        X, Y: arrays, shape (nx, ny)
            Grid coordinates.
    
    Returns:
        z_x_first: integrate dfdx along x first, then dfdy along y
        z_y_first: integrate dfdy along y first, then dfdx along x
        z_avg: average of both integration paths
    """
    gx = dfdx
    gy = dfdy
    nx, ny = gx.shape

    # Integrate along x for each row (fixed y), then along y
    z_x_first = np.zeros((nx, ny))
    for j in range(ny):
        z_x_first[:, j] = cumulative_trapezoid(gx[:, j], X[:, j], initial=0)
    for i in range(nx):
        z_x_first[i, :] += cumulative_trapezoid(gy[i, :], Y[i, :], initial=0)

    # Integrate along y for each column (fixed x), then along x
    z_y_first = np.zeros((nx, ny))
    for i in range(nx):
        z_y_first[i, :] = cumulative_trapezoid(gy[i, :], Y[i, :], initial=0)
    for j in range(ny):
        z_y_first[:, j] += cumulative_trapezoid(gx[:, j], X[:, j], initial=0)

    # Symmetric average
    z_avg = 0.5 * (z_x_first + z_y_first)

    return z_x_first, z_y_first, z_avg


if __name__ == "__main__":
    # Generate grid
    nx, ny = 200, 200
    x = np.linspace(-10, 10, nx)
    y = np.linspace(-10, 10, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Evaluate function and derivatives
    Z = monkey_saddle(X, Y)
    GX = dfdx(X, Y)
    GY = dfdy(X, Y)
    
    _,_,ZRecon = path_integration_from_derivatives(GX, GY, X, Y)
    
    # Visualization
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis')
    ax1.set_title("Monkey Saddle Surface")
    fig.colorbar(surf, ax=ax1, shrink=0.5)

    ax2 = fig.add_subplot(132)
    c1 = ax2.contourf(X, Y, GX, cmap='coolwarm')
    ax2.set_title("df/dx")
    fig.colorbar(c1, ax=ax2)

    ax3 = fig.add_subplot(133)
    c2 = ax3.contourf(X, Y, GY, cmap='coolwarm')
    ax3.set_title("df/dy")
    fig.colorbar(c2, ax=ax3)

    plt.tight_layout()
    
    fig2 = plt.figure(figsize=(15, 5))
    ax4 = fig2.add_subplot(131, projection='3d')
    surf = ax4.plot_surface(X, Y, ZRecon)
    ax4.set_title("Recon")
    
    Error = abs(Z-ZRecon)
    miner = np.min(Error) 
    print("Min Error is: ",miner)
    
    ax5 = fig2.add_subplot(132,projection='3d')
    surf = ax5.plot_surface(X,Y,Error)
    ax5.set_title("Abweichung")
    
    plt.show()