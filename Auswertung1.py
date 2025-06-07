#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 20:47:32 2025

@author: ali-mert-akcay
"""

from scipy.integrate import cumulative_trapezoid
import numpy as np


def steigungausn(n):

    # Struktur:
    zeile = len(n[:, 0])
    spalte = len(n[0, :])
    steigungsfeld = np.empty((zeile, spalte), dtype=object)

    for j in range(zeile):
        for i in range(spalte):  # anzahl_phi
            puffer = n[j, i]
            mx = -puffer[0]/puffer[2]
            my = -puffer[1]/puffer[2]
            steigungsfeld[j, i] = np.array([mx, my])

    return steigungsfeld


def normalvektor(Vein, Vout, standartVein=True):
    # Eingabe des Ausgangs Vektorfelds
    # Ausgabe des Normalvektors
    # TODO: Idee für später Matrix um Formen in eine Lange Kette
    # Sollte der Eintritsvekotr bekannt sein kann es von Aussen eingeben werden
    import numpy as np
    zeile = len(Vout[:, 0])
    spalte = len(Vout[0, :])

    n = np.empty((zeile, spalte), dtype=object)
    if standartVein:
        Vein = np.empty((zeile, spalte), dtype=object)
        Vein[:, :] = [0, 0, -1]

    for i in range(zeile):
        for j in range(spalte):
            Veinnom = np.linalg.norm(Vein[i, j])
            Voutnom = np.linalg.norm(Vout[i, j])

            Veinpuffer = Vein[i, j]/Veinnom
            Voutpuffer = Vout[i, j]/Voutnom

            # Vein = Vein/Veinnom
            # Vout = Vout/Voutnom

            Vdot = np.dot(Veinpuffer, Voutpuffer)
            # Formel aus der Doktorarbeit korekkt umgesetzt
            npuffer = (Voutpuffer-Veinpuffer)/(np.sqrt(2*(1-Vdot)))
            n[i, j] = npuffer

    return n


def Integration(steigungsfeld, X, Y):
    import numpy as np
    from scipy.integrate import cumulative_trapezoid
    # README: Angepasste Intgration Mit der neuen DatenStruktur
    # TODO: Diese Integration funktioniert nicht für Jede Funktion
    m = steigungsfeld

    fintnachy = np.zeros(m.shape)
    fintnachx = np.zeros(m.shape)

    zeile = len(steigungsfeld[:, 0])
    spalte = len(steigungsfeld[0, :])
    # matrix = np.empty((),dtype=object)

    mx = np.zeros(m.shape)
    my = np.zeros(m.shape)

    for j in range(m.shape[0]):
        for i in range(m.shape[1]):
            mx[j, i] = m[j, i][0]
            my[j, i] = m[j, i][1]

    # Integration jeweilig nach Y
    for i in range(spalte):
        fintnachy[i, :] = cumulative_trapezoid(my[:, i], Y[:, i], initial=0)
    # Integration Jeweilig nach X
    for j in range(zeile):
        fintnachx[:, j] = cumulative_trapezoid(mx[j, :], X[j, :], initial=0)

    z_combined = fintnachx+fintnachy

    return z_combined


def extraktionkordinaten():
    # Funktion zur extraktion der Kordinaten aus der RAW datei falls Nötig

    X = 1
    Y = 1
    return X, Y


def pfadintegral_cumtrapz(steigungsfeld, X, Y):
    """
    Integriert ein Vektorfeld entlang (erst X, dann Y) mit scipy.integrate.cumulative_trapezoid.
    """
    ny, nx = steigungsfeld.shape

    # Komponenten des Vektorfeldes extrahieren
    mx = np.zeros((ny, nx))
    my = np.zeros((ny, nx))

    for j in range(ny):
        for i in range(nx):
            mx[j, i] = steigungsfeld[j, i][0]
            my[j, i] = steigungsfeld[j, i][1]

    z = np.zeros((ny, nx))

    # Integration entlang X-Richtung (Zeile für Zeile)
    for j in range(ny):
        z[j, :] = cumulative_trapezoid(mx[j, :], X[j, :], initial=0)

    # Integration entlang Y-Richtung (Spalte für Spalte) wird **additiv** hinzugefügt
    for i in range(nx):
        z[:, i] += cumulative_trapezoid(my[:, i], Y[:, i], initial=0)

    return z


def integrated_potential(steigungsfeld, X, Y, ref_point=(0, 0), ref_value=0):
    """
    Reconstructs the scalar function from its gradient field using numerical integration.

    Parameters:
        steigungsfeld: 2D array of shape (rows, cols), each element is [p, q]
                        where p = ∂f/∂x, q = ∂f/∂y
        X: 2D array of x-coordinates
        Y: 2D array of y-coordinates
        ref_point: tuple (i_ref, j_ref), indices of the reference point
        ref_value: scalar, the value of the function at the reference point

    Returns:
        f: 2D array of reconstructed function values
    """
    import numpy as np
    from scipy.integrate import cumulative_trapezoid

    m = steigungsfeld
    rows, cols = m.shape

    # Extract p and q derivatives into separate matrices
    p = np.zeros((rows, cols))
    q = np.zeros((rows, cols))
    for j in range(cols):
        for i in range(rows):
            p[i, j] = m[i, j][0]
            q[i, j] = m[i, j][1]

    # Initialize potential arrays
    f_x = np.zeros_like(p)
    f_y = np.zeros_like(q)

    # Integrate along X (rows)
    for j in range(cols):
        f_x[:, j] = cumulative_trapezoid(p[:, j], X[:, j], initial=0)
        # Shift so that reference point matches ref_value
        shift_x = ref_value - f_x[ref_point]
        f_x[:, j] += shift_x

    # Integrate along Y (columns)
    for i in range(rows):
        f_y[i, :] = cumulative_trapezoid(q[i, :], Y[i, :], initial=0)
        # Shift so that reference point matches ref_value
        shift_y = ref_value - f_y[i, ref_point[1]]
        f_y[i, :] += shift_y

    # Combine the two estimates
    f_combined = (f_x + f_y) / 2

    # Adjust to ensure the reference point has the correct value
    # Calculate the difference at the reference point
    delta = ref_value - f_combined[ref_point]
    f_final = f_combined + delta

    return f_final


def poisson_solver(divergence, dx, dy):
    """
    Solve Laplace equation: ∇²f = divergence using FFT.
    """
    # Was genau ist divergence ? in diesem Sinne
    # TODO: Divergence Fixen muss hier die Divergenz des vektorfelds für die Steigung hin? 
    ny, nx = divergence.shape
    # Fourier coordinates
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky)

    # Fourier transform of divergence
    div_fft = np.fft.fft2(divergence)

    # Avoid division by zero at zero frequency
    denom = kx**2 + ky**2
    denom[0, 0] = 1  # Set zero frequency to 1 to avoid division by zero

    # Compute the Fourier transform of the solution
    f_fft = div_fft / denom

    # Set the zero frequency component to zero (or any constant)
    f_fft[0, 0] = 0

    # Inverse FFT to get the solution
    f = np.real(np.fft.ifft2(f_fft))
    return f

def divergenz(steigungsfeld):
    
    spalte = len(steigungsfeld[0,:])
    zeile = len(steigungsfeld[:,0])
    
    div = np.zeros((zeile,spalte))
    
    for i in range(zeile):
        for j in range(spalte):
            div[i,j] = steigungsfeld[i,j][0] + steigungsfeld[i,j][1]  
    
    
    
    
    return div

