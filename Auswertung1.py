#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 20:47:32 2025

@author: ali-mert-akcay
"""

from scipy.integrate import cumulative_trapezoid
import numpy as np
from numpy.linalg import lstsq

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
    z2 = z
    
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

def steigungextraktion(steigungsfeld):
    
    spalte = len(steigungsfeld[0,:])
    zeile = len(steigungsfeld[:,0])
    
    fx = np.zeros((zeile,spalte))
    fy = fx
    
    for i in range(zeile):
        for j in range(spalte):
            fx[i,j] = steigungsfeld[i,j][0]  
            fy[i,j] = steigungsfeld[i,j][1]     
    
    return fx,fy

def IntegrationPhillip(steigungsfeld):
    from scipy.ndimage import gaussian_filter
    
    # Beispiel-Steigungen (mx, my) auf Kanten, z.B. 3x3
    # mx = np.array([
    #     [0.1, 0.2, 0.1],
    #     [0.0, 0.1, 0.0],
    #     [-0.1, 0.0, -0.1]
    # ])
    # my = np.array([
    #     [0.0, 0.1, 0.0],
    #     [0.2, 0.1, 0.2],
    #     [0.1, 0.0, 0.1]
    # ])
    # """
    #np.random.seed(10)  # Für reproduzierbare Ergebnisse
    
    # Beispiel-Steigungsmatrizen (mx, my) mit Werten zwischen -0.2 und 0.2
    #mx = np.random.uniform(-0.2, 0.2, (10, 10))
    #my = np.random.uniform(-0.2, 0.2, (10, 10))
    zeile = steigungsfeld.shape[0]
    spalte = steigungsfeld.shape[1] 
    
    mx = np.zeros((zeile,spalte))
    my = mx              
    
    for i in range(zeile):
        for j in range(spalte):
            mx[i,j] = steigungsfeld[i,j][0]
    
            my[i,j] = steigungsfeld[i,j][1]
    
    n, m = mx.shape #nimmt form von mx und gibt dim in n und m zurück
    height_x = np.zeros((n, m))  # Integration: erst Zeile, dann Spalte
    height_y = np.zeros((n, m))  # Integration: erst Spalte, dann Zeile
    
    # Integration über x-Pfad (erst Zeile, dann Spalte)
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                height_x[i, j] = 0
            elif i == 0 and j > 0:
                if j-1 < m:
                    height_x[i, j] = height_x[i, j-1] + mx[i, j-1]
            elif j == 0 and i > 0:
                if i-1 < n:
                    height_x[i, j] = height_x[i-1, j] + my[i-1, j]
            else:
                if i-1 < n and j-1 < m:
                    height_x[i, j] = height_x[i, j-1] + mx[i-1, j-1]
    
    # Integration über y-Pfad (erst Spalte, dann Zeile)
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                height_y[i, j] = 0
            elif j == 0 and i > 0:
                if i-1 < n and j < m:
                    height_y[i, j] = height_y[i-1, j] + my[i-1, j]
                else:
                    height_y[i, j] = height_y[i-1, j]
            elif i == 0 and j > 0:
                if j-1 < m:
                    height_y[i, j] = height_y[i, j-1] + mx[i, j-1]
                else:
                    height_y[i, j] = height_y[i, j-1]
            else:
                if i-1 < n and j < m:
                    height_y[i, j] = height_y[i-1, j] + my[i-1, j]
                else:
                    height_y[i, j] = height_y[i, j-1]
    
    # Mittelwert der beiden Integrationspfade
    height_avg = 0.5 * (height_x + height_y)
    sigma = 1.5
    height_smoothed = gaussian_filter(height_avg, sigma=sigma)
    return height_avg

def pfadintegral_cumtrapz_mittel(mx, my, x, y):
    #TODO: Dies ist die Korekkte Integration
    n, m = mx.shape
    height_x = np.zeros((n+1, m+1))
    height_y = np.zeros((n+1, m+1))

    # Pfad 1: erst Zeile (mx), dann Spalte (my)
    height_x[0, 1:] = cumulative_trapezoid(mx[0, :], x[1:], initial=0)
    height_x[1:, 0] = cumulative_trapezoid(my[:, 0], y[1:], initial=0)
    for i in range(1, n+1):
        height_x[i, 1:] = height_x[i, 0] + cumulative_trapezoid(mx[i-1, :], x[1:], initial=0)

    # Pfad 2: erst Spalte (my), dann Zeile (mx)
    height_y[1:, 0] = cumulative_trapezoid(my[:, 0], y[1:], initial=0)
    height_y[0, 1:] = cumulative_trapezoid(mx[0, :], x[1:], initial=0)
    for j in range(1, m+1):
        height_y[1:, j] = height_y[0, j] + cumulative_trapezoid(my[:, j-1], y[1:], initial=0)

    height_avg = 0.5 * (height_x + height_y)
    return height_avg


def monkey_saddle(x, y,i):
    
    match i:
        case 1: 
            F = x**3 - 3*x*y**2 
        case 2:
            F = x**2 + y **3
        case 3: 
            F = x*y**2+x**2*y**2
        case 4:
            F = np.sin(x**2)*10+y**2
            
    return F

def dfdx(x, y, i):
    match i:
        case 1: 
            Fx = 3*x**2 - 3*y**2 
        case 2:
            Fx = 2*x
        case 3: 
            Fx = y**2+2*x*y**2
        case 4:
            Fx = 20*x*np.cos(x**2) 
    
    return Fx

def dfdy(x, y,i):
    match i:
        case 1:
            # F = x**3 - 3*x*y**2
            dF_dy = -6 * x * y
        case 2:
            # F = x**2 + y**3
            dF_dy = 3 * y**2
        case 3:
            # F = x*y**2 + x**2*y**2
            dF_dy = 2 * x * y + 2 * x**2 * y
        case 4:
            # F = np.sin(x**2)*10 + y**2
            dF_dy = 2 * y
    
    return dF_dy


def SimWerteforChunk():
    """
    Doc: Diese Funktion Nimmt später messwerte an bzw eben die Simulierten werte und bricht sie in Blöcke auf
    Diese werden dann im Speicher also Auf der festplatte abgespeichert und bleiben dort für die Verarbeitung
    Dies ist notwendig aus zwei Gründen der Labor PC hat Bestimmt nicht 64 gb Ram und wenn doch wäre es trz.
    Es schlecht alles dort zu lassen. Auch deshalb weil die Auswertung sehr lange dauern kann und dem entsprechend
    die werte zwischen zu lagern sinn voll wäre auch weil der Rechner vllt abstürzen könnte.

    Returns
    -------
    k : TYPE
        DESCRIPTION.

    """
    k = 50
    nx, ny = 10000, 10000
    s = nx/k
    
    x_vec = np.linspace(-1, 1, nx)
    y_vec = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x_vec, y_vec)
    #points = np.column_stack((X.ravel(), Y.ravel()))
    
    # Funktionswerte und Gradienten
    Z_true = monkey_saddle(X, Y,i=1).ravel()
    gx = dfdx(X, Y,i=1)
    gy = dfdy(X, Y,i=1)
    
    
    for i in range(int(s)):
        # Gitterpunkte
        
        np.save("./Data/gx/"+"MessChunk_gx"+str(i)+".npy",gx[i*k:k*(i+1),i*k:k*(i+1)])
        np.save("./Data/gy/"+"MessChunk_gy"+str(i)+".npy",gy[i*k:k*(i+1),i*k:k*(i+1)])
        
        np.save("./Data/X/"+"MessChunk_X"+str(i)+".npy",X[i*k:k*(i+1),i*k:k*(i+1)])
        np.save("./Data/Y/"+"MessChunk_Y"+str(i)+".npy",Y[i*k:k*(i+1),i*k:k*(i+1)])
        print(i)
    return k,s

def RBF_Algorihtmuss(s):
    """
    Algorihtmuss zur Rekonstruktion der Oberfläche mit hilfe der RBF Funktion die Theorie muss ich mir noch angucken
    https://en.wikipedia.org/wiki/Radial_basis_function
    https://de.wikipedia.org/wiki/Radiale_Basisfunktion
    """    
    for i in range(s):
        
        X = np.load("./Data/X/"+"MessChunk_X"+str(i)+".npy")
        Y = np.load("./Data/Y/"+"MessChunk_Y"+str(i)+".npy")
        
        gx = np.load("./Data/gx/"+"MessChunk_gx"+str(i)+".npy").ravel()
        gy = np.load("./Data/gy/"+"MessChunk_gy"+str(i)+".npy").ravel()
        
        #gxrav = gx.ravel
        #gyrav = gy.ravel
        
        points = np.column_stack((X.ravel(), Y.ravel()))    
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
        np.save("./Data/ZAprox/"+"Chunk"+str(i)+".npy",Z_rbf)
        print("Chunk: "+str(i)+"wird Aproximiert")
        
    return s
