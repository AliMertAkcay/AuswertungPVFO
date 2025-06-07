"""
Created on Sat Jun  7 12:45:45 2025

@author: ali-mert-akcay
"""

#Hier ist die Differenz sehr Klein was gut ist mit RBF Funktionen ist es gut Realisiert, dies Solten wir so verfolgen

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from numpy.linalg import lstsq
import cupy as cp
import numpy as np
#from cuml.linalg import lstsq
plt.close("all")

# Affensattel-Funktion und Ableitungen
def sklarfunction(x, y,i):
    
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
    Diese werden dann im Speicher also Auf der Festplatte abgespeichert und bleiben dort für die Verarbeitung.
    Dies ist notwendig aus zwei Gründen der Labor PC hat bestimmt nicht 64 gb Ram und wenn doch wäre es trz.
    es schlecht alles dort zu lassen. Auch deshalb weil die Auswertung sehr lange dauern kann und dem entsprechend
    die werte zwischen zu lagern sinn voll wäre auch weil der Rechner vllt abstürzen könnte.

    Returns
    -------
    k : TYPE
        Size of the Chunk
    
    s : TYPE
        Number of iterations for the calculation

    """
    k = 50 
    nx, ny = 100, 100
    s = int(nx/k)
    
    x_vec = np.linspace(-1, 1, nx)
    y_vec = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x_vec, y_vec)
    #points = np.column_stack((X.ravel(), Y.ravel()))
    
    # Funktionswerte und Gradienten
    Z_true = sklarfunction(X, Y,i=1).ravel()
    gx = dfdx(X, Y,i=1)
    gy = dfdy(X, Y,i=1)
    
    
    for i in range(int(s)):
        for j in range(int(s)):
            
            np.save("./Data/gx/"+"MessChunk_gx"+str(i)+str(j)+".npy",gx[i*k:k*(i+1),j*k:k*(i+1)])
            np.save("./Data/gy/"+"MessChunk_gy"+str(i)+str(j)+".npy",gy[i*k:k*(i+1),j*k:k*(i+1)])
            
            np.save("./Data/X/"+"MessChunk_X"+str(i)+str(j)+".npy",X[i*k:k*(i+1),j*k:k*(i+1)])
            np.save("./Data/Y/"+"MessChunk_Y"+str(i)+str(j)+".npy",Y[i*k:k*(i+1),j*k:k*(i+1)])
            print("Zeile:",i,"Spalte:",j)
        
    return k,s

#%% Algorihtmuss


def RBF_Algorihtmuss(s, epsilon=1.0):
    for i in range(s):
        for j in range(s):
            
            X = np.load(f"./Data/X/MessChunk_X{i}{j}.npy")
            Y = np.load(f"./Data/Y/MessChunk_Y{i}{j}.npy")
            gx = np.load(f"./Data/gx/MessChunk_gx{i}{j}.npy").ravel()
            gy = np.load(f"./Data/gy/MessChunk_gy{i}{j}.npy").ravel()
            
            X_gpu = cp.asarray(X.ravel())
            Y_gpu = cp.asarray(Y.ravel())
            gx_gpu = cp.asarray(gx.ravel())
            gy_gpu = cp.asarray(gy.ravel())
            points_gpu = cp.stack([X_gpu, Y_gpu], axis=1)
            N = points_gpu.shape[0]
        
            # Paarweise Distanzen auf der GPU
            diff_gpu = points_gpu[:, cp.newaxis, :] - points_gpu[cp.newaxis, :, :]
            r_gpu = cp.linalg.norm(diff_gpu, axis=2)
            Phi_gpu = cp.exp(-(epsilon * r_gpu) ** 2)
        
            # Ableitungen auf der GPU
            Phi_dx_gpu = -2 * epsilon**2 * diff_gpu[:, :, 0] * Phi_gpu
            Phi_dy_gpu = -2 * epsilon**2 * diff_gpu[:, :, 1] * Phi_gpu
        
            # Least Squares auf der GPU
            A_gpu = cp.vstack([Phi_dx_gpu, Phi_dy_gpu])
            b_gpu = cp.hstack([gx_gpu, gy_gpu])
            coeffs_gpu, *_ = lstsq(A_gpu, b_gpu)
        
            # Ergebnis berechnen
            Z_rbf_gpu = Phi_gpu @ coeffs_gpu
        
            # Ergebnis zurück auf die CPU und speichern
            #Z_rbf = cp.asnumpy(Z_rbf_gpu)
            #np.save(f"./Data/ZAprox/Chunk{i}{j}.npy", Z_rbf)
            cp.save(f"./Data/ZAprox/Chunk{i}{j}.npy", Z_rbf_gpu)
            print(f"Chunk: Zeile: {i} Spalte: {j} auf der GPU berechnet.")
            del Phi_gpu, Phi_dx_gpu, Phi_dy_gpu, r_gpu, diff_gpu, A_gpu, b_gpu, coeffs_gpu, Z_rbf_gpu
    return s


def Plotfunktion(s):
    for i in range(s):
        for j in range(s):
            
            X = np.load(f"./Data/X/MessChunk_X{i}{j}.npy")
            Y = np.load(f"./Data/Y/MessChunk_Y{i}{j}.npy")
            Z_true_grid = np.load(f"./Data/ZAprox/Chunk{i}{j}.npy")
            
            fig = plt.figure(figsize=(18, 5))
            ax1 = fig.add_subplot(111, projection='3d')
            ax1.plot_surface(X, Y, Z_true_grid, cmap='viridis')
            ax1.set_title('Original: Rekunstruktion')
    

    


#%% 
k,s = SimWerteforChunk()
RBF_Algorihtmuss(s,epsilon=1.0)
