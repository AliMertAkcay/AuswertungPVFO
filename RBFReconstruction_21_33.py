"""
Created on Sat Jun  7 12:45:45 2025

@author: ali-mert-akcay
"""

#Hier ist die Differenz sehr Klein was gut ist mit RBF Funktionen ist es gut Realisiert, dies Solten wir so verfolgen

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from numpy.linalg import lstsq
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
    k = 20 
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
    
            points = np.column_stack((X.ravel(), Y.ravel()))
            N = points.shape[0]
    
            # Fast C-based pairwise distances
            r = cdist(points, points)
            Phi = np.exp(-(epsilon * r)**2)
    
            # For derivatives, use broadcasting as before (unless you find a more efficient analytical way)
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            Phi_dx = -2 * epsilon**2 * diff[:, :, 0] * Phi
            Phi_dy = -2 * epsilon**2 * diff[:, :, 1] * Phi
    
            # Least Squares
            A = np.vstack([Phi_dx, Phi_dy])
            b = np.hstack([gx, gy])
    
            coeffs, *_ = lstsq(A, b, rcond=None)
            #Z_rbf = Phi @ coeffs
            #print("Chunk: "+"Zeile: "+ str(i)+"Spalte: "+str(j)+" wird Aproximiert")
            #np.save(f"./Data/ZAprox/Chunk{i}{j}.npy", Z_rbf)
            np.save(f"./Data/ZAprox/Chunk{i}{j}.npy", Phi @ coeffs)
        
    return s
#%% 
def Plotfunktion(s,k):
    for i in range(s):
        for j in range(s):
            
            X = np.load(f"./Data/X/MessChunk_X{i}{j}.npy")
            Y = np.load(f"./Data/Y/MessChunk_Y{i}{j}.npy")
            Z_true_grid = np.load(f"./Data/ZAprox/Chunk{i}{j}.npy").reshape(k,k)
            
            
            
            fig = plt.figure(figsize=(18, 5))
            ax1 = fig.add_subplot(111, projection='3d')
            ax1.plot_surface(X, Y, Z_true_grid, cmap='viridis')
            ax1.set_title('Original: Rekunstruktion')
            
k,s = SimWerteforChunk()
RBF_Algorihtmuss(s,epsilon=1.0)
Plotfunktion(s,k)
