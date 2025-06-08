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
import os
import shutil

import os
import shutil

trashpfad = ""

# Die zu löschenden Ordnerpfade
ordner_liste = [
    "./Data/gx",
    "./Data/gy",
    "./Data/X",
    "./Data/Y",
    "./Data/ZAprox",
    ""
]

antwort = input("Soll der Inhalt aller 5 Ordner gelöscht werden? (ok zum Bestätigen):").strip().lower()

if antwort == "ok":
    for ordner in ordner_liste:
        if os.path.exists(ordner):
            for dateiname in os.listdir(ordner):
                dateipfad = os.path.join(ordner, dateiname)
                if os.path.isfile(dateipfad) or os.path.islink(dateipfad):
                    os.unlink(dateipfad)
                elif os.path.isdir(dateipfad):
                    shutil.rmtree(dateipfad)
            print(f"Inhalt von {ordner} gelöscht.")
        else:
            print(f"Ordner {ordner} existiert nicht.")
    print("Alle Inhalte gelöscht.")
else:
    print("Abgebrochen.")

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
    nx, ny = 1000, 1000
    s = int(nx/k)
    
    x_vec = np.linspace(-10, 10, nx)
    y_vec = np.linspace(-10, 10, ny)
    X, Y = np.meshgrid(x_vec, y_vec)
    #points = np.column_stack((X.ravel(), Y.ravel()))
    
    # Funktionswerte und Gradienten
    ztrue = sklarfunction(X, Y,i=1).ravel()
    gx = dfdx(X, Y,i=1)
    gy = dfdy(X, Y,i=1)
    
    
    for i in range(int(s)):
        for j in range(int(s)):
            
            np.save("./Data/gx/"+"MessChunk_gx"+str(i)+str(j)+".npy",gx[i*k:k*(i+1),j*k:k*(j+1)])
            np.save("./Data/gy/"+"MessChunk_gy"+str(i)+str(j)+".npy",gy[i*k:k*(i+1),j*k:k*(j+1)])
            
            np.save("./Data/X/"+"MessChunk_X"+str(i)+str(j)+".npy",X[i*k:k*(i+1),j*k:k*(j+1)])
            np.save("./Data/Y/"+"MessChunk_Y"+str(i)+str(j)+".npy",Y[i*k:k*(i+1),j*k:k*(j+1)])
            print("Zeile:",i,"Spalte:",j)
        
    return k,s,ztrue,X,Y

#%% Algorihtmuss


def RBF_Algorihtmuss(s, epsilon=1.0):
    j = 0
    p = 0
    for j in range(s):
        for p in range(s):
            X = np.load(f"./Data/X/MessChunk_X{p}{j}.npy")
            Y = np.load(f"./Data/Y/MessChunk_Y{p}{j}.npy")
            gx = np.load(f"./Data/gx/MessChunk_gx{p}{j}.npy").ravel()
            gy = np.load(f"./Data/gy/MessChunk_gy{p}{j}.npy").ravel()
        
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
            Z_rbf = Phi @ coeffs
            print("Chunk: "+"Zeile: "+ str(p)+"Spalte: "+str(j)+" wird Aproximiert")
            #np.save(f"./Data/ZAprox/Chunk{i}{j}.npy", Z_rbf)
            np.save(f"./Data/ZAprox/Chunk{p}{j}.npy", Z_rbf)
            
            
    return s

def RBF_Algorihtmuss_Randkopplung(s, k, epsilon=1.0):
    for i in range(s):
        for j in range(s):
            # Eigene Daten laden
            X = np.load(f"./Data/X/MessChunk_X{i}{j}.npy")
            Y = np.load(f"./Data/Y/MessChunk_Y{i}{j}.npy")
            gx = np.load(f"./Data/gx/MessChunk_gx{i}{j}.npy").ravel()
            gy = np.load(f"./Data/gy/MessChunk_gy{i}{j}.npy").ravel()

            points = np.column_stack((X.ravel(), Y.ravel()))
            points_total = points.copy()
            gx_total = gx.copy()
            gy_total = gy.copy()

            # Ränder der Nachbarn holen (nur bereits berechnete Chunks einbeziehen)
            # Obere Nachbar (i-1,j)
            if i > 0:
                X_top = np.load(f"./Data/X/MessChunk_X{i-1}{j}.npy")[-1, :]
                Y_top = np.load(f"./Data/Y/MessChunk_Y{i-1}{j}.npy")[-1, :]
                gx_top = np.load(f"./Data/gx/MessChunk_gx{i-1}{j}.npy")[-1, :].ravel()
                gy_top = np.load(f"./Data/gy/MessChunk_gy{i-1}{j}.npy")[-1, :].ravel()
                pts_top = np.column_stack((X_top, Y_top))
                points_total = np.vstack([points_total, pts_top])
                gx_total = np.concatenate([gx_total, gx_top])
                gy_total = np.concatenate([gy_total, gy_top])
            # Linker Nachbar (i,j-1)
            if j > 0:
                X_left = np.load(f"./Data/X/MessChunk_X{i}{j-1}.npy")[:, -1]
                Y_left = np.load(f"./Data/Y/MessChunk_Y{i}{j-1}.npy")[:, -1]
                gx_left = np.load(f"./Data/gx/MessChunk_gx{i}{j-1}.npy")[:, -1].ravel()
                gy_left = np.load(f"./Data/gy/MessChunk_gy{i}{j-1}.npy")[:, -1].ravel()
                pts_left = np.column_stack((X_left, Y_left))
                points_total = np.vstack([points_total, pts_left])
                gx_total = np.concatenate([gx_total, gx_left])
                gy_total = np.concatenate([gy_total, gy_left])

            # --- KORREKTUR BEGINNT HIER ---
            # Fit: alle Stützstellen (points_total), alle Ableitungen (gx_total, gy_total)
            r_fit = cdist(points_total, points_total)
            Phi_fit = np.exp(-(epsilon * r_fit) ** 2)
            diff_fit = points_total[:, np.newaxis, :] - points_total[np.newaxis, :, :]
            Phi_dx_fit = -2 * epsilon ** 2 * diff_fit[..., 0] * Phi_fit
            Phi_dy_fit = -2 * epsilon ** 2 * diff_fit[..., 1] * Phi_fit

            A_fit = np.vstack([Phi_dx_fit, Phi_dy_fit])  # (2*N_total, N_total)
            b_fit = np.hstack([gx_total, gy_total])      # (2*N_total,)

            coeffs, *_ = lstsq(A_fit, b_fit, rcond=None)

            # Rekonstruktion: NUR auf eigenen Punkten!
            r_recon = cdist(points, points_total)
            Phi_recon = np.exp(-(epsilon * r_recon) ** 2)
            Z_rbf = Phi_recon @ coeffs

            print(f"Chunk: Zeile: {i} Spalte: {j} wird mit Randkopplung approximiert")
            np.save(f"./Data/ZAprox/Chunk{i}{j}.npy", Z_rbf)
    return s

#%% Plot funktion
def Plotfunktion(s,k):
    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(111, projection='3d')
    for i in range(s):
        for j in range(s):

            X = np.load(f"./Data/X/MessChunk_X{i}{j}.npy")
            Y = np.load(f"./Data/Y/MessChunk_Y{i}{j}.npy")
            Z_rec = np.load(f"./Data/ZAprox/Chunk{i}{j}.npy").reshape(k,k)


            ax1.plot_surface(X, Y, Z_rec, cmap='viridis')
            ax1.set_title('Rekunstruktion')
    plt.show()
#%% Zusammensetzen der Chunks
# def combine_chunks_to_surface(s, k, chunk_folder="./Data/ZAprox"):
#     # s ... Anzahl der Chunks pro Achse
#     # k ... Kantenlänge jedes Chunks
#     # chunk_folder ... Ordner mit Chunk-Dateien
#     Z_total = np.zeros((s*k, s*k))

#     for i in range(s):
#         for j in range(s):
#             # Chunk laden
#             fname = os.path.join(chunk_folder, f"Chunk{i}{j}.npy")
#             chunk = np.load(fname).reshape(k, k)
#             # An die richtige Stelle einfügen
#             Z_total[i*k:(i+1)*k, j*k:(j+1)*k] = chunk
#     return Z_total

def combine_chunks_overlap(s, k, overlap, chunk_folder="./Data/ZAprox"):
    # s ... Anzahl der Chunks pro Achse
    # k ... Kantenlänge jedes Chunks (ohne Überlappung)
    # overlap ... Breite der Überlappung in Pixeln
    total_size = s * (k - overlap) + overlap
    Z_total = np.zeros((total_size, total_size))
    count = np.zeros_like(Z_total)

    for i in range(s):
        for j in range(s):
            fname = os.path.join(chunk_folder, f"Chunk{i}{j}.npy")
            chunk = np.load(fname).reshape(k, k)
            x_start = i * (k - overlap)
            y_start = j * (k - overlap)
            Z_total[x_start:x_start+k, y_start:y_start+k] += chunk
            count[x_start:x_start+k, y_start:y_start+k] += 1
    # Mittelwert im Überlappungsbereich
    Z_total /= count
    return Z_total


#%% Main Function
    
k,s,ztrue,Xtrue,Ytrue = SimWerteforChunk()
RBF_Algorihtmuss_Randkopplung(s, k,epsilon=1)
fig2 = plt.figure(figsize=(18, 5))
ax2 = fig2.add_subplot(111, projection='3d')
# k,s,ztrue,X,Y
ax2.plot_surface(Xtrue, Ytrue, ztrue.reshape(1000,1000), cmap='viridis')
ax2.set_title('Original')

#Z_Rec = combine_chunks_overlap(s, k, overlap=0)


Plotfunktion(s,k)
#%%

