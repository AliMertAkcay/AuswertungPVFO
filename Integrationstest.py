#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 19:12:18 2025

@author: ali-mert-akcay
"""

# Erzeug einiger Oberflächen und dessen Normalvektorfeld

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import Auswertung1 as Aus1
from poisson_integration import poisson_integration
plt.close("all")


# Fest gelegte DatenStruktur: Matrix in der die Werte drin Stehen
#   x---------------------------------------------->
# y (nx,ny,nz) | () »
# | 
# |
# |
# |
# |
#

# ----------------------- Drei Dimensionale Funktionen ----------------

def AffensattelnFeld (Anzahlx = 101, Anzahly = 101):
    
    x = np.linspace(-10,10,Anzahlx)
    y = np.linspace(-10, 10,Anzahly)
    X,Y = np.meshgrid(x,y)
    #X,Y = np.meshgrid(y,x)
    fx = -(3*X**2-3*Y**2)
    fy = -(-6*X*Y)
    einsen = np.ones((Anzahlx,Anzahly))
    matrix = np.empty((Anzahlx,Anzahly),dtype = object)
    matrix = np.stack((fx,fy,einsen),axis=2)

    n = np.empty((Anzahlx,Anzahly),dtype = object)

    for k in range(Anzahlx):
        for s in range(Anzahly):
            n[k,s] = [matrix[k,s,0],matrix[k,s,1],matrix[k,s,2]]
    
    dx = x[1] -x[0]
    dy = y[1] -y[0]
    
    return n,X,Y,dx,dy,x,y

def Paraboloid (Anzahlx = 1001, Anzahly = 1001):
    
    x = np.linspace(-10,10,Anzahlx)
    y = np.linspace(-10, 10,Anzahly)
    X,Y = np.meshgrid(x,y)
    #X,Y = np.meshgrid(y,x)
    fx = -(2*X + 10)
    fy = -2*Y
    einsen = np.ones((Anzahlx,Anzahly))
    matrix = np.empty((Anzahlx,Anzahly),dtype = object)
    matrix = np.stack((fx,fy,einsen),axis=2)

    n = np.empty((Anzahlx,Anzahly),dtype = object)

    for k in range(Anzahlx):
        for s in range(Anzahly):
            n[k,s] = [matrix[k,s,0],matrix[k,s,1],matrix[k,s,2]]
    
    dx = x[1] -x[0]
    dy = y[1] -y[0]
    
    return n,X,Y,dx,dy

#n,X,Y = AffensattelnFeld()

#n,X,Y,dx,dy = Paraboloid()

#-------------------- Test poisson Integration ------------------

#steigungsfeld = Aus1.steigungausn(n)

#Z = Aus1.pfadintegral_cumtrapz(steigungsfeld, X, Y)

#fx,fy = Aus1.steigungextraktion(steigungsfeld)

#f = poisson_integration(fx,fy,dx,dy)



# fig2 = plt.figure("Oberfläche aus der Integration2", figsize=(12, 9)) # Figure mit Title
# ax2 = fig2.add_subplot(111, projection='3d')

# ax2.plot_surface(X,Y,Z,color = "Blue",label="Integration der Ableitung2") 

# #ax2.plot_surface(X,Y,X**3-3*X*Y**2,label="Ursprüngliche Funktion")
# ax2.plot_surface(X,Y,X**2+Y**2+10*X,color = "Black",label="Ursprüngliche Funktion")
# ax2.set_xlabel("X")
# ax2.set_ylabel("Y")
# ax2.set_zlabel("Z")
# fig2.label()



#--------------------- Test pfad Integral --------------------------------

n,X,Y,dx,dy,x,y = AffensattelnFeld()

steigungsfeld = Aus1.steigungausn(n)
mx,my = Aus1.steigungextraktion(steigungsfeld)
Z = Aus1.pfadintegral_cumtrapz_mittel(mx, my, x, y)

fig2 = plt.figure("Oberfläche aus der Integration2", figsize=(12, 9)) # Figure mit Title
ax2 = fig2.add_subplot(111, projection='3d')

ax2.plot_surface(X,Y,Z,color = "Blue",label="Integration der Ableitung2") 

ax2.plot_surface(X,Y,X**3-3*X*Y**2,label="Ursprüngliche Funktion")
#ax2.plot_surface(X,Y,X**2+Y**2+10*X,color = "Black",label="Ursprüngliche Funktion")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
fig2.label()

