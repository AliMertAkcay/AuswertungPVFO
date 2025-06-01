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

def AffensattelnFeld (Anzahlx = 2201, Anzahly = 2201):
    
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
    
    return n,X,Y

def Paraboloid (Anzahlx = 2001, Anzahly = 2001):
    
    x = np.linspace(-10,10,Anzahlx)
    y = np.linspace(-10, 10,Anzahly)
    X,Y = np.meshgrid(x,y)
    #X,Y = np.meshgrid(y,x)
    fx = -2*X
    fy = -2*Y
    einsen = np.ones((Anzahlx,Anzahly))
    matrix = np.empty((Anzahlx,Anzahly),dtype = object)
    matrix = np.stack((fx,fy,einsen),axis=2)

    n = np.empty((Anzahlx,Anzahly),dtype = object)

    for k in range(Anzahlx):
        for s in range(Anzahly):
            n[k,s] = [matrix[k,s,0],matrix[k,s,1],matrix[k,s,2]]
    
    return n,X,Y

n,X,Y = AffensattelnFeld()

n,X,Y = Paraboloid()


#--------------------- Test Auswertung --------------------------------
# Auswertungs Teil zum test ob diese Funktionen mit den Oberen Simulativen werten passen

steigungsfeld = Aus1.steigungausn(n)

Z = Aus1.pfadintegral_cumtrapz(steigungsfeld, X, Y)
# Alternative Integration: Aus1.Integration(steigungsfeld, X, Y)
fig = plt.figure("Oberfläche aus der Integration", figsize=(12, 9)) # Figure mit Title

ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X,Y,Z,label="Integration der Ableitung") 

#ax.plot_surface(X,Y,X**3-3*X*Y**2,label="Ursprüngliche Funktion") 
ax.plot_surface(X,Y,X**2+Y**2,label="Ursprüngliche Funktion")
plt.label()

   