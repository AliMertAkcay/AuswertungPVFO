#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 20:47:32 2025

@author: ali-mert-akcay
"""

import numpy as np

def steigungausn(n):
     
    # Struktur:
    zeile = len(n[:,0])
    spalte = len(n[0,:])     
    steigungsfeld= np.empty((zeile,spalte),dtype = object) #     


    for j in range(zeile): # 
        for i in range(spalte): # anzahl_phi 
            puffer = n[j,i]
            mx = -puffer[0]/puffer[2]
            my = -puffer[1]/puffer[2] 
            steigungsfeld[j,i] = np.array([mx,my])
            
    return steigungsfeld


def normalvektor(Vein,Vout,standartVein = True):
    # Eingabe des Ausgangs Vektorfelds
    # Ausgabe des Normalvektors
    #TODO: Idee für später Matrix um Formen in eine Lange Kette
    # Sollte der Eintritsvekotr bekannt sein kann es von Aussen eingeben werden
    import numpy as np    
    zeile = len(Vout[:,0])
    spalte = len(Vout[0,:])
    
    
    n = np.empty((zeile,spalte),dtype= object)
    if standartVein:
        Vein = np.empty((zeile,spalte),dtype= object)
        Vein[:,:] = [0,0,-1]
    
    for i in range(zeile):
        for j in range(spalte):
            Veinnom = np.linalg.norm(Vein[i,j]) 
            Voutnom = np.linalg.norm(Vout[i,j])
           
            Veinpuffer = Vein[i,j]/Veinnom 
            Voutpuffer = Vout[i,j]/Voutnom
            
            #Vein = Vein/Veinnom
            #Vout = Vout/Voutnom
            
            Vdot = np.dot(Veinpuffer,Voutpuffer)
            npuffer = (Voutpuffer-Veinpuffer)/(np.sqrt(2*(1-Vdot))) # Formel aus der Doktorarbeit korekkt umgesetzt
            n[i,j] = npuffer
            
    return n


def Integration(steigungsfeld,X,Y):
    import numpy as np
    from scipy.integrate import cumulative_trapezoid
    # README: Angepasste Intgration Mit der neuen DatenStruktur
    # TODO: Diese Integration funktioniert nicht für Jede Funktion
    m = steigungsfeld
    
    fintnachy = np.zeros(m.shape)
    fintnachx = np.zeros(m.shape)


    zeile = len(steigungsfeld[:,0])
    spalte = len(steigungsfeld[0,:])
    #matrix = np.empty((),dtype=object)
    
    mx = np.zeros(m.shape)
    my = np.zeros(m.shape)

    
    for j in range(m.shape[0]):
        for i in range(m.shape[1]):
            mx[j, i] = m[j, i][0]
            my[j, i] = m[j, i][1]    

    # Integration jeweilig nach Y
    for i in range(spalte):     
        fintnachy[i,:] = cumulative_trapezoid(my[:,i],Y[:,i],initial=0)
    # Integration Jeweilig nach X
    for j in range(zeile):
        fintnachx[:,j] = cumulative_trapezoid(mx[j,:],X[j,:],initial=0)

    z_combined = fintnachx+fintnachy

    return z_combined




def extraktionkordinaten():
    # Funktion zur extraktion der Kordinaten aus der RAW datei falls Nötig
    
    X = 1
    Y = 1
    return X,Y

import numpy as np
from scipy.integrate import cumulative_trapezoid

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

