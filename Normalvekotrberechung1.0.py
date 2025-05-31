#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 17:09:58 2025

@author: ali-mert-akcay
"""



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