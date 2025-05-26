#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 25 13:34:17 2025

@author: ali_mert_akcay
"""
import numpy as np
# TODO: Sollte so Richtig sein !
def steigungausn(n):
    # Description Berechung der Steigung aus einem Normalvektorfeld
    # Aufbau des Vektorfelds
    # [origin_x,origin_y,origin_z,nx,ny,nz] --anzahlphi->
    # |
    # anzahl_t
    # |
    # |
    # 
    # Struktur:
    steigungsfeld= np.zeros((len(n[:,0,0]),5,len(n[0,0,:])))    
    steigungsfeld[:,0:2,:] = n[:,0:2,:]


    for j in range(0,len(n[0,0,:])): # 
        for i in range(0,len(n[:,0,0])): # anzahl_phi 
            mx = -n[i,3,j]/n[i,5,j]
            my = -n[i,4,j]/n[i,5,j] 
            steigungsfeld[i,3,j] = mx
            steigungsfeld[i,4,j] = my
            
    return steigungsfeld
