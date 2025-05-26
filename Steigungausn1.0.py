#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 16:27:29 2025

@author: ali_mert_akcay
"""
import numpy as np

def steigungausn(n):
     
    # Struktur:
    zeile = len(n[:,0])
    spalte = len(n[0,:])     
    steigungsfeld= np.empty((zeile,spalte),dtype = object) #     


    for j in range(zeile): # 
        for i in range(0,spalte): # anzahl_phi 
            puffer = steigungsfeld[j,i]
            mx = -puffer[0]/puffer[2]
            my = -puffer[1]/puffer[2] 
            steigungsfeld[zeile,spalte] = np.array([mx,my])
            
    return steigungsfeld