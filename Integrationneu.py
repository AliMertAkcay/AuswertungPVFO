#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 25 19:32:21 2025

@author: ali-mert-akcay
"""

def Integration(steigungsfeld,X,Y):
    import numpy as np
    from scipy.integrate import cumulative_trapezoid
    # README: FÜR DAS EINLADEN DER MATRIX MUSS EINE GEWISSE STRUKTUR BEACHTET WERDEN

    # Berechnung der Integration
    # Extrahiere die Steigungen
    dz_dx = steigungsfeld[:, 3, :]  # Steigung in x-Richtung (Shape: [x, y])
    dz_dy = steigungsfeld[:, 4, :]  # Steigung in y-Richtung (Shape: [x, y])

    fintnachy = np.zeros(dz_dx.shape)
    fintnachx = np.zeros(dz_dy.shape)

    print("Shape von dz_dx",dz_dx.shape)
    print("Shape von dz_dy", dz_dy.shape)
    # Koordinaten


    # Integration jeweilig nach Y
    for i in range(0,len(Y[:,0])):     
        fintnachy[i,:] = cumulative_trapezoid(dz_dy[i,:],Y[i,:],initial=0)
    # Integration Jeweilig nach X
    for j in range(0,len(X[0,:])):
        fintnachx[:,j] = cumulative_trapezoid(dz_dx[:,j],X[:,j],initial=0)

    z_combined = fintnachx+fintnachy

    return z_combined