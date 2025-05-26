#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 20:29:19 2025

@author: ali-mert-akcay
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from Zylinder3D import zylinder
from Steigungausnvektor import steigungausn
from Integrationneu import Integration
#import Auswertung as aus

plt.close("all")

# 3D zylinder
normal_vektorfeld, surface = zylinder()


# Steigungswete
steigungsfeld = steigungausn(normal_vektorfeld)

fig2 = plt.figure("Test Plot")
ax = fig2.add_subplot(111, projection='3d')

ax.plot_surface(surface[0],surface[1],surface[2])

# Integration


X = surface[0]
Y = surface[1]
#z_combined= Integration(steigungsfeld,X=X,Y=Y)

#fig3 = plt.figure("Plot aus der Integration")
#ax3 = fig3.add_subplot(111,projection='3d')
#ax3.plot_surface(X,Y,z_combined)


z_combined = Integration(steigungsfeld, X, Y)

fig3 = plt.figure("Plot aus der Integration")
ax3 = fig3.add_subplot(111,projection='3d')
ax3.plot_surface(X,Y,z_combined)



# 
