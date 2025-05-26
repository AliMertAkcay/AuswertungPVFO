# Erstellt am 24.05.2025
# Ali Mert Akcay
def zylinder (r=800,h=800,my=0,anzhalphi = 1001,anzahlt = 50,axis = [0,1,0],deg=-10,step= 10,plottrue= True,Verschiebung_X=0,Verschiebung_Y= 0, Verschiebung_Z = 1):
    #TODO: Fix den Code so das die darstellung in ordnung ist da bei einem zu kleinen h die f
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)  # Normalisierung
    theta = np.radians(deg)  # Drehwinkel

    # Rodrigues-Formel zur Erzeugung der Rotationsmatrix (KORRIGIERT)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K @ K)

    # Ursprüngliche Parametervektoren des Zylinders
    f1 = np.array([0, 0, r])  # Radiale Richtung (cos-Anteil) # Das funktioniert noch nicht Richtig der Vekor hat eine Falschen Betrag
    f2 = np.array([0, r, 0])  # Radiale Richtung (sin-Anteil)
    f3 = np.array([h, 0, 0])  # Höhenrichtung
    q0 = np.array([0, 0, r])  # Verschiebungsvektor
    #translation_z = np.array([0,0,r])

    # Drehung der Parametervektoren
    f1_rot = R @ f1
    f2_rot = R @ f2
    f3_rot = R @ f3
    q0_rot = R @ q0 #+ translation_z

    # Parameterflächen
    phi1 = np.linspace(np.pi/2+np.pi-0.01,-np.pi/2+np.pi-0.01,anzhalphi) # -0.01 damit die Steigung nicht so groß wird
    #phi1 = np.linspace(0,2*np.pi,anzhalphi)
    t = np.linspace(0, 1, anzahlt)
    phi, t = np.meshgrid(phi1, t)

    # Berechnung der Zylinderpunkte (KORRIGIERT)
    X = q0_rot[0] + f1_rot[0]*np.cos(phi) + f2_rot[0]*np.sin(phi) + f3_rot[0]*t #+ Verschiebung_X
    Y = q0_rot[1] + f1_rot[1]*np.cos(phi) + f2_rot[1]*np.sin(phi) + f3_rot[1]*t #+ Verschiebung_Y
    Z = q0_rot[2] + f1_rot[2]*np.cos(phi) + f2_rot[2]*np.sin(phi) + f3_rot[2]*t #+ Verschiebung_Z

    # KORREKTE Berechnung der Normalenvektoren über Tangentenvektoren
    # Tangentialvektor in phi-Richtung
    T_phi = np.zeros((3, phi.shape[0], phi.shape[1]))
    T_phi[0] = -f1_rot[0]*np.sin(phi) + f2_rot[0]*np.cos(phi)
    T_phi[1] = -f1_rot[1]*np.sin(phi) + f2_rot[1]*np.cos(phi)
    T_phi[2] = -f1_rot[2]*np.sin(phi) + f2_rot[2]*np.cos(phi)

    # Tangentialvektor in t-Richtung (konstant)
    T_t = np.zeros((3, phi.shape[0], phi.shape[1]))
    T_t[0] = f3_rot[0] 
    T_t[1] = f3_rot[1]
    T_t[2] = f3_rot[2]

    # Normalenvektor als Kreuzprodukt T_phi × T_t
    normals_rot = np.zeros((3, phi.shape[0], phi.shape[1]))
    normals_rot[0] = T_phi[1]*T_t[2] - T_phi[2]*T_t[1]
    normals_rot[1] = T_phi[2]*T_t[0] - T_phi[0]*T_t[2]
    normals_rot[2] = T_phi[0]*T_t[1] - T_phi[1]*T_t[0]

    # Normierung der Normalenvektoren
    norm = np.sqrt(normals_rot[0]**2 + normals_rot[1]**2 + normals_rot[2]**2)
    normals_rot[0] /= norm
    normals_rot[1] /= norm
    normals_rot[2] /= norm
    
    # Diese Normalvektorfeld ist andeers Aufgebaut
    normal_vektorfeld  = np.zeros((anzahlt,6,anzhalphi))
    normal_vektorfeld[:,0,:] = X[:,:]
    normal_vektorfeld[:,1,:] = Y[:,:]
    normal_vektorfeld[:,2,:] = Z[:,:]
    
    for i in range (0,len(normal_vektorfeld[0,0,:])):
        for j in range(0,len(normal_vektorfeld[:,0,0])):
            normal_vektorfeld[j,3,i] = normals_rot[0,j,i] # Alle X-Komponenten
            normal_vektorfeld[j,4,i] = normals_rot[1,j,i] # Alle   
            normal_vektorfeld[j,5,i] = normals_rot[2,j,i]
    
    if(plottrue == True):
        # 3D-Plot
        fig = plt.figure("Zylinder", figsize=(12, 9)) # Figure mit Title
        #fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Normalenvektoren (reduziert für Übersichtlichkeit)
        #step = 10 # Auswählen der Anzahl an vektoren
        ax.quiver(X[::step, ::step], Y[::step, ::step], Z[::step, ::step],
                  normals_rot[0, ::step, ::step], normals_rot[1, ::step, ::step], normals_rot[2, ::step, ::step],
                  length=r*0.1, color='r', normalize=True, label='Normalvektoren')
        
        ax.plot_surface(X,Y,Z) # Plot der Zylinder Oberfläche
        # Basisvektoren
        ax.quiver(q0_rot[0], q0_rot[1], q0_rot[2], f1_rot[0], f1_rot[1], f1_rot[2], 
                  color='red', label="f1 Vektor")
        ax.quiver(q0_rot[0], q0_rot[1], q0_rot[2], f2_rot[0], f2_rot[1], f2_rot[2], 
                  color='green', label="f2 Vektor")
        ax.quiver(q0_rot[0], q0_rot[1], q0_rot[2], f3_rot[0], f3_rot[1], f3_rot[2], 
                  color='blue', label="f3 Vektor")

        ax.quiver(0,0,0,axis[0],axis[1],axis[2],length=r,color="Black",label ="axis")

        # Legende mit Proxy-Artist für die Oberfläche
        proxy = [Patch(facecolor='c', edgecolor='c', label='Zylinder Oberfläche')]
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(proxy + handles, ['Zylinder Oberfläche'] + labels)

        # Achsenbeschriftung
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #TODO: Fixen von der Darstellung des Zylinders
        #ax.set_zlim(0,r)
        #ax.set_ylim(-r,r)
        plt.show()
        plt.savefig("Zylinder_Parametrisiert_Gedreht_Korrekt.png", dpi=300, bbox_inches='tight')
      
        ax.quiver(normal_vektorfeld[::step,0,::step],normal_vektorfeld[::step,1,::step],normal_vektorfeld[::step,2,::step],normal_vektorfeld[::step,3,::step],normal_vektorfeld[::step,4,::step],normal_vektorfeld[::step,5,::step],length=r*0.1, color='b', normalize=True,label="Vektorfeld")
        ax.legend()
        plt.show()
        surface = np.array([X,Y,Z])
    return normal_vektorfeld,surface
 


