# Info_Notizen:
um die Fläche zu Rekunstruieren kann man die RBF Funktionen verwenden diese Aproximations Art ist geigenet da der Fehler sehr giring ist.
Das Problem ist noch das für größere Matrizen (Messwerte) diese Matrix für die Berechung erstmal in kleine Blöcke aufgeteilt werden müssen bevor man die Matrix Berechenn kann.
So ist man in der lage mit weniger Ram Nutzung die Fläche zu rekunstruiere.

# Aktuelles Problem:
das problem ist noch aus den einzelen Teilflächen welche aus der Aproximation der Blöcke entstanden ist in eine Große Oberfläche zurück zu führen.
Sollte dies Problem jedoch behoben sein dann stellt das Programm: RBFReconstruction_21_33.py ein Soliden Algorihtmuss für die Oberflächen Reconstruction.

# Nächste Schritte:
Die nächsten Schritte für das Auswerte Skript sollte sein die Vorverarbeitung zu überarbeiten das diese Berechnung nur Vektoriell in Pytohn durchgeführt wird. Und so der Reconstructionsalgorihtmuss die steigungs werte aus den gemessene Austritsvektoren hat.

# Randbedinungen:
Es müsste eine Möglichkeit geben aus der die Oberfläche Rekunstruiert wird mit den Randbedingungen siehe RBFMitrandbedinungen.py

# Fazit: 
Sonst für messwerte in der form von 50x50 werden wir aufjeden fall eine Auswertung durch führen können Bei größeren Matrizen muss man eben mit Chunks arbeiten.
