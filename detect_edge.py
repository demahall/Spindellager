import cv2
import numpy as np

# Lade das Bild
image = cv2.imread('/mnt/data/7023415d-5698-49fc-b00e-050b725cddac.png')

# Konvertiere das Bild in Graustufen
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Wende GaussianBlur an, um Rauschen zu reduzieren
blurred = cv2.GaussianBlur(gray, (15, 15), 0)

# Verwende die Canny-Kantenerkennung
edges = cv2.Canny(blurred, 50, 150)

# Erkenne Kreise im Bild mit der Hough-Transformation
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=50, param2=30, minRadius=50, maxRadius=200)

# Wenn Kreise erkannt wurden, zeichne sie auf das Originalbild
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        # Zeichne den Kreis
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)
        # Zeichne den Mittelpunkt des Kreises
        cv2.circle(image, (x, y), 3, (0, 0, 255), 3)

# Zeige das Bild an
cv2.imshow("Detected Circles", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
