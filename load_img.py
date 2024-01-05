import time, cv2

img_path = 'assets/img/Test1.jpg'
img = cv2.imread(img_path)

if img is not None:
    cv2.imshow('Image', img)
    cv2.waitKey(0)  # Attendre indéfiniment jusqu'à ce qu'une touche soit enfoncée
    cv2.destroyAllWindows()  # Fermer toutes les fenêtres après avoir appuyé sur une touche
else:
    print("Impossible de charger l'image. Vérifiez le chemin d'accès.")