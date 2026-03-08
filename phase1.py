import cv2
import os
import numpy as np
from PIL import Image

# 1. Fonction pour extraire un visage d'une image
def extract_face(image_path, required_size=(160, 160)):
    """
    Charge l'image, détecte le visage avec HAAR, le redimensionne et le retourne.
    """
    # Charger l'image avec OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"Impossible de lire l'image: {image_path}")
        return None
    
    # Convertir en niveaux de gris (HAAR fonctionne mieux)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Charger le classificateur HAAR pour la détection de visages
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Détection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    
    if len(faces) == 0:
        print(f"Aucun visage détecté dans {image_path}")
        return None
    
    # On prend le premier visage détecté (le plus grand souvent)
    x, y, w, h = faces[0]
    
    # Extraire la région du visage
    face = img[y:y+h, x:x+w]
    
    # Redimensionner à la taille souhaitée
    face = cv2.resize(face, required_size)
    
    # Convertir en RGB (car OpenCV utilise BGR)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    
    return face

# 2. Fonction pour charger tous les visages d'un répertoire (une personne)
def load_faces(directory):
    """
    Parcourt toutes les images d'un dossier et applique extract_face.
    Retourne une liste de visages (tableaux numpy).
    """
    faces = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        # Vérifier que c'est bien une image (optionnel)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            face = extract_face(path)
            if face is not None:
                faces.append(face)
    return faces

# 3. Fonction pour charger tout le dataset (train ou test)
def load_dataset(parent_dir):
    """
    Parcourt les sous-dossiers du répertoire parent.
    Chaque sous-dossier correspond à une personne (label).
    Retourne :
        X : liste de visages
        y : liste d'étiquettes (sous forme de chaînes ou d'entiers)
    """
    X = []
    y = []
    
    # Chaque sous-dossier = une classe
    for label in os.listdir(parent_dir):
        person_dir = os.path.join(parent_dir, label)
        if os.path.isdir(person_dir):
            faces = load_faces(person_dir)
            X.extend(faces)
            y.extend([label] * len(faces))  # étiquette = nom du dossier
            print(f"Chargé {len(faces)} visages pour {label}")
    
    return X, y