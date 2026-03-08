from PIL import Image
import numpy as np
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
def extract_face(filename, required_size=(160,160)):

    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)

    detector = MTCNN()
    results = detector.detect_faces(pixels)

    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    face = pixels[y1:y2, x1:x2]

    image = Image.fromarray(face)
    image = image.resize(required_size)

    face_array = np.asarray(image)

    return face_array