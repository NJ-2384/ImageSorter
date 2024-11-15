import os
import face_recognition
import pickle
from PIL import Image as im
from numpy import asarray

name = "Kottu"

print("[INFO] start processing faces...")

imagePath = (os.listdir(f"dataset/{name}"))
imagePaths = []

for j in imagePath:
    imagePaths.append(os.path.join(os.getcwd(),f"dataset/{name}",j))

knownEncodings = []
knownNames = []

for (i, imagepath) in enumerate(imagePaths):
    print(f"[INFO] processing image {i + 1}/{len(imagePaths)}")
    
    image = asarray(im.open(imagepath))
    
    boxes = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, boxes, model='large')
    
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

print("[INFO] serializing encodings...")

with open(f"{name}.p", "wb") as f:
    f.write(pickle.dumps(knownEncodings))

print(f"[INFO] Training complete. Encodings saved to '{name}.p'")