import cv2
import insightface
from insightface.app import FaceAnalysis
import os
import glob
import numpy as np

# Preparing model
app_face = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
app_face.prepare(ctx_id=0, det_size=(640, 640))

# Loading faces from the dataset
KNOWN_FACES = {}
for img_path in glob.glob("dataset/*.*"):
    name = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    if img is not None:
        faces = app_face.get(img)
        if len(faces) > 0:
            KNOWN_FACES[name] = faces[0].embedding
            print(f"loaded: {name}")

# opening camera
cap = cv2.VideoCapture(0)

print("opening cam.... to exit press 'q'.")

while cap.isOpened():
    ret, frame = cap.isOpened() and cap.read()
    if not ret:
        break

    # Discover faces in the  current frame
    faces = app_face.get(frame)
    
    for face in faces:
        # drowing box on face
        bbox = face.bbox.astype(int)
        current_embedding = face.embedding
        
        detected_name = "Unknown"
        highest_similarity = 0.0
        threshold = 0.45  # similarity score
        
        # compare face with stored images
        for name, ref_embedding in KNOWN_FACES.items():
            similarity = np.dot(current_embedding, ref_embedding) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(ref_embedding)
            )
            if similarity > highest_similarity and similarity > threshold:
                highest_similarity = similarity
                detected_name = name

        # drowing box and person name on screen
        color = (0, 255, 0) if detected_name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        text = f"{detected_name} ({highest_similarity:.2f})"
        cv2.putText(frame, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Local Face Recognition Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()