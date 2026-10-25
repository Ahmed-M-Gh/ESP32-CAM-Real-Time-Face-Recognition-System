import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from insightface.app import FaceAnalysis
import os
import glob

app = FastAPI()

# Detection model
app_face = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider']) #If you don't have GPU use 'CPUExecutionProvider'
app_face.prepare(ctx_id=0, det_size=(640, 640))

KNOWN_FACES = {}

def load_known_faces():
    global KNOWN_FACES
    dataset_path = "dataset"
    
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print(f"تم إنشاء فولدر الـ {dataset_path}, يرجى وضع صور الأشخاص فيه.")
        return

    image_paths = glob.glob(os.path.join(dataset_path, "*.*"))
    
    for img_path in image_paths:
        # using person name from filename
        filename = os.path.basename(img_path)
        name, _ = os.path.splitext(filename)
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        faces = app_face.get(img)
        if len(faces) > 0:
            #storing embeddings for person
            KNOWN_FACES[name] = faces[0].embedding
            print(f"person face loaded: {name}")
        else:
            print(f"Warning face not found in: {filename}")

    print(f"The total number of people currently known: {len(KNOWN_FACES)}")

@app.on_event("startup")
async def startup_event():
    load_known_faces()

@app.websocket("/ws/face-recognition")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("The ESP32 Connected and ready to receive images...")
    
    try:
        while True:
            # Receive images in binary
            image_bytes = await websocket.receive_bytes()
            
            # Convert thte  Bytes to image can be understood by OpenCV
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                await websocket.send_json({"name": "Unknown", "error": "Invalid Image"})
                continue

            # Discover face and load stored Embeddings for received images
            faces = app_face.get(frame)
            
            detected_name = "Unknown"
            
            if len(faces) > 0:
                current_embedding = faces[0].embedding
                
                # Using Cosine Similarity to get similarity between images embeddings
                highest_similarity = 0.0
                threshold = 0.4  
                
                for name, ref_embedding in KNOWN_FACES.items():
                    if ref_embedding is None:
                        continue
                    
                    # Calculate the Cosine Similarity between current face and stored faces
                    similarity = np.dot(current_embedding, ref_embedding) / (
                        np.linalg.norm(current_embedding) * np.linalg.norm(ref_embedding)
                    )
                    
                    if similarity > highest_similarity and similarity > threshold:
                        highest_similarity = similarity
                        detected_name = name

                print(log_msg := f"Result: {detected_name} (Similarity Score: {highest_similarity:.2f})")
            
            else:
                print("No Face in the image.")

            # Response on ESP32 By recognized name
            await websocket.send_json({"name": detected_name})
            
    except WebSocketDisconnect:
        print("The ESP32 Cut the connection.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)