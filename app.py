from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import joblib
import tempfile
import os
import mediapipe as mp

app = FastAPI()

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load models ---
loaded_model = joblib.load("svm_facial_shape_model.pkl")
loaded_scaler = joblib.load("scaler_facial_shape.pkl")

# --- Mediapipe face mesh ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# --- Skin tone helper ---
def detect_skin_tone(image, landmarks):
    if landmarks is None:
        return "unknown"
    h, w, _ = image.shape
    # Use nose tip (landmark 1) as sample
    x = int(landmarks[1][0] * w)
    y = int(landmarks[1][1] * h)
    color = image[y, x]
    v = np.mean(color)  # approximate brightness
    if v > 170:
        return "light"
    elif v > 100:
        return "neutral"
    else:
        return "dark"

# --- Process image for prediction ---
def process_image_for_prediction(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    predicted_shape = "no_face"
    skin_tone = "unknown"

    if results.multi_face_landmarks:
        landmarks = []
        for lm in results.multi_face_landmarks[0].landmark:
            landmarks.append([lm.x, lm.y])
        landmarks = np.array(landmarks)

        # Normalize by nose tip (landmark 1 in Mediapipe)
        nose_tip = landmarks[1]
        normalized_coords = landmarks - nose_tip

        # Use distance between eyes for scaling (landmarks 33 and 263)
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        eye_distance = np.linalg.norm(left_eye - right_eye)

        if eye_distance != 0:
            normalized_coords = normalized_coords / eye_distance
            flattened = normalized_coords.flatten().reshape(1, -1)
            processed = loaded_scaler.transform(flattened)
            predicted_shape = loaded_model.predict(processed)[0]

        skin_tone = detect_skin_tone(image, landmarks)

    return predicted_shape, skin_tone

# --- Routes ---
@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "Server is running!"}

@app.post("/api/vision/face-shape")
async def detect_face_shape(image: UploadFile = File(...)):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(await image.read())
        temp_file.close()

        frame = cv2.imread(temp_file.name)
        os.unlink(temp_file.name)

        if frame is None:
            return JSONResponse({"error": "Invalid image"}, status_code=400)

        shape, tone = process_image_for_prediction(frame)
        return {"face_shape": shape, "skin_tone": tone}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# --- Run locally ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8800, reload=True)
