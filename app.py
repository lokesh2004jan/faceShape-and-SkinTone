from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import dlib
import numpy as np
import joblib
import tempfile
import os

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
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
loaded_model = joblib.load("svm_facial_shape_model.pkl")
loaded_scaler = joblib.load("scaler_facial_shape.pkl")

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

# --- Skin tone helper ---
def detect_skin_tone(image, face_rects):
    if len(face_rects) == 0:
        return "unknown"
    x, y, w, h = face_rects[0].left(), face_rects[0].top(), face_rects[0].width(), face_rects[0].height()
    face_roi = image[max(0, y):y + h, max(0, x):x + w]
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    v = np.mean(hsv[:, :, 2])
    if v > 170:
        return "light"
    elif v > 100:
        return "neutral"
    else:
        return "dark"

# --- Process for prediction ---
def process_image_for_prediction(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)
    predicted_shape = "no_face"
    skin_tone = "unknown"

    if len(faces) > 0:
        face = faces[0]
        landmarks = landmark_predictor(gray, face)
        coords = np.array([[p.x, p.y] for p in landmarks.parts()])
        nose_tip = coords[30]
        normalized_coords = coords - nose_tip

        left_eye_center = np.mean(coords[36:42], axis=0)
        right_eye_center = np.mean(coords[42:48], axis=0)
        eye_distance = np.linalg.norm(left_eye_center - right_eye_center)

        if eye_distance != 0:
            normalized_coords = normalized_coords / eye_distance
            normalized_landmarks_flat = normalized_coords.flatten()
            processed_landmarks = loaded_scaler.transform(normalized_landmarks_flat.reshape(1, -1))
            predicted_shape = loaded_model.predict(processed_landmarks)[0]

        skin_tone = detect_skin_tone(image, faces)

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

        return {
            "face_shape": shape,
            "skin_tone": tone
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# --- Run locally ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
