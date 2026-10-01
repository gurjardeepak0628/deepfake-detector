import os
os.environ["HF_TOKEN"] = "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"

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from database import get_connection
import jwt, time, shutil, uuid
import numpy as np
import cv2
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image, ImageChops, ImageEnhance
import io

router = APIRouter()
security = HTTPBearer()
SECRET = os.getenv("JWT_SECRET", "deepguard_secret_key_2024")
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------- Load Model (single model to save memory) --------
print("⏳ Loading model (dima806)...")
MODEL_1_NAME = "dima806/deepfake_vs_real_image_detection"
model1 = ViTForImageClassification.from_pretrained(MODEL_1_NAME, low_cpu_mem_usage=True)
processor1 = ViTImageProcessor.from_pretrained(MODEL_1_NAME)
model1.eval()
print("✅ Model loaded!")

# -------- Face Detector (Haar Cascade, built into OpenCV) --------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

MAX_FACES_TO_CHECK = 5  # avoid checking too many faces in huge group photos (speed)

def detect_all_faces(pil_img: Image.Image, margin: float = 0.35):
    """
    Detects ALL faces in a PIL image and returns a list of cropped PIL images
    (largest faces first, capped at MAX_FACES_TO_CHECK). Returns [] if none found.
    """
    img_array = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return []

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:MAX_FACES_TO_CHECK]

    crops = []
    for (x, y, w, h) in faces:
        mx, my = int(w * margin), int(h * margin)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(img_array.shape[1], x + w + mx)
        y2 = min(img_array.shape[0], y + h + my)
        face_crop = img_array[y1:y2, x1:x2]
        crops.append(Image.fromarray(face_crop))
    return crops

# -------- ELA Analysis --------
def ela_analysis(image_path: str, quality: int = 90) -> float:
    try:
        original = Image.open(image_path).convert('RGB')
        buffer = io.BytesIO()
        original.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert('RGB')
        diff = ImageChops.difference(original, compressed)
        enhancer = ImageEnhance.Brightness(diff)
        diff_enhanced = enhancer.enhance(20)
        diff_array = np.array(diff_enhanced).astype(float)
        ela_score = np.mean(diff_array) / 255.0
        return ela_score
    except Exception:
        return 0.0

# -------- Single-model helper --------
def _run_model1(img: Image.Image) -> float:
    inputs = processor1(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model1(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()
    id2label = model1.config.id2label
    label_probs = {id2label[i]: probs[i] for i in range(len(probs))}
    fake_prob = 0
    for label, p in label_probs.items():
        if "fake" in label.lower() or "deepfake" in label.lower():
            fake_prob = p
    return fake_prob

def ensemble_fake_prob(img: Image.Image) -> float:
    fake_prob = _run_model1(img)
    print(f"🔍 Model={fake_prob:.3f}")
    return fake_prob

def vit_detection(image_path: str) -> dict:
    """
    Checks ALL faces in the image. If ANY face looks manipulated,
    the whole image is flagged suspicious (uses the highest fake_prob found).
    """
    img = Image.open(image_path).convert('RGB')
    faces = detect_all_faces(img)

    if not faces:
        fake_prob = ensemble_fake_prob(img)
        return {"fake_prob": fake_prob, "real_prob": 1 - fake_prob, "faces_checked": 0}

    face_probs = []
    for i, face in enumerate(faces):
        print(f"  → Checking face {i+1}/{len(faces)}")
        face_probs.append(ensemble_fake_prob(face))

    worst_fake_prob = max(face_probs)
    return {"fake_prob": worst_fake_prob, "real_prob": 1 - worst_fake_prob, "faces_checked": len(faces)}

# -------- Combined Analysis --------
def analyze_image(image_path: str) -> dict:
    start = time.time()
    try:
        vit_result = vit_detection(image_path)
        vit_fake_prob = vit_result["fake_prob"]
        faces_checked = vit_result.get("faces_checked", 0)

        ela_score = ela_analysis(image_path)
        ela_manipulation = ela_score > 0.15

        details = []
        details.append(
            f"A total of {faces_checked} face(s) were detected and analyzed in this image."
            if faces_checked > 0 else
            "No face was detected, the full image was analyzed instead."
        )
        details.append(f"AI deepfake model manipulation score: {round(vit_fake_prob*100,1)}% (higher means more suspicious).")
        details.append(f"Compression/editing analysis (ELA) score: {round(ela_score*100,1)}% (indicates possible photo editing or splicing).")

        if vit_fake_prob > 0.45:
            prediction = "FAKE"
            confidence = round(vit_fake_prob * 100, 1)
            details.append("⚠️ Strong signs of AI face-swap or deepfake manipulation were found in the face — such as texture, edge, or lighting mismatches.")
        elif ela_manipulation and ela_score > 0.25:
            prediction = "FAKE"
            confidence = round(min(ela_score * 200, 99.0), 1)
            details.append("⚠️ A part of the image has a compression pattern that doesn't match the rest — suggesting that section may have been edited or pasted in separately.")
        elif ela_manipulation and vit_fake_prob > 0.3:
            prediction = "FAKE"
            combined = (vit_fake_prob + ela_score) / 2
            confidence = round(combined * 100, 1)
            details.append("⚠️ Mild manipulation signals were found in both the face and the overall image, and combined these indicate a FAKE.")
        else:
            prediction = "REAL"
            confidence = round(vit_result["real_prob"] * 100, 1)
            details.append("✅ No signs of face-swapping and no editing fingerprints in compression were found — the image appears genuine.")

        reason = " ".join(details)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis error: {str(e)}")

    detection_time = round(time.time() - start, 2)
    return {"prediction": prediction, "confidence": confidence, "detection_time": detection_time, "reason": reason}

# -------- Video Analysis (face-aware, multi-frame voting) --------
def analyze_video(video_path: str) -> dict:
    start = time.time()
    try:
        cap = cv2.VideoCapture(video_path)
        frame_results = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_count = 16
        interval = max(1, total // sample_count)
        count = 0
        checked = 0
        while cap.isOpened() and len(frame_results) < 10 and checked < sample_count:
            ret, frame = cap.read()
            if not ret:
                break
            if count % interval == 0:
                checked += 1
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                faces = detect_all_faces(pil_img)
                if not faces:
                    count += 1
                    continue

                frame_face_probs = [ensemble_fake_prob(f) for f in faces]
                frame_results.append(max(frame_face_probs))
            count += 1
        cap.release()

        if not frame_results:
            cap = cv2.VideoCapture(video_path)
            count = 0
            while cap.isOpened() and len(frame_results) < 8:
                ret, frame = cap.read()
                if not ret:
                    break
                if count % max(1, total // 8) == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    fake_prob = ensemble_fake_prob(pil_img)
                    frame_results.append(fake_prob)
                count += 1
            cap.release()

        if not frame_results:
            raise ValueError("No frames extracted")

        fake_votes = sum(1 for p in frame_results if p > 0.45)
        avg_fake_prob = np.mean(frame_results)
        max_fake_prob = max(frame_results)

        is_fake = (
            fake_votes > len(frame_results) / 2
            or avg_fake_prob > 0.42
            or max_fake_prob > 0.85
        )

        confidence = round((avg_fake_prob if is_fake else (1 - avg_fake_prob)) * 100, 1)
        prediction = "FAKE" if is_fake else "REAL"

        if is_fake:
            reason = (
                f"A total of {len(frame_results)} frames were sampled from the video. "
                f"{fake_votes} of these frames showed strong signs of face manipulation "
                f"(average suspicion score: {round(avg_fake_prob*100,1)}%, highest frame: {round(max_fake_prob*100,1)}%). "
                f"⚠️ This pattern indicates AI face-swapping or deepfake generation."
            )
        else:
            reason = (
                f"A total of {len(frame_results)} frames were sampled from the video. "
                f"Most frames appeared genuine (average suspicion score was only {round(avg_fake_prob*100,1)}%). "
                f"✅ No consistent manipulation pattern was found."
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video analysis error: {str(e)}")
    detection_time = round(time.time() - start, 2)
    return {"prediction": prediction, "confidence": confidence, "detection_time": detection_time, "reason": reason}

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")

@router.post("/detect")
async def detect(
    file: UploadFile = File(...),
    user=Depends(get_current_user)
):
    allowed = ["image/jpeg", "image/png", "image/webp",
               "video/mp4", "video/avi", "video/quicktime", "video/webm"]
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    ext = file.filename.split(".")[-1]
    temp_name = f"{uuid.uuid4().hex}.{ext}"
    temp_path = os.path.join(UPLOAD_DIR, temp_name)
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    file_type = "video" if file.content_type.startswith("video") else "image"
    try:
        if file_type == "image":
            result = analyze_image(temp_path)
        else:
            result = analyze_video(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """INSERT INTO detections
               (user_id, filename, file_type, prediction, confidence, detection_time)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (user["user_id"], file.filename, file_type,
             result["prediction"], result["confidence"], result["detection_time"])
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()
    return {
        "filename": file.filename,
        "file_type": file_type,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "detection_time": result["detection_time"],
        "reason": result.get("reason", ""),
    }