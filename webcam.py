import argparse
import json
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=None, help="webcam index (0, 1, 2...)")
args = parser.parse_args()

cfg = json.load(open("webcam_config.json"))
CLASSES = cfg["classes"]
IMG_SIZE = cfg["img_size"]
SRC_SIZE = cfg.get("src_size", 75)   # native size of the FER2025 images (75x75)
MIN_CONF = 0.40                      # below this: "uncertain" (the model has no neutral class)
device = "cuda" if torch.cuda.is_available() else "cpu"


class GlobalMean(nn.Module):
    def forward(self, x):
        return x.mean(dim=(2, 3), keepdim=True)


def build_resnet(n_classes):
    m = models.get_model(cfg["arch"], weights=None)
    m.conv1 = nn.Conv2d(1, m.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    if not cfg["use_maxpool"]:
        m.maxpool = nn.Identity()
    m.avgpool = GlobalMean()
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, n_classes))
    return m


model = build_resnet(len(CLASSES))
model.load_state_dict(torch.load("best_resnet.pt", map_location=device))
model.to(device).eval()

# --- face detector -----------------------------------------------------------
if not hasattr(cv2, "data"):
    raise SystemExit("OpenCV without Haar cascades (headless or 5.x build?). Fix with:\n"
                     "  pip uninstall -y opencv-python opencv-python-headless\n  pip install \"opencv-python<5\"")
face_det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_det.empty():
    raise SystemExit("Could not load haarcascade_frontalface_default.xml")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess(gray_face):
    # same chain as training: native 75x75 -> 128x128 (INTER_AREA) -> CLAHE.
    # Going through 75x75 first gives the same blur as the training images.
    img = cv2.resize(gray_face, (SRC_SIZE, SRC_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    if cfg["use_clahe"]:
        img = clahe.apply(img)
    x = (img.astype("float32") / 255.0 - cfg["mean"]) / cfg["std"]
    return torch.from_numpy(x)[None, None].to(device)


def square_crop(gray, x0, y0, w, h, margin=0.10):
    # SQUARE crop (no distortion when resizing), kept inside the image
    H, W = gray.shape
    cx, cy = x0 + w // 2, y0 + h // 2
    half = int(max(w, h) * (0.5 + margin))
    half = min(half, cx, cy, W - cx, H - cy)
    return gray[cy - half:cy + half, cx - half:cx + half], (cx - half, cy - half, cx + half, cy + half)


# --- opening the webcam ------------------------------------------------------
def open_camera(index):
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif sys.platform == "darwin":
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    indexes = [index] if index is not None else [0, 1, 2]
    for i in indexes:
        for b in backends:
            cap = cv2.VideoCapture(i, b)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            good = False
            for _ in range(30):               # the first frames are often empty/black
                ok, frame = cap.read()
                if ok and frame is not None and frame.mean() > 1:
                    good = True
                    break
            if good:
                print(f"Webcam opened: index {i}, backend {cap.getBackendName()}")
                return cap
            cap.release()
    return None


cap = open_camera(args.camera)
if cap is None:
    raise SystemExit("Could not open the webcam. Check: (1) no other app is using it "
                     "(Zoom, Teams, browser), (2) camera permission (Windows: Settings > Privacy > "
                     "Camera; macOS: Settings > Privacy > Camera for Terminal/VS Code), "
                     "(3) try  python webcam.py --camera 1")

smooth = None
fails = 0
print("Press q to quit")

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        fails += 1
        if fails > 30:
            print("The webcam stopped returning frames.")
            break
        continue
    fails = 0

    frame = cv2.flip(frame, 1)   # mirror view (text stays readable)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    if len(faces) > 0:
        x0, y0, w, h = max(faces, key=lambda f: f[2] * f[3])
        face, (x1, y1, x2, y2) = square_crop(gray, x0, y0, w, h)

        if face.size > 0:
            with torch.inference_mode():
                p = model(preprocess(face)).softmax(1)[0].cpu().numpy()
            smooth = p if smooth is None else 0.6 * smooth + 0.4 * p

            k = int(smooth.argmax())
            label = CLASSES[k] if smooth[k] >= MIN_CONF else "uncertain"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {smooth[k]:.0%}", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            for i, (c, v) in enumerate(zip(CLASSES, smooth)):
                cv2.rectangle(frame, (10, 10 + 22 * i), (10 + int(150 * v), 28 + 22 * i), (0, 200, 255), -1)
                cv2.putText(frame, c, (15, 25 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    else:
        smooth = None
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Emotions", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
