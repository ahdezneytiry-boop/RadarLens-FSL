import cv2
import torch
import numpy as np
from model import EMA_FSL
from config import Config

model = EMA_FSL()
model.load_state_dict(torch.load(Config.save_path, map_location="cpu"))
model.eval()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (64, 64))
    img = img[np.newaxis, np.newaxis, :, :].astype(np.float32)

    radar = np.random.randn(1, 30, 16).astype(np.float32)

    radar = torch.tensor(radar)
    visual = torch.tensor(img)

    with torch.no_grad():
        output = model(radar, visual)
        pred = torch.argmax(output, dim=-1).item()

    text = "Danger!" if pred == 1 else "Normal"

    cv2.putText(frame, text, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if pred == 1 else (0, 255, 0), 2)

    cv2.imshow("EMA-FSL Demo", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
