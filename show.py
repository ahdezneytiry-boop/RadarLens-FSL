import cv2
import time
import os
from ultralytics import YOLO

# ======================
# 初始化
# ======================
model = YOLO(r"E:\2026计算机设计\yolov8n.pt")

cap = cv2.VideoCapture(0)

prev_y = None
prev_h = None

danger_counter = 0
last_save_time = 0

SAVE_DIR = "danger_records"
os.makedirs(SAVE_DIR, exist_ok=True)

print("按 ESC 退出")

# ======================
# 主循环
# ======================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    results = model(frame, verbose=False)[0]

    person_box = None

    for box in results.boxes:
        if int(box.cls[0]) == 0:  # 人
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_box = (x1, y1, x2, y2)
            break

    status = "NO PERSON"

    if person_box:
        x1, y1, x2, y2 = person_box
        w, h = x2 - x1, y2 - y1

        # 框人（稳定）
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, "Person", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # ======================
        # 核心：行为检测（稳定版）
        # ======================
        fall_detected = False

        if prev_y is not None and prev_h is not None:
            # 1. 向下移动速度
            delta_y = y1 - prev_y

            # 2. 高度变化（蹲下会变矮）
            delta_h = prev_h - h

            # 判定（关键调参）
            if delta_y > 20 and delta_h > 10:
                fall_detected = True

        prev_y = y1
        prev_h = h

        # ======================
        # Danger持续机制（核心优化）
        # ======================
        if fall_detected:
            danger_counter = 25   # 持续时间更长

        if danger_counter > 0:
            status = "DANGER"
            color = (0,0,255)
            danger_counter -= 1

            # ======================
            # 自动截图
            # ======================
            now = time.time()
            if now - last_save_time > 2:
                filename = time.strftime("%Y%m%d_%H%M%S") + ".jpg"
                path = os.path.join(SAVE_DIR, filename)
                cv2.imwrite(path, frame)
                print("保存异常:", path)
                last_save_time = now

        else:
            status = "NORMAL"
            color = (0,255,0)

    else:
        prev_y = None
        prev_h = None

    # ======================
    # UI显示
    # ======================
    cv2.putText(frame, status, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    cv2.imshow("EMA-FSL Stable Demo", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    time.sleep(0.02)

cap.release()
cv2.destroyAllWindows()
