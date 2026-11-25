
import cv2
from ultralytics import YOLO

model = YOLO("best260408.pt")

img = cv2.imread("ex2-26.png")

all_boxes = []
scores = []

# ① 元画像全体から検出
results = model.predict(
    "ex2-26.png",
    conf=0.03,
    imgsz=1792,
    iou=0.2,
    augment=True
)

for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    if 100 < cx < 1200 and 250 < cy < 550:
        all_boxes.append([x1, y1, x2-x1, y2-y1])
        scores.append(float(box.conf[0]))

# ② 選手が密集している部分を追加検出
left = 700
top = 320
right = 950
bottom = 480

crop = img[top:bottom, left:right].copy()

results = model.predict(
    crop,
    conf=0.1,
    imgsz=640,
    iou=0.3
)

for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    x1 += left
    x2 += left
    y1 += top
    y2 += top

    all_boxes.append([x1, y1, x2-x1, y2-y1])
    scores.append(float(box.conf[0]))

# ③ 重複した検出枠を除去
indices = cv2.dnn.NMSBoxes(
    all_boxes,
    scores,
    0.05,
    0.3
)

# ④ 元画像に赤枠を描画
count = 0

for i in indices:
    i = int(i)
    x, y, w, h = all_boxes[i]

    cv2.rectangle(
        img,
        (x, y),
        (x+w, y+h),
        (0, 0, 255),
        thickness=2
    )

    count += 1

print("検出したヘルメットの数:", count)

cv2.imshow("ex4", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
