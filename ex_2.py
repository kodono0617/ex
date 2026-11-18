
# ex2-26.png中のフィールド上の人物を抽出し，
# その領域を赤枠で元の画像に描画する．
# また抽出した人物の数を示す．

import cv2
from ultralytics import YOLO

model = YOLO("yolov8x.pt")

results = model.predict(
    "ex2-26.png",
    conf=0.15,
    iou=0.7,
    imgsz=1280
)

# 入力画像
img = results[0].orig_img

# 認識した物体領域を取得する．
boxes = results[0].boxes

count = 0

for box in boxes:
    # personのみ抽出
    if int(box.cls[0]) == 0:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # 人物の足元の座標
        center_x = (x1 + x2) // 2
        bottom_y = y2

        # フィールド上の人物だけを抽出
        if 250 < bottom_y < 550 and 100 < center_x < 1200:

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                thickness=3,
            )

            count += 1

print("検出した人物の数:", count)

cv2.imshow("ex2", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
