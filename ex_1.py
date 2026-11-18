import cv2
from ultralytics import YOLO

model = YOLO("yolov8x.pt")

results = model.predict("ex1-26.png", conf=0.25)

# 入力画像
img = results[0].orig_img

# 認識した物体領域を取得する．
boxes = results[0].boxes

count = 0

for box in boxes:
    # 人物(person)のみ抽出する．
    if int(box.cls[0]) == 0:
        # 物体領域のxy座標を取得する．
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            thickness=3,
        )
        count += 1

print("検出した人物の数:", count)

cv2.imshow("", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
