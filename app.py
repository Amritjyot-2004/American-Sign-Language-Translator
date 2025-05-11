from flask import Flask, request, render_template
import cv2
import numpy as np
from ultralytics import YOLO
from cvzone.HandTrackingModule import HandDetector
import math
import cvzone

app = Flask(__name__)

model = YOLO("best.pt")
classNames = ["A", "F", "L", "Y"]
detector = HandDetector(maxHands=1)
imgSize = 640
offset = 20
img = cv2.imread("A1.jpg")

_ = model.predict(img, stream = False)
print(_)

_ = model.predict(img, stream = False)
print(_)

_ = model.predict(img, stream = False)
print(_)

_ = model.predict(img, stream = False)
print(_)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
async def process_frame():
    # Read image file from request
    file = request.files['frame']            # name 'frame' matches FormData key
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        try:
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hGap + hCal, :] = imgResize

            results = model.predict(imgWhite, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    color = (0, 255, 0) if classNames[cls] == "A" else (255, 0, 0)
                    cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                    cvzone.putTextRect(img, f"{classNames[cls].upper()} {int(conf * 100)}%",
                                    (max(0, x1), max(35, y1)), scale=2, thickness=4,
                                    colorR=color, colorB=color)
        except cv2.error as e:
            cv2.putText(img, "Going out of frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            print(f"Resize failed: {e}")
    else:
        cv2.putText(img, "No hand detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    ret, jpeg = cv2.imencode('.jpg', img)
    return app.response_class(jpeg.tobytes(), mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(host = 0.0.0.0, port = 5000)
