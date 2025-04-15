import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 20
imgSize = 580
counter = 0

folder = "Data/Hello"

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture video. Exiting...")
        break
    
    hands, img = detector.findHands(img)
    if hands:
        x_min = min([hand['bbox'][0] for hand in hands])
        y_min = min([hand['bbox'][1] for hand in hands])
        x_max = max([hand['bbox'][0] + hand['bbox'][2] for hand in hands])
        y_max = max([hand['bbox'][1] + hand['bbox'][3] for hand in hands])
        
        x1, x2 = max(0, x_min - offset), min(img.shape[1], x_max + offset)
        y1, y2 = max(0, y_min - offset), min(img.shape[0], y_max + offset)

        imgCrop = img[y1:y2, x1:x2]
        if imgCrop.size == 0:
            print("Empty crop detected, skipping frame.")
            continue

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        h, w = y2 - y1, x2 - x1
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
    elif key == 27:
        print("Escape key pressed. Exiting...")
        break

cap.release()
cv2.destroyAllWindows()