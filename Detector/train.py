import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)

offset = 15
imgSize = 300
#hola
# obtiene la ruta de la carpeta actual
current_directory = os.getcwd()
folder_A= os.path.join(current_directory, "Detector", "data", "A")
folder_B= os.path.join(current_directory, "Detector", "data", "B")
folder_C= os.path.join(current_directory, "Detector", "data", "C")
counter=0

while True :
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        

        aspectRatio = h/w
        #print(aspectRatio)

        if aspectRatio > 1 :
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop,(imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap:hGap + hCal,:] = imgResize



        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("ImageWhite",imgWhite)


    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord("a"):
        counter += 1
        cv2.imwrite(f'{folder_A}/Image_{time.time()}.jpg',imgWhite)

    if key == ord("b"):
        counter += 1
        cv2.imwrite(f'{folder_B}/Image_{time.time()}.jpg',imgWhite)

    if key == ord("c"):
        counter += 1
        cv2.imwrite(f'{folder_C}/Image_{time.time()}.jpg',imgWhite)

    