from pathlib import Path
import numpy as np
import time
import cv2

# Takes photos by size of 70x70
# If you saved photos before it continue to save last number of photo 

cap = cv2.VideoCapture(0)

kind = "two"
file_path = "./Dataset_70x70/"
ext = ".png"

def file_check(counter = 0):
    counter = 0
    while True:
        if Path((file_path+kind+str(counter)+ext)).is_file():
            counter = counter + 1
        else:
            return counter

counter = file_check()
while True:
    _,img = cap.read()
    img = cv2.resize(img,(720,480))
    img = cv2.flip(img,+1)
    hand_capture = img[65:330,400:700]
    hand_capture = cv2.cvtColor(hand_capture,cv2.COLOR_BGR2GRAY)
    img = cv2.rectangle(img, (400,65), (700,330), (128,0,128), 3)

    cv2.namedWindow("Final",cv2.WINDOW_NORMAL)
    cv2.imshow("Final",hand_capture)
    cv2.namedWindow("Original",cv2.WINDOW_NORMAL)
    cv2.imshow("Original",img)
    save_or_exit = cv2.waitKey(50)
    if save_or_exit & 0xFF==ord("s"):
        filename = file_path + kind + str(counter) + ext
        hand_capture = cv2.resize(hand_capture,(70,70))
        cv2.imwrite(filename,hand_capture)
        print(counter)
        counter = file_check(counter = counter)
    elif save_or_exit & 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
