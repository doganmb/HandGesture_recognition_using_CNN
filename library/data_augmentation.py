from pathlib import Path
import numpy as np
import time
import cv2

kinds = ["zero","one","two","three","four","five"]
kind = kinds[1]
file_path = "./Dataset_70x70/"
file_path_2 = "./Dataset_70x70/"
ext = ".png"
counter = 0
def file_check(counter = 0): # finds how many photos do u have?
    counter = 0
    while True:
        if Path((file_path+kind+str(counter)+ext)).is_file():
            counter = counter + 1
        else:
            return counter

counter_2 = file_check()
"""
while True: # Resize photos for keras
    if Path((file_path+kind+str(counter)+ext)).is_file():
            filename = file_path + kind + str(counter) +ext
            img = cv2.imread(filename,0)
            img = cv2.resize(img,(70,70))
            filename_2 = file_path_2 + kind + str(counter) +ext


            cv2.imwrite(filename_2,img)

            counter = counter + 1
            counter_2 = counter_2 + 1
    else:
        print("Mission Completed.")
        break

"""
limit = counter_2

while counter<limit:  # resimleri döndürür
    if Path((file_path+kind+str(counter)+ext)).is_file():
            filename = file_path + kind + str(counter) +ext
            img = cv2.imread(filename,0)
            img = cv2.flip(img, flipCode=1)
            filename_2 = file_path_2 + kind + str(counter_2) +ext
            cv2.imwrite(filename_2,img)

            counter = counter + 1
            counter_2 = counter_2 + 1
    else:
        print("Mission Completed.")
        print(counter,counter_2)
        break
