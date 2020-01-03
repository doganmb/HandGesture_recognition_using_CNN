#import tensorflow as tf
from pathlib import Path
import numpy as np
import time
import cv2
np.set_printoptions(suppress=True)

kinds = ["zero","one","two","three","four","five"]
kind = kinds[0]
file_path = "./Dataset_70x70/"
ext = ".png"

def file_check(counter = 0):
    counter2 = 0
    for i in range(len(kinds)):
        kind = kinds[i]
        counter = 0
        while True:
            if Path((file_path+kind+str(counter)+ext)).is_file():
                counter = counter + 1
                counter2= counter2 + 1
            else:
                break
    return counter2


data = np.empty((file_check(),4901),np.uint8)

counter = 0
counter2 = 0
for i in range(6):
    kind = kinds[i]
    counter = 0
    while True:
        if Path((file_path+kind+str(counter)+ext)).is_file():
            filename = file_path + kind + str(counter) + ext
            img = cv2.imread(filename,0)
            img = img.flatten()
            img = np.append(img,i)
            data[counter2] = img
            counter = counter + 1
            counter2 = counter2 +1
            print(filename,counter2)
        else:
            break

np.savetxt("new_data.csv",data,fmt='%d')
