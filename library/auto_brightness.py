import os
import cv2
import numpy as np
import pyautogui
from time import sleep
import threading



class bright: # you can change your computer brightness
    def __init__(self):
        self.light = None
    def change_brightness(self):
        while True:
            sleep(1)
            if(self.light != None and self.light != 777):
                active = "POWERCFG /S 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
                ac_change = "POWERCFG /SETACVALUEINDEX 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c 7516b95f-f776-4464-8c53-06167f40cc99 aded5e82-b909-4619-9949-f5d71dac0bcb " + str(self.light)
                dc_change = "POWERCFG /SETDCVALUEINDEX 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c 7516b95f-f776-4464-8c53-06167f40cc99 aded5e82-b909-4619-9949-f5d71dac0bcb " + str(self.light)
                os.system(dc_change)
                os.system(ac_change)
                os.system(active)
            elif (self.light == 777):
                break
    def run(self):
        th = threading.Thread(target=self.change_brightness)
        th.start()
