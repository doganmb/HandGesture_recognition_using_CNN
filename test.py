from keras.models import load_model
import cv2
import numpy as np
from library import auto_brightness
import threading


cb = auto_brightness.bright()
cb.run()
model_best = load_model('./Model_Patch/lastest_model_save.h5')

emotion_dict = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}
webcam = cv2.VideoCapture(0)
cb.light = 50
while True:
    ret,img = webcam.read()
    if (ret == True):
        img = cv2.resize(img,(720,480))
        img = cv2.flip(img,+1)
        hand_capture = img[65:330,400:700]
        hand_capture = cv2.cvtColor(hand_capture,cv2.COLOR_BGR2GRAY)
        img = cv2.rectangle(img, (400,65), (700,330), (128,0,128), 3)
        cv2.imshow("Final",hand_capture)
        hand_capture = cv2.resize(hand_capture,(70,70))


        final = np.expand_dims(np.expand_dims(hand_capture,-1),0)
        cv2.normalize(final, final, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        prediction = model_best.predict(final)
        if (np.max(prediction)>0.90):
            cv2.putText(img, emotion_dict[int(np.argmax(prediction))], (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 5, cv2.LINE_AA)
            cv2.putText(img, str(np.max(prediction)), (100,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 5, cv2.LINE_AA)
            if emotion_dict[int(np.argmax(prediction))] == "Zero": # computer brightness is 0
                    cb.light = 0
            elif emotion_dict[int(np.argmax(prediction))] == "One": # computer brightness is 20
                    cb.light = 20
            elif emotion_dict[int(np.argmax(prediction))] == "Two": # computer brightness is 20
                    cb.light = 40
            elif emotion_dict[int(np.argmax(prediction))] == "Three": # computer brightness is 20
                    cb.light = 60
            elif emotion_dict[int(np.argmax(prediction))] == "Four": # computer brightness is 20
                    cb.light = 80
            elif emotion_dict[int(np.argmax(prediction))] == "Five": # computer brightness is 20
                    cb.light = 100

            #elif emotion_dict[int(np.argmax(prediction))] == "Four":
                #cb.light = 777
                #break

        cv2.imshow("Original",img)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            cb.light = 777
            break

webcam.release()
cv2.destroyAllWindows()
