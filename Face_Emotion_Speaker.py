from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import dlib
import numpy as np
import time

face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
emotion_classifier =load_model('./Emotion_Detection.h5')
landmarks_classifier = dlib.shape_predictor("shape_predictor_68_face_landmarks_GTX.dat")

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']
FRAME_COLOR = (255,0,0)
HOLD_TIME = 3.0       # Hold the speaker frame 2 seconds
is_speaker = False
start = None


cap = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if is_speaker:
        if time.time() - start >=HOLD_TIME:
            is_speaker = False
            print("Hold End")
            FRAME_COLOR = (255, 0, 0)

    # Find All face Candidates in the Frames
    for (x,y,w,h) in faces:
        # Draw Face Frame
        cv2.rectangle(frame,(x,y),(x+w,y+h),FRAME_COLOR,2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        # Extract the Lips LandMarks
        face = dlib.rectangle(x, y, x+w, y+h)
        landmarks = landmarks_classifier(gray, face)
        pnts = landmarks.parts()
        upper_lips = np.array([(pnt.x,pnt.y) for pnt in pnts[61:64]])
        lower_lips = np.array([(pnt.x,pnt.y) for pnt in pnts[68:64:-1]])
        # Draw Lips LandMarks
        for pnt in upper_lips:
            cv2.circle(frame, pnt, 3,(255,0,0), -1)
        for pnt in lower_lips:
            cv2.circle(frame, pnt, 3,(255,255,0), -1)

        dist = np.linalg.norm(lower_lips - upper_lips, axis=1)
        if np.any(dist > 7):    # Lips Movement Detected
            # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            print("Speaker Detected")

            is_speaker = True
            start = time.time()
            FRAME_COLOR = (255, 255, 255)

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class
            preds = emotion_classifier.predict(roi)[0]
            # print("\nprediction = ",preds)
            label=class_labels[preds.argmax()]
            # print("\nprediction max = ",preds.argmax())
            # print("\nlabel = ",label)
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        # print("\n\n")
    cv2.imshow('Emotion Detector',frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()