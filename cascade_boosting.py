from tkinter import Scale
import cv2
import requests
import imutils
import numpy as np
#Flag é uma string [haar = faz o processo com o haar | hog = faz o processo com o hog]
#Flag cam vai dizer se vai usar web cam tradicional ou do android 
 # {'webcam' ou 'android'}




url = "http://10.0.50.5:8080/shot.jpg"



def get_template_Cascade(current_frame, classifier):
    gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.equalizeHist(gray_frame)
    faces = classifier.detectMultiScale(
                    gray_frame,     
                    scaleFactor=1.2,
                    minNeighbors=5,     
                    minSize=(20, 20)
                )
    templates = []

    for face in faces:
        (x,y,w,h) = face
        face = gray_frame[y:y+h, x: x+h]
        templates.append(face)
    
    return templates


def face_detection_with_cam(classifier, flag, flag_cam):
    
    if (flag_cam == 'webcam'):
        cap = cv2.VideoCapture(0)
        cap.set(3,640) # set Width
        cap.set(4,480) # set Height
    
        while True:
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if (flag == 'haar'):
                faces = classifier.detectMultiScale(
                    gray,     
                    scaleFactor=1.2,
                    minNeighbors=5,     
                    minSize=(20, 20)
                )
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = img[y:y+h, x:x+w]  
            else:
                faces = classifier(gray, 1)
                for (i, rect) in enumerate(faces):
                    x = rect.left()
                    y = rect.top()
                    w = rect.right() - x
                    h = rect.bottom() - y
                    #draw a rectangle
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow('video',img)
            k = cv2.waitKey(30) & 0xff
            if k == 27: # press 'ESC' to quit
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        while True:
            img_resp = requests.get(url)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            img = cv2.imdecode(img_arr, -1)
            img = imutils.resize(img, width=1000, height=1800)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if (flag == 'haar'):
                faces = classifier.detectMultiScale(
                    gray,     
                    scaleFactor=1.2,
                    minNeighbors=5,     
                    minSize=(20, 20)
                )
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = img[y:y+h, x:x+w]  
            else:
                faces = classifier(gray, 1)
                for (i, rect) in enumerate(faces):
                    x = rect.left()
                    y = rect.top()
                    w = rect.right() - x
                    h = rect.bottom() - y
                    #draw a rectangle
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Android_cam", img)
        
            # Press Esc key to exit
            if cv2.waitKey(1) == 27:
                break
    
        cv2.destroyAllWindows()





def face_detection_image(classifier, path_image, flag):
    img = cv2.imread(path_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if (flag == 'haar'):
        faces = classifier.detectMultiScale(
            gray,
            scaleFactor = 1.2, #ver com 1.2
            minNeighbors = 5,
            minSize = (20,20)
        )
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img, (x,y), (x+w, y+w), (255,0,0),2)
    else:
        faces = classifier(gray, 1)
        for (i, rect) in enumerate(faces):
            x = rect.left()
            y = rect.top()
            w = rect.right() - x
            h = rect.bottom() - y
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    resize = cv2.resize(img, (500,500))
    cv2.imwrite('img/img_detectadas/resultado.jpg', resize)
    #cv2.imshow('img', resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
