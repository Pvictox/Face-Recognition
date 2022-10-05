import numpy as np
import cv2
import dlib




def face_detection_with_cam():
    hogFaceDetector = dlib.get_frontal_face_detector()

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # set Width
    cap.set(4, 480)  # set Height
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = hogFaceDetector(gray, 1)

        for (i, rect) in enumerate(faces):
            x = rect.left()
            y = rect.top()
            w = rect.right() - x
            h = rect.bottom() - y
            #draw a rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow("Image", img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:  # press 'ESC' to quit
            break
    cap.release()
    cv2.destroyAllWindows()


face_detection_with_cam()
