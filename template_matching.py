import cv2
import numpy as np

cap = cv2.VideoCapture(0)
template = cv2.imread("template3.png", cv2.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]
threshold = 0.8  

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= threshold) 
    for pt in zip(*loc[::-1]):   
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)  
        
    cv2.imshow("Template Matching", frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()