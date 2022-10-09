import cv2
import numpy as np
import requests
import imutils
import cascade_boosting

url = "http://10.0.50.5:8080/shot.jpg"



def template_matching_with_cascade(flag_cam, cascade_classifier):
    template_faces = []
    if (flag_cam == 'webcam'):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not template_faces:  #sem templates
                template_faces = cascade_boosting.get_template_Cascade(frame, cascade_classifier)
            
            for face in template_faces:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                result = cv2.matchTemplate(gray_frame, face, cv2.TM_CCOEFF_NORMED)
                threshold = 0.8  
                loc = np.where(result > threshold)
                loc = list(zip(*loc[::-1]))

                (h, w) = face.shape
                print(len(loc))
                for point in loc:
                    cv2.rectangle(frame, point, (point[0] + w, point[1] + h), (255, 0, 0))

                cv2.imshow("Template Matching", frame)

                k = cv2.waitKey(30) & 0xff
                if k == 27: # press 'ESC' to quit
                    break

                cap.release()
                cv2.destroyAllWindows()
    else: #android
        while True:
            img_resp = requests.get(url)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            img = cv2.imdecode(img_arr, -1)
            img = imutils.resize(img, width=1000, height=1800)

            if not template_faces:  #sem templates
                template_faces = cascade_boosting.get_template_Cascade(img, cascade_classifier)

            for face in template_faces:
                gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                result = cv2.matchTemplate(gray_frame, face, cv2.TM_CCOEFF_NORMED)
                threshold = 0.8  
                loc = np.where(result > threshold)
                loc = list(zip(*loc[::-1]))

                (h, w) = face.shape
                print(len(loc))
                for point in loc:
                    cv2.rectangle(img, point, (point[0] + w, point[1] + h), (255, 0, 0))
                
            cv2.imshow("Template Matching", img)

            # Press Esc key to exit
            if cv2.waitKey(1) == 27:
                break
    
        cv2.destroyAllWindows()



def face_detection_with_Cam(flag_cam, path_template):
    if (flag_cam == 'webcam'):    
        cap = cv2.VideoCapture(0)
        template = cv2.imread(path_template, cv2.IMREAD_GRAYSCALE)
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
    else:
        template = cv2.imread(path_template, cv2.IMREAD_GRAYSCALE)
        w, h = template.shape[::-1]
        template = cv2.resize(template, (500,500))
        threshold = 0.8  
        while True:
            img_resp = requests.get(url)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            img = cv2.imdecode(img_arr, -1)
            img = imutils.resize(img, width=1000, height=1800)
            gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(result >= threshold) 
            for pt in zip(*loc[::-1]):   
                cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)  
                
            cv2.imshow("Template Matching", img)

            # Press Esc key to exit
            if cv2.waitKey(1) == 27:
                break
    
        cv2.destroyAllWindows()
    


def face_detection_with_image(path_template, path_imagem):
    img = cv2.imread(path_imagem, 0)
    copia_img = img.copy()
    template = cv2.imread(path_template, 0)
    w,h = template.shape[::-1]

    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    for meth in methods:
        img = copia_img.copy()
        method = eval(meth)

        resultado = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(resultado)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img,top_left, bottom_right, 255, 2)
        
    cv2.imwrite('img/img_detectadas/resultado_TM.jpg', img)
    cv2.destroyAllWindows()
    