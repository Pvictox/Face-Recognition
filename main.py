from importlib import import_module
import cascade_boosting
import cv2
import template_matching
import andoid_cam
#ainda to tendo erro com o dlib testar depois é só tirar os comentários
#import dblib


haar_Classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#template_matching.face_detection_with_image('img/template_jao.png', 'img/teste_img.jpg')

#andoid_cam.start()
#hogFaceDetector = dlib.get_frontal_face_detector()

#Caso teste HAAR
#cascade_boosting.face_detection_image(haar_Classifier, 'img/teste_img.jpg', 'haar')
cascade_boosting.face_detection_with_cam(haar_Classifier, 'haar', 'android')

#Caso teste HOG
#cascade_boosting.face_detection_image(hogFaceDetector, 'img/teste_img.jpg', 'hog')