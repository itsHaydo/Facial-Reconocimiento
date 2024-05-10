import cv2 as cv
import os
import imutils
from datetime import datetime

# Ingresamos el nombre de la persona a capturar sus fotografias
nombre = "Cristobal"
# datosPersona sera la carpeta donde se ingresaran las personas registradas
datosPersona = "personas/"
# personaDir es el directorio completo de la persona elegida nueva a registrar
personaDir = datosPersona + nombre

if not os.path.exists(personaDir):
    print('Carpeta creada:' + personaDir)
    os.makedirs(personaDir)

cap = cv.VideoCapture(0)

faceClassifier = cv.CascadeClassifier("xml/haarcascade_frontalface_default.xml")
c = 0
print("Este procesos puede tardar un poco :D")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = imutils.resize(frame,width=640)
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    auxFrame = frame.copy()

    caras = faceClassifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in caras:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cara = auxFrame[y:y+h,x:x+w]
        cara = cv.resize(cara,(150,150),interpolation=cv.INTER_CUBIC)
        # Use the current date and time as part of the filename
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        cv.imwrite(personaDir + '/cara_{}_{}.jpg'.format(c, timestamp), cara)
        c += 1
    cv.imshow('Capturando...',frame)
    k = cv.waitKey(1)

    if k == 27 or c >= 300:
        print('El proceso de capturacion ha terminado')
        break

cap.release()
cv.destroyAllWindows()
