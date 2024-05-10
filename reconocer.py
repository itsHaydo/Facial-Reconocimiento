import cv2
import os

personas = 'personas/'
imgPaths = os.listdir(personas)
print(imgPaths)

reconocerJeta = cv2.face.LBPHFaceRecognizer_create()
reconocerJeta.read('training/modelo2.xml')

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

caraClasificador = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    caras = caraClasificador.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in caras:
        cara = auxFrame[y:y + h, x:x + w]
        cara = cv2.resize(cara, (150, 150), interpolation=cv2.INTER_CUBIC)
        resultado = reconocerJeta.predict(cara)

        #cv2.putText(frame, '{}'.format(resultado), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

        print(resultado)
        if resultado[1] < 70:
            cv2.putText(frame, '{}'.format(imgPaths[resultado[0]]),(x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Persona no reconocida', (x, y - 20), 2, 0.8, (0, 0, 255),1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('Reconociendo caras', frame)
    k = cv2.waitKey(10)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
