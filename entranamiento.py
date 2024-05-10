import cv2
import numpy as np
import os

personas = "personas/"
listaPersonas = os.listdir(personas)
print(listaPersonas)

etiquetas = []
datosCaras = []
etiqueta = 0

for persona in listaPersonas:
    datosPersona = personas + persona
    print('Leyendo datos de persona ' + persona)

    for cara in os.listdir(datosPersona):
        # print('Rostros: ', persona + '/' + cara)
        etiquetas.append(etiqueta)
        datosCaras.append(cv2.imread(datosPersona + '/' + cara, 0))
        image = cv2.imread(datosPersona + '/' + cara, 0)
    etiqueta += 1

print("Entrenando... 🏋️")

face_reconocer = cv2.face.LBPHFaceRecognizer_create()

face_reconocer.train(datosCaras, np.array(etiquetas))

face_reconocer.write('training/modelo2.xml')

print("Quedo bien mamadisimo 😮‍💨")
