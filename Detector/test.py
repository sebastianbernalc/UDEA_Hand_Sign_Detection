#--------------------------------------------------------------------------
#------------------------ -Alfabeto Dactilológico -------------------------
#--------------------------------------------------------------------------
#-------------------------Coceptos básicos de PDI--------------------------
#--------------------------------------------------------------------------
#-------------------------Sebastian Bernal Cuaspa--------------------------
#----------------------sebastian.bernalc@udea.edu.co-----------------------
#--------------------------------------------------------------------------
#-----------------------Kevin David Martinez Zapata------------------------   
#-----------------------kevin.martinez1@udea.edu.co------------------------
#--------------------------------------------------------------------------
#------------------------Universidad De Antioquia--------------------------
#-------------------------Ingenieria Electronica---------------------------
#--------------------Procesamiento Digital De Imagenes I-------------------
#-----------------------------Proyecto #2----------------------------------


#--------------------------------------------------------------------------
#--1. Inicializo el sistema -----------------------------------------------
#--------------------------------------------------------------------------

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from cvzone.ClassificationModule import Classifier
import time
import os

#--------------------------------------------------------------------------
#---------------------2 Configuracion del proyecto.------------------------
#--------------------------------------------------------------------------

current_directory = os.getcwd()                                                         # obtiene la ruta de la carpeta actual
model_folder = os.path.join(current_directory, "Detector", "model", "keras_model.h5")   #Carpeta del modelo entrenado
labels_folder = os.path.join(current_directory, "Detector", "model", "labels.txt")      #Carpeta de los labels (Letras entrenadas)

labels = ["A","B","C"]                              #Labels para mostrar letra en pantalla
cap = cv2.VideoCapture(0)                           #Captura de camara
detector = HandDetector(maxHands = 1)               #Detector de manos activas
classifier = Classifier(model_folder,labels_folder) #Modelo de aprendizaje automatico entrenado

#Parametros recuadro mapeo de mano
offset = 15
imgSize = 300

#--------------------------------------------------------------------------
#---------------------3 Loop principal.------------------------------------
#--------------------------------------------------------------------------

while True :
    success, img = cap.read()            #Lectura de camara y almacenamiento de variables
    imgOutuput = img.copy()              #Copia del cuadro de video leído para poder dibujar visualizaciones y resultados de detección en ella sin modificar el cuadro original.
    hands, img = detector.findHands(img) #Detectar las manos en el cuadro de video leído

    #--------------------------------------------------------------------------
    #---------------------4 detector de manos.---------------------------------
    #--------------------------------------------------------------------------
    
    #si se detectaron manos en el cuadro de video leído 
    if hands:
        hand = hands[0]                                                     #Se extrae la primera mano de la lista 
        x, y, w, h = hand['bbox']                                           #La región de interés de la mano se define mediante las coordenadas del cuadro delimitador
        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255                #Se crea una nueva imagen  y se llena con el valor blanco
        #Esta imagen blanca se utilizará más adelante como fondo para mostrar la mano recortada.
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset] #Se extrae el cuadro de la imagen que rodea la mano y se agrega un margen  a su alrededor para que la mano no quede demasiado cerca de los bordes de la imagen.
        imgCropShape = imgCrop.shape                                        #Se calcula la forma (dimensiones) de la imagen de la mano recortada
        aspectRatio = h/w                                                   #Se calcula el aspectRatio de la mano a partir de su altura (h) y ancho (w).

        #--------------------------------------------------------------------------
        #---------------------5 Reajuste de identificacion de manos.---------------
        #--------------------------------------------------------------------------
        #Si la mano es más alta que ancha
        if aspectRatio > 1 :
            k = imgSize / h                                         #Se calcula una escala k que se usará para redimensionar la imagen de la mano (imgCrop) para que su altura sea igual a imgSize
            wCal = math.ceil(k * w)                                 #Se calcula la anchura redimensionada wCal utilizando esta escala y el ancho original de la imagen de la mano.
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))          #Redimensionar la imagen de la mano
            imgResizeShape = imgResize.shape                        #Se calcula la forma (dimensiones) de la imagen de la mano redimensionada
            wGap = math.ceil((imgSize - wCal)/2)                    #Se calcula la cantidad de espacio en blanco que se debe agregar a cada lado de la imagen redimensionada para que tenga el mismo ancho que imgSize
            imgWhite[:, wGap:wGap + wCal] = imgResize               #Se actualiza la imagen en blanco imgWhite para agregar la imagen redimensionada de la mano (imgResize) en el centro de la imagen blanca
            prediction,index = classifier.getPrediction(imgWhite)   #Se utiliza el clasificador de gestos de mano (classifier) para predecir el gesto de la mano en la imagen blanca redimensionada.
            print(prediction,index)                                 #Muestra de datos
            
        #Si la mano es más ancha que alta
        else:
            k = imgSize / w                                         #Se calcula una escala k que se usará para redimensionar la imagen de la mano (imgCrop) para que su altura sea igual a imgSize
            hCal = math.ceil(k * h)                                 #Se calcula la anchura redimensionada wCal utilizando esta escala y el ancho original de la imagen de la mano.
            imgResize = cv2.resize(imgCrop,(imgSize, hCal))         #Redimensionar la imagen de la mano
            imgResizeShape = imgResize.shape                        #Se calcula la forma (dimensiones) de la imagen de la mano redimensionada
            hGap = math.ceil((imgSize - hCal)/2)                    #Se calcula la cantidad de espacio en blanco que se debe agregar a cada lado de la imagen redimensionada para que tenga el mismo ancho que imgSize
            imgWhite[hGap:hGap + hCal,:] = imgResize                #Se actualiza la imagen en blanco imgWhite para agregar la imagen redimensionada de la mano (imgResize) en el centro de la imagen blanca
            prediction,index = classifier.getPrediction(imgWhite)   #Se utiliza el clasificador de gestos de mano (classifier) para predecir el gesto de la mano en la imagen blanca redimensionada.

        #--------------------------------------------------------------------------
        #---------------------6. Muestra de resultados.----------------------------
        #--------------------------------------------------------------------------
        cv2.putText(imgOutuput, labels[index], (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)  #se utiliza para dibujar texto en una imagen.

        cv2.imshow("ImageCrop",imgCrop)   #Muestra imagen recortada de la deteccion de la mano
        cv2.imshow("ImageWhite",imgWhite) #Muestra imagen redimencionada que es usada para entrenar y predecir


    cv2.imshow("Image",imgOutuput) #Muestra imagen completa en tiempo real
    key = cv2.waitKey(1)


