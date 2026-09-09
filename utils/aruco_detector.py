import cv2
import numpy as np
import math

class ArucoScaleEstimator:
    def __init__(self, dictionary_type=cv2.aruco.DICT_4X4_50, marker_size_cm=50.0):
        # Configuração do ArUco para o dicionário especificado pelo usuário (4x4 50)
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        self.marker_size_cm = marker_size_cm
        self.last_known_scale = None
        
    def get_scale(self, frame):
        """
        Processa o frame, procura marcadores ArUco e retorna a escala em cm/px.
        Se não encontrar neste frame específico, retorna a última escala conhecida.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            # Desenhar as bordas do ArUco encontrado no frame
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Extrair os quatro cantos do primeiro marcador encontrado
            c = corners[0][0]
            
            # Calcular os pixels de cada um dos 4 lados do quadrado preto
            side1 = math.dist(c[0], c[1])
            side2 = math.dist(c[1], c[2])
            side3 = math.dist(c[2], c[3])
            side4 = math.dist(c[3], c[0])
            
            # Usar a média para obter o valor mais preciso possível do lado
            avg_side_pixels = (side1 + side2 + side3 + side4) / 4.0
            
            if avg_side_pixels > 0:
                # O quadrado interno preto mede exatamente 50.0 cm
                scale = self.marker_size_cm / avg_side_pixels
                self.last_known_scale = scale
                return scale, frame
        
        return self.last_known_scale, frame
