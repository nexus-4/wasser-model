import cv2
import numpy as np
import time

print("[1/7] Importando a Inteligencia Artificial (YOLO)...")
from ultralytics import YOLO

from utils.aruco_detector import ArucoScaleEstimator
from utils.weight_calculator import WeightCalculator

def draw_hud_box(img, text, pos, bg_color=(0, 0, 0), text_color=(255, 255, 255)):
    """Desenha um texto com um fundo escuro elegante."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    
    # Calcular tamanho do texto
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = pos
    # Desenhar retângulo de fundo
    cv2.rectangle(img, (x, y - text_height - 10), (x + text_width + 10, y + baseline), bg_color, -1)
    
    # Desenhar o texto
    cv2.putText(img, text, (x + 5, y - 5), font, font_scale, text_color, thickness)

def main():
    print("[2/7] Iniciando Teste em Imagem Unica...")

    MODEL_PATH = "yolov8n-seg.pt"
    IMAGE_PATH = "media/teste_frame.png"
    
    scale_estimator = ArucoScaleEstimator(marker_size_cm=50.0)
    weight_calc = WeightCalculator(calibration_weight_kg=500.0, base_calibration_area_m2=2.0)

    print(f"[4/7] Carregando o modelo de segmentacao ({MODEL_PATH})...")
    model = YOLO(MODEL_PATH)
    
    frame = cv2.imread(IMAGE_PATH)

    if frame is None:
        print("ERRO FATAL: Nao encontrei a imagem!")
        return

    print("[6/7] Procurando pelo quadrado do ArUco na imagem...")
    current_scale, annotated_frame = scale_estimator.get_scale(frame)
    
    # Interface superior (Escala)
    scale_text = f"Escala: {current_scale:.3f} cm/px" if current_scale else "Escala: Nao encontrada"
    draw_hud_box(annotated_frame, scale_text, (20, 40), bg_color=(50, 50, 50), text_color=(0, 255, 255))

    print("[7/7] Escaneando a imagem com o YOLO (buscando possiveis gados)...")
    # Aumentando a confiança para 0.60 para remover a sombra (30%) e a cabeça (32%)
    results = model.predict(frame, conf=0.60, verbose=False)
    result = results[0]

    gado_count = 0

    if result.boxes is not None and result.masks is not None:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        print(f"      -> O YOLO encontrou {len(boxes_xyxy)} coisas na imagem!")
        
        for box_xyxy, conf, cls, mask_data in zip(boxes_xyxy, confs, classes, result.masks):
            x1, y1, x2, y2 = map(int, box_xyxy)
            polygon = mask_data.xy[0] 
            
            if len(polygon) > 0:
                pixel_area = cv2.contourArea(polygon)
                
                # Filtrar sombras/ruídos muito pequenos
                if pixel_area < 10000:
                    continue
                    
                gado_count += 1
                color = (0, 200, 0)
                
                # Desenhar apenas o contorno suave, sem a caixa quadrada dura
                cv2.polylines(annotated_frame, [polygon.astype(np.int32)], True, color, 3)
                
                weight_text = "Sem ArUco"
                if current_scale is not None:
                    estimated_weight = weight_calc.calculate_weight(pixel_area, current_scale)
                    weight_text = f"{estimated_weight:.1f} kg"
                
                nome_da_classe = model.names[int(cls)].capitalize()
                label = f"{nome_da_classe} | {weight_text}"
                
                # Desenhar etiqueta elegante em cima do animal
                draw_hud_box(annotated_frame, label, (x1, max(30, y1 - 15)), bg_color=(0,0,0))
                print(f"         - Objeto {gado_count}: {nome_da_classe} (Area: {pixel_area:.1f}) -> {weight_text}")

    draw_hud_box(annotated_frame, f"Total Detectado: {gado_count}", (20, 80), bg_color=(50, 50, 50), text_color=(0, 255, 0))

    cv2.imwrite("resultado_frame.jpg", annotated_frame)
    print("\nCONCLUIDO! Imagem salva. Feche o VS Code ou abra o arquivo resultado_frame.jpg para ver o novo visual.\n")

if __name__ == "__main__":
    main()
