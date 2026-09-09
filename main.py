import cv2
import numpy as np
from ultralytics import YOLO
from utils.aruco_detector import ArucoScaleEstimator
from utils.weight_calculator import WeightCalculator

def main():
    print("Iniciando Wasser tracking com estimativa de peso (ArUco + Segmentacao)...")

    # Configurações de inicialização
    video_path = "videos/video-teste-wasser.mp4" # Agora usando a pasta videos nativa
    model_path = "yolov8n-seg.pt"

    model = YOLO(model_path)

    scale_estimator = ArucoScaleEstimator(marker_size_cm=50.0)
    weight_calc = WeightCalculator(calibration_weight_kg=500.0, base_calibration_area_m2=2.0)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detectar ArUco e atualizar escala
        current_scale, frame = scale_estimator.get_scale(frame)

        # 2. Inferência com YOLO-Seg (apenas classe cow = 19)
        results = model.track(frame, persist=True, classes=[19], verbose=False)
        result = results[0]

        gado_count = 0

        if result.boxes is not None and result.masks is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else [None] * len(boxes)
            masks = result.masks.xy

            for box, track_id, polygon in zip(boxes, track_ids, masks):
                gado_count += 1
                x1, y1, x2, y2 = map(int, box)

                # Desenhar contorno da máscara
                if len(polygon) > 0:
                    cv2.polylines(frame, [polygon.astype(np.int32)], True, (0, 255, 0), 2)

                    # Calcular área em pixels da máscara
                    pixel_area = cv2.contourArea(polygon)

                    weight_text = "N/A"
                    if current_scale is not None:
                        weight_kg = weight_calc.calculate_weight(pixel_area, current_scale)
                        weight_text = f"{weight_kg:.1f} kg"

                    # Desenhar ID e Peso
                    label = f"ID: {int(track_id) if track_id else 'N/A'} - {weight_text}"
                    cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # HUD com informações globais
        scale_text = f"Escala: {current_scale:.3f} cm/px" if current_scale else "Escala: Pendente"
        cv2.putText(frame, scale_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Total Gado Visivel: {gado_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Wasser - Pesagem e Rastreamento", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Processamento finalizado!")

if __name__ == "__main__":
    main()
