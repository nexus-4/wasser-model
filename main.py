import cv2
from ultralytics import YOLO

def main():
    print("Inicializando Sistema Wasser...")

    MODEL_PATH   = "yolo26x.pt" 
    VIDEO_PATH   = "media/video-teste-wasser.mp4"
    TRACKER_PATH = "wasser_tracker.yaml"

    # Carregar o modelo YOLOv10x
    model = YOLO(MODEL_PATH)
    cap   = cv2.VideoCapture(VIDEO_PATH)

    # Pegamos a largura, altura e FPS originais do vídeo para o arquivo final ficar com a mesma qualidade
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Criamos o arquivo de saída 'resultado_wasser.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec de vídeo
    out    = cv2.VideoWriter('resultado_wasser.mp4', fourcc, fps, (width, height))

    gado_contado = set()
    nome_gado = {
        1: "Mimosa",
        2: "Biscoito",
        3: "Pipoca",
        4: "Bolinha"
    }

    print("Pronto para iniciar o rastreamento...")

    while cap.isOpened():
        success, frame = cap.read()
        
        if success:
            # Executar o rastreamento
            results = model.track(
                frame, 
                persist=True, 
                tracker=TRACKER_PATH, 
                classes=[19],
                verbose=False,
            )
            
            result = results[0]
            annotated_frame = frame.copy() 

            if result.boxes is not None and result.boxes.id is not None:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                track_ids  = result.boxes.id.int().cpu().tolist()

                for box_xyxy, track_id in zip(boxes_xyxy, track_ids):
                    
                    # 1. Registrar na contagem única
                    gado_contado.add(track_id) 
                    
                    # 2. Pegar o nome do animal
                    nome = nome_gado.get(track_id, f"Gado {track_id}") 

                    # 3. Desenhar a caixa e o nome na tela
                    x1, y1, x2, y2 = map(int, box_xyxy)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"{nome} (ID:{track_id})", (x1, max(0, y1 - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Exibir o total no canto da tela
            cv2.putText(annotated_frame, f"Total Unico Contado: {len(gado_contado)}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Salvando o frame rabiscado no nosso arquivo de vídeo ---
            out.write(annotated_frame)

            cv2.imshow("Wasser Tracking", annotated_frame)

            # Quebrar o loop se 'q' for pressionado
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    # Fechando e salvando o arquivo final
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Processamento concluído. Vídeo salvo como 'resultado_wasser.mp4'")

if __name__ == "__main__":
    main()