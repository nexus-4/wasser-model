import cv2
from ultralytics import YOLO

def get_color_and_status(conf):
    # BGR no OpenCV
    if conf >= 0.8:
        return (0, 255, 0), "Alto"  # Verde para alta confiança
    elif conf >= 0.5:
        return (0, 255, 255), "Médio" # Amarelo para média confiança
    else:
        return (0, 0, 255), "Baixo"   # Vermelho para baixa confiança

def draw_legend(frame):
    x, y        = 20, 80
    line_height = 30

    # Fundo da legenda para melhor leitura
    cv2.rectangle(frame, (x - 10, y - 30), (420, y + 3 * line_height + 10), (0, 0, 0), - 1 )
    cv2.putText(frame, "Diagnostico por Confiança: ", (x, y - 8), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, "VERDE >= 80% (ALTA)", (x, y + 1 * line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.putText(frame, "AMARELO >= 50% (MÉDIA)", (x, y + 2 * line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.putText(frame, "VERMELHO < 50% (BAIXA)", (x, y + 3 * line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def main():
    
    print("Inicializando Sistema Wasser...")

    MODEL_PATH   = "yolo26x.pt" 
    VIDEO_PATH   = "media/video-teste-wasser.mp4"
    TRACKER_PATH = "wasser_tracker.yaml"

    # Carregar o modelo YOLOv10x
    model  = YOLO(MODEL_PATH)
    video  = cv2.VideoCapture(VIDEO_PATH)

    # Pegamos a largura, altura e FPS originais do vídeo para o arquivo final ficar com a mesma qualidade
    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(video.get(cv2.CAP_PROP_FPS))
    
    # Criamos o arquivo de saída 'resultado_wasser.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec de vídeo
    out    = cv2.VideoWriter('resultado_wasser.mp4', fourcc, fps, (width, height))

    gado_contado = set()
    nome_gado = {
        1: "Mimosa",
        2: "Biscoito",
        3: "Pipoca",
        4: "Bolinha",
        5: "Urso",
        6: "Pé de Pano",
    }

    print("Pronto para iniciar o rastreamento...")

    while video.isOpened():
        success, frame = video.read()
        
        if success:
            # Executar o rastreamento
            results = model.track(
                frame, 
                persist=True, 
                tracker=TRACKER_PATH, 
                classes=[19],
                verbose=False,
            )
            
            result          = results[0]
            annotated_frame = frame.copy() 

            if result.boxes is not None and result.boxes.id is not None:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                track_ids  = result.boxes.id.int().cpu().tolist()
                confs      = result.boxes.conf.cpu().numpy()

                for box_xyxy, track_id, conf in zip(boxes_xyxy, track_ids, confs):
                    
                    # 1. Registrar na contagem única
                    gado_contado.add(track_id)
                    
                    # 2. Pegar o nome do animal
                    nome = nome_gado.get(track_id, f"Gado {track_id}")

                    # 3. Determinar a cor e status com base na confiança
                    color, status = get_color_and_status(float(conf))

                    # 3. Desenhar a caixa e o nome na tela
                    x1, y1, x2, y2 = map(int, box_xyxy)

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                    cv2.putText(
                        annotated_frame,
                        f"{nome} (ID:{track_id}) {conf*100:.1f}% [{status}]",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        color, 
                        2
                    )

            # Exibir o total no canto da tela
            cv2.putText(annotated_frame, f"Total Unico Contado: {len(gado_contado)}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Desenhando a legenda explicativa no canto inferior esquerdo
            draw_legend(annotated_frame)

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
    video.release()
    cv2.destroyAllWindows()
    print(f"Processamento concluído. Vídeo salvo como 'resultado_wasser.mp4'")

if __name__ == "__main__":
    main()