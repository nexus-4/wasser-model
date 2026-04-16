from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_MODEL_PATH = "yolo26x.pt"
DEFAULT_VIDEO_PATH = "media/video-teste-wasser.mp4"
DEFAULT_TRACKER_PATH = "wasser_tracker.yaml"
DEFAULT_OUTPUT_PATH = "resultado_wasser.mp4"
DEFAULT_INFERENCE_SIZE = 640
COW_CLASS_ID = 19

DEFAULT_CATTLE_NAMES = {
    1: "Mimosa",
    2: "Biscoito",
    3: "Pipoca",
    4: "Bolinha",
    5: "Urso",
    6: "Pe de Pano",
}


def load_model(model_path=DEFAULT_MODEL_PATH):
    return YOLO(model_path)


def get_color_and_status(conf):
    if conf >= 0.8:
        return (0, 255, 0), "High"
    if conf >= 0.5:
        return (0, 255, 255), "Medium"
    return (0, 0, 255), "Low"


def draw_legend(frame):
    x, y = 20, 86
    line_height = 30

    cv2.rectangle(frame, (x - 10, y - 35), (430, y + 3 * line_height + 10), (0, 0, 0), -1)
    cv2.putText(frame, "Confidence Legend", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "GREEN >= 80% (HIGH)", (x, y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, "YELLOW >= 50% (MEDIUM)", (x, y + 2 * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, "RED < 50% (LOW)", (x, y + 3 * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def draw_counter(frame, count, mode="acumulado"):
    label = "Total Detected" if mode == "acumulado" else "Current Frame"
    cv2.rectangle(frame, (10, 10), (410, 62), (0, 0, 0), -1)
    cv2.putText(frame, f"{label}: {count}", (20, 47), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)


def draw_detection(frame, box_xyxy, track_id, conf, cattle_names=None):
    cattle_names = cattle_names or DEFAULT_CATTLE_NAMES
    color, status = get_color_and_status(float(conf))
    x1, y1, x2, y2 = map(int, box_xyxy)
    name = cattle_names.get(track_id, f"Cattle {track_id}")

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        f"{name} ID:{track_id} {conf * 100:.1f}% {status}",
        (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def add_detection_to_heatmap(heatmap_accum, box_xyxy):
    x1, y1, x2, y2 = map(int, box_xyxy)
    center_x = max(0, min(heatmap_accum.shape[1] - 1, (x1 + x2) // 2))
    center_y = max(0, min(heatmap_accum.shape[0] - 1, (y1 + y2) // 2))
    radius = max(14, min(heatmap_accum.shape[:2]) // 28)
    cv2.circle(heatmap_accum, (center_x, center_y), radius, 1.0, -1)


def render_activity_heatmap(heatmap_accum, background_frame):
    if heatmap_accum.max() <= 0:
        heatmap_accum[:] = 0

    blurred = cv2.GaussianBlur(heatmap_accum, (0, 0), sigmaX=25, sigmaY=25)
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    if background_frame is None:
        background_frame = np.zeros((*heatmap_accum.shape, 3), dtype=np.uint8)

    overlay = cv2.addWeighted(background_frame, 0.55, color_heatmap, 0.45, 0)
    return overlay


def save_activity_heatmap(heatmap_accum, background_frame, output_heatmap_path):
    overlay = render_activity_heatmap(heatmap_accum, background_frame)
    cv2.imwrite(str(output_heatmap_path), overlay)


def validate_processing_inputs(video_path, model_path, tracker_path):
    missing = []
    for label, path in (("video", video_path), ("tracker", tracker_path)):
        if not Path(path).exists():
            missing.append(f"{label}: {path}")

    if missing:
        raise FileNotFoundError("Required files not found: " + "; ".join(missing))


def process_video(
    video_path=DEFAULT_VIDEO_PATH,
    output_path=DEFAULT_OUTPUT_PATH,
    model_path=DEFAULT_MODEL_PATH,
    tracker_path=DEFAULT_TRACKER_PATH,
    model=None,
    counter_mode="acumulado",
    show_window=False,
    progress_callback=None,
    preview_callback=None,
    output_heatmap_path=None,
    imgsz=DEFAULT_INFERENCE_SIZE,
    preview_interval=5,
):
    validate_processing_inputs(video_path, model_path, tracker_path)

    model = model or load_model(model_path)
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = str(output_path)
    output_heatmap_path = output_heatmap_path or str(Path(output_path).with_name(f"{Path(output_path).stem}_heatmap.jpg"))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        video.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    counted_ids = set()
    last_frame_count = 0
    frames_processed = 0
    heatmap_accum = np.zeros((height, width), dtype=np.float32)
    heatmap_background = None

    try:
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            if heatmap_background is None:
                heatmap_background = frame.copy()

            results = model.track(
                frame,
                persist=True,
                tracker=tracker_path,
                classes=[COW_CLASS_ID],
                imgsz=imgsz,
                verbose=False,
            )

            annotated_frame = frame.copy()
            frame_track_ids = []
            result = results[0]

            if result.boxes is not None and result.boxes.id is not None:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()
                confs = result.boxes.conf.cpu().numpy()
                frame_track_ids = track_ids

                for box_xyxy, track_id, conf in zip(boxes_xyxy, track_ids, confs):
                    counted_ids.add(track_id)
                    add_detection_to_heatmap(heatmap_accum, box_xyxy)
                    draw_detection(annotated_frame, box_xyxy, track_id, conf)

            last_frame_count = len(frame_track_ids)
            display_count = len(counted_ids) if counter_mode == "acumulado" else last_frame_count
            draw_counter(annotated_frame, display_count, counter_mode)
            draw_legend(annotated_frame)
            out.write(annotated_frame)

            frames_processed += 1
            if progress_callback and total_frames:
                progress_callback(min(frames_processed / total_frames, 1.0))
            if preview_callback and frames_processed % preview_interval == 0:
                heatmap_preview = render_activity_heatmap(heatmap_accum, heatmap_background)
                preview_callback(annotated_frame, display_count, heatmap_preview)

            if show_window:
                cv2.imshow("Wasser Tracking", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        out.release()
        video.release()
        if show_window:
            cv2.destroyAllWindows()

    save_activity_heatmap(heatmap_accum, heatmap_background, output_heatmap_path)

    return {
        "output_path": output_path,
        "heatmap_path": str(output_heatmap_path),
        "total_unique": len(counted_ids),
        "last_frame_count": last_frame_count,
        "frames_processed": frames_processed,
        "width": width,
        "height": height,
        "fps": fps,
    }
