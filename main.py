import argparse

from processor import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TRACKER_PATH,
    DEFAULT_VIDEO_PATH,
    process_video,
)


def main():
    parser = argparse.ArgumentParser(description="Contagem de gado em vídeo")
    parser.add_argument("--pasto", action="store_true", help="conta animais visíveis por frame no pasto")
    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--model", help="caminho do peso YOLO")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    result = process_video(
        video_path=args.video,
        output_path=args.output,
        model_path=args.model,
        tracker_path=DEFAULT_TRACKER_PATH,
        counter_mode="pasto" if args.pasto else "acumulado",
        show_window=not args.headless,
    )
    print(f"Processing complete. Video saved as '{result['output_path']}'")
    if args.pasto:
        print(f"Animals detected in last frame: {result['last_frame_count']}")
    else:
        print(f"Total unique animals counted: {result['total_unique']}")


if __name__ == "__main__":
    main()
