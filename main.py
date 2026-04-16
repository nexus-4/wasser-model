from processor import (
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TRACKER_PATH,
    DEFAULT_VIDEO_PATH,
    process_video,
)


def main():
    print("Starting Wasser tracking...")
    result = process_video(
        video_path=DEFAULT_VIDEO_PATH,
        output_path=DEFAULT_OUTPUT_PATH,
        model_path=DEFAULT_MODEL_PATH,
        tracker_path=DEFAULT_TRACKER_PATH,
        counter_mode="acumulado",
        show_window=True,
    )
    print(f"Processing complete. Video saved as '{result['output_path']}'")
    print(f"Total unique animals counted: {result['total_unique']}")


if __name__ == "__main__":
    main()
