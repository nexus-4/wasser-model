from pathlib import Path
from textwrap import dedent
import tempfile

import streamlit as st

from processor import (
    DEFAULT_INFERENCE_SIZE,
    DEFAULT_MODEL_PATH,
    DEFAULT_TRACKER_PATH,
    load_model,
    process_video,
)


st.set_page_config(page_title="Vigilio Monitor", layout="wide", initial_sidebar_state="collapsed")


@st.cache_resource
def get_cached_model(model_path):
    return load_model(model_path)


def save_uploaded_video(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_input.write(uploaded_file.getbuffer())
    temp_input.close()
    return temp_input.name


def create_output_path():
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_output.close()
    return temp_output.name


def create_heatmap_output_path():
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_output.close()
    return temp_output.name


def validate_app_inputs(tracker_path):
    missing = []
    if not Path(tracker_path).exists():
        missing.append(f"Tracker not found: {tracker_path}")
    return missing


def render_html(markup):
    st.markdown(dedent(markup).strip(), unsafe_allow_html=True)


def render_farm_placeholder():
    render_html(
        """
        <div class="farm-view">
            <div class="farm-sky"></div>
            <div class="farm-field"></div>
            <div class="farm-fence fence-a"></div>
            <div class="farm-fence fence-b"></div>

            <div class="cow cow-a">
                <span class="bbox box-green"></span><span class="cow-body"></span><span class="cow-head"></span>
            </div>
            <div class="cow cow-b">
                <span class="bbox box-green"></span><span class="cow-body"></span><span class="cow-head"></span>
            </div>
            <div class="cow cow-c">
                <span class="bbox box-red"></span><span class="cow-body"></span><span class="cow-head"></span>
            </div>
            <div class="cow cow-d">
                <span class="bbox box-green"></span><span class="cow-body"></span><span class="cow-head"></span>
            </div>

            <div class="detection-card">
                <div class="detected-count">23 Animals Detected</div>
                <div class="confidence-row"><span class="confidence-dot"></span>Confidence: 92%</div>
            </div>

            <div class="playback-bar">
                <div class="playback-controls">
                    <span class="play-icon"></span><span class="control-line"></span><span class="control-line short"></span>
                </div>
                <div class="video-progress"><span></span></div>
                <div class="timestamp">01:24 / 04:10</div>
            </div>
        </div>
        """
    )


def render_upload_notice():
    render_html(
        """
        <div class="upload-notice">
            Fa&ccedil;a o upload do v&iacute;deo clicando no bot&atilde;o ou arraste o v&iacute;deo para c&aacute;
        </div>
        """
    )


def render_heatmap_placeholder():
    render_html(
        """
        <div class="map-view">
            <div class="map-dropdown">Activity Heatmap</div>
            <div class="heatmap-empty">O mapa de calor real sera gerado apos a analise do video.</div>
        </div>
        """
    )


render_html(
    """
    <style>
        :root {
            --bg: #050912;
            --panel: rgba(12, 18, 31, 0.82);
            --panel-strong: rgba(8, 13, 24, 0.94);
            --line: rgba(106, 132, 173, 0.28);
            --text: #f5f8ff;
            --muted: #93a4bd;
            --green: #22f06b;
            --blue: #2387ff;
            --red: #ff385c;
        }

        .stApp {
            background:
                radial-gradient(circle at 22% 16%, rgba(28, 119, 255, 0.18), transparent 30%),
                radial-gradient(circle at 80% 0%, rgba(34, 240, 107, 0.12), transparent 28%),
                linear-gradient(135deg, #03060d 0%, #08111f 48%, #03060a 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1920px;
            padding: 1.1rem 2rem 1.4rem;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .top-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            min-height: 72px;
            padding: 0 1.25rem;
            margin-bottom: 1rem;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: linear-gradient(90deg, rgba(9, 15, 28, 0.94), rgba(14, 32, 58, 0.88), rgba(6, 12, 22, 0.94));
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.35);
            backdrop-filter: blur(18px);
        }

        .brand-wrap {
            display: flex;
            align-items: center;
            gap: 1.2rem;
        }

        .brand-title {
            font-size: 1.55rem;
            line-height: 1;
            font-weight: 800;
            color: var(--text);
        }

        .processing-pill,
        .sync-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            color: #dfffea;
            font-weight: 700;
        }

        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--green);
            box-shadow: 0 0 16px rgba(34, 240, 107, 0.9);
        }

        .header-icons {
            display: flex;
            gap: 0.6rem;
        }

        .header-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 44px;
            height: 38px;
            border: 1px solid rgba(127, 153, 190, 0.24);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.05);
            color: #dce7f7;
            font-size: 0.78rem;
            font-weight: 800;
        }

        .st-key-left_panel,
        .st-key-right_panel {
            aspect-ratio: 1 / 1;
            min-height: 520px;
            padding: 0.75rem;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--panel);
            box-shadow: 0 26px 70px rgba(0, 0, 0, 0.42), inset 0 0 0 1px rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(18px);
            overflow: hidden;
        }

        .st-key-left_panel video,
        .st-key-left_panel img {
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.12);
        }

        .st-key-video_upload_zone {
            position: relative;
            min-height: 490px;
            aspect-ratio: 1 / 1;
            border: 1px dashed rgba(34, 240, 107, 0.48);
            border-radius: 8px;
            background:
                radial-gradient(circle at 50% 40%, rgba(35, 135, 255, 0.16), transparent 28%),
                rgba(5, 10, 18, 0.72);
            box-shadow: inset 0 0 60px rgba(35, 135, 255, 0.08);
        }

        .upload-notice {
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            max-width: 420px;
            width: 72%;
            margin: 0;
            color: rgba(245, 248, 255, 0.48);
            font-size: 1.05rem;
            line-height: 1.45;
            font-weight: 700;
            text-align: center;
            z-index: 1;
        }

        .st-key-video_upload_zone [data-testid="stFileUploader"] {
            position: absolute;
            inset: 0;
            z-index: 3;
            width: 100%;
            height: 100%;
            opacity: 0;
            margin: 0;
        }

        .st-key-video_upload_zone [data-testid="stFileUploader"] section {
            min-height: 490px;
            height: 100%;
        }

        .st-key-video_upload_zone [data-testid="stFileUploader"] small {
            display: none;
        }

        .st-key-video_upload_zone [data-testid="stFileUploaderDropzoneInstructions"] {
            display: none;
        }

        .farm-view,
        .map-view {
            position: relative;
            width: 100%;
            aspect-ratio: 1 / 1;
            min-height: 490px;
            overflow: hidden;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: #111927;
        }

        .farm-sky {
            position: absolute;
            inset: 0 0 45% 0;
            background:
                radial-gradient(circle at 72% 22%, rgba(255, 255, 255, 0.65), transparent 8%),
                linear-gradient(180deg, #253b55 0%, #142536 100%);
        }

        .farm-field {
            position: absolute;
            inset: 38% 0 0 0;
            background:
                linear-gradient(12deg, rgba(22, 240, 107, 0.08) 0 10%, transparent 10% 18%, rgba(35, 135, 255, 0.08) 18% 25%, transparent 25%),
                linear-gradient(180deg, #20391f 0%, #102410 100%);
        }

        .farm-fence {
            position: absolute;
            left: -8%;
            width: 116%;
            height: 2px;
            background: rgba(230, 236, 245, 0.5);
            transform: rotate(-4deg);
        }

        .fence-a { top: 49%; }
        .fence-b { top: 55%; }

        .cow {
            position: absolute;
            width: 118px;
            height: 70px;
        }

        .cow-a { left: 13%; top: 48%; transform: scale(1.05); }
        .cow-b { left: 53%; top: 43%; transform: scale(0.96); }
        .cow-c { left: 68%; top: 65%; transform: scale(1.08); }
        .cow-d { left: 27%; top: 67%; transform: scale(0.9); }

        .cow-body {
            position: absolute;
            left: 22px;
            top: 22px;
            width: 72px;
            height: 36px;
            border-radius: 48% 42% 45% 45%;
            background:
                radial-gradient(circle at 28% 42%, #171717 0 14%, transparent 15%),
                radial-gradient(circle at 67% 45%, #1b1b1b 0 12%, transparent 13%),
                #f3f0e8;
        }

        .cow-head {
            position: absolute;
            left: 88px;
            top: 18px;
            width: 26px;
            height: 28px;
            border-radius: 50%;
            background: #eee8dc;
        }

        .cow-body:before,
        .cow-body:after {
            content: "";
            position: absolute;
            bottom: -18px;
            width: 7px;
            height: 18px;
            border-radius: 4px;
            background: #e5dfd4;
        }

        .cow-body:before { left: 15px; }
        .cow-body:after { right: 14px; }

        .bbox {
            position: absolute;
            inset: 8px 0 0 8px;
            border-radius: 4px;
            z-index: 3;
        }

        .box-green {
            border: 2px solid var(--green);
            box-shadow: 0 0 14px rgba(34, 240, 107, 0.75);
        }

        .box-red {
            border: 2px solid var(--red);
            box-shadow: 0 0 14px rgba(255, 56, 92, 0.75);
        }

        .detection-card {
            position: absolute;
            top: 24px;
            left: 24px;
            min-width: 260px;
            padding: 1rem 1.1rem;
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 8px;
            background: rgba(5, 10, 18, 0.72);
            box-shadow: 0 12px 34px rgba(0, 0, 0, 0.32);
            backdrop-filter: blur(14px);
        }

        .detected-count {
            font-size: 1.55rem;
            line-height: 1.2;
            font-weight: 900;
            color: #ffffff;
        }

        .confidence-row {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-top: 0.5rem;
            color: #d9ffe5;
            font-weight: 700;
        }

        .confidence-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--green);
            box-shadow: 0 0 12px rgba(34, 240, 107, 0.9);
        }

        .playback-bar {
            position: absolute;
            left: 20px;
            right: 20px;
            bottom: 20px;
            display: flex;
            align-items: center;
            gap: 0.9rem;
            padding: 0.75rem 0.9rem;
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 8px;
            background: rgba(3, 8, 15, 0.78);
            backdrop-filter: blur(14px);
        }

        .playback-controls {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            min-width: 96px;
        }

        .play-icon {
            width: 0;
            height: 0;
            border-top: 8px solid transparent;
            border-bottom: 8px solid transparent;
            border-left: 13px solid #ffffff;
        }

        .control-line {
            width: 24px;
            height: 6px;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.55);
        }

        .control-line.short { width: 14px; }

        .video-progress {
            flex: 1;
            height: 8px;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.18);
            overflow: hidden;
        }

        .video-progress span {
            display: block;
            width: 44%;
            height: 100%;
            border-radius: 8px;
            background: linear-gradient(90deg, var(--blue), var(--green));
        }

        .timestamp {
            color: #dce7f7;
            font-size: 0.82rem;
            font-weight: 700;
        }

        .map-view {
            background:
                linear-gradient(135deg, #142031 0%, #07101d 52%, #102235 100%);
        }

        .map-view:before {
            content: "";
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(rgba(255, 255, 255, 0.07) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.06) 1px, transparent 1px);
            background-size: 54px 54px;
            opacity: 0.42;
        }

        .map-dropdown {
            position: absolute;
            top: 24px;
            left: 24px;
            z-index: 3;
            padding: 0.75rem 1rem;
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 8px;
            background: rgba(5, 10, 18, 0.74);
            color: #ffffff;
            font-weight: 800;
            backdrop-filter: blur(14px);
        }

        .heatmap-empty {
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            width: 70%;
            color: rgba(245, 248, 255, 0.52);
            font-size: 1.05rem;
            line-height: 1.45;
            font-weight: 700;
            text-align: center;
            z-index: 5;
        }

        .map-road {
            position: absolute;
            z-index: 1;
            height: 8px;
            border-radius: 999px;
            background: rgba(194, 205, 218, 0.16);
            transform-origin: left center;
        }

        .road-a { width: 82%; left: 5%; top: 34%; transform: rotate(18deg); }
        .road-b { width: 72%; left: 10%; top: 64%; transform: rotate(-21deg); }
        .road-c { width: 55%; left: 38%; top: 50%; transform: rotate(66deg); }

        .heat {
            position: absolute;
            z-index: 2;
            border-radius: 50%;
            filter: blur(16px);
            opacity: 0.86;
        }

        .heat-a { width: 220px; height: 220px; left: 18%; top: 26%; background: rgba(35, 135, 255, 0.72); }
        .heat-b { width: 250px; height: 250px; left: 45%; top: 35%; background: rgba(34, 240, 107, 0.48); }
        .heat-c { width: 230px; height: 230px; left: 42%; top: 50%; background: rgba(255, 202, 40, 0.5); }
        .heat-d { width: 170px; height: 170px; left: 51%; top: 57%; background: rgba(255, 56, 92, 0.72); }

        .map-pin {
            position: absolute;
            z-index: 4;
            width: 14px;
            height: 14px;
            border: 3px solid #ffffff;
            border-radius: 50%;
            background: var(--green);
            box-shadow: 0 0 16px rgba(34, 240, 107, 0.9);
        }

        .pin-a { left: 42%; top: 43%; }
        .pin-b { left: 58%; top: 62%; background: var(--red); box-shadow: 0 0 16px rgba(255, 56, 92, 0.9); }

        .map-label {
            position: absolute;
            right: 24px;
            bottom: 24px;
            z-index: 5;
            padding: 0.65rem 0.9rem;
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 8px;
            background: rgba(5, 10, 18, 0.72);
            color: #dce7f7;
            font-weight: 800;
        }

        .st-key-bottom_toolbar {
            margin-top: 1rem;
            min-height: 84px;
            padding: 1rem 1.25rem;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(8, 13, 24, 0.9);
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.36);
            backdrop-filter: blur(16px);
        }

        .st-key-bottom_toolbar [data-testid="stMetric"] {
            padding: 0;
        }

        .st-key-bottom_toolbar div[data-testid="stMetricValue"] {
            color: #ffffff;
            font-size: 1.25rem;
        }

        .st-key-bottom_toolbar [data-testid="stFileUploader"] section {
            min-height: 42px;
            padding: 0.25rem 0.5rem;
            border: 1px solid rgba(127, 153, 190, 0.26);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.05);
        }

        .st-key-bottom_toolbar [data-testid="stFileUploader"] small {
            display: none;
        }

        .sync-check {
            display: inline-block;
            width: 18px;
            height: 18px;
            border-right: 4px solid var(--green);
            border-bottom: 4px solid var(--green);
            transform: rotate(45deg);
            margin-left: 0.3rem;
            filter: drop-shadow(0 0 8px rgba(34, 240, 107, 0.8));
        }

        .stButton > button {
            border-radius: 8px;
            border: 1px solid rgba(83, 139, 255, 0.55);
            background: linear-gradient(180deg, #246bff, #123c9c);
            color: white;
            font-weight: 900;
            min-height: 44px;
        }

        .stDownloadButton > button {
            border-radius: 8px;
            border: 1px solid rgba(34, 240, 107, 0.5);
            background: rgba(34, 240, 107, 0.12);
            color: #dcffe6;
            font-weight: 800;
        }
    </style>
    """
)

if "processed_video_path" not in st.session_state:
    st.session_state.processed_video_path = None
if "heatmap_path" not in st.session_state:
    st.session_state.heatmap_path = None
if "detected_count" not in st.session_state:
    st.session_state.detected_count = 0
if "status_label" not in st.session_state:
    st.session_state.status_label = "Processing"

with st.sidebar:
    st.header("Technical Configuration")
    model_path = st.text_input("YOLO model", value=DEFAULT_MODEL_PATH)
    tracker_path = st.text_input("BoT-SORT tracker", value=DEFAULT_TRACKER_PATH)
    inference_size = st.selectbox(
        "Inference size",
        options=(416, 512, 640, 960, 1280),
        index=(416, 512, 640, 960, 1280).index(DEFAULT_INFERENCE_SIZE),
        help="Lower values process faster. The exported video keeps the original resolution.",
    )
    counter_mode_label = st.radio(
        "Counter mode",
        options=("Accumulated total", "Current frame"),
        index=0,
    )
    counter_mode = "acumulado" if counter_mode_label == "Accumulated total" else "frame"

render_html(
    """
    <div class="top-header">
        <div class="brand-wrap">
            <div class="brand-title">Vigilio Monitor</div>
            <div class="processing-pill"><span class="status-dot"></span><span>Processing</span></div>
        </div>
        <div class="header-icons">
            <span class="header-icon">SET</span>
            <span class="header-icon">NET</span>
            <span class="header-icon">ALR</span>
        </div>
    </div>
    """
)

left_col, right_col = st.columns(2, gap="large")
drop_uploaded_video = None

with left_col:
    with st.container(key="left_panel"):
        left_view = st.empty()
        with left_view.container():
            if st.session_state.processed_video_path:
                st.video(st.session_state.processed_video_path)
            else:
                with st.container(key="video_upload_zone"):
                    render_upload_notice()
                    drop_uploaded_video = st.file_uploader(
                        "Video source",
                        type=("mp4", "avi", "mov"),
                        label_visibility="collapsed",
                        key="drop_video_source",
                    )

with right_col:
    with st.container(key="right_panel"):
        right_view = st.empty()
        with right_view.container():
            if st.session_state.heatmap_path:
                st.image(st.session_state.heatmap_path, use_container_width=True)
            else:
                render_heatmap_placeholder()

with st.container(key="bottom_toolbar"):
    total_col, upload_col, filter_col, export_col, sync_col = st.columns([1.25, 1.55, 1.1, 1.1, 1.1])

    with total_col:
        st.metric("Total Detected", f"{st.session_state.detected_count} Animals")

    with upload_col:
        toolbar_uploaded_video = st.file_uploader(
            "Video source",
            type=("mp4", "avi", "mov"),
            label_visibility="collapsed",
            key="toolbar_video_source",
        )

    with filter_col:
        st.selectbox(
            "Time filter",
            options=("Last 15 Minutes", "Last 30 Minutes", "Last 2 Hours"),
            index=2,
            label_visibility="collapsed",
        )

    uploaded_video = toolbar_uploaded_video or drop_uploaded_video

    with export_col:
        process_clicked = st.button("Export Video", type="primary", disabled=uploaded_video is None)

    with sync_col:
        render_html(
            """
            <div class="sync-pill">
                <span class="status-dot"></span>
                <span>Synchronized</span>
                <span class="sync-check"></span>
            </div>
            """
        )

progress_placeholder = st.empty()
message_placeholder = st.empty()

if st.session_state.processed_video_path:
    processed_data = Path(st.session_state.processed_video_path).read_bytes()
    st.download_button(
        "Download processed video",
        data=processed_data,
        file_name="resultado_wasser.mp4",
        mime="video/mp4",
        use_container_width=True,
    )

if process_clicked and uploaded_video is not None:
    missing_inputs = validate_app_inputs(tracker_path)
    if missing_inputs:
        for message in missing_inputs:
            message_placeholder.error(message)
        st.stop()

    st.session_state.status_label = "Processing"
    progress_bar = progress_placeholder.progress(0)
    input_path = save_uploaded_video(uploaded_video)
    output_path = create_output_path()
    heatmap_path = create_heatmap_output_path()
    preview_state = {"frames": 0}

    def update_progress(value):
        progress_bar.progress(value)

    def update_preview(frame, display_count, heatmap_frame):
        preview_state["frames"] += 1

        st.session_state.detected_count = display_count
        rgb_frame = frame[:, :, ::-1]
        rgb_heatmap = heatmap_frame[:, :, ::-1]
        with left_view.container():
            st.image(rgb_frame, channels="RGB", use_container_width=True)
        with right_view.container():
            st.image(rgb_heatmap, channels="RGB", use_container_width=True)

    try:
        model = get_cached_model(model_path)
        result = process_video(
            video_path=input_path,
            output_path=output_path,
            model_path=model_path,
            tracker_path=tracker_path,
            model=model,
            counter_mode=counter_mode,
            show_window=False,
            progress_callback=update_progress,
            preview_callback=update_preview,
            output_heatmap_path=heatmap_path,
            imgsz=inference_size,
        )
    except Exception as exc:
        st.session_state.status_label = "Error"
        message_placeholder.error(f"Processing error: {exc}")
        st.stop()

    st.session_state.processed_video_path = output_path
    st.session_state.heatmap_path = result["heatmap_path"]
    st.session_state.detected_count = (
        result["total_unique"] if counter_mode == "acumulado" else result["last_frame_count"]
    )
    progress_bar.progress(1.0)
    st.rerun()
