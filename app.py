from pathlib import Path
from textwrap import dedent
import tempfile

import streamlit as st

from processor import (
    DEFAULT_INFERENCE_SIZE,
    DEFAULT_MODEL_PATH,
    PASTURE_MODEL_PATH,
    PASTURE_INFERENCE_SIZE,
    DEFAULT_TRACKER_PATH,
    prepare_browser_video,
    process_video,
    resolve_device,
)


st.set_page_config(page_title="Vigilio Monitor", layout="wide", initial_sidebar_state="collapsed")


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


def discard_temp_file(path):
    """Remove um temporario sem quebrar o app se ele ja sumiu ou esta em uso."""
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def discard_previous_results():
    """Limpa a saida da rodada anterior antes de comecar uma nova."""
    discard_temp_file(st.session_state.get("processed_video_path"))
    discard_temp_file(st.session_state.get("heatmap_path"))
    st.session_state.processed_video_path = None
    st.session_state.heatmap_path = None


def validate_app_inputs(tracker_path):
    missing = []
    if tracker_path is not None and not Path(tracker_path).exists():
        missing.append(f"Tracker not found: {tracker_path}")
    return missing


def render_html(markup):
    st.markdown(dedent(markup).strip(), unsafe_allow_html=True)


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
with st.sidebar:
    st.header("Technical Configuration")
    modes = {"Accumulated total": "acumulado", "Current frame": "frame",
             "Drone pasture (per frame)": "pasto"}
    counter_mode = modes[st.radio("Counter mode", options=list(modes))]
    pasture = counter_mode == "pasto"
    model_path = st.text_input("YOLO model", value=PASTURE_MODEL_PATH if pasture else DEFAULT_MODEL_PATH,
                               key="pasture_model" if pasture else "tracking_model")
    tracker_path = None if pasture else st.text_input("BoT-SORT tracker", value=DEFAULT_TRACKER_PATH)
    sizes = (416, 512, 640, 960, 1280)
    inference_size = st.selectbox("Inference size", options=sizes,
                                  index=sizes.index(PASTURE_INFERENCE_SIZE if pasture else DEFAULT_INFERENCE_SIZE),
                                  key="pasture_size" if pasture else "tracking_size")

    device_label = st.radio(
        "Processing device",
        options=("Auto", "GPU (CUDA)", "GPU (Apple MPS)", "CPU"),
        index=0,
        help="Auto usa CUDA, depois MPS, depois CPU. CUDA exige o extra cu130 instalado.",
    )
    device = {
        "Auto": None,
        "GPU (CUDA)": "cuda",
        "GPU (Apple MPS)": "mps",
        "CPU": "cpu",
    }[device_label]
    st.caption(f"Em uso: {resolve_device(device)}")

render_html(
    """
    <div class="top-header">
        <div class="brand-wrap">
            <div class="brand-title">Vigilio Monitor</div>
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
                if st.session_state.get("browser_video_ready", False):
                    st.video(st.session_state.processed_video_path)
                else:
                    st.info("Browser preview unavailable. Download the processed video below.")
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
    total_col, upload_col, export_col = st.columns([1.25, 2, 1])

    with total_col:
        count_metric = st.empty()
        count_metric.metric("Animals detected", st.session_state.detected_count)

    with upload_col:
        toolbar_uploaded_video = st.file_uploader(
            "Video source",
            type=("mp4", "avi", "mov"),
            label_visibility="collapsed",
            key="toolbar_video_source",
        )

    uploaded_video = toolbar_uploaded_video or drop_uploaded_video

    with export_col:
        process_clicked = st.button("Export Video", type="primary", disabled=uploaded_video is None)

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

    progress_bar = progress_placeholder.progress(0)
    input_path = save_uploaded_video(uploaded_video)
    output_path = create_output_path()
    heatmap_path = create_heatmap_output_path()

    def update_progress(value):
        progress_bar.progress(value)

    def update_preview(frame, display_count, heatmap_frame):
        count_metric.metric("Animals detected", display_count)
        rgb_frame = frame[:, :, ::-1]
        rgb_heatmap = heatmap_frame[:, :, ::-1]
        with left_view.container():
            st.image(rgb_frame, channels="RGB", use_container_width=True)
        with right_view.container():
            st.image(rgb_heatmap, channels="RGB", use_container_width=True)

    try:
        result = process_video(
            video_path=input_path,
            output_path=output_path,
            model_path=model_path,
            tracker_path=tracker_path,
            counter_mode=counter_mode,
            show_window=False,
            progress_callback=update_progress,
            preview_callback=update_preview,
            output_heatmap_path=heatmap_path,
            imgsz=inference_size,
            device=device,
        )
        progress_placeholder.caption("Preparing video for playback…")
        browser_video_ready = prepare_browser_video(output_path)
    except Exception as exc:
        discard_temp_file(output_path)
        discard_temp_file(heatmap_path)
        message_placeholder.error(f"Processing error: {exc}")
        st.stop()
    finally:
        # O video de entrada ja foi consumido; so a saida precisa sobreviver
        # ao rerun do Streamlit.
        discard_temp_file(input_path)

    discard_previous_results()
    st.session_state.browser_video_ready = browser_video_ready
    st.session_state.processed_video_path = output_path
    st.session_state.heatmap_path = result["heatmap_path"]
    st.session_state.detected_count = (
        result["total_unique"] if counter_mode == "acumulado" else result["last_frame_count"]
    )
    st.rerun()
