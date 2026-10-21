# ============================================================
# STREAMLIT
# LUMPY SKIN DISEASE CLASSIFICATION
# SCENARIO MODEL SELECTOR + U2NET SEGMENTATION
# ============================================================

import os
import hashlib
import time
from pathlib import Path

import cv2
import h5py
import json
import numpy as np
import streamlit as st
from dashboard_style import apply_dashboard_style
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image
from streamlit.errors import StreamlitSecretNotFoundError
from rembg import new_session, remove
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Layer,
)
from tensorflow.keras.models import Model, load_model


# ============================================================
# PAGE CONFIG
# ============================================================

IMG_SIZE = (224, 224)
THRESHOLD = 0.5

CLASS_NAMES = {
    0: "Lumpy Skin",
    1: "Normal Skin",
}

SCENARIOS = {
    "Skenario 1 - VGG19 + Triple Attention": {
        "path": Path("Model/best_model.h5"),
        "filename": "best_model.h5",
        "architecture": "triple_attention",
        "description": "Dataset_NoBG, split 80/10/10, VGG19 + TripleAttention",
    },
    "Skenario 2 - VGG19": {
        "path": Path("Model/best_model2.h5"),
        "filename": "best_model2.h5",
        "architecture": "vgg19",
        "description": "Dataset_NoBG, split 80/10/10, VGG19",
    },
    "Skenario 3 - VGG19 + Triple Attention": {
        "path": Path("Model/best_model3.h5"),
        "filename": "best_model3.h5",
        "architecture": "triple_attention",
        "description": "Dataset_NoBG, split 70/20/10, VGG19 + TripleAttention",
    },
    "Skenario 4 - VGG19": {
        "path": Path("Model/best_model4.h5"),
        "filename": "best_model4.h5",
        "architecture": "vgg19",
        "description": "Dataset_NoBG, split 70/20/10, VGG19",
    },
}


# ============================================================
# HUGGING FACE HELPERS
# ============================================================


def get_hf_token():
    env_path = Path(__file__).resolve().parent / ".env"
    token = None

    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "TOKEN-HG":
                token = value.strip().strip('"').strip("'")
                break

    token = token or os.getenv("TOKEN-HG") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token
    try:
        return st.secrets.get("TOKEN-HG") or st.secrets.get("HF_TOKEN") or st.secrets.get("HUGGINGFACE_HUB_TOKEN")
    except StreamlitSecretNotFoundError:
        return None


def download_model_from_huggingface(model_filename, destination_path):
    repo_id = "erwinfirmansyah/ta-skripsi"
    token = get_hf_token()
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    candidate_files = [model_filename, f"model/{model_filename}"]
    last_error = None

    for candidate in candidate_files:
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=candidate,
                repo_type="model",
                token=token,
                local_dir=str(destination_path.parent),
            )

            downloaded_file = Path(downloaded_path)
            if downloaded_file != destination_path:
                if destination_path.exists():
                    destination_path.unlink()
                downloaded_file.replace(destination_path)

            return destination_path
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Gagal mengunduh model {model_filename} dari Hugging Face: {last_error}")


def resolve_model_path(scenario):
    base_dir = Path(__file__).resolve().parent
    cached_path = base_dir / ".cache" / "models" / scenario["filename"]
    local_path = base_dir / scenario["path"]
    if local_path.is_file():
        return local_path
    if cached_path.is_file():
        return cached_path
    return download_model_from_huggingface(scenario["filename"], cached_path)


# ============================================================
# CUSTOM LAYERS
# ============================================================

class ChannelAttention(Layer):
    def __init__(self, reduction_ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.fc1 = Dense(
            channels // self.reduction_ratio,
            activation="relu",
            use_bias=False,
        )
        self.fc2 = Dense(channels, activation="sigmoid", use_bias=False)

    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=[1, 2])
        max_pool = tf.reduce_max(x, axis=[1, 2])

        avg_out = self.fc2(self.fc1(avg_pool))
        max_out = self.fc2(self.fc1(max_pool))

        scale = avg_out + max_out
        scale = tf.reshape(scale, (-1, 1, 1, x.shape[-1]))

        return x * scale

    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config


class SpatialAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = Conv2D(
            filters=1,
            kernel_size=7,
            padding="same",
            activation="sigmoid",
            use_bias=False,
        )

    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        return x * self.conv(concat)


class TripleAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.channel_att = ChannelAttention()
        self.spatial_att = SpatialAttention()

    def call(self, x):
        x_c = self.channel_att(x)

        x_h = tf.transpose(x, [0, 2, 1, 3])
        x_h = self.spatial_att(x_h)
        x_h = tf.transpose(x_h, [0, 2, 1, 3])

        x_w = tf.transpose(x, [0, 3, 2, 1])
        x_w = self.spatial_att(x_w)
        x_w = tf.transpose(x_w, [0, 3, 2, 1])

        return (x_c + x_h + x_w) / 3.0


# ============================================================
# MODEL HELPERS
# ============================================================

def get_h5_layer_names(model_path):
    with h5py.File(model_path, "r") as file:
        config = json.loads(file.attrs["model_config"])

    return [
        (layer["class_name"], layer["config"].get("name"))
        for layer in config["config"]["layers"]
    ]


def get_top_layer_name(layer_names, class_name, fallback):
    matches = [name for layer_class, name in layer_names if layer_class == class_name]
    return matches[-1] if matches else fallback


def build_model_from_h5_config(model_path, architecture):
    layer_names = get_h5_layer_names(model_path)

    gap_name = get_top_layer_name(layer_names, "GlobalAveragePooling2D", "global_average_pooling2d")
    batch_norm_name = get_top_layer_name(layer_names, "BatchNormalization", "batch_normalization")
    dropout_name = get_top_layer_name(layer_names, "Dropout", "dropout")

    dense_names = [name for layer_class, name in layer_names if layer_class == "Dense"]
    dense_hidden_name = dense_names[-2] if len(dense_names) >= 2 else "dense_hidden"
    dense_output_name = dense_names[-1] if dense_names else "dense_output"

    base_model = VGG19(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output

    if architecture == "triple_attention":
        triple_name = get_top_layer_name(layer_names, "TripleAttention", "triple_attention")
        x = TripleAttention(name=triple_name)(x)

    x = GlobalAveragePooling2D(name=gap_name)(x)
    x = Dense(512, activation="relu", name=dense_hidden_name)(x)
    x = BatchNormalization(name=batch_norm_name)(x)
    x = Dropout(0.5, name=dropout_name)(x)
    output = Dense(1, activation="sigmoid", name=dense_output_name)(x)

    return Model(inputs=base_model.input, outputs=output)


@st.cache_resource(show_spinner="Memuat model klasifikasi...")
def load_classification_model(model_path_string, architecture):
    model_path = Path(model_path_string)

    if not model_path.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {model_path}")

    custom_objects = {
        "TripleAttention": TripleAttention,
        "ChannelAttention": ChannelAttention,
        "SpatialAttention": SpatialAttention,
    }

    try:
        return load_model(model_path, custom_objects=custom_objects, compile=False)
    except Exception:
        model = build_model_from_h5_config(model_path, architecture)
        model.load_weights(model_path, by_name=True)
        return model


@st.cache_resource(show_spinner="Memuat U2Net segmentation...")
def load_u2net():
    return new_session("u2net")


# ============================================================
# IMAGE PROCESSING
# ============================================================

def segment_u2net(pil_img, session):
    result = remove(pil_img, session=session)
    rgba = np.array(result)

    mask = rgba[:, :, 3]
    segmented_rgb = rgba[:, :, :3].astype(np.uint8)

    white_bg = np.full_like(segmented_rgb, 255)
    segmented_preview = np.where(mask[:, :, np.newaxis] > 0, segmented_rgb, white_bg)

    return mask, segmented_rgb, segmented_preview.astype(np.uint8)


def preprocess_classification(img):
    resized = cv2.resize(img, IMG_SIZE)
    preprocessed = resized.astype("float32")
    preprocessed = preprocess_input(preprocessed)
    return np.expand_dims(preprocessed, axis=0)


def predict_image(model, img):
    start_time = time.time()
    normal_probability = float(model.predict(img, verbose=0).ravel()[0])
    inference_time = time.time() - start_time

    predicted_class = 1 if normal_probability > THRESHOLD else 0
    confidence = normal_probability if predicted_class == 1 else 1.0 - normal_probability

    return predicted_class, normal_probability, confidence, inference_time


# ============================================================
# APP
# ============================================================

def main():
    st.set_page_config(
        page_title="Analisis kulit sapi",
        page_icon=":material/biotech:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.markdown("### :material/biotech: Studi kulit sapi")
        st.caption("Lumpy Skin Disease")
        st.space("medium")
        st.markdown("**Pengaturan analisis**")
        scenario_name = st.selectbox(
            "Model skenario", list(SCENARIOS),
            format_func=lambda name: name.split(" - ")[0],
        )
        scenario = SCENARIOS[scenario_name]
        previous_scenario = st.session_state.get("selected_scenario", scenario_name)
        scenario_changed = previous_scenario != scenario_name
        st.session_state.selected_scenario = scenario_name
        st.markdown(f"**{scenario_name.split(' - ', 1)[1]}**")
        split = "80 / 10 / 10" if scenario_name.startswith(("Skenario 1", "Skenario 2")) else "70 / 20 / 10"
        st.caption(f"Latih / validasi / uji: {split}")
        st.caption("Ganti skenario untuk otomatis menganalisis ulang citra yang tersedia.")
        st.space("medium")
        with st.expander("Panduan penggunaan", icon=":material/help_outline:"):
            st.write("Unggah foto kulit sapi yang jelas dalam format JPG atau PNG, lalu tekan Jalankan analisis.")
            st.write("Hasil segmentasi ditampilkan lebih dahulu, diikuti hasil klasifikasi.")


    title_col, theme_col = st.columns([5, 1], vertical_alignment="center")
    with title_col:
        st.title("Analisis kulit sapi")
        st.caption("Lumpy Skin Disease / Segmentasi dan klasifikasi citra")
    with theme_col:
        dark_mode = st.toggle("Mode gelap", key="dashboard_dark", value=False,
                              help="Aktifkan untuk tema gelap; nonaktifkan untuk tema terang.")
        st.caption(":material/dark_mode: Gelap" if dark_mode else ":material/light_mode: Terang")
    apply_dashboard_style(dark_mode)
    st.space("small")

    with st.container(border=False, key="input_panel"):
        st.subheader("01 / Input citra")
        st.caption("Unggah foto kulit sapi dan periksa pratinjau sebelum menjalankan analisis.")
        upload_col, preview_col = st.columns([1, 1.2], gap="large")
        with upload_col:
            uploaded_file = st.file_uploader(
                "Pilih citra", type=["jpg", "jpeg", "png"],
                help="Pilih satu foto kulit sapi yang tajam dengan pencahayaan merata.",
            )
            pil_img = None
            image_key = None
            if uploaded_file is not None:
                try:
                    pil_img = Image.open(uploaded_file).convert("RGB")
                    image_key = (hashlib.sha256(uploaded_file.getvalue()).hexdigest(), scenario_name)
                except (OSError, ValueError, Image.DecompressionBombError):
                    st.error("Citra tidak dapat dibaca. Pilih berkas JPG atau PNG yang valid.")
            run_analysis = st.button(
                "Jalankan analisis", type="primary", icon=":material/arrow_forward:",
                width="stretch", disabled=pil_img is None,
            )
        with preview_col:
            st.caption("PRATINJAU CITRA")
            with st.container(horizontal_alignment="center", vertical_alignment="center", border=False):
                if pil_img is not None:
                    preview = pil_img.copy()
                    preview.thumbnail((800, 280))
                    st.image(preview, alt="Pratinjau citra kulit sapi")
                    st.caption(f"{pil_img.width} × {pil_img.height} piksel · {uploaded_file.size / 1024:.0f} KB")
                else:
                    st.caption("Belum ada citra. Unggah foto untuk melihat pratinjau.")

    # Results belong to both the image and the selected scenario.
    if st.session_state.get("analysis_key") != image_key:
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("analysis_key", None)

    image_digest = image_key[0] if image_key is not None else None
    if st.session_state.get("segmentation_key") != image_digest:
        st.session_state.pop("segmentation_result", None)
        st.session_state.pop("segmentation_key", None)

    if (run_analysis or scenario_changed) and pil_img is not None:
        st.session_state.pop("analysis_result", None)
        try:
            with st.spinner("Menyiapkan model dan menganalisis citra..."):
                model_path = resolve_model_path(scenario)
                model = load_classification_model(str(model_path), scenario["architecture"])
                if "segmentation_result" not in st.session_state:
                    session = load_u2net()
                    st.session_state.segmentation_result = segment_u2net(pil_img, session)
                    st.session_state.segmentation_key = image_digest
                mask, segmented, preview = st.session_state.segmentation_result
                prediction = predict_image(model, preprocess_classification(segmented))
                st.session_state.analysis_result = (mask, preview, prediction)
                st.session_state.analysis_key = image_key
        except Exception as exc:
            st.error("Analisis belum berhasil. Coba jalankan kembali.")
            with st.expander("Detail kendala"):
                st.text(str(exc))

    result = st.session_state.get("analysis_result")
    st.space("small")
    with st.container(border=False, key="segmentation_panel"):
        st.subheader("02 / Hasil segmentasi")
        st.caption("Pemisahan objek dari latar menggunakan U2Net.")
        if result is None:
            st.markdown(":material/layers: **Menunggu citra diproses**")
            st.caption("Citra asli, mask, dan hasil segmentasi akan ditampilkan berdampingan.")
        else:
            mask, segmented_preview, prediction = result
            for col, label, description, img in zip(
                st.columns(3, gap="medium"),
                ["Citra asli", "Mask segmentasi", "Hasil segmentasi"],
                ["Foto yang diunggah", "Area objek yang dipertahankan", "Objek pada latar putih"],
                [pil_img, mask, segmented_preview],
            ):
                with col.container(border=False):
                    st.markdown(f"**{label}**")
                    display_img = img.copy() if isinstance(img, Image.Image) else Image.fromarray(img)
                    display_img.thumbnail((640, 240))
                    with st.container(horizontal_alignment="center", vertical_alignment="center"):
                        st.image(display_img, alt=label)
                    st.caption(description)

    st.space("small")
    with st.container(border=False, key="classification_panel"):
        st.subheader("03 / Hasil klasifikasi")
        st.caption("Prediksi kelas berdasarkan citra yang telah melalui segmentasi.")
        if result is None:
            st.markdown(":material/query_stats: **Belum ada hasil klasifikasi**")
            st.caption("Kelas prediksi dan probabilitas tampil setelah analisis selesai.")
        else:
            predicted_class, normal_probability, confidence, inference_time = prediction
            summary_col, probability_col = st.columns([1, 1.2], gap="large")
            with summary_col.container(border=False):
                st.caption("KELAS PREDIKSI")
                st.subheader(f":orange[{CLASS_NAMES[predicted_class]}]" if predicted_class == 0 else f":green[{CLASS_NAMES[predicted_class]}]")
                confidence_col, time_col = st.columns(2)
                confidence_col.metric("Keyakinan model", f"{confidence * 100:.2f}%")
                time_col.metric("Waktu inferensi", f"{inference_time:.2f} dtk")
            with probability_col.container(border=False):
                st.markdown("**Probabilitas kelas**")
                st.progress(1.0 - normal_probability, text=f"Lumpy Skin · {(1.0 - normal_probability) * 100:.2f}%")
                st.progress(normal_probability, text=f"Normal Skin · {normal_probability * 100:.2f}%")
                st.caption(scenario_name)
            with st.expander("Detail pemrosesan", icon=":material/tune:"):
                st.caption(f"Model: {scenario_name}")
                st.caption(f"Input klasifikasi: {IMG_SIZE[0]} × {IMG_SIZE[1]} piksel · Preprocessing VGG19")
                st.caption(f"Ambang keputusan: sigmoid > {THRESHOLD} diklasifikasikan sebagai Normal Skin.")
                st.caption("Waktu inferensi hanya mengukur prediksi klasifikasi, tidak termasuk pemuatan model dan segmentasi.")

    st.space("medium")
    st.caption("Lumpy Skin Disease · Klasifikasi citra berbasis VGG19")


if __name__ == "__main__":
    main()
