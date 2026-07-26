# ============================================================
# STREAMLIT
# LUMPY SKIN DISEASE CLASSIFICATION
# SCENARIO MODEL SELECTOR + U2NET SEGMENTATION
# ============================================================

import os
import time
from pathlib import Path

import cv2
import h5py
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image
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

    return token or os.getenv("TOKEN-HG") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or st.secrets.get("TOKEN-HG") or st.secrets.get("HF_TOKEN") or st.secrets.get("HUGGINGFACE_HUB_TOKEN")


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
                local_dir_use_symlinks=False,
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

    try:
        return download_model_from_huggingface(scenario["filename"], cached_path)
    except Exception:
        local_path = base_dir / scenario["path"]
        if local_path.exists():
            return local_path
        if cached_path.exists():
            return cached_path
        raise


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
        page_title="Lumpy Skin Disease Classification",
        layout="wide",
    )

    st.title("Lumpy Skin Disease Classification")
    st.caption("U2Net segmentation + selectable VGG19 scenario model")

    scenario_name = st.sidebar.selectbox("Pilih model skenario", list(SCENARIOS.keys()))
    scenario = SCENARIOS[scenario_name]
    model_path = resolve_model_path(scenario)

    st.sidebar.caption(scenario["description"])
    st.sidebar.caption(f"File model: {model_path}")

    missing_models = [
        f"{name}: {config['path']}"
        for name, config in SCENARIOS.items()
        if not (Path(__file__).resolve().parent / config["path"]).exists()
    ]

    if missing_models:
        st.sidebar.info(
            "📦 Model disimpan di Hugging Face dan diunduh otomatis "
            "saat skenario dipilih. Setelah unduhan pertama, model di-cache "
            "sehingga skenario berikutnya lebih cepat."
        )

    if not model_path.exists():
        st.error(
            f"File model untuk {scenario_name} belum ada di workspace: {model_path}. "
            "Letakkan file .h5 sesuai nama tersebut atau ubah path pada konfigurasi SCENARIOS."
        )
        st.stop()

    classification_model = load_classification_model(
        str(model_path),
        scenario["architecture"],
    )
    u2net_session = load_u2net()

    uploaded_file = st.file_uploader(
        "Upload citra kulit sapi",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("Silakan upload gambar untuk memulai klasifikasi.")
        st.stop()

    pil_img = Image.open(uploaded_file).convert("RGB")
    original = np.array(pil_img)

    with st.spinner("Melakukan segmentasi U2Net dan klasifikasi..."):
        mask, segmented_for_model, segmented_preview = segment_u2net(pil_img, u2net_session)
        input_img = preprocess_classification(segmented_for_model)
        predicted_class, normal_probability, confidence, inference_time = predict_image(
            classification_model,
            input_img,
        )

    st.subheader("Preprocessing U2Net")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("Original Image")
        st.image(original, use_column_width=True)

    with col2:
        st.caption("Segmentation Mask")
        st.image(mask, use_column_width=True, clamp=True)

    with col3:
        st.caption("Segmented Result")
        st.image(segmented_preview, use_column_width=True)

    st.divider()

    result_label = CLASS_NAMES[predicted_class]
    lumpy_probability = 1.0 - normal_probability

    result_col, detail_col = st.columns([1, 1])

    with result_col:
        if predicted_class == 0:
            st.error(f"Prediction: {result_label}")
        else:
            st.success(f"Prediction: {result_label}")

        st.metric("Confidence", f"{confidence * 100:.2f}%")

    with detail_col:
        st.metric("Lumpy Skin Probability", f"{lumpy_probability:.4f}")
        st.metric("Normal Skin Probability", f"{normal_probability:.4f}")
        st.metric("Inference Time", f"{inference_time:.2f} detik")
        st.caption(f"Threshold: sigmoid > {THRESHOLD} berarti Normal Skin")
        st.caption(f"Model aktif: {scenario_name}")


if __name__ == "__main__":
    main()
