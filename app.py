import os

# Prevent TensorFlow segfaults on macOS
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import io

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CLASS_NAMES = [
    "✈️ Airplane", "🚗 Automobile", "🐦 Bird", "🐱 Cat", "🦌 Deer",
    "🐶 Dog", "🐸 Frog", "🐴 Horse", "🚢 Ship", "🚚 Truck",
]

CLASS_NAMES_PLAIN = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck",
]

MODEL_PATH = "model_clfar-10.keras"

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Hero Title ── */
    .hero-title {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 25%, #FEC163 50%, #45B7D1 75%, #6C5CE7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0;
        line-height: 1.2;
        animation: fadeInDown 0.8s ease-out;
    }

    .hero-subtitle {
        font-size: 1.25rem;
        color: #B0B8C8;
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 400;
        animation: fadeInUp 0.8s ease-out;
    }

    /* ── Glass Card ── */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1rem 0;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.35);
    }

    .glass-card h3 {
        margin-top: 0;
        color: #FF8E53;
    }

    /* ── Stat Cards ── */
    .stat-card {
        background: linear-gradient(135deg, rgba(255,107,107,0.12), rgba(108,92,231,0.12));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.4rem;
        text-align: center;
        transition: transform 0.25s ease;
    }
    .stat-card:hover { transform: scale(1.04); }
    .stat-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B6B, #FEC163);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #8892A4;
        margin-top: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }

    /* ── Prediction Result ── */
    .prediction-box {
        background: linear-gradient(135deg, rgba(69,183,209,0.15), rgba(108,92,231,0.15));
        border: 1px solid rgba(69,183,209,0.25);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .pred-label {
        font-size: 2.4rem;
        font-weight: 800;
        color: #45B7D1;
    }
    .pred-confidence {
        font-size: 1.1rem;
        color: #B0B8C8;
        margin-top: 0.5rem;
    }

    /* ── Architecture Layer ── */
    .arch-layer {
        background: rgba(255,255,255,0.03);
        border-left: 3px solid #FF6B6B;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        border-radius: 0 10px 10px 0;
        font-size: 0.92rem;
        transition: background 0.2s ease;
    }
    .arch-layer:hover {
        background: rgba(255,107,107,0.08);
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin-top: 3rem;
        color: #6B7280;
        font-size: 0.85rem;
    }
    .footer a {
        color: #FF6B6B;
        text-decoration: none;
        font-weight: 600;
    }
    .footer a:hover { text-decoration: underline; }

    /* ── Animations ── */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ── Sidebar polish ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #151B28 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(255,107,107,0.35);
        border-radius: 14px;
        padding: 1rem;
        transition: border-color 0.3s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #FF6B6B;
    }

    /* ── Hide default Streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Model Loading (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CNN model …")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data(show_spinner="Loading CIFAR-10 samples …")
def load_cifar_samples():
    """Load a small set of sample images from CIFAR-10 for the dataset page."""
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    samples = {}
    for class_idx in range(10):
        idxs = np.where(y_train.flatten() == class_idx)[0][:5]
        samples[class_idx] = x_train[idxs]
    return samples


# ─────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Navigation")
    st.markdown("---")
    page = st.radio(
        "Go to",
        ["🏠 Home", "📊 About Dataset", "🧬 Model Info", "🔮 Try Prediction"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align:center; color:#6B7280; font-size:0.8rem;'>
            Built with ❤️ using<br>
            <b>Streamlit</b> & <b>TensorFlow</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Helper: render footer
# ─────────────────────────────────────────────
def render_footer():
    st.markdown(
        """
        <div class="footer">
            <p>
                🎓 Built by <b>Md Nazmul Huda Nabil</b> &nbsp;|&nbsp;
                CIFAR-10 Image Classification using CNN &nbsp;|&nbsp;
                <a href="https://github.com/NazmulHudaNabil/cifar10-cnn-streamlit-app" target="_blank">GitHub ↗</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════
# PAGE: Home
# ═════════════════════════════════════════════
def page_home():
    st.markdown('<h1 class="hero-title">CIFAR-10 Image Classifier</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Classify images into 10 categories using a Convolutional Neural Network</p>',
        unsafe_allow_html=True,
    )

    st.write("")

    # Project Description
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.markdown(
            """
            <div class="glass-card" style="text-align:center;">
                <h3>🚀 About This Project</h3>
                <p style="color:#B0B8C8; line-height:1.8;">
                    This deep-learning web application leverages a <b>Convolutional Neural Network (CNN)</b>
                    trained on the <b>CIFAR-10</b> dataset to classify images into one of ten categories:
                    airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.
                    <br><br>
                    The model was built with <b>TensorFlow / Keras</b>, and this interactive front-end is
                    powered by <b>Streamlit</b>. Simply upload an image and watch the model predict in real time!
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Category showcase
    st.markdown("### 🏷️ Supported Categories")
    cols = st.columns(5)
    for i, name in enumerate(CLASS_NAMES):
        with cols[i % 5]:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value" style="font-size:2rem;">{name.split(' ')[0]}</div>
                    <div class="stat-label">{name.split(' ')[1] if len(name.split(' '))>1 else ''}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Second row of categories
    cols2 = st.columns(5)
    for i, name in enumerate(CLASS_NAMES[5:]):
        with cols2[i % 5]:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value" style="font-size:2rem;">{name.split(' ')[0]}</div>
                    <div class="stat-label">{name.split(' ')[1] if len(name.split(' '))>1 else ''}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_footer()


# ═════════════════════════════════════════════
# PAGE: About Dataset
# ═════════════════════════════════════════════
def page_dataset():
    st.markdown('<h1 class="hero-title">📊 About The Dataset</h1>', unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="glass-card">
                <h3>📝 What is CIFAR-10?</h3>
                <p style="color:#B0B8C8; line-height:1.8;">
                    The <b>CIFAR-10</b> dataset is one of the most widely used benchmark datasets in
                    computer vision and machine learning. It was collected by
                    <b>Alex Krizhevsky</b>, <b>Vinod Nair</b>, and <b>Geoffrey Hinton</b>.
                    <br><br>
                    It consists of <b>60,000 color images</b> of size <b>32 × 32 pixels</b>,
                    divided equally into <b>10 mutually exclusive classes</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="glass-card">
                <h3>📈 Dataset Statistics</h3>
                <p style="color:#B0B8C8; line-height:1.8;">
                    • <b>Total Images:</b> 60,000<br>
                    • <b>Training Set:</b> 50,000 images<br>
                    • <b>Test Set:</b> 10,000 images<br>
                    • <b>Image Size:</b> 32 × 32 × 3 (RGB)<br>
                    • <b>Classes:</b> 10<br>
                    • <b>Images per Class:</b> 6,000
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("### 🖼️ Sample Images from Each Class")

    samples = load_cifar_samples()

    for class_idx in range(0, 10, 2):
        cols = st.columns(2)
        for offset, col in enumerate(cols):
            idx = class_idx + offset
            if idx >= 10:
                break
            with col:
                st.markdown(f"**{CLASS_NAMES[idx]}**")
                fig, axes = plt.subplots(1, 5, figsize=(10, 2.5))
                for ax_i, ax in enumerate(axes):
                    ax.imshow(samples[idx][ax_i])
                    ax.axis("off")
                fig.patch.set_facecolor("#0E1117")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    render_footer()


# ═════════════════════════════════════════════
# PAGE: Model Info
# ═════════════════════════════════════════════
def page_model():
    st.markdown('<h1 class="hero-title">🧬 Model Information</h1>', unsafe_allow_html=True)
    st.write("")

    # Architecture
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            """
            <div class="glass-card">
                <h3>🏗️ CNN Architecture</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        layers = [
            ("Input", "32 × 32 × 3 RGB Image"),
            ("Conv2D", "32 filters, 3×3 kernel, ReLU"),
            ("MaxPool2D", "2×2 pool size"),
            ("BatchNorm", "Normalize activations"),
            ("Conv2D", "64 filters, 3×3 kernel, ReLU"),
            ("MaxPool2D", "2×2 pool size"),
            ("BatchNorm", "Normalize activations"),
            ("Conv2D", "128 filters, 3×3 kernel, ReLU"),
            ("MaxPool2D", "2×2 pool size"),
            ("BatchNorm", "Normalize activations"),
            ("Flatten", "Convert to 1D vector"),
            ("Dense", "128 units, ReLU"),
            ("Dropout", "50% regularization"),
            ("Dense", "10 units, Softmax (Output)"),
        ]

        for name, desc in layers:
            st.markdown(
                f'<div class="arch-layer"><b>{name}</b> &nbsp;—&nbsp; {desc}</div>',
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown(
            """
            <div class="glass-card">
                <h3>⚙️ Training Details</h3>
                <p style="color:#B0B8C8; line-height:2;">
                    <b>Optimizer:</b> Adam<br>
                    <b>Loss Function:</b> Sparse Categorical Crossentropy<br>
                    <b>Epochs:</b> 20<br>
                    <b>Batch Size:</b> 32<br>
                    <b>Framework:</b> TensorFlow / Keras
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metrics
        st.markdown(
            """
            <div class="stat-card" style="margin-top:1rem;">
                <div class="stat-value">~87.65%</div>
                <div class="stat-label">Training Accuracy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="stat-card" style="margin-top:1rem;">
                <div class="stat-value">~73.33%</div>
                <div class="stat-label">Validation Accuracy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="glass-card" style="margin-top:1rem;">
                <h3>📌 Key Highlights</h3>
                <p style="color:#B0B8C8; line-height:1.8;">
                    • Three convolutional blocks with increasing filter depth (32 → 64 → 128)<br>
                    • Batch Normalization after each pooling layer for stable training<br>
                    • Dropout (50%) to prevent overfitting<br>
                    • Softmax output for multi-class probability distribution
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Model Summary (optional) ──
    st.write("")
    with st.expander("📄 Show Full Model Summary"):
        model = load_model()
        summary_buf = io.StringIO()
        model.summary(print_fn=lambda x: summary_buf.write(x + "\n"))
        st.code(summary_buf.getvalue(), language="text")

    render_footer()


# ═════════════════════════════════════════════
# PAGE: Prediction
# ═════════════════════════════════════════════
def page_prediction():
    st.markdown('<h1 class="hero-title">🔮 Try Prediction</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Upload an image and let the CNN classify it!</p>',
        unsafe_allow_html=True,
    )
    st.write("")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown(
            """
            <div class="glass-card">
                <h3>📤 Upload Image</h3>
                <p style="color:#8892A4;">Supported formats: JPG, JPEG, PNG</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Choose an image …",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

    with col_right:
        if uploaded_file is not None:
            model = load_model()

            # ── Preprocess ──
            with st.spinner("🔍 Analyzing image …"):
                img_resized = image.resize((32, 32))
                img_array = np.array(img_resized).astype("float32") / 255.0
                img_input = img_array.reshape(1, 32, 32, 3)
                predictions = model.predict(img_input, verbose=0)
                probs = predictions[0]

            # ── Top prediction ──
            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx]) * 100

            st.markdown(
                f"""
                <div class="prediction-box">
                    <div class="pred-label">{CLASS_NAMES[top_idx]}</div>
                    <div class="pred-confidence">Confidence: {top_conf:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")

            # ── Top-3 Predictions ──
            st.markdown("#### 🏆 Top-3 Predictions")
            top3_idxs = np.argsort(probs)[::-1][:3]

            for rank, idx in enumerate(top3_idxs, start=1):
                conf = float(probs[idx]) * 100
                medal = ["🥇", "🥈", "🥉"][rank - 1]
                st.markdown(f"**{medal} {CLASS_NAMES[idx]}**")
                st.progress(min(conf / 100, 1.0))
                st.caption(f"{conf:.2f}%")

            st.write("")

            # ── Full probability distribution ──
            with st.expander("📊 Show All Class Probabilities"):
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = [
                    "#FF6B6B" if i == top_idx else "#3B4252" for i in range(10)
                ]
                bars = ax.barh(
                    CLASS_NAMES_PLAIN,
                    probs * 100,
                    color=colors,
                    edgecolor="none",
                    height=0.65,
                )
                ax.set_xlabel("Confidence (%)", color="#B0B8C8", fontsize=10)
                ax.set_xlim(0, 105)
                ax.tick_params(colors="#B0B8C8", labelsize=9)
                ax.invert_yaxis()
                fig.patch.set_facecolor("#0E1117")
                ax.set_facecolor("#0E1117")
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.grid(axis="x", color="rgba(255,255,255,0.06)", linestyle="--")

                # Value labels
                for bar, val in zip(bars, probs * 100):
                    ax.text(
                        bar.get_width() + 1,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%",
                        va="center",
                        color="#B0B8C8",
                        fontsize=8,
                    )

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.markdown(
                """
                <div class="glass-card" style="text-align:center; min-height:300px; display:flex; align-items:center; justify-content:center;">
                    <div>
                        <p style="font-size:3rem;">📷</p>
                        <p style="color:#6B7280;">Upload an image to get started</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_footer()


# ═════════════════════════════════════════════
# Router
# ═════════════════════════════════════════════
if page == "🏠 Home":
    page_home()
elif page == "📊 About Dataset":
    page_dataset()
elif page == "🧬 Model Info":
    page_model()
elif page == "🔮 Try Prediction":
    page_prediction()
