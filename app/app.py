# app/app.py
import os
import urllib.request

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.pipeline import ModelBundle, compare_tiger_stripes

# ============================
# Page config
# ============================
st.set_page_config(
    page_title="Tiger Stripe Matcher",
    page_icon="🐯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================
# Premium CSS (UI polish)
# ============================
st.markdown(
    """
<style>
/* Layout */
.block-container { padding-top: 1.35rem; padding-bottom: 2rem; max-width: 1200px; }
hr { border: none; height: 1px; background: rgba(255,255,255,0.10); margin: 14px 0 16px; }

/* Typography */
.big-title { font-size: 2.35rem; font-weight: 900; letter-spacing: -0.02em; line-height: 1.08; }
.subtle { opacity: 0.86; font-size: 1.02rem; margin-top: 0.25rem; }
.mini { font-size: 0.9rem; opacity: 0.82; }

/* Cards */
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 18px;
  padding: 16px 16px 12px 16px;
  box-shadow: 0 14px 50px rgba(0,0,0,0.18);
}
.card-tight { padding: 12px 14px; }

/* Badges */
.badge {
  display: inline-flex;
  gap: 8px;
  align-items: center;
  padding: 6px 12px;
  border-radius: 999px;
  font-weight: 800;
  border: 1px solid rgba(255,255,255,0.16);
  background: rgba(124,58,237,0.16);
}
.badge-good { background: rgba(34,197,94,0.14); border-color: rgba(34,197,94,0.25); }
.badge-bad  { background: rgba(239,68,68,0.14); border-color: rgba(239,68,68,0.25); }

/* Step chips */
.chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
.chip {
  display:inline-flex; align-items:center; gap:8px;
  padding:6px 10px; border-radius:999px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  font-size: 0.9rem; opacity: 0.92;
}

/* Buttons */
.stButton > button {
  border-radius: 14px !important;
  font-weight: 800 !important;
  padding: 0.85rem 1rem !important;
}

/* Image rounding */
img { border-radius: 14px; }

/* Hide Streamlit chrome (optional-ish) */
[data-testid="stToolbar"] { visibility: hidden; height: 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================
# SAM checkpoint auto-download
# ============================
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SAM_PATH = "sam_vit_h_4b8939.pth"


def ensure_sam_checkpoint() -> None:
    if os.path.exists(SAM_PATH):
        return
    with st.status("Downloading SAM checkpoint (first run only)…", expanded=False) as s:
        urllib.request.urlretrieve(SAM_URL, SAM_PATH)
        s.update(label="SAM checkpoint downloaded ✅", state="complete", expanded=False)


@st.cache_resource
def load_models() -> ModelBundle:
    ensure_sam_checkpoint()
    return ModelBundle(sam_ckpt_path=SAM_PATH)


# ============================
# Helpers
# ============================
def pil_to_rgb_np(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))


def overlay_mask(img_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    vis = img_rgb.copy()
    color = np.array([0, 255, 160], dtype=np.uint8)  # mint
    m = mask.astype(bool)
    vis[m] = (vis[m] * (1 - alpha) + color * alpha).astype(np.uint8)
    return vis


def draw_bbox(img_rgb: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    out = img_rgb.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 220, 80), 3)
    return out


def clamp_2d(img2d: np.ndarray) -> np.ndarray:
    """Ensure Streamlit renders grayscale nicely."""
    if img2d.dtype != np.uint8:
        img2d = np.clip(img2d, 0, 255).astype(np.uint8)
    return img2d


# ============================
# Header
# ============================
h1, h2 = st.columns([0.72, 0.28], vertical_alignment="top")

with h1:
    st.markdown('<div class="big-title">🐯 Tiger Stripe Matcher</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtle">Upload two tiger images and verify whether the stripe patterns match. '
        'Background foliage is automatically ignored using segmentation.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="chips">
  <div class="chip">📦 ROI detection</div>
  <div class="chip">✂️ Body segmentation</div>
  <div class="chip">🌓 Stripe enhancement</div>
  <div class="chip">🧠 Pattern matching</div>
</div>
""",
        unsafe_allow_html=True,
    )

with h2:
    st.markdown('<div class="card card-tight">', unsafe_allow_html=True)
    st.markdown("**⚡ What you get**")
    st.markdown(
        '<div class="mini">• Positive/Negative label<br>'
        '• Confidence (0–100)<br>'
        '• ROI box + mask overlays<br>'
        '• Stripe-enhanced + edge views</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ============================
# Upload section
# ============================
u1, u2 = st.columns(2, vertical_alignment="top")

with u1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🖼️ Tiger Image #1")
    f1 = st.file_uploader("Upload first image", type=["jpg", "jpeg", "png"], key="img1")
    if f1 is not None:
        img1_preview = Image.open(f1).convert("RGB")
        st.image(img1_preview, use_container_width=True, caption="Preview #1")
    st.markdown("</div>", unsafe_allow_html=True)

with u2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🖼️ Tiger Image #2")
    f2 = st.file_uploader("Upload second image", type=["jpg", "jpeg", "png"], key="img2")
    if f2 is not None:
        img2_preview = Image.open(f2).convert("RGB")
        st.image(img2_preview, use_container_width=True, caption="Preview #2")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ============================
# Controls
# ============================
cL, cR = st.columns([0.62, 0.38], vertical_alignment="top")

with cL:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🚀 Run Matching")
    run = st.button("Compare Stripe Patterns", use_container_width=True, type="primary")
    st.markdown(
        '<div class="mini">Tip: side-profile or full-body images give the cleanest stripe signals.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with cR:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Options")
    show_debug = st.toggle("Show debug details", value=False)
    st.markdown(
        '<div class="mini">Shows the individual score components used to compute the confidence.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ============================
# Run
# ============================
if run:
    if f1 is None or f2 is None:
        st.error("Please upload **both** images first.")
        st.stop()

    # Re-open from uploader buffers (important: file pointer may move)
    f1.seek(0)
    f2.seek(0)
    img1_pil = Image.open(f1).convert("RGB")
    img2_pil = Image.open(f2).convert("RGB")
    img1 = pil_to_rgb_np(img1_pil)
    img2 = pil_to_rgb_np(img2_pil)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Progress UX
    prog = st.progress(0, text="Initializing models…")
    try:
        bundle = load_models()
    except Exception as e:
        st.error(
            "Model initialization failed. If this is Streamlit Cloud, ensure you pinned Python 3.11 "
            "and used opencv-python-headless. Error: " + str(e)
        )
        st.stop()

    prog.progress(20, text="Detecting tiger ROI…")
    prog.progress(45, text="Segmenting tiger body (ignoring background)…")
    prog.progress(70, text="Enhancing stripe patterns…")
    prog.progress(85, text="Matching stripes + computing confidence…")

    artifacts, result = compare_tiger_stripes(bundle, img1, img2)

    prog.progress(100, text="Done ✅")

    if artifacts is None:
        st.error(result.get("reason", "Failed to process images."))
        st.stop()

    label = result["label"]
    conf = int(result["confidence"])

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Result hero card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if label == "positive":
        st.markdown(
            f'## ✅ Match Found <span class="badge badge-good">🎯 Confidence: {conf}/100</span>',
            unsafe_allow_html=True,
        )
        st.success("Stripe patterns look consistent across both images.")
    else:
        st.markdown(
            f'## ❌ No Match <span class="badge badge-bad">🧾 Confidence: {conf}/100</span>',
            unsafe_allow_html=True,
        )
        st.warning("Stripe patterns appear different across the two tigers.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Visual panels
    st.markdown("<hr/>", unsafe_allow_html=True)
    p1, p2 = st.columns(2, vertical_alignment="top")

    with p1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🐯 Image #1 — ROI + Mask")
        bbox1 = artifacts["bbox1"]
        mask1 = artifacts["mask1"]
        vis1 = overlay_mask(draw_bbox(img1, bbox1), mask1)
        st.image(vis1, use_container_width=True, caption="ROI box + segmentation mask")

        st.markdown("**🌓 Stripe Enhanced**")
        st.image(clamp_2d(artifacts["dog1"]), use_container_width=True)

        st.markdown("**🧩 Edges**")
        st.image(clamp_2d(artifacts["edge1"]), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with p2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🐯 Image #2 — ROI + Mask")
        bbox2 = artifacts["bbox2"]
        mask2 = artifacts["mask2"]
        vis2 = overlay_mask(draw_bbox(img2, bbox2), mask2)
        st.image(vis2, use_container_width=True, caption="ROI box + segmentation mask")

        st.markdown("**🌓 Stripe Enhanced**")
        st.image(clamp_2d(artifacts["dog2"]), use_container_width=True)

        st.markdown("**🧩 Edges**")
        st.image(clamp_2d(artifacts["edge2"]), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if show_debug:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("### 🧾 Debug Details")
        st.json(result, expanded=False)

# ============================
# Footer
# ============================
st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("Built with YOLOv8 + Segment Anything + robust stripe preprocessing. © Tiger Stripe Matcher")
