import os
import numpy as np
import streamlit as st
from PIL import Image
import cv2

from src.pipeline import ModelBundle, compare_tiger_stripes

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Tiger Stripe Matcher",
    page_icon="🐯",
    layout="wide"
)

# ----------------------------
# CSS for premium UI
# ----------------------------
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
.big-title { font-size: 2.2rem; font-weight: 800; line-height: 1.1; }
.subtle { opacity: 0.85; font-size: 1.0rem; }
.card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 16px 16px 10px 16px;
}
.badge {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 700;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(124,58,237,0.10);
}
.hr { height: 1px; background: rgba(255,255,255,0.10); margin: 10px 0 14px; }
.small { font-size: 0.9rem; opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Helpers
# ----------------------------
def pil_to_rgb_np(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))

def overlay_mask(img_rgb, mask, alpha=0.45):
    vis = img_rgb.copy()
    color = np.array([0, 255, 160], dtype=np.uint8)  # mint
    m = mask.astype(bool)
    vis[m] = (vis[m] * (1-alpha) + color * alpha).astype(np.uint8)
    return vis

def draw_bbox(img_rgb, bbox_xyxy):
    x1,y1,x2,y2 = [int(v) for v in bbox_xyxy]
    out = img_rgb.copy()
    cv2.rectangle(out, (x1,y1), (x2,y2), (255, 220, 80), 3)
    return out

@st.cache_resource
import os
import urllib.request
import streamlit as st

SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SAM_PATH = "sam_vit_h_4b8939.pth"

def ensure_sam_checkpoint():
    if os.path.exists(SAM_PATH):
        return
    with st.spinner("Downloading SAM model (first run only)…"):
        urllib.request.urlretrieve(SAM_URL, SAM_PATH)

@st.cache_resource
def load_models():
    ensure_sam_checkpoint()
    return ModelBundle(sam_ckpt_path=SAM_PATH)


# ----------------------------
# Header
# ----------------------------
left, right = st.columns([0.72, 0.28], vertical_alignment="top")

with left:
    st.markdown('<div class="big-title">🐯 Tiger Stripe Matcher</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Upload two tiger images and verify whether the stripe patterns match. '
                'Background foliage is automatically ignored using segmentation.</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**⚡ Pipeline**")
    st.markdown('<div class="small">1) Detect tiger ROI<br>2) Segment tiger body<br>3) Enhance stripes<br>4) Compare patterns</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ----------------------------
# Upload section
# ----------------------------
u1, u2 = st.columns(2)

with u1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🖼️ Tiger Image #1")
    f1 = st.file_uploader("Upload first image", type=["jpg","jpeg","png"], key="img1")
    st.markdown('</div>', unsafe_allow_html=True)

with u2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🖼️ Tiger Image #2")
    f2 = st.file_uploader("Upload second image", type=["jpg","jpeg","png"], key="img2")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Controls
# ----------------------------
controls = st.columns([0.60, 0.40])
with controls[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧪 Run Matching")
    run = st.button("🚀 Compare Stripe Patterns", use_container_width=True, type="primary")
    st.markdown('<div class="small">Tip: try side-profile images for best results.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with controls[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Display Options")
    show_debug = st.toggle("Show debug details", value=False)
    st.markdown('<div class="small">Includes similarity components and raw scores.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Run
# ----------------------------
if run:
    if f1 is None or f2 is None:
        st.error("Please upload BOTH images first.")
        st.stop()

    img1_pil = Image.open(f1).convert("RGB")
    img2_pil = Image.open(f2).convert("RGB")
    img1 = pil_to_rgb_np(img1_pil)
    img2 = pil_to_rgb_np(img2_pil)

    # Fancy progress
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    prog = st.progress(0, text="Initializing models…")
    bundle = load_models()
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
    conf = result["confidence"]

    # Result hero card
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if label == "positive":
        st.markdown(f'## ✅ Match Found <span class="badge">Confidence: {conf}/100</span>', unsafe_allow_html=True)
        st.success("Stripe patterns look consistent across both images.")
    else:
        st.markdown(f'## ❌ No Match <span class="badge">Confidence: {conf}/100</span>', unsafe_allow_html=True)
        st.warning("Stripe patterns appear different across the two tigers.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Visual panels
    c1, c2 = st.columns(2)

    # Image 1 panel
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🐯 Image #1 — ROI + Mask")
        bbox1 = artifacts["bbox1"]
        mask1 = artifacts["mask1"]
        vis1 = overlay_mask(draw_bbox(img1, bbox1), mask1)
        st.image(vis1, use_container_width=True)
        st.markdown("**Stripe Enhanced / Edges**")
        st.image(artifacts["dog1"], clamp=True, use_container_width=True)
        st.image(artifacts["edge1"], clamp=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Image 2 panel
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🐯 Image #2 — ROI + Mask")
        bbox2 = artifacts["bbox2"]
        mask2 = artifacts["mask2"]
        vis2 = overlay_mask(draw_bbox(img2, bbox2), mask2)
        st.image(vis2, use_container_width=True)
        st.markdown("**Stripe Enhanced / Edges**")
        st.image(artifacts["dog2"], clamp=True, use_container_width=True)
        st.image(artifacts["edge2"], clamp=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if show_debug:
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.markdown("### 🧾 Debug Details")
        st.json(result, expanded=False)

# Footer
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.caption("Built with YOLOv8 + Segment Anything + robust stripe preprocessing. © Tiger Stripe Matcher")
