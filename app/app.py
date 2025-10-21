import streamlit as st
from predict import predict_image, gradcam_on_image
from PIL import Image
import io

# ---------- Page setup ----------
st.set_page_config(
    page_title="🧠 LDH Classifier + Grad-CAM",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------- Minimal custom style ----------
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #f7fafc 0%, #ffffff 60%); }
html, body, [class*="css"] { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial; }

.block-container { padding-top: 2rem; padding-bottom: 2.5rem; }
.app-card {
  background:#ffffff; border:1px solid #eaeaea; border-radius:16px;
  padding:24px; box-shadow: 0 6px 24px rgba(0,0,0,0.06);
}

h1, h2, h3 { letter-spacing: 0.2px; }
.small-muted { color:#6b7280; font-size:0.92rem; }

.result-badge {
  display:inline-block; padding:10px 14px; border-radius:999px;
  background:#ecfdf5; color:#065f46; font-weight:600; border:1px solid #d1fae5;
}

.footer { color:#6b7280; font-size:0.85rem; margin-top: 16px; }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("<div class='app-card'>", unsafe_allow_html=True)
st.title("🧠 Lumbar Disc Herniation Classifier with Grad-CAM")
st.markdown("<p class='small-muted'>Upload an MRI image to classify and visualize important regions via Grad-CAM.</p>", unsafe_allow_html=True)

# ---------- Tabs ----------
tab_all, tab_about = st.tabs(["🖼️ Prediction + Grad-CAM", "📘 About"])

with tab_all:
    st.write("Upload a spine image, get a prediction, and visualize Grad-CAM — all in one place.")

    uploaded = st.file_uploader("Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if uploaded:
        st.session_state.uploaded_bytes = uploaded.read()
        st.image(uploaded, caption="Uploaded image", use_column_width=True)

    if "uploaded_bytes" in st.session_state and st.session_state.uploaded_bytes:
        if st.button("✨ Predict and Show Grad-CAM", use_container_width=True):
            # Predict
            preds = predict_image(st.session_state.uploaded_bytes, topk=2)
            top1 = preds[0]
            st.markdown(
                f"<span class='result-badge'>Top-1: {top1[0]} ({top1[1]:.1%})</span>",
                unsafe_allow_html=True
            )
            st.subheader("Probabilities")
            st.table({
                "label": [l for l, _ in preds],
                "probability": [f"{p:.2%}" for _, p in preds],
            })

            # Grad-CAM
            overlay, label, prob, topk = gradcam_on_image(st.session_state.uploaded_bytes)

            # Side-by-side display
            col1, col2 = st.columns(2)
            with col1:
                st.image(st.session_state.uploaded_bytes, caption="Original", use_column_width=True)
            with col2:
                st.image(overlay, caption=f"Grad-CAM — Predicted: {label} ({prob:.1%})", use_column_width=True)

            # Optional: download overlay
            buf = io.BytesIO()
            Image.fromarray(overlay).save(buf, format="PNG")
            st.download_button(
                "⬇️ Download Grad-CAM PNG",
                data=buf.getvalue(),
                file_name=f"gradcam_{label}.png",
                mime="image/png",
            )
    else:
        st.info("Upload an image to begin.")

with tab_about:
    st.markdown("### ℹ️ How it works")
    st.markdown(
        "- **Model:** ResNet-18 adapted for grayscale MRI inputs.\n"
        "- **Preprocessing:** Grayscale → Resize 224 → Normalize (mean=0.5, std=0.5).\n"
        "- **Grad-CAM:** Highlights the regions most influential for the predicted class.\n"
        "- **Disclaimer:** This demo is **not for clinical use** — educational purposes only."
    )

st.markdown("</div>", unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown(
    """
    <div class='footer'>
        Built with Streamlit • LDH Project • Demo for MLOps delivery<br>
        <span style='color:#9CA3AF; font-size:0.9em;'>Version: v1.0.1</span>
    </div>
    """,
    unsafe_allow_html=True
)
