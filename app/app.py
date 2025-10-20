
import streamlit as st
from predict import predict_image, gradcam_on_image

if "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes = None

tab_all, tab_about = st.tabs(["🖼️ Prediction + Grad-CAM", "📘 About"])

with tab_all:
    st.write("Upload a spine image, get a prediction, and visualize Grad-CAM — all in one place.")

    uploaded = st.file_uploader("Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if uploaded:
        st.session_state.uploaded_bytes = uploaded.read()
        st.image(uploaded, caption="Uploaded image", use_column_width=True)

    if st.session_state.uploaded_bytes:
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
            st.image(overlay, caption=f"Grad-CAM — Predicted: {label} ({prob:.1%})", use_column_width=True)

            # Optional: download overlay
            import io
            from PIL import Image
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
    st.markdown("**How it works**")
    st.markdown(
        "- ResNet-18 adapted for 1-channel inputs (grayscale).\n"
        "- Preprocessing: Grayscale → Resize 224 → Normalize mean=0.5, std=0.5.\n"
        "- Predict → Grad-CAM highlights the most influential regions.\n"
        "- Demo is **not for clinical use**."
    )
