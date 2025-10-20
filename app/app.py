
import streamlit as st
from predict import predict_image

st.set_page_config(page_title="LDH Classifier", layout="centered")

st.markdown('<div class="app-card">', unsafe_allow_html=True)
st.title("Lumbar Disc Herniation Classifier")
st.caption("Upload an MRI slice to predict whether the case shows herniation or not. (Educational demo only)")

uploaded = st.file_uploader("Upload a spine image", type=["png","jpg","jpeg"])
if uploaded:
    st.image(uploaded, use_column_width=True)
    if st.button("✨ Predict", use_container_width=True):
        preds = predict_image(uploaded.read(), topk=2)
        top1 = preds[0]
        st.markdown(f"<span style='color:green;font-weight:600;'>Prediction: {top1[0]} ({top1[1]:.1%})</span>", unsafe_allow_html=True)
        st.table({"label": [l for l,_ in preds], "probability": [f"{p:.2%}" for _,p in preds]})

st.markdown("<div style='color:gray;font-size:0.85rem;margin-top:1rem;'>Built with Streamlit • Not for clinical use</div>", unsafe_allow_html=True)
