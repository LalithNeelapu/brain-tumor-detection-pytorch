import sys
import os

ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
from PIL import Image
from utils.predict import predict_image


# ==================================
# PAGE CONFIG
# ==================================

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)


# ==================================
# SESSION
# ==================================

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# ==================================
# SIDEBAR
# ==================================

with st.sidebar:

    st.title("🧠 Model Information")

    st.success("CNN (PyTorch)")

    st.markdown("### Dataset")
    st.info("9618 Images")

    st.markdown("### Outputs")

    st.markdown("""
- CT / MRI
- Healthy / Tumor
""")

    st.markdown("### Performance")

    st.success("Tumor Accuracy ~96%")

    st.success("Modality Accuracy 100%")


# ==================================
# HEADER
# ==================================

st.title("🧠 Brain Tumor Detection")

st.markdown("""
Upload a CT or MRI image to predict:

- Imaging Modality
- Tumor Status
""")


# ==================================
# FILE UPLOADER
# ==================================

uploaded = st.file_uploader(
    "Upload Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    key=f"upload_{st.session_state.uploader_key}"
)


# ==================================
# PREDICTION
# ==================================

if uploaded is not None:

    image = Image.open(
        uploaded
    ).convert(
        "RGB"
    )

    st.image(
        image,
        caption="Uploaded Image",
        width=450
    )

    st.caption(
        f"Image Size: {image.size[0]} × {image.size[1]}"
    )

    with st.spinner(
        "Running Prediction..."
    ):

        modality, tumor, probs = predict_image(
            image
        )

    modality_conf = (
        probs[0]
        if modality == "MRI"
        else (1 - probs[0])
    )

    tumor_conf = (
        probs[1]
        if tumor == "Tumor"
        else (1 - probs[1])
    )

    st.success(
        "Prediction Completed"
    )

    st.subheader(
        "Result"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Modality",
            modality
        )

        st.write(
            "Confidence"
        )

        st.progress(
            int(
                modality_conf
                * 100
            )
        )

        st.write(
            f"{modality_conf*100:.2f}%"
        )



    with c2:

        if tumor == "Tumor":

            st.error(
                "⚠ Tumor Detected"
            )

        else:

            st.success(
                "✅ Healthy"
            )

        st.write(
            "Confidence"
        )

        st.progress(
            int(
                tumor_conf
                * 100
            )
        )

        st.write(
            f"{tumor_conf*100:.2f}%"
        )

    with st.expander(
        "Model Details"
    ):

        st.markdown("""
CNN Architecture

Input:
- RGB Image

Outputs:
- CT / MRI
- Healthy / Tumor

Framework:
- PyTorch

Deployment:
- FastAPI + Streamlit
""")

    st.warning(
        "For educational purposes only. Not medical advice."
    )


# ==================================
# CLEAR BUTTON
# ==================================

if st.button(
    "🗑 Clear Prediction"
):

    st.session_state.uploader_key += 1

    st.rerun()


# ==================================
# FOOTER
# ==================================

st.markdown("---")

st.caption(
    "Built using PyTorch • FastAPI • Streamlit"
)