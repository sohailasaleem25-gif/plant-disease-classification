import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "sohaila-12345/plant-disease-vit"

st.set_page_config(
    page_title="Plant Disease Classification",
    page_icon="🌿",
    layout="centered"
)

@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model

st.title("🌿 Plant Disease Classification")
st.write("Upload a plant leaf image to identify its disease.")

try:
    processor, model = load_model()
    st.success("Model loaded successfully!")

    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Leaf Image", width=300)

        if st.button("Predict Disease"):
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                predicted_id = outputs.logits.argmax(dim=1).item()

            label = model.config.id2label[predicted_id]
            confidence = probabilities[0, predicted_id].item()

            st.subheader("Prediction")
            st.write(f"**Class:** {label}")
            st.write(f"**Confidence:** {confidence:.2%}")

except Exception as e:
    st.error(f"Error loading model: {e}")
