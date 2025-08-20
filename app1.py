import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Vitiligo Skin Classifier",
    page_icon="🔬",
    layout="centered"
)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    """Load the trained Keras model."""
    try:
        model = tf.keras.models.load_model('vitiligo_classifier.keras')
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}", icon="🚨")
        return None

model = load_model()

# --- USER INTERFACE ---
st.title("🔬 Vitiligo Skin Classifier")
st.write(
    "Upload an image of a skin patch, and this app will predict whether it shows signs of Vitiligo or is Non-Vitiligo (Healthy)."
)

uploaded_file = st.file_uploader(
    "Choose a skin image...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    
    with st.spinner("Classifying..."):
        # Preprocess the image for the model
        image_resized = image.resize((224, 224))
        image_array = np.array(image_resized)
        image_array = image_array / 255.0  # Rescale
        image_batch = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Make a prediction
        prediction = model.predict(image_batch)
        score = prediction[0][0]
        
        # Display the result
        st.subheader("Prediction")
        if score > 0.5:
            st.success(f"**Result: Vitiligo** (Confidence: {score:.2%})", icon="✅")
        else:
            st.success(f"**Result: Non-Vitiligo** (Confidence: {1 - score:.2%})", icon="✅")
        
        st.info("This is a preliminary analysis. Please consult a dermatologist for a professional diagnosis.", icon="ℹ️")