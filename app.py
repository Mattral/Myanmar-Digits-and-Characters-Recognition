import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Function to preprocess the image
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict the digit
def predict_digit(model, image):
    processed_image = preprocess_image(image, (200, 200))  # Match your model's input size
    prediction = model.predict(processed_image)
    return np.argmax(prediction), np.max(prediction)

# Load your trained model
model = load_model("last_burmese_Digit_recognizer_model.h5")

# Streamlit app
st.title("Burmese Digit Recognizer")

# Upload image file or draw
st.markdown("## Upload an Image or Draw")
col1, col2 = st.columns(2)

with col1:
    file = st.file_uploader("Upload Here", type=['png', 'jpg', 'jpeg'])

with col2:
    # Drawable canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Drawing parameters
        stroke_width=3,
        stroke_color="#ffffff",
        background_color="#000000",
        background_image=None if file else st.session_state.get("background", None),
        update_streamlit=True,
        width=400,
        height=400,
        drawing_mode="freedraw",
        key="canvas",
    )

image = None  # Initialize image variable

# Process uploaded image or drawing
if file is not None:
    image = Image.open(file)  # Read image with PIL
elif canvas_result.image_data is not None:
    image = Image.fromarray(np.array(canvas_result.image_data, dtype=np.uint8)).convert('RGB')

if image is not None:
    st.image(image, caption='Uploaded Image')  # Display the uploaded/drawn image

    # Predict the digit
    digit, confidence = predict_digit(model, image)
    st.write(f"Predicted Digit: {digit} with confidence {confidence:.2f}")
else:
    st.write("Please upload an image or use the canvas to draw.")

