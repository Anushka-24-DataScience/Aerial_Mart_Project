import streamlit as st
import base64
import os
from AerialDetection.pipeline.training_pipeline import TrainPipeline
from AerialDetection.utils.main_utils import decodeImage, encodeImageIntoBase64

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.filepath = os.path.join(os.getcwd(), self.filename)  # Full path to the image

clApp = ClientApp()

def encode_image(image_bytes):
    """Encodes image bytes to base64"""
    encoded = base64.b64encode(image_bytes).decode('utf-8')
    return encoded

# Streamlit app title and description
st.title("Aerial Detection Streamlit App")

# Sidebar for navigation
option = st.sidebar.selectbox(
    'Choose an action:',
    ['Home', 'Train Model', 'Predict Image', 'Live Detection']
)

# Home page
if option == 'Home':
    st.write("Welcome to the Aerial Detection App!")

# Train Model Page
elif option == 'Train Model':
    st.header("Train the Model")
    if st.button("Start Training"):
        try:
            obj = TrainPipeline()
            obj.run_pipeline()
            st.success("Training Successful!!")
        except Exception as e:
            st.error(f"Training failed: {e}")

# Predict Image Page
elif option == 'Predict Image':
    st.header("Predict Image")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        
        # Save uploaded image to the file system
        with open(clApp.filepath, "wb") as f:
            f.write(image_bytes)
        
        if st.button("Predict"):
            try:
                # Run the YOLOv5 model on the image
                os.system(f"cd yolov5/ && python detect.py --weights best.pt --img 416 --conf 0.5 --source {clApp.filepath}")

                # Display the output image
                output_image_path = "yolov5/runs/detect/exp/inputImage.jpg"
                st.image(output_image_path, caption="Predicted Image")
                os.system("rm -rf yolov5/runs")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# Live Detection Page
elif option == 'Live Detection':
    st.header("Live Detection")
    if st.button("Start Live Detection"):
        try:
            os.system("cd yolov5/ && python detect.py --weights my_model.pt --img 416 --conf 0.5 --source 0")
            st.success("Camera started!!")
            os.system("rm -rf yolov5/runs")
        except Exception as e:
            st.error(f"Error starting live prediction: {e}")
