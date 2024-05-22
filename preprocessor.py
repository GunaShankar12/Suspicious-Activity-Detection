import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import model_from_json, load_model
import io
import tempfile
import os
import json
import matplotlib.pyplot as plt

def main():
    st.title("Video Input App")
    
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
    
    if uploaded_file is not None:
        # Read video file
        video_bytes = uploaded_file.read()
        st.video(video_bytes)

        # Process the frames and make predictions
        frames = extract_and_process_frames(video_bytes)
        fashion_model = load_model_with_weights()
        if fashion_model:
            predictions = predict_frames(frames, fashion_model)
            display_predictions(predictions)

def extract_and_process_frames(video_bytes):
    """
    Extracts frames from the given video file, converts each frame to grayscale,
    resizes it to 50x50 pixels, and returns them in a NumPy array.

    Parameters:
    - video_bytes: The bytes of the video file.

    Returns:
    - A NumPy array containing processed frames of shape (num_frames, 50, 50, 1).
    """
    # Create a temporary file to write video bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(video_bytes)
        temp_file_path = temp_file.name
    
    # Open the temporary file with cv2.VideoCapture
    cap = cv2.VideoCapture(temp_file_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        resized_frame = cv2.resize(gray_frame, (50, 50))  # Resize the frame
        frames.append(resized_frame.reshape(50, 50, 1))  # Add an extra dimension for consistency

    cap.release()

    # Remove the temporary file
    if temp_file_path:
        os.unlink(temp_file_path)

    return np.array(frames)

def load_model_with_weights():
    """
    Load the model architecture and weights.

    Returns:
    - The loaded model.
    """
    # Load the model architecture from JSON
    with open("model_architecture.json", "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

    # Load the weights into the model
    loaded_model.load_weights("CNN_LSTM.h5")
    

    return loaded_model

def predict_frames(frames, fashion_model):
    """
    Predicts the classes of frames using the fashion_model.

    Parameters:
    - frames: The frames to predict, processed for both CNN and LSTM parts.
    - fashion_model: The trained model that will make the prediction.

    Returns:
    - The predicted classes of the frames.
    """
    # Assuming frames_lstm is processed for LSTM part, though this part is unclear in your code
    frames_lstm = frames.reshape(frames.shape[0], -1, 1) 
    test_prob = fashion_model.predict([frames, frames_lstm])
    y_pred = np.argmax(test_prob, axis=1)
    return y_pred

def display_predictions(predictions):
    """
    Display the predictions.

    Parameters:
    - predictions: The predicted classes of frames.
    """
    categories_labels = {
        0: 'Abuse',
        1: 'Arrest',
        2: 'Arson',
        3: 'Assault',
        4: 'Burglary',
        5: 'Explosion',
        6: 'Fighting',
        7: 'Normal',
        8: 'RoadAccidents',
        9: 'Robbery',
        10: 'Shooting',
        11: 'Shoplifting',
        12: 'Stealing',
        13: 'Vandalism'
    }

    st.write("Predicted Classes:")
   
    unique_values, counts = np.unique(predictions, return_counts=True)
   
    value_counts_dict = dict(zip(unique_values.tolist(), counts.tolist()))

    

    labels = []
    values = []
    for label in categories_labels.keys():
        if label not in value_counts_dict:
            continue
        labels.append(categories_labels[label])
        values.append(value_counts_dict.get(label, 0))
    print(labels, values)

    # Create a bar graph for the predictions
    fig, ax = plt.subplots()
    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.legend(title="Categories", loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.title('Pie Chart')

    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.axis('equal')

    # Display the pie chart using Streamlit
    st.pyplot(plt)

if __name__ == '__main__':
    main()
