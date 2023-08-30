import tkinter as tk
import cv2
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from utils import num_to_char
from modelutil import load_model

# Load the lip-reading model
lip_reader_model = load_model()
sequence_length = 75
frame_buffer = []

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (140, 46))
    frame = tf.expand_dims(frame, axis=-1)
    frame = tf.cast(frame, tf.float32)
    frame = (frame - tf.math.reduce_mean(frame)) / tf.math.reduce_std(frame)
    return frame

def predict_lip(frame):
    global frame_buffer

    # Preprocess the frame to match the model input shape
    frame = preprocess_frame(frame)

    # Add the frame to the buffer
    frame_buffer.append(frame)

    # If the buffer size exceeds the sequence length, remove the oldest frames
    if len(frame_buffer) > sequence_length:
        frame_buffer = frame_buffer[-sequence_length:]

    # Make predictions when the buffer contains enough frames
    if len(frame_buffer) == sequence_length:
        # Prepare the input sequence for the model
        input_sequence = tf.stack(frame_buffer, axis=0)
        input_sequence = tf.expand_dims(input_sequence, axis=0)

        # Make predictions with the lip-reading model
        prediction = lip_reader_model.predict(input_sequence)
        predicted_label = tf.argmax(prediction, axis=-1)[0]

        # Convert the predicted label to the corresponding character
        predicted_chars = num_to_char(predicted_label).numpy().astype(str).tolist()
        predicted_string = "".join(predicted_chars)

        return predicted_string

    return ""

def update_frame():
    ret, frame = cap.read()
    if ret:
        predicted_lip = predict_lip(frame)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        canvas.img = img
        canvas.itemconfig(canvas_image, image=img)
        label.config(text="Predicted Lip: " + predicted_lip)
    root.after(30, update_frame)

def on_close():
    cap.release()
    root.destroy()

# Initialize Tkinter
root = tk.Tk()
root.title("Live Lip-Reading App")

# Create canvas to display video
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

# Create label for predicted lip
label = tk.Label(root, text="", font=("Helvetica", 16))
label.pack()

# Get a video capture object
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    label.config(text="Error: Unable to access the camera.")
else:
    canvas_image = canvas.create_image(0, 0, anchor=tk.NW)
    update_frame()

# Bind the close event to release the camera before closing the app
root.protocol("WM_DELETE_WINDOW", on_close)

# Start the main loop
root.mainloop()
