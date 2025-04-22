import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# Class labels
class_labels = {
    0: 'Aphids',
    1: 'Army worm',
    2: 'Bacterial Blight',
    3: 'Powdery Mildew',
    4: 'Target spot',
    5: 'Healthy'
}

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="models/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0,1]
    return img_array

def predict_disease(img_path):
    # Preprocess the image
    img_array = preprocess_image(img_path)
    
    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    # Run the inference
    interpreter.invoke()
    
    # Extract the output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(output_data[0])
    
    return class_labels[pred]

if __name__ == "__main__":
    img_path = input("Enter the path to the cotton leaf image: ")
    if os.path.exists(img_path):
        prediction = predict_disease(img_path)
        print(f"Prediction: {prediction}")
    else:
        print("File not found. Please check the path.")