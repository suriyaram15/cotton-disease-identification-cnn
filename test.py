import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('models/best_model.h5')

# Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('models/model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted to TFLite and saved.")

# Optional: Quantization for smaller size (uncomment if needed)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# quantized_tflite_model = converter.convert()
# with open('models/model_quantized.tflite', 'wb') as f:
#     f.write(quantized_tflite_model)
# print("Quantized model saved.")