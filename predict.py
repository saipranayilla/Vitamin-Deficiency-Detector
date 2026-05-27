from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("vitamin_model.h5")

# Class labels (must match training order)
class_names = ['Vitamin_A', 'Vitamin_B1', 'Vitamin_C', 'Vitamin_D']

@app.route("/", methods=["GET", "POST"])
def predict():
    result = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            img_path = "uploaded_image.jpg"
            file.save(img_path)

            # Load and preprocess image
            img = image.load_img(img_path, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]

            result = predicted_class

            # Remove uploaded image
            os.remove(img_path)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)