from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Importing deps for image prediction
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route("/")
def home():
    return {"message": "Hello from backend"}

@app.route("/upload", methods=['POST'])
def upload():
    file = request.files['file']
    file.save('uploads/' + file.filename)

    # Load the image to predict
    img_path = f"./uploads/{file.filename}"
    img = image.load_img(img_path, target_size=(128, 128))
    x = image.img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)

    loaded_model = load_model('./unet.h5')

    # 4 CLASSES
    classes = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

    # Make the prediction
    prediction = loaded_model.predict(x)
    if os.path.exists(f"./uploads/{file.filename}"):
        os.remove(f"uploads/{file.filename}")
        
    prob = loaded_model.predict(x)
    print(prob)
    top = np.argmax(prob[0])
    print(classes[top])
    print(prob[0][top])
    # plt.imshow(image)
    
    return jsonify({"message": classes[top]})


if __name__ == '__main__':
    app.run(debug=True)

