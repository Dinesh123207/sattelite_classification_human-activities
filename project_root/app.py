import os
import time
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin
from changeSegmentation.utils.main_utils import decodeImage, encodeImageIntoBase64
from changeSegmentation.constant.application import APP_HOST, APP_PORT
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"

def resize_image(image, target_size):
    target_size = (max(target_size[0], 3), max(target_size[1], 3))
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def generate_heatmap(image1, image2):
    # Convert images to grayscale
    target_size = (min(image1.shape[1], image2.shape[1]), min(image1.shape[0], image2.shape[0]))
    image1_resized = resize_image(image1, target_size)
    image2_resized = resize_image(image2, target_size)

    gray1 = cv2.cvtColor(image1_resized, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference
    diff = cv2.absdiff(gray1, gray2)

    # Apply a colormap for visualization
    heatmap_color = cv2.applyColorMap(diff, cv2.COLORMAP_JET)

    # Convert the heatmap to grayscale
    heatmap_gray = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2GRAY)

    return heatmap_gray

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST', 'GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        clApp = ClientApp()

        # Save the image to the data folder with the name inputImage.jpg
        print("Decoding and saving image...")
        decodeImage(image, clApp.filename)

        # Time tracking for YOLO execution
        start_time = time.time()
        print("Running the YOLO command...")
        os.system(
            "yolo task=segment mode=predict model=artifacts/model_trainer/best.pt conf=0.15 source=data/inputImage.jpg save=true")

        # Load the model and make predictions
        model = YOLO("artifacts/model_trainer/best.pt")
        results = model.predict(source="data/inputImage.jpg", conf=0.50, save=True)

        # Encode the result image to base64
        print("Encoding the result image to base64...")
        result_image_base64 = encodeImageIntoBase64("runs/segment/predict/inputImage.jpg")

        # Create the result dictionary
        result = {"image": result_image_base64.decode('utf-8')}
        end_time = time.time()
        print("Time taken to run YOLO:", end_time - start_time)

        # Print classes and confidence scores
        print("Classes:", results[0].boxes.cls.numpy())
        print("Confidence Scores:", results[0].boxes.conf.numpy())

        # Clean up temporary files
        print("Cleaning up temporary files...")
        os.system("rm -rf runs")
        os.remove("data/inputImage.jpg")

    except ValueError as val:
        print(val)
        return Response("Value not found inside JSON data")
    except KeyError:
        return Response("Key value error: incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"

    return jsonify(result)

@app.route('/generate_heatmap', methods=['POST'])
def generate_heatmap_api():
    try:
        # Assuming images are sent as files with names 'image1' and 'image2'
        image1_file = request.files['image1']
        image2_file = request.files['image2']

        # Read images from file objects
        image1 = cv2.imdecode(np.frombuffer(image1_file.read(), np.uint8), cv2.IMREAD_COLOR)
        image2 = cv2.imdecode(np.frombuffer(image2_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Generate the black and white heatmap
        heatmap_bw = generate_heatmap(image1, image2)

        # Encode the heatmap to base64 and return
        _, heatmap_encoded = cv2.imencode('.png', heatmap_bw)
        heatmap_base64 = base64.b64encode(heatmap_encoded).decode('utf-8')

        return jsonify({'heatmap': heatmap_base64})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT)
