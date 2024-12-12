# main_utils.py

import base64
import cv2
import numpy as np

def decodeImage(image_base64, output_filename):
    # Decode a base64 encoded image and save it as a file
    img_data = base64.b64decode(image_base64)
    with open(output_filename, "wb") as f:
        f.write(img_data)

def encodeImageIntoBase64(image_path):
    # Encode an image into base64
    with open(image_path, "rb") as f:
        img_data = f.read()
    return base64.b64encode(img_data)
