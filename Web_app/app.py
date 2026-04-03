from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import tensorflow as tf
import os

from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess


app = Flask(__name__)

# Load models once when the server starts
xception_model = tf.keras.models.load_model("models/xception_model.keras")
resnet_model = tf.keras.models.load_model("models/resnet_model.keras")
mobilenet_model = tf.keras.models.load_model("models/mobilenet_model.keras")


def prepare_image(image, size, preprocess):
    
    image = image.resize(size)
    image = np.array(image)

    image = preprocess(image)

    image = np.expand_dims(image, axis=0)

    return image

def generate_explanation(result, confidence, model_name):

    if result == "Fake":
        if confidence >= 90:
            explanation = (
                f"The {model_name} model classified this image as a deepfake "
                f"with high confidence ({confidence}%). The model detected strong "
                f"visual patterns commonly associated with AI-generated faces, such "
                f"as inconsistencies in facial texture, unnatural skin smoothness, "
                f"or subtle asymmetries in facial features that are difficult for "
                f"generative models to replicate accurately."
            )
        elif confidence >= 70:
            explanation = (
                f"The {model_name} model classified this image as a deepfake "
                f"with moderate-to-high confidence ({confidence}%). Several visual "
                f"features suggest AI manipulation, though the model's certainty "
                f"is not absolute. Common indicators at this confidence level include "
                f"slight irregularities in lighting consistency, edge artefacts around "
                f"facial boundaries, or unnatural blending between the face and "
                f"background regions."
            )
        else:
            explanation = (
                f"The {model_name} model classified this image as a deepfake "
                f"with low-to-moderate confidence ({confidence}%). This suggests "
                f"the image contains some characteristics associated with AI-generated "
                f"faces, but the model is not highly certain. This may indicate a "
                f"high-quality deepfake that is difficult to detect, or an image with "
                f"naturally unusual lighting or compression artefacts. Independent "
                f"verification is recommended."
            )
    else:
        if confidence >= 90:
            explanation = (
                f"The {model_name} model classified this image as real "
                f"with high confidence ({confidence}%). The facial features, "
                f"lighting, texture and overall image composition are consistent "
                f"with authentic photographic content. No strong indicators of "
                f"AI-generated manipulation were detected."
            )
        elif confidence >= 70:
            explanation = (
                f"The {model_name} model classified this image as real "
                f"with moderate-to-high confidence ({confidence}%). The image "
                f"appears to contain natural facial characteristics consistent "
                f"with authentic photography, though some ambiguous features "
                f"were present. The model found no dominant indicators of "
                f"deepfake manipulation."
            )
        else:
            explanation = (
                f"The {model_name} model classified this image as real "
                f"with low-to-moderate confidence ({confidence}%). While no "
                f"clear manipulation was detected, the model's uncertainty "
                f"is relatively high. This could be due to unusual image "
                f"quality, lighting conditions, or facial angles that differ "
                f"from the training distribution. Caution is advised."
            )

    return explanation


@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    confidence = None
    error = None
    image_url = None
    explanation = None

    if request.method == "POST":

        selected_model = request.form.get("model")
        file = request.files.get("image")

        if not file:
            error = "Please upload an image."

        else:

            filepath = os.path.join("static", "uploaded_image.jpg")
            file.save(filepath)

            image_url = filepath

            img = Image.open(filepath).convert("RGB")

            if selected_model == "resnet50":

                processed = prepare_image(img, (224,224), resnet_preprocess)
                prediction = resnet_model.predict(processed)[0][0]

            elif selected_model == "mobilenetv2":

                processed = prepare_image(img, (224,224), mobilenet_preprocess)
                prediction = mobilenet_model.predict(processed)[0][0]

            else:

                processed = prepare_image(img, (299,299), xception_preprocess)
                prediction = xception_model.predict(processed)[0][0]


            model_display_names = {
                "resnet50": "ResNet50",
                "mobilenetv2": "MobileNetV2",
                "xception": "Xception"
            }

            if prediction > 0.5:
                result = "Real"
                confidence = round(prediction * 100, 2)
            else:
                result = "Fake"
                confidence = round((1 - prediction) * 100, 2)

            explanation = generate_explanation(
                result, 
                confidence, 
                model_display_names.get(selected_model, selected_model)
            )


    return render_template(
    "index.html",
    result=result,
    confidence=confidence,
    error=error,
    image_url=image_url,
    explanation=explanation
    )


if __name__ == "__main__":
    app.run(debug=True)