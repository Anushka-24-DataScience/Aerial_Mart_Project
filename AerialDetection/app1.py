import sys
import os
from AerialDetection.pipeline.training_pipeline import TrainPipeline
from AerialDetection.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin
from AerialDetection.constant.application import APP_HOST, APP_PORT

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"

clApp = ClientApp()  # Ensure the ClientApp instance is initialized here

@app.route("/train")
def trainRoute():
    obj = TrainPipeline()
    obj.run_pipeline()
    return "Training Successful!!" 

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST','GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)

        os.system("cd yolov5/ && python detect.py --weights best.pt --img 416 --conf 0.5 --source ../data/inputImage.jpg")

        opencodedbase64 = encodeImageIntoBase64("yolov5/runs/detect/exp/inputImage.jpg")
        result = {"image": opencodedbase64.decode('utf-8')}
        os.system("rm -rf yolov5/runs")

    except ValueError as val:
        print(f"ValueError: {val}")
        return Response("Value not found inside JSON data", status=400)
    except KeyError as key_err:
        print(f"KeyError: {key_err}")
        return Response("Incorrect key passed in JSON data", status=400)
    except Exception as e:
        print(f"Exception: {e}")
        return Response("Invalid input", status=500)

    return jsonify(result)

@app.route("/live", methods=['GET'])
@cross_origin()
def predictLive():
    try:
        os.system("cd yolov5/ && python detect.py --weights my_model.pt --img 416 --conf 0.5 --source 0")
        os.system("rm -rf yolov5/runs")
        return "Camera starting!!" 
    except ValueError as val:
        print(f"ValueError: {val}")
        return Response("Value not found inside JSON data", status=400)
    except Exception as e:
        print(f"Exception: {e}")
        return Response("Invalid input", status=500)

if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT)  # Corrected the syntax here
