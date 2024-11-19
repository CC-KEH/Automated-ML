import os
import sys
import json
import shutil
import logger
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file

from AutoML.logger import logger
from AutoML.pipelines.prediction import PredictionPipeline

app = Flask(__name__)

# Folder where uploaded files will be saved
UPLOAD_FOLDER = "uploads/"
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Check if the file has an allowed extension
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def reset():
    # Remove the manual config file and the uploads directory
    if os.path.exists("manual_config.json"):
        os.remove("manual_config.json")

    # if os.path.exists('uploads'):
    #     shutil.rmtree('uploads')

    if os.path.exists("artifacts"):
        shutil.rmtree("artifacts")
    logger.info("Configurations reset successfully!")


@app.route("/", methods=["GET"])
def homepage():
    reset()
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "dataset" not in request.files:
        return "No file part"

    file = request.files["dataset"]

    if file.filename == "":
        return "No selected file"

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

        os.system(f"{sys.executable} main.py")

    return "Invalid file type"



@app.route("/manual_config", methods=["GET"])
def manual_config():
    return render_template("manual.html")

@app.route("/train", methods=["POST"])
def train():

    os.system(f"{sys.executable} main.py") 
    
    algorithm_name, algorithm_info, metrics = get_model_info()
    data = pd.read_csv("artifacts/data_ingestion/data.csv")
    features = data.columns.tolist()
    features.remove("target")
    return render_template("model.html", 
                           model_name=algorithm_name, 
                           algorithm_info=algorithm_info, 
                           metrics=metrics, 
                           features=features)


def manually_train():
    os.system(f"{sys.executable} main.py") 
    
    algorithm_name, algorithm_info, metrics = get_model_info()
    data = pd.read_csv("artifacts/data_ingestion/data.csv")
    features = data.columns.tolist()
    features.remove("target")
    return algorithm_name, algorithm_info, metrics, features

@app.route("/manual_train", methods=["POST", "GET"])
def manual_train():
    # Parse the received JSON data
    config_data = request.json
    
    # Save the config data to a JSON file
    with open("manual_config.json", "w") as config_file:
        json.dump(config_data, config_file, indent=4)

    algorithm_name, algorithm_info, metrics, features = manually_train()
    print(algorithm_name, algorithm_info, metrics, features)
    return render_template(
        "model.html",
        model_name=algorithm_name,
        algorithm_info=algorithm_info,
        metrics=metrics,
        features=features,
    )

@app.route('/model')
def model():
    return render_template('model.html')

def get_model_info():
    algorithm = os.listdir("artifacts/model_trainer")[0].split(".")[0]

    # name, description, hyperparameters, metrics
    with open("models_info.json", "r") as f:
        models_info = json.load(f)
        algorithm_name = models_info[algorithm]["name"]
        algorithm_info = models_info[algorithm]["description"]

    with open("artifacts/model_evaluation/metrics.json", "r") as f:
        metrics = json.load(f)

    return algorithm_name, algorithm_info, metrics


@app.route('/download_model', methods=['GET'])
def download_model():
    model_dir = 'artifacts/model_trainer'
    
    try:
        model_files = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f)) and f.endswith('.pkl')]
        
        if len(model_files) == 0:
            return jsonify({"error": "No model file found in the directory"})
        elif len(model_files) > 1:
            return jsonify({"error": "Multiple model files found. Please ensure only one model file exists."})
        
        # If there is exactly one model file, proceed to send it
        model_file_path = os.path.join(model_dir, model_files[0])
        return send_file(model_file_path, as_attachment=True)
    
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/predict", methods=["POST"])
def predict():
    # Parse the received JSON data
    prediction_data = request.json
    prediction_dict = {}
    for key, value in prediction_data.items():
        prediction_dict[key] = value
    for key, value in prediction_dict.items():
        # If value can be converted to float or int, convert it
        try:
            prediction_dict[key] = int(value)
        except ValueError:
            try:
                prediction_dict[key] = float(value)
            except ValueError:
                pass
    data = [val for val in prediction_dict.values()]
    print(data)
    # Fix the issue of no of features mismatch, model might be trained on different no of features
    data = np.array(data).reshape(1,len(data))
    
    pipeline = PredictionPipeline()
    output = pipeline.predict(data)
    if output == 1:
        output = 'True'
    else:
        output = 'False'
    return render_template("result.html", output=output)

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host="0.0.0.0", port=8000, debug=True)