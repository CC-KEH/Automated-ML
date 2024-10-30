import os
import sys
import json
import shutil
import logger
import pandas as pd
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, jsonify

from AutoML.constants import MANUAL_CONFIG_PATH

app = Flask(__name__)

# Folder where uploaded files will be saved
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def reset():
    # Remove the manual config file and the uploads directory
    if os.path.exists('manual_config.json'):
        os.remove('manual_config.json')
    if os.path.exists('uploads'):
        shutil.rmtree('uploads')
    logger.info('Configurations reset successfully!')
    

@app.route('/', methods=['GET'])
def homepage():
    reset()
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'dataset' not in request.files:
        return 'No file part'
    
    file = request.files['dataset']
    
    if file.filename == '':
        return 'No selected file'
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        os.system(f'{sys.executable} main.py')
    
    return 'Invalid file type'

@app.route('/train', methods=['POST'])
def train():
    # Simulate training by calling an external script
    os.system(f'{sys.executable} main.py')  # Assuming this is your training script
    return 'Training Successful!'

@app.route('/manual_config',methods=['GET'])
def manual_config():
    return render_template('manual.html')

@app.route('/manual_train', methods=['POST'])
def manual_train():
    # Parse the received JSON data
    config_data = request.json
    dataset = config_data['dataset']
    # copy dataset in uploads folder
    print(dataset)
    # shutil.copy(dataset, 'uploads/')

    
    # Save the config data to a JSON file
    with open('manual_config.json', 'w') as config_file:
        json.dump(config_data, config_file, indent=4)

    # Placeholder message - replace with actual training logic
    response_message = 'Configuration received and saved. Model training would start here.'

    os.system(f'{sys.executable} main.py')
    # Return a response
    return jsonify({'message': response_message})


@app.route('/model', methods=['GET'])
def result():

    with open("manual_config.json", "r") as f:
        algorithm = json.load(f)['algorithm']
    
    # name, description, hyperparameters, metrics
    with open("models_info.json", "r") as f:
        models_info = json.load(f)
        algorithm_info = models_info["models"].get(algorithm, None)
    
    with open("artifacts/model_evaluation/metrics.json", "r") as f:
        metrics = json.load(f)

    # For testing send form with features and target will be predicted
    data = pd.read_csv("artifacts/data_ingestion/data.csv")
    features = data.columns.tolist().remove('target')
    
    return render_template('model.html', algorithm_info=algorithm_info, metrics=metrics, features=features)


@app.route("/predict", methods=["POST"])
def predict():
    pass

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host="0.0.0.0", port=8000, debug=True)