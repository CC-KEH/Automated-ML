import os
import sys
import json

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

@app.route('/', methods=['GET'])
def homepage():
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

    # Save the config data to a JSON file
    with open('manual_config.json', 'w') as config_file:
        json.dump(config_data, config_file, indent=4)

    # Placeholder message - replace with actual training logic
    response_message = 'Configuration received and saved. Model training would start here.'

    os.system(f'{sys.executable} main.py')
    # Return a response
    return jsonify({'message': response_message})

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host="0.0.0.0", port=8000, debug=True)