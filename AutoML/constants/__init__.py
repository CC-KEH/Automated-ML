from pathlib import Path
import os
from datetime import datetime

CONFIG_PATH = Path("config/config.yaml")
PARAMS_PATH = Path("params.yaml")
MANUAL_CONFIG_PATH = Path("manual_config.json")

PIPELINE_NAME = "AutoML"
ARTIFACT_DIR = "artifacts"
TIMESTAMP = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")