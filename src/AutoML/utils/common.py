import os
import yaml
import json
import joblib
from src.AutoML.utils import logger
from typing import Any
from pathlib import Path
from box.exceptions import BoxValueError  # For Custom Exceptions
from box import ConfigBox
from ensure import ensure_annotations

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    '''
    Read Yaml file and returns

    Args:
        path_to_yaml(str): path as input
    Raises:
        ValueError if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox Type
    '''
    try:
        with open(path_to_yaml, 'r', encoding="utf8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f'Yaml file:{path_to_yaml} loaded successfully')
            return ConfigBox(content)

    except BoxValueError:
        raise ValueError('Yaml file is empty')

    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    '''
    Creates list of directories
    Args:
        path_to_directories: list of path of directories
        verbose: For Logging
    '''
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f'Created directory at: {path}')


@ensure_annotations
def save_json(path: Path, data: dict):
    ''' save json data
    Args:
        path: destination path for json
        data: data to be stored
    '''
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

    logger.info(f'Json file saved at: {path}')


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    '''loads json file
    Args:
        path: Path for Json file to load
    '''
    with open(path) as file:
        content = json.load(file)

    logger.info(f'Json file loaded from: {path}')
    return ConfigBox(content)


@ensure_annotations
def save_bin(path: Path, data: Any):
    '''Save Binary File
    '''
    joblib.dump(value=data, filename=path)
    logger.info(f'Saved binary file at: {path}')


@ensure_annotations
def load_bin(path: Path) -> Any:
    ''' loads binary file
    '''
    data = joblib.load(path)
    logger.info(f'Successfully loaded binary file from: {path}')
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    ''' Get Size in KB
    Args:
        path: path.
    Returns:
        str: size.
    '''
    size_in_kb = round(os.path.getsize(path)/1024)
    return f'~ {size_in_kb} KB'