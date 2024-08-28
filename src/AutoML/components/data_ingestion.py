import os
import zipfile
from src.AutoML.utils import logger
from src.AutoML.entity.config_entity import Data_Ingestion_Config

class Data_Ingestion:
    def __init__(self,config: Data_Ingestion_Config):
        self.config = config
    
    def save_data_to_path(self, zipped_data):
        ''' Saves the zipped data to the path
        '''
        data_path = self.config.data_path
        os.makedirs(os.path.dirname(data_path),exist_ok=True)
        with open(data_path,'wb') as f:
            f.write(zipped_data)
    
    def initiate_data_ingestion(self, zipped_data):
        ''' Initiates the data ingestion process
        '''
        logger.info("Initiating Data Ingestion")
        self.save_data_to_path(zipped_data)
        self.extract_zip_data()
        logger.info("Data Ingestion Completed")
        
    
    def extract_zip_data(self):
        ''' Extracts the zip file
        '''
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path,exist_ok=True)
        with zipfile.ZipFile(self.config.data_path,'r') as zip_ref:
            zip_ref.extractall(unzip_path)