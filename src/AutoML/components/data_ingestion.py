import os
import zipfile
import pandas as pd
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
    
    def check_zip_file(self):
        ''' Checks if the file is a zip file
        '''
        for file in os.listdir('uploads/'):
            if file.endswith('.zip'):
                zipped_data = file
                return zipped_data
        return None

    def extract_zip_data(self):
        ''' Extracts the zip file
        '''
        logger.info("Extracting data from zip file")
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path,exist_ok=True)
        with zipfile.ZipFile(self.config.data_path,'r') as zip_ref:
            zip_ref.extractall(unzip_path)
    
    def convert_to_csv(self):
        ''' If the data is in excel format or any other, converts it to csv
        '''
        logger.info("Converting data to csv format")
        for file in os.listdir(self.config.unzip_dir):
            if file.endswith('.csv'):
                logger.info("Data is already in csv format")
                return
            elif file.endswith('.xlsx'):
                logger.info("Converting xlsx file to csv")
                data = pd.read_excel(os.path.join(self.config.unzip_dir,file))
                data.to_csv(os.path.join(self.config.unzip_dir,file.split('.')[0]+'.csv'),index=False)
            elif file.endswith('.xls'):
                logger.info("Converting xls file to csv")
                data = pd.read_excel(os.path.join(self.config.unzip_dir,file))
                data.to_csv(os.path.join(self.config.unzip_dir,file.split('.')[0]+'.csv'),index=False)
    
    def initiate_data_ingestion(self):
        ''' Initiates the data ingestion process
        '''
        logger.info("Initiating Data Ingestion")
        zip_file = self.check_zip_file()
        if zip_file:
            self.save_data_to_path(zip_file)
        self.extract_zip_data()
        self.convert_to_csv()
        logger.info("Data Ingestion Completed")