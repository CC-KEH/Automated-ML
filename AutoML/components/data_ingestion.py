import os
import shutil
import zipfile
import pandas as pd
from AutoML.utils import logger
from AutoML.entity.config_entity import Data_Ingestion_Config

class Data_Ingestion:
    def __init__(self,config: Data_Ingestion_Config):
        self.config = config
    
    def save_data_to_path(self):
        ''' Saves the first file in the source directory to the specified data path
        '''
        # Get the full path of the source file
        source_dir = self.config.source_path
        source_file_name = os.listdir(source_dir)[0]  # Assuming there's only one file in the directory
        source_file = os.path.join(source_dir, source_file_name)
        
        # Ensure the destination directory exists
        data_path = self.config.data_path
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
        # Copy the file to the destination
        shutil.copy(source_file, data_path)

        print(f"File {source_file_name} copied successfully to {data_path}")
    
    def convert_to_csv(self):
        ''' If the data is in excel format or any other, converts it to csv
        '''
        logger.info("Converting data to csv format")
        for file in os.listdir(self.config.data_path):
            if file.endswith('.csv'):
                logger.info("Data is already in csv format")
                return
            elif file.endswith('.xlsx'):
                logger.info("Converting xlsx file to csv")
                data = pd.read_excel(os.path.join(self.config.data_path,file))
                data.to_csv(os.path.join(self.config.data_path,file.split('.')[0]+'.csv'),index=False)
            elif file.endswith('.xls'):
                logger.info("Converting xls file to csv")
                data = pd.read_excel(os.path.join(self.config.data_path,file))
                data.to_csv(os.path.join(self.config.data_path,file.split('.')[0]+'.csv'),index=False)
    
    def initiate_data_ingestion(self):
        ''' Initiates the data ingestion process
        '''
        logger.info("Initiating Data Ingestion")
        self.save_data_to_path()
        # self.convert_to_csv()
        logger.info("Data Ingestion Completed")