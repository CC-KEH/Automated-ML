import os
import shutil
import pandas as pd
import pymongo
from AutoML.logger import logger
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
        # Rename the file to data.csv
        data_path = os.path.join(data_path, "data.csv")
        
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
    
    def store_to_mongo(self):
        client = pymongo.MongoClient(self.config.connection_url)
        db = client[self.config.database_name]
        collection = db[self.config.collection_name]
        data = pd.read_csv(self.config.data_path)
        records = data.to_dict(orient='records')
        collection.insert_many(records)
        logger.info("Data stored in MongoDB")
        
    def fetch_data_from_mongo(self):
        client = pymongo.MongoClient(self.config.connection_url)
        db = client[self.config.database_name]
        collection = db[self.config.collection_name]
        data = pd.DataFrame(list(collection.find()))
        return data
    
    def initiate_data_ingestion(self, manual_config=None):
        ''' Initiates the data ingestion process
        '''
        if manual_config == 'local':
            logger.info("Initiating Data Ingestion")
            self.save_data_to_path()
            # self.store_to_mongo()
            # self.convert_to_csv()
            logger.info("Data Ingestion Completed") 
        else:
            # Load data from MongoDB
            logger.info("Initiating Data Ingestion")
            data = self.fetch_data_from_mongo()
            data.to_csv(self.config.data_path,index=False)
            logger.info("Data Ingestion Completed")