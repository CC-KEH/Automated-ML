import os
import numpy as np
import pandas as pd
from src.AutoML.utils import logger
from src.AutoML.entity.config_entity import Data_Transformation_Config

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Regression_Data_Transformation:
    def __init__(self,config: Data_Transformation_Config):
        self.config = config
        self.data = self.config.data_path
        self.max_allowed_features = 10
        self.max_selected_features = 20
        self.pca_components = 10
        
        
    def reduce_dimensionality(self):
        pca = PCA(n_components=self.pca_components)
        pca.fit(self.data)
        self.data = pca.transform(self.data)
    
    def standardize_data(self):
        standard_scaler = StandardScaler()
        standard_scaler.fit(self.data)
        self.data = standard_scaler.transform(self.data)
    
    def select_features(self):
        corr_matrix = self.data.corr()
        corr_matrix = np.abs(corr_matrix)
        target_corr = corr_matrix['target']
        target_corr = target_corr.sort_values(ascending=False)
        positive_features = target_corr[:10].index
        negative_features = target_corr[-10:].index
        selected_features = positive_features.append(negative_features) 
        self.data = self.data[selected_features]
        
    def extract_features(self):
        pass
    
    def save_data(self):
        pass
    
    def initiate_data_transformation(self):
        logger.info("Initiating Data Transformation")
        # Count no of features
        feature_count = len(self.data.columns)
        if feature_count >= 30:
            logger.info("Feature count is greater than 20, selecting top 10 features using correlation")
            self.select_features()
        
        if feature_count >= 20:
            logger.info("Feature count is greater than 20, reducing dimensionality using PCA")
            self.reduce_dimensionality()
            
        self.standardize_data()

class Classification_Data_Transformation:
    def __init__(self,config: Data_Transformation_Config):
        self.config = config
    
    def reduce_dimensionality(self):
        pass
    
    def standardize_data(self):
        pass
    
    def select_features(self):
        pass
    
    def extract_features(self):
        pass
    
    def save_data(self):
        pass
    
    def initiate_data_transformation(self):
        logger.info("Initiating Data Transformation")
        # Count no of features
        feature_count = len(self.data.columns)
        if feature_count >= 30:
            logger.info("Feature count is greater than 20, selecting top 10 features using correlation")
            self.select_features()
        
        if feature_count >= 20:
            logger.info("Feature count is greater than 20, reducing dimensionality using PCA")
            self.reduce_dimensionality()
            
        self.standardize_data()
        