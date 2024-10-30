import os
from constants import MANUAL_CONFIG_PATH
import json
import numpy as np
import pandas as pd
from pathlib import Path

from AutoML.logger import logger
from AutoML.entity.config_entity import Data_Transformation_Config

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, LDA
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder,OneHotEncoder


class Data_Transformation:
    def __init__(self, config: Data_Transformation_Config):
        """Initializes the regression data transformation with config settings."""
        self.config = config
        self.max_allowed_features = 10
        self.max_selected_features = 20
        self.pca_components = 10

    def reduce_dimensionality(self, technique='pca'):
        if technique == 'pca':
            pca = PCA(n_components=self.pca_components)
            self.data = pca.fit_transform(self.data)
        else:
            lda = LDA(n_components=self.pca_components)
            self.data = lda.fit_transform(self.data)

    def standardize_data(self, features=None):
        standard_scaler = StandardScaler()
        if features:
            self.data[features] = standard_scaler.fit_transform(self.data[features])
        else:
            # Apply to all numerical features
            numerical_features, _ = self.get_features()
            self.data[numerical_features] = standard_scaler.fit_transform(self.data[numerical_features])

    def select_features(self):
        corr_matrix = self.data.corr().abs()  # Get absolute correlation matrix
        target_corr = corr_matrix['target'].sort_values(ascending=False)

        # Top positive and negative correlated features
        positive_features = target_corr[:self.max_allowed_features].index
        negative_features = target_corr[-self.max_allowed_features:].index

        selected_features = positive_features.append(negative_features)
        self.data = self.data[selected_features]

    def split_and_save_data(self):
        """Saves the transformed data to the output path defined in config."""
        X_train,X_test,y_train,y_test = train_test_split(self.data.drop('target', axis=1), self.data['target'], test_size=0.2, random_state=42)
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        train_data.to_csv(self.config.train_path, index=False)
        test_data.to_csv(self.config.test_path, index=False)
        logger.info("Splitted and saved the transformed data.")
        
    def get_features(self):
        """Returns numerical and object (categorical) features from the dataset."""
        numerical_features = self.data.select_dtypes(include=[np.number]).columns
        object_features = self.data.select_dtypes(include=[object]).columns
        return numerical_features, object_features

    def get_categorical_features(self, object_features):
        """
        Identifies categorical features from object columns with unique values below a threshold.

        Args:
            object_features: List of object type features.

        Returns:
            Tuple containing categorical features and non-categorical object features.
        """
        categorical_features = [feature for feature in object_features if self.data[feature].nunique() < 10]
        object_features = [feature for feature in object_features if feature not in categorical_features]
        return categorical_features, object_features

    def apply_encoding(self, technique='label'):
        """Applies encoding to categorical features based on the technique provided."""
        if technique == 'label':
            encoder = LabelEncoder()
            for feature in self.data.columns:
                if self.data[feature].dtype == 'object':
                    self.data[feature] = encoder.fit_transform(self.data[feature])

        else:
            encoder = OneHotEncoder()
            for feature in self.data.columns:
                if self.data[feature].dtype == 'object':
                    encoded_categories = encoder.fit_transform(self.data[feature])
                    self.data = self.data.drop(feature, axis=1)
                    self.data = pd.concat([self.data, encoded_categories], axis=1)
    
    
    def initiate_data_transformation(self,manual_config):
        """Initiates the complete data transformation process."""
        logger.info("Initiating Data Transformation")
       
        self.data = pd.read_csv(self.config.data_path)
        self.target = self.data['target']
        self.data = self.data.drop('target', axis=1)
        
        # Count number of features
        feature_count = len(self.data.columns)
        
        # Remove duplicate features
        self.data = self.data.drop_duplicates()
        
        numerical_features, object_features = self.get_features()
        
        categorical_features, object_features = self.get_categorical_features(object_features)
        
        if manual_config=='auto':
            if categorical_features:
                # ! Might Cause Error
                # Perform encoding on categorical features
                encoder = LabelEncoder()
                encoded_categories = encoder.fit_transform(self.data[categorical_features])
                self.data = self.data.drop(categorical_features, axis=1)

                # Standardize the data
                self.standardize_data()

                # Combine the encoded categories with the data
                self.data = pd.concat([self.data, encoded_categories], axis=1)

            else:
                self.standardize_data()

            self.data['target'] = self.target

            # Perform feature selection and dimensionality reduction based on feature count
            if feature_count >= 30:
                logger.info("Feature count is greater than 30, selecting top 10 features using correlation")
                self.select_features()

            if feature_count >= 20:
                logger.info("Feature count is greater than 20, reducing dimensionality using PCA")
                self.reduce_dimensionality()

            # Save the transformed data
            self.split_and_save_data()
        
        else:
            # Train only on numerical features
            if manual_config['train_numerical']:
                self.data = self.data[numerical_features]
                
            
            # Train on all features        
            else:
                # Encoding type for categorical features
                if not manual_config['n']:
                    self.apply_encoding(technique='label')

                else: # One-hot encoding
                    self.apply_encoding(technique='one-hot')
            
            self.standardize_data()
                 
            # Dimensionality reduction technique
            if manual_config['dimension_reduction'] == 'pca':
                self.reduce_dimensionality() # PCA by default
            
            elif manual_config['dimension_reduction'] == 'lda':
                self.reduce_dimensionality(technique='lda')
                
            # Save the transformed data
            self.split_and_save_data()