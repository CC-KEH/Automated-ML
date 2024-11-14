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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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
            logger.info("Reducing dimensionality using PCA")
            logger.info("Data shape before PCA: {}".format(self.data.shape))
            pca = PCA(n_components=self.pca_components)
            self.data = pca.fit_transform(self.data)
            logger.info("Data shape after PCA: {}".format(self.data.shape))
        else:
            logger.info("Reducing dimensionality using LDA")
            logger.info("Data shape before LDA: {}".format(self.data.shape))
            lda = LDA(n_components=self.pca_components)
            self.data = lda.fit_transform(self.data,self.target)
            logger.info("Data shape after LDA: {}".format(self.data.shape))

    def standardize_data(self, features=None):
        logger.info("Standardizing the data")
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
        # If self.data is not a DataFrame, convert it back
        if isinstance(self.data, np.ndarray):
            self.data = pd.DataFrame(self.data, columns=[f'feature_{i}' for i in range(self.data.shape[1])])

        # Ensure that self.target is a Series
        if not isinstance(self.target, pd.Series):
            self.target = pd.Series(self.target, name='target')

        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, test_size=0.2, random_state=47)

        # Concatenate the splits for saving
        train_data = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
        test_data = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)
        
        train_data.dropna(inplace=True)
        test_data.dropna(inplace=True)
                
        logger.info("Train set shape: {}".format(train_data.shape))
        logger.info("Test set shape: {}".format(test_data.shape))
        
        # Save to CSV
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
            logger.info("Applying Label Encoding")
            encoder = LabelEncoder()
            for feature in self.data.columns:
                if self.data[feature].dtype == 'object':
                    self.data[feature] = encoder.fit_transform(self.data[feature])

        else:
            logger.info("Applying One Hot Encoding")
            encoder = OneHotEncoder()
            for feature in self.data.columns:
                if self.data[feature].dtype == 'object':
                    encoded_categories = encoder.fit_transform(self.data[feature])
                    self.data = self.data.drop(feature, axis=1)
                    self.data = pd.concat([self.data, encoded_categories], axis=1)
    
    def initiate_data_transformation(self, manual_config):
        logger.info("Initiating Data Transformation")
        self.data = pd.read_csv(self.config.data_path)
        
        # 1. Remove duplicates and handle NaNs.
        self.data = self.data.drop_duplicates().fillna(method='ffill')
        self.target = self.data['target']
        self.data = self.data.drop('target', axis=1)
        
        # 2. Get numerical and object features.
        numerical_features, object_features = self.get_features()

        # 3. Apply transformations based on config.
        if manual_config == None:
            self.auto_transformation(numerical_features, object_features)
        else:
            self.manual_transformation(manual_config, numerical_features, object_features)

        # 4. Save the transformed data.
        self.split_and_save_data()

    def auto_transformation(self, numerical_features, object_features):
        """Handles auto data transformation workflow."""
        categorical_features, _ = self.get_categorical_features(object_features)
        if categorical_features:
            self.apply_encoding(technique='label')
        self.standardize_data()
        if len(self.data.columns) >= 30:
            self.select_features()
        if len(self.data.columns) >= 20:
            self.reduce_dimensionality()

    def manual_transformation(self, manual_config, numerical_features, object_features):
        """Handles manual data transformation based on provided config."""
        if manual_config['train_numerical']:
            self.data = self.data[numerical_features]
        else:
            self.apply_encoding(technique=manual_config['encoding_type'])
        self.standardize_data()
        
        if manual_config['dimension_reduction'] == 'pca':
            self.reduce_dimensionality(technique='pca')
        
        elif manual_config['dimension_reduction'] == 'lda':
            self.reduce_dimensionality(technique='lda')