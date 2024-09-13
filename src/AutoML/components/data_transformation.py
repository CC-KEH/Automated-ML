import os
import numpy as np
import pandas as pd

from src.AutoML.utils import logger
from src.AutoML.entity.config_entity import Data_Transformation_Config

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


class Data_Transformation:
    def __init__(self, config: Data_Transformation_Config):
        """Initializes the regression data transformation with config settings."""
        self.config = config
        self.data = pd.read_csv(self.config.data_path)  # Load data from CSV
        self.max_allowed_features = 10
        self.max_selected_features = 20
        self.pca_components = 10

    def reduce_dimensionality(self):
        pca = PCA(n_components=self.pca_components)
        self.data = pca.fit_transform(self.data)

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

    def save_data(self):
        """Saves the transformed data to the output path defined in config."""
        output_path = os.path.join(self.config.output_dir, "transformed_data.csv")
        self.data.to_csv(output_path, index=False)
        logger.info(f"Transformed data saved at {output_path}")

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

    def initiate_data_transformation(self):
        """Initiates the complete data transformation process."""
        logger.info("Initiating Regression Data Transformation")

        # Count number of features
        feature_count = len(self.data.columns)
        numerical_features, object_features = self.get_features()
        categorical_features, object_features = self.get_categorical_features(object_features)

        # Standardize numerical features
        self.standardize_data(numerical_features)

        # Perform feature selection and dimensionality reduction based on feature count
        if feature_count >= 30:
            logger.info("Feature count is greater than 30, selecting top 10 features using correlation")
            self.select_features()

        if feature_count >= 20:
            logger.info("Feature count is greater than 20, reducing dimensionality using PCA")
            self.reduce_dimensionality()

        # Standardize the final data
        self.standardize_data()

        # Save the transformed data
        self.save_data()