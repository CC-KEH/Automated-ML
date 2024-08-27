import joblib
import pandas as pd
from src.AutoML.utils import logger
from src.AutoML.utils.common import save_model
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.mode_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, precision_score, recall_score

from src.AutoML.config.configuration import Configuration_Manager

class Regression_Model_Trainer:
    def __init__(self) -> None:
        self.config = Configuration_Manager().get_regression_model_trainer_config()
        self.models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'ElasticNet': ElasticNet(),
            'DecisionTreeRegressor': DecisionTreeRegressor(),
            'RandomForestRegressor': RandomForestRegressor(),
            'SVR': SVR()
        }
        self.X_train = None
        self.y_train = None
    
    def initiate_model_training(self):
        logger.info("Initiating Regression Model Training")
        for model_name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            save_model(model, model_name)
            logger.info(f'{model_name} model trained successfully')

class Classification_Model_Trainer:
    def __init__(self) -> None:
        self.config = Configuration_Manager().get_classification_model_trainer_config()
        self.models = {
            'LogisticRegression': LogisticRegression(),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
            'SVC': SVC(),
            'GaussianNB': GaussianNB(),
            'KNeighborsClassifier': KNeighborsClassifier()
        }
    
    def initiate_model_training(self):
        logger.info("Initiating Classification Model Training")
        for model_name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            save_model(model, model_name)
            logger.info(f'{model_name} model trained successfully')