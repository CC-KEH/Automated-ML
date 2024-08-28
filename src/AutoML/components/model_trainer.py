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
from sklearn.mode_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, precision_score, recall_score

from src.AutoML.config.configuration import Configuration_Manager

class Regression_Model_Trainer:
    def __init__(self) -> None:
        self.config, self.params = Configuration_Manager().get_regression_model_trainer_config()
        self.models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'ElasticNet': ElasticNet(),
            'DecisionTreeRegressor': DecisionTreeRegressor(),
            'RandomForestRegressor': RandomForestRegressor(),
            'SVR': SVR()
        }
        self.X_train, self.X_val, 
        self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train, 
            test_size=self.config['test_size'], 
            random_state=self.config['random_state']
        )
    
    def get_metrics(self, y_true, y_pred):
        return {
            'mean_squared_error': mean_squared_error(y_true, y_pred),
            'mean_absolute_error': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred)
        }
    
    def initiate_model_training(self):
        logger.info("Initiating Regression Model Training")
        for model_name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            save_model(model, model_name)
            logger.info(f'{model_name} model trained successfully')
            metrics = self.get_metrics(self.y_val,model.predict(self.X_val))
            logger.info(f'{model_name} \n Metrics: {metrics}')
            
class Classification_Model_Trainer:
    def __init__(self) -> None:
        self.config, self.params = Configuration_Manager().get_classification_model_trainer_config()
        self.models = {
            'LogisticRegression': LogisticRegression(),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
            'SVC': SVC(),
            'GaussianNB': GaussianNB(),
            'KNeighborsClassifier': KNeighborsClassifier()
        }
        self.X_train, self.X_val,
        self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train, 
            test_size=self.config['test_size'], 
            random_state=self.config['random_state']
        )
    
    def get_metrics(self, y_true, y_pred):
        return {
            'accuracy_score': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'precision_score': precision_score(y_true, y_pred),
            'recall_score': recall_score(y_true, y_pred)
        }
    
    def initiate_model_training(self):
        logger.info("Initiating Classification Model Training")
        for model_name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            save_model(model, model_name)
            logger.info(f'{model_name} model trained successfully')
            metrics = self.get_metrics(self.y_val,model.predict(self.X_val))
            logger.info(f'{model_name} \n Metrics: {metrics}')