import pandas as pd
from pathlib import Path
import mlflow
import mlflow.sklearn
import joblib
from sklearn.metrics import (accuracy_score, mean_squared_error, mean_absolute_error,
                             r2_score, f1_score, precision_score, recall_score)

from urllib.parse import urlparse
from src.AutoML.entity.config_entity import Model_Evaluation_Config
from src.AutoML.utils.common import save_json


class Classification_Model_Evaluation:
    def __init__(self,config:Model_Evaluation_Config):
        self.config = config
    
    def evaluate_metrics(self,actual,pred):
        accuracy = accuracy_score(actual,pred)
        mse = mean_squared_error(actual,pred)
        f1 = f1_score(actual,pred)
        precision = precision_score(actual,pred)
        recall = recall_score(actual,pred)
        return accuracy,mse,f1,precision,recall

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        X_test = test_data.drop([self.config.target_column],axis=1)
        y_test  = test_data[[self.config.target_column]]
        
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            predictions = model.predict(X_test)
            (accuracy,mse,f1,precision,recall) = self.evaluate_metrics(y_test,predictions)
            
            # Save Metrics as local
            
            scores = {"Accuracy": accuracy,"Mean Squared Error": mse,"F1 Score": f1,"Precision Score": precision,"Recall Score": recall}
            save_json(path=Path(self.config.metric_file_name),data=scores)
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("Accuracy",accuracy)
            mlflow.log_metric("Mean Squared Error",mse)
            mlflow.log_metric("F1 Score",f1)
            mlflow.log_metric("Precision Score",precision)
            mlflow.log_metric("Recall Score",recall)
            
            
            # Model registry does not work with file store
            if tracking_url_type_store !='file':
                mlflow.sklearn.log_model(model,"model",registered_model_name="Logistic Regression Model")
            else:
                mlflow.sklearn.load_model(model,'model')
    
    def initiate_model_evaluation(self):
        self.log_into_mlflow()
    
class Regression_Model_Evaluation:
    def __init__(self,config:Model_Evaluation_Config):
        self.config = config
    
    def evaluate_metrics(self,actual,pred):
        mae = mean_absolute_error(actual,pred)
        mse = mean_squared_error(actual,pred)
        r2 = r2_score(actual,pred)
        return mae,mse,r2
    
    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        X_test = test_data.drop([self.config.target_column],axis=1)
        y_test  = test_data[[self.config.target_column]]
        
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            predictions = model.predict(X_test)
            (mae,mse,r2) = self.evaluate_metrics(y_test,predictions)
            
            # Save Metrics as local
            
            scores = {"Mean Absolute Error": mae,"Mean Squared Error": mse,"R2 Score": r2}
            save_json(path=Path(self.config.metric_file_name),data=scores)
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("Accuracy",mae)
            mlflow.log_metric("Mean Squared Error",mse)
            mlflow.log_metric("R2 Score",r2)
            
            # Model registry does not work with file store
            if tracking_url_type_store !='file':
                mlflow.sklearn.log_model(model,"model",registered_model_name="Logistic Regression Model")
            else:
                mlflow.sklearn.load_model(model,'model')
            
    def initiate_model_evaluation(self):
        self.log_into_mlflow()