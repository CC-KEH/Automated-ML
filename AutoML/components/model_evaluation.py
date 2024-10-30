import os
import pandas as pd
from pathlib import Path
import mlflow
import mlflow.sklearn
import joblib
from sklearn.metrics import (accuracy_score, mean_squared_error, mean_absolute_error,
                             r2_score, f1_score, precision_score, recall_score,silhouette_score, davies_bouldin_score)

from urllib.parse import urlparse
from AutoML.entity.config_entity import Model_Evaluation_Config
from AutoML.utils.main_utils import save_json

class Regression_Model_Evaluation:
    def __init__(self,config:Model_Evaluation_Config):
        self.config = config
    
    def evaluate_metrics(self,actual,pred):
        mae = mean_absolute_error(actual,pred)
        mse = mean_squared_error(actual,pred)
        r2 = r2_score(actual,pred)
        return mae,mse,r2
    
    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_path)
        models = os.listdir(self.config.model_path)
        model = models[0]
        model = joblib.load(os.path.join(self.config.model_path,model))
        X_test = test_data.drop(['target'],axis=1)
        y_test  = test_data[['target']]
        
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
                mlflow.sklearn.log_model(model,"model",registered_model_name="AutoML Regression Model")
            else:
                mlflow.sklearn.load_model(model,'model')
            
    def initiate_model_evaluation(self,manual_config=None):
        self.log_into_mlflow()
        
        
class Classification_Model_Evaluation:
    def __init__(self,config:Model_Evaluation_Config):
        self.config = config
    
    def evaluate_metrics(self,actual,pred):
        accuracy = accuracy_score(actual,pred)
        f1 = f1_score(actual,pred)
        precision = precision_score(actual,pred)
        recall = recall_score(actual,pred)
        return accuracy,f1,precision,recall

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_path)
        models = os.listdir(self.config.model_path)
        model = models[0]
        model = joblib.load(os.path.join(self.config.model_path, model))
        X_test = test_data.drop(['target'],axis=1)
        y_test  = test_data[['target']]
        
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            predictions = model.predict(X_test)
            (accuracy,f1,precision,recall) = self.evaluate_metrics(y_test,predictions)
            
            # Save Metrics as local
            
            scores = {"Accuracy": accuracy,"F1 Score": f1,"Precision Score": precision,"Recall Score": recall}
            save_json(path=Path(self.config.metric_file_name),data=scores)
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("Accuracy",accuracy)
            mlflow.log_metric("F1 Score",f1)
            mlflow.log_metric("Precision Score",precision)
            mlflow.log_metric("Recall Score",recall)
            
            
            # Model registry does not work with file store
            if tracking_url_type_store !='file':
                mlflow.sklearn.log_model(model,"model",registered_model_name="AutoML Classification Model")
            else:
                mlflow.sklearn.load_model(model,'model')
    
    def initiate_model_evaluation(self,manual_config=None):
        self.log_into_mlflow()
        
        
class Clustering_Model_Evaluation:
    def __init__(self, config: Model_Evaluation_Config):
        self.config = config
    
    def evaluate_metrics(self, actual, pred) -> Dict[str, float]:
        """
        Evaluate and return clustering metrics.
        """
        scores = {
            'silhouette_score': silhouette_score(actual, pred),
            'davies_bouldin_score': davies_bouldin_score(actual, pred)
            # Add other clustering metrics if needed
        }
        return scores
    
    def log_into_mlflow(self):
        """
        Logs model and metrics into MLflow.
        """
        test_data = pd.read_csv(self.config.test_path)
        models = os.listdir(self.config.model_path)
        model = models[0]
        model = joblib.load(os.path.join(self.config.model_path, model))
        
        X_test = test_data.drop(['target'], axis=1)
        y_test = test_data[['target']]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predictions = model.predict(X_test)
            scores = self.evaluate_metrics(X_test, predictions)
            
            # Save metrics locally
            save_json(path=Path(self.config.metric_file_name), data=scores)
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(scores)
            
            # Model registry compatibility
            if tracking_url_type_store != 'file':
                mlflow.sklearn.log_model(model, "model", registered_model_name="AutoML Clustering Model")
            else:
                mlflow.sklearn.log_model(model, 'model')
    
    def initiate_model_evaluation(self, is_auto: bool):
        """
        Initiates model evaluation.
        """
        if is_auto:
            self.log_into_mlflow()