import joblib
import pandas as pd
from AutoML.logger import logger
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, Birch

from AutoML.entity.config_entity import Model_Trainer_Config


class Regression_Model_Trainer:
    def __init__(self,config: Model_Trainer_Config) -> None:
        self.config = config
        self.models = {
            'LinearRegression': LinearRegression(),
            'ElasticNetRegressor': ElasticNet(),
            'DecisionTreeRegressor': DecisionTreeRegressor(),
            'RandomForestRegressor': RandomForestRegressor(),
            'SVR': SVR(),
            'GradientBoostingRegressor': GradientBoostingRegressor(),
            'KNeighborsRegressor': KNeighborsRegressor(),
        }
        self.train_data = pd.read_csv(self.config.train_path)
        self.y = self.train_data['target']
        self.x = self.train_data.drop(columns=['target'])
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.x,self.y, 
            test_size=0.2, 
            random_state=42
        )
        self.best_model_info = None
        self.best_model = None
        
    
    def get_metrics(self, y_true, y_pred):
        if self.manual_config and self.manual_config['evaluation_metric']!='auto':
            if self.manual_config['evaluation_metric']=='mean_squared_error':
                return {
                    'mean_squared_error': mean_squared_error(y_true, y_pred)
                }
            elif self.manual_config['evaluation_metric']=='mean_absolute_error':
                return {
                    'mean_absolute_error': mean_absolute_error(y_true, y_pred)
                }
            elif self.manual_config['evaluation_metric']=='r2_score':
                return {
                    'r2_score': r2_score(y_true, y_pred)
                }
        else:
            return {
                'mean_squared_error': mean_squared_error(y_true, y_pred),
                'mean_absolute_error': mean_absolute_error(y_true, y_pred),
                'r2_score': r2_score(y_true, y_pred)
            }
    
    def fine_tune_hyperparameters(self, model, model_name):
        params_grid = getattr(self.config.params, model_name)
        gs = GridSearchCV(model, param_grid=params_grid)
        gs.fit(self.X_train, self.y_train)
        logger.info(f'Finetuning {model} model')
        logger.info(f'Best Parameters for {model}: {gs.best_params_}')
        logger.info(f'Best Score for {model}: {gs.best_score_}')
        finetuned_model = model.set_params(**gs.best_params_)
        return finetuned_model
    
    def finetune_model(self, model, model_name):
        best_estimator = self.fine_tune_hyperparameters(model, model_name)
        return best_estimator
    
    def save_model(self, model, model_name):
        joblib.dump(model, f'{self.config.root_dir}/{model_name}.pkl')
        logger.info(f'{model_name} model saved successfully')
        
    def initiate_model_training(self,manual_config=None):
        logger.info("Initiating Regression Model Training")
        self.manual_config = manual_config
            
        if manual_config == None or manual_config['algorithm']=='auto':
            for model_name, model in self.models.items():
                model.fit(self.X_train, self.y_train)

                logger.info(f'{model_name} model trained successfully')

                metrics = self.get_metrics(self.y_val,model.predict(self.X_val))

                logger.info(f'{model_name} \n Metrics: {metrics}')


                if self.best_model_info is None:
                    self.best_model_info = {model_name: [model, metrics]}
                    self.best_model = model

                elif metrics['mean_squared_error'] < list(self.best_model_info.values())[0][1]['mean_squared_error']:
                    self.best_model_info = {model_name: [model, metrics]}
                    self.best_model = model

            best_model = self.finetune_model(self.best_model, list(self.best_model_info.keys())[0])
            self.save_model(best_model, list(self.best_model_info.keys())[0])
            
        else:
            model = self.models[manual_config['algorithm']]
            
            if "hyperparameters" in manual_config:
                model.set_params(**manual_config["hyperparameters"].get(manual_config['algorithm'], {}))
            logger.info(f'Training {manual_config["algorithm"]} model on Hyperparameters: {manual_config["hyperparameters"].get(manual_config["algorithm"], {})}')

            model.fit(self.X_train, self.y_train)

            logger.info(f'{manual_config["algorithm"]} model trained successfully')

            metrics = self.get_metrics(self.y_val,model.predict(self.X_val))

            logger.info(f'{manual_config["algorithm"]} \n Metrics: {metrics}')

            self.save_model(model, manual_config['algorithm'])
                    
class Classification_Model_Trainer:
    def __init__(self,config: Model_Trainer_Config) -> None:
        self.config = config
        self.models = {
            'LogisticRegression': LogisticRegression(),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
            'SVC': SVC(),
            'GaussianNB': GaussianNB(),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
        }
        self.train_data = pd.read_csv(self.config.train_path)
        self.y = self.train_data['target']
        self.x = self.train_data.drop(columns=['target'])
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.x,self.y, 
            test_size=0.2, 
            random_state=42
        )
        self.best_model_info = None
        self.best_model = None
        
    def get_metrics(self, y_true, y_pred):
        
        if self.manual_config and self.manual_config['evaluation_metric']!='auto':
            
            if self.manual_config['evaluation_metric']=='accuracy_score':
                return {
                    'accuracy_score': accuracy_score(y_true, y_pred)
                }
            elif self.manual_config['evaluation_metric']=='f1_score':
                return {
                    'f1_score': f1_score(y_true, y_pred)
                }
            elif self.manual_config['evaluation_metric']=='precision_score':
                return {
                    'precision_score': precision_score(y_true, y_pred)
                }
            elif self.manual_config['evaluation_metric']=='recall_score':
                return {
                    'recall_score': recall_score(y_true, y_pred)
                }
        else:
            return {
                'accuracy_score': accuracy_score(y_true, y_pred),
                'f1_score': f1_score(y_true, y_pred),
                'precision_score': precision_score(y_true, y_pred),
                'recall_score': recall_score(y_true, y_pred)
            }
    
    def fine_tune_hyperparameters(self, model, model_name):
        params_grid = getattr(self.config.params, model_name)
        gs = GridSearchCV(model, param_grid=params_grid)
        gs.fit(self.X_train, self.y_train)
        logger.info(f'Finetuning {model} model')
        logger.info(f'Best Parameters for {model}: {gs.best_params_}')
        logger.info(f'Best Score for {model}: {gs.best_score_}')
        finetuned_model = model.set_params(**gs.best_params_)
        return finetuned_model
    
    def finetune_model(self, model, model_name):
        best_estimator = self.fine_tune_hyperparameters(model, model_name)
        return best_estimator
    
    def save_model(self, model, model_name):
        joblib.dump(model, f'{self.config.root_dir}/{model_name}.pkl')
        logger.info(f'{model_name} model saved successfully')
    
    def initiate_model_training(self,manual_config=None):
        logger.info("Initiating Classification Model Training")
        self.manual_config = manual_config
        
        
        if manual_config == None or manual_config['algorithm']=='auto':
            for model_name, model in self.models.items():
                if model_name == 'Lasso Classifier' or model_name == 'Ridge Classifier':
                    continue
                
                model.fit(self.X_train, self.y_train)

                logger.info(f'{model_name} model trained successfully')

                metrics = self.get_metrics(self.y_val,model.predict(self.X_val))

                logger.info(f'{model_name} \n Metrics: {metrics}')


                if self.best_model_info is None:
                    self.best_model_info = {model_name: [model, metrics]}
                    self.best_model = model

                elif metrics['accuracy_score'] < list(self.best_model_info.values())[0][1]['accuracy_score']:
                    self.best_model_info = {model_name: [model, metrics]}
                    self.best_model = model

            best_model= self.finetune_model(self.best_model, list(self.best_model_info.keys())[0])
            self.save_model(best_model, list(self.best_model_info.keys())[0])
            
        else:
            model = self.models[manual_config['algorithm']]
                        
            if "hyperparameters" in manual_config:
                model.set_params(**manual_config["hyperparameters"].get(manual_config['algorithm'], {}))
            
            logger.info(f'Training {manual_config["algorithm"]} model on Hyperparameters: {manual_config["hyperparameters"].get(manual_config["algorithm"], {})}')
            
            model.fit(self.X_train, self.y_train)

            logger.info(f'{manual_config["algorithm"]} model trained successfully')

            metrics = self.get_metrics(self.y_val,model.predict(self.X_val))

            logger.info(f'{manual_config["algorithm"]} \n Metrics: {metrics}')

            self.save_model(model, manual_config['algorithm'])
            
        
class Clustering_Model_Trainer:
    def __init__(self,config: Model_Trainer_Config) -> None:
        self.config = config
        self.models = {
            'KMeans': KMeans(),
            'DBSCAN': DBSCAN(),
            'AgglomerativeClustering': AgglomerativeClustering(),
            'MeanShift': MeanShift(),
            'Birch': Birch(),
        }
        self.train_data = pd.read_csv(self.config.train_path)
        self.y = self.train_data['target']
        self.x = self.train_data.drop(columns=['target'])
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.x,self.y, 
            test_size=0.2, 
            random_state=42
        )
        self.best_model_info = None
        self.best_model = None
        
    def get_metrics(self, y_true, y_pred):
        if self.manual_config['evaluation_metric']=='auto':
            return {
                'accuracy_score': accuracy_score(y_true, y_pred),
                'f1_score': f1_score(y_true, y_pred),
                'precision_score': precision_score(y_true, y_pred),
                'recall_score': recall_score(y_true, y_pred)
            }
        else:
            if self.manual_config['evaluation_metric']=='accuracy_score':
                return {
                    'accuracy_score': accuracy_score(y_true, y_pred)
                }
            elif self.manual_config['evaluation_metric']=='f1_score':
                return {
                    'f1_score': f1_score(y_true, y_pred)
                }
            elif self.manual_config['evaluation_metric']=='precision_score':
                return {
                    'precision_score': precision_score(y_true, y_pred)
                }
            elif self.manual_config['evaluation_metric']=='recall_score':
                return {
                    'recall_score': recall_score(y_true, y_pred)
                }
    
    def fine_tune_hyperparameters(self, model, model_name):
        params_grid = getattr(self.config.params, model_name)
        gs = GridSearchCV(model, param_grid=params_grid)
        gs.fit(self.X_train, self.y_train)
        logger.info(f'Finetuning {model} model')
        logger.info(f'Best Parameters for {model}: {gs.best_params_}')
        logger.info(f'Best Score for {model}: {gs.best_score_}')
        finetuned_model = model.set_params(**gs.best_params_)
        return finetuned_model
    
    def finetune_model(self, model, model_name):
        best_estimator = self.fine_tune_hyperparameters(model, model_name)
        return best_estimator
    
    def save_model(self, model, model_name):
        joblib.dump(model, f'{self.config.root_dir}/{model_name}.pkl')
        logger.info(f'{model_name} model saved successfully')
        
    def initiate_model_training(self,manual_config=None):
        logger.info("Initiating Clustering Model Training")
        self.manual_config = manual_config
        
        if manual_config['algorithm']=='auto':
            for model_name, model in self.models.items():
                model.fit(self.X_train, self.y_train)

                logger.info(f'{model_name} model trained successfully')

                metrics = self.get_metrics(self.y_val,model.predict(self.X_val))

                logger.info(f'{model_name} \n Metrics: {metrics}')


                if self.best_model_info is None:
                    self.best_model_info = {model_name: [model, metrics]}
                    self.best_model = model

                elif metrics['accuracy_score'] < list(self.best_model_info.values())[0][1]['accuracy_score']:
                    self.best_model_info = {model_name: [model, metrics]}
                    self.best_model = model

            best_model = self.finetune_model(self.best_model, list(self.best_model_info.keys())[0])
            logger.info(f'Finetuned: {list(self.best_model_info.keys())[0]}')
            self.save_model(best_model, list(self.best_model_info.keys())[0])
        
        else:
            model = self.models[manual_config['algorithm']]
                        
            if "hyperparameters" in manual_config:
                model.set_params(**manual_config["hyperparameters"].get(manual_config['algorithm'], {}))
            logger.info(f'Training {manual_config["algorithm"]} model on Hyperparameters: {manual_config["hyperparameters"].get(manual_config["algorithm"], {})}')
            
            model.fit(self.X_train, self.y_train)

            logger.info(f'{manual_config["algorithm"]} model trained successfully')

            metrics = self.get_metrics(self.y_val,model.predict(self.X_val))

            logger.info(f'{manual_config["algorithm"]} \n Metrics: {metrics}')

            self.save_model(model, manual_config['algorithm'])