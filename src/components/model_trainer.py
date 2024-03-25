import numpy as np
import pandas as pd
import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from src.exception import customException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class model_trainer_config:
    trained_model_filepath = os.path.join('artifact','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = model_trainer_config()
        
    def initiate_model_tainer(self,train_array,test_array):
        try:
            logging.info("split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "AdaBoost Regressor": AdaBoostRegressor()
                }
            
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            
            best_model_score = max(sorted(model_report.values())) #sorting and choosing the highest score model
            #to get the best model name 
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise customException('No good model found')
            
            logging.info(f'Best found model on both training and testing dataset')
            
            save_object(
                file_path=self.model_trainer_config.trained_model_filepath,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            r2_score_model = r2_score(y_test,predicted)
            return r2_score_model
        
        
        except Exception as e:
            raise customException(e,sys)
            