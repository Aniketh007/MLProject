import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.logger import logging
from src.exception import CustomException
from src.utils import save_model,evaluate_model

@dataclass
class ModelTrainerConfig:
    trainer_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting test and train data")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models={
                "RandomForestRegressor":RandomForestRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "LinearRegression":LinearRegression(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "CatBoostRegressor":CatBoostRegressor(),
                "XGBRegressor":XGBRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor()
            }

            model_report=evaluate_model(x_train,y_train,x_test,y_test,models)
            best_score=max(model_report.values())
            best_score_model=list(model_report.keys())[
                list(model_report.values()).index(best_score)
            ]
            best_model=models[best_score_model]
            if best_score<0.6:
                raise CustomException("No best model found") 
            logging.info("Best model found for train and test")
            save_model(
                path=self.model_config.trainer_path,
                obj=best_model
            )
            pred=best_model.predict(x_test)
            return r2_score(y_test,pred)
        except Exception as e:
            raise CustomException(e,sys)