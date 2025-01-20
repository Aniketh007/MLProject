import sys
import os
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils import save_model
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformConfig:
    preprocessor=os.path.join("artifacts","preprocessor.pkl")
class DataTransform:
    def __init__(self):
        self.data_transform=DataTransformConfig()

    def get_data_transform(self):
        try:
            num_features=['reading_score', 'writing_score']
            col_features=[
                'gender', 
                'race_ethnicity', 
                'parental_level_of_education', 
                'lunch', 
                'test_preparation_course'
            ]

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            col_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehot",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical and Categorical Feature Scaling Done")
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_features),
                    ("col_pipeline",col_pipeline,col_features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transform(self,train_data,test_path):
        try:
            train_df=pd.read_csv(train_data)
            test_df=pd.read_csv(test_path)
            logging.info("Data loaded")

            preprocess_obj=self.get_data_transform()
            target_col="math_score"
            input_df=train_df.drop(target_col,axis=1)
            tar_df=train_df[target_col]

            input_test_df=test_df.drop(target_col,axis=1)
            tar_test_df=test_df[target_col]

            input_arr=preprocess_obj.fit_transform(input_df)
            input_test_arr=preprocess_obj.transform(input_test_df)
            train_arr=np.c_[
                input_arr,np.array(tar_df)
            ]
            test_arr=np.c_[input_test_arr,np.array(tar_test_df)]
            logging.info("Saved preprocessing obj")
            save_model(
                path=self.data_transform.preprocessor,
                obj=preprocess_obj
            )
            return(
                train_arr,test_arr,self.data_transform.preprocessor
            )
        except Exception as e:
            raise CustomException(e,sys)