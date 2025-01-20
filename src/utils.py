import numpy as np
import os
import sys
import pandas as pd
import dill
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.exception import CustomException

def save_model(path,obj):
    try:
        dir_path=os.path.dirname(path)
        os.makedirs(dir_path,exist_ok=True)
        with open(path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    