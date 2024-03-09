import numpy as np
import pandas as pd
import os, sys, dill
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:

        under_sampler=RandomUnderSampler(random_state=42)
        X_train_under, y_train_under=under_sampler.fit_resample(X_train, y_train)

        over_sampler=RandomOverSampler(random_state=42)
        X_train_over, y_train_over=over_sampler.fit_resample(X_train, y_train)

        report={}

        for i in range(len(list(models))):
            model_reg =list(models.values())[i]
            model_reg.fit(X_train, y_train)

            train_model_score = model_reg.score(X_train, y_train)
            test_model_score = model_reg.score(X_test, y_test)

            report[list(models.keys())[i]] =(train_model_score, test_model_score)

            model_under=list(models.values())[i]
            model_under.fit(X_train_under, y_train_under)

            train_model_score = model_under.score(X_train, y_train)
            test_model_score = model_under.score(X_test, y_test)

            report[list(models.keys())[i]] =(train_model_score, test_model_score )

                      
            model_over=list(models.values())[i]
            model_over.fit(X_train_over, y_train_over)

            train_model_score = model_over.score(X_train, y_train)
            test_model_score = model_over.score(X_test, y_test)

            report[list(models.keys())[i]] =(train_model_score, test_model_score)

            return report

    except Exception as e:
        raise CustomException(e, sys)
