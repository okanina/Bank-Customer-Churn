import pandas as pd
import numpy as np
import sys, os
from operator import itemgetter
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, 
                            ConfusionMatrixDisplay, 
                            confusion_matrix, 
                            classification_report)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self, X_train, y_train, X_test, y_test):
        try:                             
            models={"LogisticRegression": LogisticRegression(),
                    "DecisionTreeClassifier": DecisionTreeClassifier(),
                    "RandomForestClassifier": RandomForestClassifier(),
                    "GradientBoostingClassifier": GradientBoostingClassifier()}
            
            params={

                    "LogisticRegression":{},
                    "DecisionTreeClassifier":{"criterion":['gini', 'entropy', 'log_loss'],
                                                "max_depth":range(10,50,10)},
                    
                    "RandomForestClassifier":{"n_estimators":range(25, 100, 25),
                                                "max_depth":range(10,50,10)},

                    "GradientBoostingClassifier": {
                                                    "n_estimators":range(20,40,5),
                                                    "max_depth":range(2,10)},
                    }
            
            model_report:dict=evaluate_model(X_train, y_train, X_test, y_test, models, params=params)
            
            
            best_model_score=sorted(model_report.values(), key= lambda score: score[1])[-1]

            best_model_name= list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if best_model_score[1]<0.7:
                raise CustomException("No best model found")
            logging.info("Best model found.")

            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            predicted=best_model.predict(X_test)

            predicted_score=best_model.score(X_test, predicted)

            print(f"{best_model} is the best model.")

            print(classification_report(y_test, best_model.predict(X_test)))

            return predicted_score

        except Exception as e:
            raise Exception(e, sys)
