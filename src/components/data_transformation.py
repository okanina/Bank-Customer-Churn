import numpy as np
import pandas as pd
import sys, os
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.components import data_ingestion

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        try:
            numerical_features=["active_member" ,"credit_card", "credit_score", "age", "tenure","balance", "products_number", "estimated_salary"]
            
            categorical_fetures=["country", "gender"]

            numerical_pipeline=Pipeline(
                steps=[("impute", SimpleImputer(strategy="median")),
                       ("scaler", StandardScaler())]
                       )

            logging.info("Numerical scalling completed.")

            categorical_pipeline=Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore")),
                                                 ("scaler", StandardScaler(with_mean=False))])
            
            logging.info("Categorical encoding completed.")

            preprocessor=ColumnTransformer(transformers=[("Numerical features", numerical_pipeline, numerical_features), 
                                  ("Categorical fetures", categorical_pipeline, categorical_fetures)])
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed.")

            preprocessor_obj=self.get_data_transformer_obj()

            
            train_features=list(train_df.columns.drop("churn"))
            X_train=train_df[train_features]
            y_train=train_df["churn"]

            test_features=list(test_df.columns.drop("churn"))
            X_test=test_df[test_features]
            y_test=test_df["churn"]

            logging.info("Dataset Splitting completed.")
                        
            X_train_array=preprocessor_obj.fit_transform(X_train)
            X_test_array=preprocessor_obj.transform(X_test)

            logging.info("Data transformation completed.")

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessor_obj)
            
            logging.info("File saved.")
            return (X_train_array,
                    y_train, 
                    X_test_array,
                    y_test,
                    self.data_transformation_config.preprocessor_obj_file_path,
                )

        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
   obj=data_ingestion.dataIngestion()
   train_data, test_data=obj.initiate_data_ingestion()

   data_transformation=DataTransformation()
   data_transformation.initiate_data_transformation(train_data, test_data)