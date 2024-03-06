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

            target="churn"
            train_features=train_df.drop(target, axis=1)

            test_features=test_df.drop(target, axis=1)

            logging.info("Dataset Splitting completed.")
                        
            train_features_array=preprocessor_obj.fit_transform(train_features)
            test_features_array=preprocessor_obj.transform(test_features)

            logging.info("Data transformation completed.")
            
            train_array=np.c_[train_features_array, np.array(train_features)]

            test_array=np.c_[test_features_array, np.array(test_features)]

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessor_obj)
            
            logging.info("File saved.")
            return (train_array, 
                    test_array,
                    self.data_transformation_config.preprocessor_obj_file_path,
                )

        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
   obj=data_ingestion.dataIngestion()
   train_data, test_data=obj.initiate_data_ingestion()

   data_transformation=DataTransformation()
   data_transformation.initiate_data_transformation(train_data, test_data)