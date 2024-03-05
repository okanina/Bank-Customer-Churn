import pandas as pd
import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class dataIngestionConfig:
    train_data=os.path.join('artifacts',"train.csv")
    test_data=os.path.join("artifacts", "test.csv")
    raw_data=os.path.join("artifacts", "data.csv")

class dataIngestion:
    def __init__(self):
        self.ingestion_config=dataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method.")
        try:
            df=pd.read_csv("data\Bank_Customer_Churn_Prediction.csv")
            logging.info("Read the dataset to datframe.")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data, index=False, header=True)

            logging.info("Train, Test split initiated.")

            train_set, test_set=train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data, index=False, header=True)

            logging.info("Data ingestion, complete")

            return (self.ingestion_config.train_data, self.ingestion_config.test_data)
        except Exception as e:
            raise CustomException(e,sys)
        

