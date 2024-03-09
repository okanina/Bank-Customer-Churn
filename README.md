## 1.Problem statement

The problem statement for this project is customers who churns banking industry.  Customer churn occurs when the customer terminate his/her relationship with a business or organization.

Customer churn affect any business or organization future growth. In this project I intend to use machine Learning techniques to find patterns and develop a model  that will whether the customer will stay or would leave Bank XYZ.

## 2. Data Source

I downloaded the dataset from [Kaggle.com](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction).It is in a csv file, I used pandas to read it into a dataframe, which makes it simpler to analyze the data.

It contains 10000 observations with 12 features including the target variable.This is a supervised machine learning project which predict a yes or no making it a binary classification problem.

The data has no missing or duplicate values. It consist of 3 datatypes, int64, floats, and objects(string values). 

### The following are the features present in the dataset:

 0   customer_id        
 1   credit_score      
 2   country           
 3   gender            
 4   age                 
 5   tenure           
 6   balance           
 7   products_number   
 8   credit_card       
 9   active_member     
 10  estimated_salary  
 11  churn  

 ### 4.Unique Column Values:
1.country - 'France' 'Spain' 'Germany'
2.gender - 'Female' 'Male'

### 5.Data Preprocessing and Feature engineering:

1.OneHotEncoder- used for categorical variables
2.StandardScaler- used for numerical variables to bring features to the same scale.
3.ColumnTransformer- to transform the columns.

### 6.Data limitations:

Most features are not clearly explained, there are no units.

### 7.Data Splitting Strategy and Evaluation Metrics

I split the data into traing set and testing set, 20% for testing and 80% for training. I used "imblearn" to handle the imbalance classes.

For evalution, I used the model score test to get the best model. I have alse used Precision and recall.

## Basic Statistical Summary:

In this dataset 20.4 % of the customers left Bank XYZ and 79.63 % remained. It appears that less than 25% of customers has been with the bank for 3 years while less than 50% has been with the bank for 7 years and less than 75% has been with the bank for 7 year and there are customers who have been with the bank for 10 years.

The youngest customer in this dataset is 18 years old while the oldest is 92 years old. The estimated varies between 11.58 and 199992.48 which is highly varied. The dataset has 51.51% of customers who are active and 48.49% are not active customers, however some of none active customers appear to have a balance. It seems like there are customers who were once active then cancelled their subscription.








