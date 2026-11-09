# Importing Packages required for this assignment
import pandas as pd # Pandas for the DataFrame
import numpy as np # Numpy for array manipulation

# Importing the sklearn LogisticRegression Models for the Logistic Regression
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Read the data in using the read_csv command
train_data=pd.read_csv("src/data/train.csv")
gender_submission=pd.read_csv("src/data/gender_submission.csv")
test_data=pd.read_csv("src/data/test.csv")

# Data Cleaning Step Right Here
print("Missing values BEFORE cleaning:")
print(train_data.isna().sum()) 

train_data=train_data.drop(columns=['Cabin']) # Dropped the Cabin Column, due to the majority of values being N/A (327)
 # Filled the N/A values in Age column with the median, so that there is no skewed distribution
median=train_data['Age'].median()
train_data['Age']=train_data['Age'].fillna(median)
print(f"Filled missing Age values with the training median: {median}")

 # Filled the N/A values in Embarked column with the mode, so that there is no skewed distribution, as its a categorical variable
train_data['Embarked']=train_data['Embarked'].fillna(train_data['Embarked'].mode()[0]) 
mode=train_data['Embarked'].mode()[0]
train_data['Embarked']=train_data['Embarked'].fillna(mode)
print(f"Filled missing Embarked values with the training mode: {mode}")

train_data['Is_Female']=train_data['Sex'].map({'female':1, 'male':0}) # Created an Is_Female Column, where we classify if the passenger is a woman (1 for Yes, 0 for No), as Sex is a string
print(train_data['Is_Female'])

print("Missing values AFTER cleaning:")
print(train_data.isna().sum()) # Determining how many IS/NA Values there are after cleaning

# Used all 6 numerical columns as feature columns and dropped each one individually to see the training/validation accuracy- learned that accuracy started at 79.61% (all 6 variables) and then dropped to 78.45% with just is_female.
# Learned that the dropping of columns did not change the training/validation accuracy.
feature_columns=['Is_Female','Pclass','Parch','SibSp','Age','Fare'] # Created the feature_columns for the predictor variables as X-Variables
X_training=train_data[feature_columns] #X Variables where we refer to the feature_columns
y_training=train_data['Survived'] # Y_Variable is the Survived column

# Performed a Train_Test_split on the training variables, assuming the test size is 0.3, and a random State of 55
X_train,X_test,y_train,y_test=train_test_split(X_training,y_training, test_size=0.3, random_state=55)

# Run the Logistic Regression Model
model=LogisticRegression()
# Fit the model on thee training data
model.fit(X_train,y_train)

# Use model.score to determine the training_accuracy and the Validation_Accuracy
accuracy_train = model.score(X_train, y_train)
accuracy_val=model.score(X_test,y_test)

#Printed out the values from lines 36 & 37
print(f"Feature_Selection (Validation accuracy) (held-out 30%): {accuracy_val:.2%}")
print(f"4 features (no Age/Fare): 80.97% (rs=42), 77.61% (rs=55)")
print(f"6 features (Include Age/Fare):  81.34% (rs=42), 77.99% (rs=55)")
print(f"Final Features Used: {feature_columns}")

#Cleaned out the test_data as before
print("Missing values BEFORE cleaning:")
print(test_data.isna().sum()) 

test_data=test_data.drop(columns=['Cabin']) # Dropped the Cabin Column, due to the majority of values being N/A 

median_test=test_data['Age'].median()
test_data['Age']=test_data['Age'].fillna(median_test)
print(f"Filled missing Age values with the test median: {median_test}")

test_data['Is_Female']=test_data['Sex'].map({'female':1, 'male':0}) # Created an Is_Female Column, where we classify if the passenger is a woman (1 for Yes, 0 for No), as Sex is a string
print(train_data['Is_Female'])

fare_median_test=test_data['Fare'].median()
test_data['Fare']=test_data['Fare'].fillna(fare_median_test)
print(f"Filled missing Fare values with the test median: {fare_median_test}") # Filled the N/A values in Fare column with the median, so that there is no skewed distribution

print("Missing values AFTER cleaning:")
print(test_data.isna().sum()) 

feature_columns_test=['Is_Female','Pclass','Parch','SibSp','Age','Fare'] # Created the feature_columns_test for the predictor variables as X-Variables

X_testing=test_data[feature_columns_test] #X Variables where we refer to the feature_columns

y_pred=model.predict(X_testing) # Use the training data to predict on the feature_columns_test

print(f"Number of test predictions: {len(y_pred)}") # Printing the number of test predictions here
print("Predicted outcomes on test/csv (0=died, 1=survived): ") # Predict how many died/surived (0=died, 1=survived)
print(pd.Series(y_pred).value_counts()) # Print the value_counts of 0 and 1 as a series

print("Accuracy Score: ") # Print the Accuracy Score
print(accuracy_score(gender_submission['Survived'], y_pred)) #Accuracy Score comparing the y_pred and the survived column in Gender_Submission

# What we learned here is that using 6 features, the model outperforms the 4-feature model (dropping age and fare)
# for validation across the splits. We include Age/Fare to let the model capture patterns beyond gender.





