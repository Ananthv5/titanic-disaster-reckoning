
# Reading data from all files and declaring the CSV's as train, test, and gender_submission
train=read.csv("src/data/train.csv", stringsAsFactors=FALSE)
test=read.csv("src/data/test.csv",stringsAsFactors=FALSE)
gender_submission=read.csv("src/data/gender_submission.csv", stringsAsFactors=FALSE)

#Printed the head of all dataframes to see what data needs to be cleaned (EDA)
print(head(test))
print(head(train))
print(head(gender_submission))

#Printed out the missing values in the training data
print("Missing Values before cleaning:\n")
print(colSums(is.na(train)))

# Changing All values in Cabin column to NUll to get rid of the column itself
train$Cabin = NULL

# Filled the N/A values in Age column with the median, so that there is no skewed distribution
median_age = median(train$Age, na.rm=TRUE)
train$Age[is.na(train$Age)] = median_age
cat(sprintf("Filled missing Age values with the training median: %f\n", median_age))

 # Filled the N/A values in Embarked column with the mode, so that there is no skewed distribution, as its a categorical variable
mode_embarked= names(sort(table(train$Embarked), decreasing=TRUE))[1]
train$Embarked[is.na(train$Embarked) | train$Embrasked == ""]= mode_embarked
cat(sprintf("Filled missing Embarked values with the training mode: %s\n", mode_embarked))

# Created an Is_Female Column, where we classify if the passenger is a woman (1 for Yes, 0 for No), as Sex is a string
train$Is_Female = ifelse(train$Sex == 'female',1,0)

# Determining how many IS/NA Values there are after cleaning
cat("Missing values AFTER cleaning:\n")
print(colSums(is.na(train)))

# Used all 6 numerical columns as feature columns and dropped each one individually to see the training/validation accuracy- learned that accuracy started at 79.61% (all 6 variables) and then dropped to 78.45% with just is_female.
# Learned that the dropping of columns did not change the training/validation accuracy.
feature_columns = c("Is_Female","Pclass","Parch","SibSp","Age", "Fare")

#Drawing a random sample of rows assuming the test size is 0.3, and a random State of 55 (set.seed)
set.seed(55)
n_train = nrow(train)
train_sample= sample(seq_len(n_train), size=floor(0.70 * n_train))

# Generating the training and validation data from above
train_split = train[train_sample, ]
validation_split= train[-train_sample, ]

#Using glm to generate the logistic regression model assuming a binomial family distribution and modeling it based on training data
formula_model = as.formula(paste("Survived ~", paste(feature_columns, collapse = " + ")))
logistic_model=glm(formula_model, data=train_split, family=binomial)

# Predicting the Validation accuracy here of this data, 
val_probs = predict(logistic_model, newdata=validation_split, type="response")
val_preds = ifelse(val_probs >= 0.5, 1, 0)
val_accuracy = mean(val_preds == validation_split$Survived)


# Generating the final features variables used and what the validation accuracy was assuming a test_set of 0.3
cat(sprintf("Feature_Selection (Validation accuracy) (held-out 30%%): %.2f%%\n", val_accuracy*100))
cat(sprintf("Final Features Used: %s\n", paste(feature_columns, collapse = ", ")))

#Printed out the missing values in the test data
cat("Test missing values BEFORE cleaning:\n")
print(colSums(is.na(test)))

# Changing All values in Cabin column of the test data to NUll to get rid of the column itself
test$Cabin= NULL

# Filled the N/A values in Age column with the median, so that there is no skewed distribution
median_test_age = median(test$Age, na.rm=TRUE)
test$Age[is.na(test$Age)]=median_test_age
cat(sprintf("Filled missing Age values with the test median: %f\n", median_test_age))

# Created an Is_Female Column, where we classify if the passenger is a woman (1 for Yes, 0 for No), as Sex is a string
test$Is_Female <-ifelse(test$Sex == "female",1,0)

 # Filled the N/A values in Fare column with the median, so that there is no skewed distribution.
median_test_fare = median(test$Fare, na.rm=TRUE)
test$Fare[is.na(test$Fare)]= median_test_fare
cat(sprintf("Filled missing Fare values with the test median: %f\n", median_test_fare))

# Determining how many IS/NA Values there are after cleaning
cat("Test missing values AFTER cleaning: \n")
print(colSums(is.na(test)))

# Predicting the new outcomes from the training data on the test data assuming the Logistic Regression
test_probs = predict(logistic_model, newdata=test, type="response")
y_pred = ifelse(test_probs >= 0.5, 1, 0)

# Generating the Predicted Outcomes on the test.csv, assuming 0 is the people who died and 1 is people who survived
cat("Predicted outcomes on test.csv(0=died, 1=survived:\n)")
print(table(y_pred))
# Generating the Accuracy Score comparing the Logistic Regression Model to the Gender_Submission csv file.
test_accuracy <- mean(y_pred == gender_submission$Survived)
cat(sprintf("Accuracy Score: %.4f\n", test_accuracy))

# What we learned here is that using 6 features, the model outperforms the 4-feature model (dropping age and fare)
# for validation across the splits. We include Age/Fare to let the model capture patterns beyond gender.
# We also observe that the accuracy score for this model (0.9306), is lower than the the Python model (0.9498).
# This is attributed to the fact that in Python, using scikit-learn for the logistic regression, has a default
# setting of applying a ridge penalty (Ridge Regression), assuming the parameter C=1.0, to reduce overfitting.
# in R, the model uses an unreguarlized GLM using pure maximum likelihood estimation, so it penalizes nothing.
