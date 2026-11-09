# titanic-disaster-reckoning

### Overview
This project aims to build a Logistic Regression model that predicts each passenger's survival on the Titanic in 1912. The data tracks passengers on the ill-fated ship and describes them by gender, name, age, and socioeconomic class. While chance played a role in the passengers' survival, demographic and socioeconomic indicators influenced who survived this deadly accident.

### Dataset

The dataset originated from the "Titanic- Machine Learning from Disaster" dataset on Kaggle. Here, the dataset was divided into 3 files for analysis:

- **train.csv**: The outcome of each passenger based on ID and whether they survived, along with predictor variables such as Sex, Age, sibsp (# of step/siblings), Parch (# of parents/children), Fare, Cabin, and Port of Embarkation. This dataset is where I trained my Logistic Regression Model.

- **test.csv**: The test data for the model to predict whether the passengers survived. It contained the same predictor variables, but did not indicate survivability. To Compare the Accuracy Score, the following dataset was used.
- **gender_submission.csv**: The test data for the model that contained only the survival data of all passengers. The twist here is that the model assumes all of the female passengers survive and all of the male passengers die.

## Dataset Schema & Preprocessing Strategy

The data source follows the standard Kaggle manifest stored under `src/data/`:

| Feature | Type | Description | Handling Strategy |
| :--- | :--- | :--- | :--- |
| `Pclass` | Categorical | Ticket class (1st, 2nd, 3rd) | Ignored as Pclass categorical variable |
| `Survival`| Binary| Binary Variable (`Survived = 1 `, `Not Survived=0`)| Response Variable|
| `Name` | Text | Full legal passenger name | Ignored as Name would not be a precdictor variable|
| `Sex` | Categorical | Biological sex | Convert Gender into a Binary Varible (See `Is_Gender`) below |
| `Age` | Continuous | Age in years | Median imputation by `Age`-> Used as a predictor variable |
| `SibSp` | Discrete | Siblings, spouses, parents, children-> Used as a predictor Variable | Used as a predictor variable |
| `Parch` | Discrete | Siblings, spouses, parents, children-> Used as a predictor variable | Used as a predictor variable |
| `Fare` | Continuous | Passenger ticket fare | Numerical Variable -> Used as a Predictor Variable|
| `Cabin` | Alphanumeric | Cabin identifier | Got Rid of Cabin due to the growing number of N/A Variables that comprised 77% of the column |
| `Embarked` | Categorical | Port of embarkation (`C`, `Q`, `S`) -> (C=Cherbourg, Q=Quueenstown, S=Southampton) |Mode imputation by `Embarked`|
| `Is_Gender` | Binary Variable | Binary Variable (`female = 1 `, `male=0`) | Converted the Sex column into Binary Variable before preprocessing|



'''
## Execution & Quickstart

### 1. Python Workflow

**Setup Environment**
```bash
python -m venv titanic-disaster-reckoning
source .venv/bin/activate  
pip install -r requirements.txt

# Using Conda

conda activate titanic-disaster-reckoning
pip install -r requirements.txt
```

**Run Training & Files**
```bash
python src/scripts/main.py
```
---

### 2. R Workflow

**Install Dependencies**
```bash
Rscript src/R_Scripts/install_packages.R
```

**Run Pipeline**
```bash
Rscript src/R_Scripts/main.R
```
---

### 3. Docker Workflows

**Python Environment ('titanic-disaster')**
```bash
# Build Python container image
docker build -t titanic-disaster:latest -f Dockerfile .

# Run Python Logistic Regression Model in Container Image
docker run --rm -v $(pwd)/src/data:/app/src/data titanic-disaster:latest

# Using Conda

# Build Python container image
docker build -t titanic-disaster .

# Running the Python Logistic Regression Model in Container Image
docker run titanic-disaster

```

**R Environment**
```bash
# Build R-specific container image
docker build -t titanic-r:latest -f src/R_Scripts/Dockerfile src/R_Scripts/

# Run Python Logistic Regression Model
docker run --rm -v $(pwd)/src/data:app/src/data titanic-r:latest

# Using Conda

# Build Python container image
docker build -t titanic-r .

# Running the Python Logistic Regression Model in Container Image
docker run titanic-r

```
---

## Repository Structure

- Repository maintainers and review paths are designated in `.github/CODEOWNERS`.
- Version pins for Python libraries are isolated in `requirements.txt`. 
- R library configurations and CRAN mirror bindings are defined in `src/R_Scripts/install_packages.R`.
```text
titanic-disaster-reckoning/
├── .github/
│   └── CODEOWNERS                  # Code ownership and review tracking
├── .venv/                          # Local Python virtual environment
├── src/
│   ├── data/
│   │   ├── gender_submission.csv   # Kaggle submission benchmark
│   │   ├── test.csv                # Out-of-sample test features
│   │   └── train.csv               # Ground truth training manifest
│   ├── R_Scripts/
│   │   ├── Dockerfile              # Docker container definition for R runtime (R Dockerfile)
│   │   ├── install_packages.R      # R package Dependencies 
│   │   └── main.R                  # R data loading, EDA, and LogisticRegression Model
│   └── scripts/
│       └── main.py                 # Python training, feature pipeline, and Python Logistic Regressionn Model
├── .gitignore                      # Git exclusion rules
├── Dockerfile                      # Root Python container definition (Python Dockerfile)
├── README.md                       # Repository documentation
└── requirements.txt                # Python package dependencies
```
## Results
The results stored in this file were primarily the accuracy score based on the training/test data.
We assumed a test_split of 0.3 (70% of data used for training, 30% for testing), we used all 6 numerical columns as feature columns and dropped each one individually to see the training/validation accuracy- learned that accuracy started at 77.99% (all 6 numerical variables) as we dropped one by one then dropped to 77.20% with just is_female. 
What we learned here is that using 6 features, the model outperforms the 4-feature model (dropping age and fare), for validation across the splits. We include Age/Fare to let the model capture patterns beyond gender. We learned that the dropping of columns did not change the training/validation accuracy.

# Python Model
| Validation Accuracy (Feature_Selection) | Score |
| :--- | :--- | 
| `Assuming 30% train_test_split`| 77.99% |
| `4 features (no Age/Fare)`| 80.97% (rs=42), 77.61% (rs=55)| 
| `6 features ( Include Age/Fare)` | 81.34% (rs=42), 77.99% (rs=55)| 

# R Model
| Validation Accuracy (Feature_Selection) | Score |
| :--- | :--- | 
| `Assuming 30% train_test_split`| 77.99% |
| `4 features (no Age/Fare)`| 80.97% (rs=42), 77.61% (rs=55)| 
| `6 features ( Include Age/Fare)` | 81.34% (rs=42), 77.99% (rs=55)| 

# Python Model -> Test Data Accuracy
| Test Data | Results |
| :--- | :--- | 
| Total Test Predictions (Gender Submission)| 418 |
| Passengers who died (0)| 267| 
| Passengers who survived (1)| 151| 
| Accuracy Score| 94.976%|

# R Model -> Test Data Accuracy
| Test Data | Results |
| :--- | :--- | 
| Total Test Predictions (Gender Submission)| 418 |
| Passengers who died (0)| 261| 
| Passengers who survived (1)| 157| 
| Accuracy Score| 93.06%|

 What we learned here is that using 6 features, the model outperforms the 4-feature model (dropping age and fare)
for validation across the splits. We include Age/Fare to let the model capture patterns beyond gender.
 We also observe that the accuracy score for this model (0.9306), is lower than the the Python model (0.9498).
This is attributed to the fact that in Python, using scikit-learn for the logistic regression, has a default
setting of applying a ridge penalty (Ridge Regression), assuming the parameter C=1.0, to reduce overfitting.
in R, the model uses an unreguarlized GLM using pure maximum likelihood estimation, so it penalizes nothing.


