# Diabetes Prediction Using Machine Learning

## 1. Project Overview

This project aims to predict the likelihood of diabetes using health-related indicators and demographic information.

A machine learning classification model is developed using the Diabetes Health Indicators dataset. The project includes data exploration, preprocessing, feature selection, model comparison, hyperparameter tuning, model export, and deployment through a Flask REST API and Docker.

**Objective:** Predict whether an individual is likely to have diabetes based on selected health indicators.

**Problem Statement:** The goal is to build a binary classification model
that identifies individuals who may be at risk of diabetes based on
demographic information and health-related indicators. This can help
demonstrate how machine learning techniques can be applied to public
health data for preliminary risk assessment.

> Disclaimer: This project is developed for educational purposes only. Predictions should not be considered medical advice or a clinical diagnosis.

## 2. Dataset

**Dataset Name:** Diabetes Health Indicators Dataset

**Source:** Kaggle

**Dataset Author/Uploader:** Alex Teboul

**Kaggle Dataset Link:** https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

**Dataset File Used:** `diabetes_binary_5050split_health_indicators_BRFSS2015.csv`

**Local Path:** `data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv`

The dataset is based on the 2015 Behavioral Risk Factor Surveillance System (BRFSS) health indicators.

### Dataset Characteristics

- Total records: 70,692
- Total original columns: 22
- Target classes: Balanced (35,346 records in each class)
- Missing values: None observed

The dataset was downloaded from Kaggle and used for exploratory data analysis, feature selection, model training, and evaluation.

**Data Attribution:** Dataset sourced from Alex Teboul's Kaggle repository. Please refer to the original Kaggle page for dataset details and licensing information.

## 3. Selected Features

The following 11 features were selected for model development:

| Feature | Description |
|---|---|
| genhlth | General health |
| highbp | High blood pressure |
| age | Age category |
| highchol | High cholesterol |
| diffwalk | Difficulty walking |
| income | Income category |
| heartdiseaseorattack | History of heart disease or heart attack |
| education | Education level |
| physactivity | Physical activity |
| physhlth | Number of days of poor physical health |
| bmi | Body Mass Index |

**Target variable:** `diabetes_binary`

## 4. Exploratory Data Analysis (EDA)

The notebook includes:

- Dataset structure and statistical analysis
- Target variable distribution
- Missing value analysis
- Distribution of health indicators
- Feature relationships and correlation analysis
- Mutual information analysis
- Feature selection and preprocessing

Refer to `notebooks/Midterm_Project_Diabetes_Pred.ipynb` for detailed analysis and visualizations.

## 5. Model Training and Evaluation

Multiple classification algorithms were evaluated using validation ROC-AUC.

| Model | Validation ROC-AUC |
|---|---:|
| Logistic Regression | 0.8256 |
| Decision Tree | 0.6549 |
| Tuned Decision Tree | 0.8101 |
| Random Forest | 0.7667 |
| Tuned Random Forest | 0.8205 |
| XGBoost | 0.8239 |
| Tuned XGBoost | 0.8276 |

**Selected model:** Tuned XGBoost

The tuned XGBoost model achieved the best validation ROC-AUC of approximately 0.8276.

### Test Evaluation

The notebook's test evaluation results were:

- Test ROC-AUC: 0.8222
- Test Accuracy: 74.46%

The validation and test ROC-AUC scores are relatively close, indicating consistent performance on these data splits.

## 6. Project Structure

```text
diabetes project midterm/
│
├── data/
│   └── diabetes_binary_5050split_health_indicators_BRFSS2015.csv
│
├── models/
│   └── xgb_tuned_eta_0_1_depth_4_min_child_60.bin
│
├── notebooks/
│   └── Midterm_Project_Diabetes_Pred.ipynb
│
├── src/
│
├── deployment/
│
├── diabetes_env/
│
├── train.py
├── predict.py
├── main.py
├── requirements.txt
├── Pipfile
├── Pipfile.lock
├── Dockerfile
├── .dockerignore
└── README.md
```

## 7. Environment Setup

### Prerequisites

- Python 3.11
- pip
- Pipenv
- Docker Desktop (for container deployment)

### Option 1: Setup using Pipenv (Recommended)

Pipenv manages project dependencies using `Pipfile` and `Pipfile.lock`. The lock file ensures consistent dependency versions across environments.

Install Pipenv:

```powershell
python -m pip install pipenv
```

Install dependencies using the lock file:

```powershell
python -m pipenv sync
```

Run the training script:

```powershell
python -m pipenv run python train.py
```

Run the Flask application:

```powershell
python -m pipenv run python predict.py
```

### Option 2: Setup using requirements.txt

Create and activate a virtual environment:

```powershell
python -m venv diabetes_env

.\diabetes_env\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```



## 8. Train the Model

Run the following command from the project root directory:

```powershell
python -m pipenv run python train.py
```

The training script performs the following tasks:

1. Loads the dataset.
2. Selects the required features.
3. Performs preprocessing and feature encoding.
4. Splits the data for model training and evaluation.
5. Trains the tuned XGBoost model.
6. Saves the trained model and feature vectorizer.

The model is saved at:

```text
models/xgb_tuned_eta_0_1_depth_4_min_child_60.bin
```

## 9. Run the Flask Prediction API

Start the Flask application:

```powershell
python -m pipenv run python predict.py
```

The API runs on:

```text
http://127.0.0.1:9696
```

### API Endpoint

**Method:** POST

**Endpoint:**

```text
http://127.0.0.1:9696/predict
```

### Sample Request

```json
{
    "genhlth": "fair",
    "highbp": "high_bp",
    "age": "80",
    "highchol": "no_high_col",
    "diffwalk": "yes",
    "income": "15,000_to_20,000",
    "heartdiseaseorattack": "yes",
    "education": "high_school_graduate_/_ged",
    "physactivity": "no",
    "physhlth": 15.0,
    "bmi": 36.0
}
```
### How to Test the API

You can test the prediction API using **PowerShell**

1. Start the Flask application:

   ```powershell
   python -m pipenv run python predict.py
   ```

2. Open another PowerShell terminal.
3. Execute the following command:

   ```powershell
   $body = @{
       genhlth = "fair"
       highbp = "high_bp"
       age = "80"
       highchol = "no_high_col"
       diffwalk = "yes"
       income = "15,000_to_20,000"
       heartdiseaseorattack = "yes"
       education = "high_school_graduate_/_ged"
       physactivity = "no"
       physhlth = 15.0
       bmi = 36.0
   } | ConvertTo-Json

   Invoke-RestMethod -Uri "http://127.0.0.1:9696/predict" `
       -Method Post `
       -Body $body `
       -ContentType "application/json"
   ```

4. The prediction response will be displayed in PowerShell.

**Note:** Ensure that the Flask application is running before sending the API request. The categorical input values must match the encoding used during model training.

### Sample Response

```json
{
    "diabetes probability": 0.8415,
    "diabetes_risk": true
}
```

The probability is illustrative and may vary depending on the trained model.

## 10. Docker Deployment

Docker is used to containerize the Flask prediction API along with its required dependencies and trained model.

**Note:** Pipenv is not required for Docker deployment. Docker installs the dependencies using `requirements.txt` as defined in the Dockerfile.

### 10.1 Build Docker Image

Run the following command from the project root directory:

```powershell
docker build -t diabetes-prediction:v1 .
```

### 10.2 Run Local Docker Container

```powershell
docker run -d --name diabetes-container -p 9697:9696 diabetes-prediction:v1
```

The API will be available at:

`http://127.0.0.1:9697/predict`

### 10.3 Test Local Docker API

Use PowerShell to test the prediction endpoint:

```powershell
$body = @{
    genhlth = "fair"
    highbp = "high_bp"
    age = "80"
    highchol = "no_high_col"
    diffwalk = "yes"
    income = "15,000_to_20,000"
    heartdiseaseorattack = "yes"
    education = "high_school_graduate_/_ged"
    physactivity = "no"
    physhlth = 15.0
    bmi = 36.0
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:9697/predict" `
    -Method Post `
    -Body $body `
    -ContentType "application/json"
```

### 10.4 Docker Hub

Docker Hub repository:

[https://hub.docker.com/r/rakeshsharma1712/diabetes-prediction](https://hub.docker.com/r/rakeshsharma1712/diabetes-prediction)

**Push the Docker Image to Docker Hub**

After building the image, tag and push it:

```powershell
docker tag diabetes-prediction:v1 rakeshsharma1712/diabetes-prediction:v1

docker push rakeshsharma1712/diabetes-prediction:v1
```

**Pull the Docker Image**

```powershell
docker pull rakeshsharma1712/diabetes-prediction:v1
```

**Run the Docker Hub Image**

```powershell
docker run -d --name diabetes-hub-test -p 9698:9696 rakeshsharma1712/diabetes-prediction:v1
```

The API will be available at:

`http://127.0.0.1:9698/predict`

### 10.5 Test Docker Hub API

Use the following PowerShell command:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:9698/predict" `
    -Method Post `
    -Body $body `
    -ContentType "application/json"
```

### Expected Response

```json
{
    "diabetes probability": 0.8415,
    "diabetes_risk": true
}
```

The probability is illustrative and may vary depending on the trained model and Docker image version.

**Note:** If a container with the same name already exists, remove the old container before running a new one:

```powershell
docker rm -f diabetes-container
```

For the Docker Hub container, use:

```powershell
docker rm -f diabetes-hub-test
```



## 11. AWS Elastic Beanstalk Deployment

The Flask prediction API was successfully deployed on AWS Elastic Beanstalk using a Docker image hosted on Docker Hub.

### Deployment Details

| Configuration | Value |
|---|---|
| Cloud Provider | AWS |
| Deployment Service | Elastic Beanstalk |
| Region | Asia Pacific (Mumbai) - ap-south-1 |
| Environment | diabetes-prediction-env |
| Platform | Docker on Amazon Linux 2023 |
| EC2 Instance | t3.micro |
| Deployment Type | Single Instance |
| Environment Health | Ok |

### Public Application URL

http://diabetes-prediction-env.eba-mnm2ayx3.ap-south-1.elasticbeanstalk.com

### Prediction API Endpoint

**Method:** POST

**Endpoint:**

http://diabetes-prediction-env.eba-mnm2ayx3.ap-south-1.elasticbeanstalk.com/predict

The deployed API was successfully tested using a sample patient input and returned a diabetes probability of approximately 0.8388 with `diabetes_risk` set to `true`.

### Deployment Verification

- Docker image pulled successfully from Docker Hub.
- Elastic Beanstalk environment launched successfully.
- Public prediction API tested successfully.
- Environment health verified as `Ok`.
- EC2 instance status verified as `Ok`.


## 12. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Matplotlib and Seaborn
- Jupyter Notebook
- Flask
- Docker
- Docker Hub

## 13. Future Enhancements

- Improve model performance through additional feature engineering and tuning.
- Develop a user-friendly web interface.
- Monitor model performance and prediction quality.

## 14. Author

**Project:** Diabetes Prediction Using Machine Learning

**Purpose:** Machine Learning Zoomcamp – Midterm Project

## AWS Deployment Demo

The following video demonstrates the end-to-end testing of the Diabetes Prediction API deployed on AWS Elastic Beanstalk.

### Video Demonstration

[![Watch AWS Diabetes Prediction API Demo](https://img.youtube.com/vi/cssQyGtI-9Y/0.jpg)](https://youtu.be/cssQyGtI-9Y)

**[▶ Watch AWS Diabetes Prediction API Demo on YouTube](https://youtu.be/cssQyGtI-9Y)**

### Demo Highlights

- Sending patient health information through PowerShell using a POST request.
- Processing the request through the Flask API hosted on AWS.
- Generating diabetes risk predictions using the deployed XGBoost model.
- Receiving the diabetes probability and risk classification in the PowerShell terminal.

**Note:** This video was recorded while the AWS Elastic Beanstalk environment was active. The environment was subsequently terminated to avoid ongoing AWS infrastructure costs. The video is retained as deployment evidence.

