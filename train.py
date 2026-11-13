"""
Train the final Diabetes Prediction XGBoost model.

Run this script from the project root:
    python train.py

Expected project structure:
    data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv
    models/
    train.py
"""

from pathlib import Path
import pickle

import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


# ---------------------------------------------------------------------
# 1. Paths and model configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "xgb_tuned_eta_0_1_depth_4_min_child_60.bin"

TARGET = "diabetes_binary"

# These are the exact 11 features selected in the notebook.
SELECTED_FEATURES = [
    "genhlth",
    "highbp",
    "age",
    "highchol",
    "diffwalk",
    "income",
    "heartdiseaseorattack",
    "education",
    "physactivity",
    "physhlth",
    "bmi",
]

# Tuned XGBoost parameters from the notebook.
XGB_PARAMS = {
    "eta": 0.1,
    "max_depth": 4,
    "min_child_weight": 60,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "nthread": 8,
    "seed": 1,
    "verbosity": 1,
}

NUM_BOOST_ROUND = 200
SPLIT_RANDOM_STATE = 11


# ---------------------------------------------------------------------
# 2. Load data and retain the notebook's selected features
# ---------------------------------------------------------------------

def load_and_prepare_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATA_PATH}\n"
            "Check that the CSV file is inside the project's data folder."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.lower()

    required_columns = SELECTED_FEATURES + [TARGET]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {missing_columns}")

    # The notebook first selects these 11 features and then adds the target.
    df = df[required_columns].copy()

    # Apply the same category mappings used in the notebook.
    categorical_mappings = {
        "highbp": {
            0: "no high bp",
            1: "high bp",
        },
        "highchol": {
            0: "no high col",
            1: "high col",
        },
        "heartdiseaseorattack": {
            0: "no",
            1: "yes",
        },
        "physactivity": {
            0: "no",
            1: "yes",
        },
        "genhlth": {
            1: "excellent",
            2: "very good",
            3: "good",
            4: "fair",
            5: "poor",
        },
        "diffwalk": {
            0: "no",
            1: "yes",
        },
        "age": {
            1: "24",
            2: "29",
            3: "34",
            4: "39",
            5: "44",
            6: "49",
            7: "54",
            8: "59",
            9: "64",
            10: "69",
            11: "74",
            12: "79",
            13: "80",
        },
        "education": {
            1: "Never attended Kindergarten only",
            2: "Elementary (Grades 1-8)",
            3: "Some High School (Grades 9-11)",
            4: "High School Graduate / GED",
            5: "Some College / Technical School",
            6: "College Graduate (4+ years)",
        },
        "income": {
            1: "Less than 10,000",
            2: "10,000 to 15,000",
            3: "15,000 to 20,000",
            4: "20,000 to 25,000",
            5: "25,000 to 35,000",
            6: "35,000 to 50,000",
            7: "50,000 to 75,000",
            8: "75,000 or more",
        },
    }

    for column, mapping in categorical_mappings.items():
        df[column] = df[column].map(mapping)

    # Match the notebook's lowercase and space-to-underscore normalization.
    for column in categorical_mappings:
        df[column] = (
            df[column]
            .astype("string")
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )

    if df[required_columns].isnull().any().any():
        null_counts = df[required_columns].isnull().sum()
        null_counts = null_counts[null_counts > 0]
        raise ValueError(
            "Missing values found after category mapping:\n"
            f"{null_counts.to_string()}"
        )

    return df


# ---------------------------------------------------------------------
# 3. Split into train, validation and test sets (60/20/20)
# ---------------------------------------------------------------------

def split_data(df):
    df_full_train, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=SPLIT_RANDOM_STATE,
    )

    df_train, df_val = train_test_split(
        df_full_train,
        test_size=0.25,
        random_state=SPLIT_RANDOM_STATE,
    )

    y_train = (df_train[TARGET] == 1).astype(int).values
    y_val = (df_val[TARGET] == 1).astype(int).values
    y_test = (df_test[TARGET] == 1).astype(int).values
    y_full_train = (df_full_train[TARGET] == 1).astype(int).values

    X_train = df_train[SELECTED_FEATURES]
    X_val = df_val[SELECTED_FEATURES]
    X_test = df_test[SELECTED_FEATURES]
    X_full_train = df_full_train[SELECTED_FEATURES]

    return (
        X_train,
        X_val,
        X_test,
        X_full_train,
        y_train,
        y_val,
        y_test,
        y_full_train,
    )


# ---------------------------------------------------------------------
# 4. Fit DictVectorizer on training data and validate the tuned model
# ---------------------------------------------------------------------

def evaluate_validation(X_train, X_val, y_train, y_val):
    train_dicts = X_train.to_dict(orient="records")
    val_dicts = X_val.to_dict(orient="records")

    dv_validation = DictVectorizer(sparse=False)
    X_train_encoded = dv_validation.fit_transform(train_dicts)
    X_val_encoded = dv_validation.transform(val_dicts)

    feature_names = dv_validation.get_feature_names_out().tolist()

    dtrain = xgb.DMatrix(
        X_train_encoded,
        label=y_train,
        feature_names=feature_names,
    )
    dval = xgb.DMatrix(
        X_val_encoded,
        label=y_val,
        feature_names=feature_names,
    )

    validation_model = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
    )

    val_probabilities = validation_model.predict(dval)
    val_predictions = (val_probabilities >= 0.5).astype(int)

    print("\nValidation results")
    print("------------------")
    print(f"Validation ROC-AUC: {roc_auc_score(y_val, val_probabilities):.4f}")
    print("Validation confusion matrix:")
    print(confusion_matrix(y_val, val_predictions))
    print("Validation classification report:")
    print(
        classification_report(
            y_val,
            val_predictions,
            target_names=["No Diabetes", "Diabetes"],
        )
    )


# ---------------------------------------------------------------------
# 5. Retrain final model on all 80% training data and save it
# ---------------------------------------------------------------------

def train_and_save_final_model(X_full_train, y_full_train, X_test, y_test):
    full_train_dicts = X_full_train.to_dict(orient="records")
    test_dicts = X_test.to_dict(orient="records")

    # Fit a fresh vectorizer on the complete 80% training portion.
    dv = DictVectorizer(sparse=False)
    X_full_train_encoded = dv.fit_transform(full_train_dicts)
    X_test_encoded = dv.transform(test_dicts)

    feature_names = dv.get_feature_names_out().tolist()

    dfulltrain = xgb.DMatrix(
        X_full_train_encoded,
        label=y_full_train,
        feature_names=feature_names,
    )
    dtest = xgb.DMatrix(
        X_test_encoded,
        label=y_test,
        feature_names=feature_names,
    )

    final_model = xgb.train(
        XGB_PARAMS,
        dfulltrain,
        num_boost_round=NUM_BOOST_ROUND,
    )

    test_probabilities = final_model.predict(dtest)
    test_predictions = (test_probabilities >= 0.5).astype(int)

    print("\nFinal test results")
    print("------------------")
    print(f"Test ROC-AUC: {roc_auc_score(y_test, test_probabilities):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, test_predictions):.4f}")
    print("Test confusion matrix:")
    print(confusion_matrix(y_test, test_predictions))
    print("Test classification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=["No Diabetes", "Diabetes"],
        )
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Keep the exact tuple format expected by predict.py: (dv, XGBoost Booster).
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump((dv, final_model), model_file)

    print("\nModel saved successfully:")
    print(MODEL_PATH)
    print(f"Encoded feature count: {len(feature_names)}")


def main():
    print("Loading and preparing the diabetes dataset...")
    df = load_and_prepare_data()

    print(f"Dataset shape (11 features + target): {df.shape}")

    (
        X_train,
        X_val,
        X_test,
        X_full_train,
        y_train,
        y_val,
        y_test,
        y_full_train,
    ) = split_data(df)

    print(f"Training rows: {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")
    print(f"Full training rows: {len(X_full_train)}")
    print(f"Test rows: {len(X_test)}")

    evaluate_validation(X_train, X_val, y_train, y_val)

    train_and_save_final_model(
        X_full_train,
        y_full_train,
        X_test,
        y_test,
    )


if __name__ == "__main__":
    main()
