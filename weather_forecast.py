"""
Skeleton Code for Weather Classification Midterm assessment (25%)

Instructions:
1. Complete each function where indicated.
2. You may add helper functions or import additional libraries if necessary.
3. Ensure you do not rename any of the existing function names, as the autograder relies on them.
4. Save and run: python weather_forecast.py
"""

#####################
# Import required libraries
#####################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#####################
# PART A: DATA LOADING
#####################

def load_data(csv_filepath):
    """
    Load the weather_data CSV into a pandas DataFrame.
    :param csv_filepath: str, path to weather_data.csv
    :return: pd.DataFrame
    """
    df = pd.read_csv(csv_filepath)
    return df


def clean_data(df):
    """
    Perform basic data cleaning:
    - Convert date to datetime
    - Drop rows with missing values
    :param df: pd.DataFrame
    :return: pd.DataFrame (cleaned)
    """
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna()
    return df

#####################
# PART B: FEATURE ENGINEERING
#####################

def feature_engineering(df):
    """
    Create new features:
    - temp_range = temp_max - temp_min
    - month from date
    - is_freezing if temp_min < 0
    - encode weather according to specified mapping
    :param df: pd.DataFrame
    :return: pd.DataFrame with new features + encoded target
    """
    df['temp_range'] = df['temp_max'] - df['temp_min']
    df['month'] = df['date'].dt.month
    df['is_freezing'] = (df['temp_min'] < 0).astype(int)

    # Map weather labels explicitly to match spec
    mapping = {'drizzle': 0, 'rain': 1, 'sun': 2, 'snow': 3, 'fog': 4}
    df['weather_label'] = df['weather'].map(mapping)

    return df

#####################
# PART C: DATA SPLITTING
#####################

def split_data(df):
    """
    Split the data into train/test sets (80/20).
    :param df: pd.DataFrame with features and 'weather_label'
    :return: X_train, X_test, y_train, y_test
    """
    feature_cols = [
        'precipitation', 'temp_max', 'temp_min', 'wind',
        'temp_range', 'month', 'is_freezing'
    ]
    X = df[feature_cols]
    y = df['weather_label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

#####################
# PART D: MODEL TRAINING
#####################

def train_model_logreg(X_train, y_train):
    """
    Train a LogisticRegression model.
    :return: fitted model
    """
    model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_model_rf(X_train, y_train):
    """
    Train a RandomForestClassifier.
    :return: fitted model
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

#####################
# PART E: EVALUATION
#####################

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model: accuracy, classification report, and confusion matrix.
    :return: (accuracy, report_text)
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Print metrics before plotting
    print(f"Accuracy: {acc}")
    print(report)

    # Plot confusion matrix
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return acc, report

#####################
# DEMO MAIN
#####################

if __name__ == "__main__":
    csv_path = "./data/weather-data.csv"
    df_loaded = load_data(csv_path)
    df_cleaned = clean_data(df_loaded)
    df_features = feature_engineering(df_cleaned)
    X_train, X_test, y_train, y_test = split_data(df_features)

    modelA = train_model_logreg(X_train, y_train)
    modelB = train_model_rf(X_train, y_train)

    accA, repA = evaluate_model(modelA, X_test, y_test)
    accB, repB = evaluate_model(modelB, X_test, y_test)

    print(f"\nLogistic Regression Accuracy: {accA}")
    print(repA)
    print(f"\nRandom Forest Accuracy: {accB}")
    print(repB)
