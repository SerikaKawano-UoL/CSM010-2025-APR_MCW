"""
Model Solution for Weather Classification Assignment
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def load_data(csv_filepath):
    df = pd.read_csv(csv_filepath)
    return df

def clean_data(df):
    df = df.dropna()
    df['date'] = pd.to_datetime(df['date'])
    return df

def feature_engineering(df):
    df['temp_range'] = df['temp_max'] - df['temp_min']
    df['month'] = df['date'].dt.month
    df['is_freezing'] = df['temp_min'] < 0
    df['is_freezing'] = df['is_freezing'].astype(int)
    
    le = LabelEncoder()
    df['weather_label'] = le.fit_transform(df['weather'])
    #print(df['is_freezing'].head())
    return df

def split_data(df):
    features = ['precipitation', 'temp_max', 'temp_min', 'wind', 'temp_range', 'month', 'is_freezing']
    X = df[features]
    y = df['weather_label']
    #print(df[features].head())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def train_model_logreg(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_model_rf(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    print(f"\nModel Accuracy: {acc}")
    print("Classification Report:\n", report)

if __name__ == "__main__":
    csv_path = "/home/codio/workspace/data/weather-data.csv"

    df_loaded = load_data(csv_path)
    df_cleaned = clean_data(df_loaded)
    df_features = feature_engineering(df_cleaned)

    X_train, X_test, y_train, y_test = split_data(df_features)

    model_logreg = train_model_logreg(X_train, y_train)
    model_rf = train_model_rf(X_train, y_train)

    print("\nEvaluating Logistic Regression:")
    evaluate_model(model_logreg, X_test, y_test)

    print("\nEvaluating Random Forest:")
    evaluate_model(model_rf, X_test, y_test)
