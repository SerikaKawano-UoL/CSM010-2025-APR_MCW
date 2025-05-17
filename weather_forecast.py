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

#####################
# PART A: DATA LOADING
#####################

def load_data(csv_filepath):
    """
    Load the weather_data CSV into a pandas DataFrame.
    :param csv_filepath: str, path to weather_data.csv
    :return: pd.DataFrame
    """
    # TODO: Implement loading the CSV file into a DataFrame called df
    # Then return df
    raise NotImplementedError


def clean_data(df):
    """
    Perform basic data cleaning:
    - Drop or fill missing values
    - Convert date column to datetime
    :param df: pd.DataFrame
    :return: pd.DataFrame (cleaned)
    """
    # TODO: Implement data cleaning
    # remove missing values convert the date to date time.
    raise NotImplementedError


#####################
# PART B: FEATURE ENGINEERING
#####################

def feature_engineering(df):
    """
    Create or transform features, for example:
    - temp_range = temp_max - temp_min
    - month from date
    - is_freezing if temp_min < 0
    - convert 'weather' to numeric labels
    :param df: pd.DataFrame
    :return: pd.DataFrame with new features + encoded target
    """
    # TODO: Implement feature creation
    # Make sure to store the binary encoded weather in a new column, e.g. 'weather_label'
    raise NotImplementedError


#####################
# PART C: DATA SPLITTING
#####################

def split_data(df):
    """
    Split the data into train/test sets.
    :param df: pd.DataFrame with feature columns and 'weather_label' (the target)
    :return: X_train, X_test, y_train, y_test
    """
    # TODO: Decide which columns to drop and which to keep
    # Then do train_test_split
    raise NotImplementedError


#####################
# PART D: MODEL TRAINING
#####################

def train_model_logreg(X_train, y_train):
    """
    Train a LogisticRegression model using X_train, y_train.
    :return: fitted logistic regression model
    """
    # TODO: Create a logistic regression model, fit it
    # Return the fitted model
    raise NotImplementedError


def train_model_rf(X_train, y_train):
    """
    Train a RandomForestClassifier using X_train, y_train.
    :return: fitted random forest model
    """
    # TODO: Create and fit a random forest model
    raise NotImplementedError


#####################
# PART E: EVALUATION
#####################

def evaluate_model(model, X_test, y_test):
    """
    Generate predictions for X_test using the trained model
    and print (or return) the accuracy and classification report.
    :return: (accuracy, report_text) or something similar
    """
    # TODO: Generate predictions, compute accuracy, classification report
    raise NotImplementedError


#####################
# DEMO MAIN (optional)
#####################

if __name__ == "__main__":
    """
    Optional: You can test your functions here.
    This block won't affect the autograder.
    """
    csv_path = "./data/weather-data.csv"
    try:
        df_loaded = load_data(csv_path)
        df_cleaned = clean_data(df_loaded)
        df_features = feature_engineering(df_cleaned)
        X_train, X_test, y_train, y_test = split_data(df_features)

        modelA = train_model_logreg(X_train, y_train)
        modelB = train_model_rf(X_train, y_train)

        accA, repA = evaluate_model(modelA, X_test, y_test)
        accB, repB = evaluate_model(modelB, X_test, y_test)

        print(f'\nLogistic Regression Accuracy: {accA} {repA}')
        print(f'\nRandom Forest Accuracy: {accB} {repB}')
    except NotImplementedError:
        print("Please implement the required functions before running.")
