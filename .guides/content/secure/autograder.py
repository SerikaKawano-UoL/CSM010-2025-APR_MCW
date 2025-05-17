import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from importlib import import_module

# Load model solution
import model_solution as ms

AUTOGRADER_ROOT = "/home/codio/workspace/.guides/content/secure/"

# Scoring system
total_score = 0
max_score = 100
section_scores = {
    "load_data": 10,
    "clean_data": 10,
    "feature_engineering": 15,
    "split_data": 15,
    "train_model_logreg": 20,
    "train_model_rf": 20,
    "accuracy": 10
}

feedback = ""

def grade_submission(submission_path, csv_path):
    total_score = 0
    feedback = ""
    try:
        sys.path.insert(0, submission_path)
        student_submission = import_module("weather_forecast")
    except Exception as e:
        print(f"Error importing student submission: {e}")
        return

    try:
        # Load and process data using model solution
        #df_model = ms.load_data(csv_path)
        #df_model = ms.clean_data(df_model)
        #df_model = ms.feature_engineering(df_model)
        #X_train_m, X_test_m, y_train_m, y_test_m = ms.split_data(df_model)
        
        # Load and process data using student solution
        #df_student = student_submission.load_data(csv_path)
        #df_student = student_submission.clean_data(df_student)
        #df_student = student_submission.feature_engineering(df_student)
        #X_train_s, X_test_s, y_train_s, y_test_s = student_submission.split_data(df_student)
        
        # Train models
        #model_logreg_s = student_submission.train_model_logreg(X_train_s, y_train_s)
        #model_rf_s = student_submission.train_model_rf(X_train_s, y_train_s)
        
        # Evaluate models
        #preds_logreg_s = model_logreg_s.predict(X_test_s)
        #preds_rf_s = model_rf_s.predict(X_test_s)
        
        #acc_logreg = accuracy_score(y_test_s, preds_logreg_s)
        #acc_rf = accuracy_score(y_test_s, preds_rf_s)
        
        #print(f"Logistic Regression Accuracy: {acc_logreg:.4f}")
        #print(f"Random Forest Accuracy: {acc_rf:.4f}")
        
        #print("\nLogistic Regression Classification Report:")
        #print(classification_report(y_test_s, preds_logreg_s))
        
        #print("\nRandom Forest Classification Report:")
        #print(classification_report(y_test_s, preds_rf_s))
        

        score_breakdown = {}
        feedback += "-----\n\n"
        try:
            student_submission.load_data(csv_path)
            total_score += section_scores["load_data"]
            score_breakdown["load_data"] = section_scores["load_data"]
        except:
            print("load_data function failed.")
            score_breakdown["load_data"] = 0
            feedback += "load_data: load_data function failed.\n"
        try:
            df_student = student_submission.load_data(csv_path)
            student_submission.clean_data(df_student)

            total_score += section_scores["clean_data"]
            score_breakdown["clean_data"] = section_scores["clean_data"]
        except:
            print("clean_data function failed.")
            score_breakdown["clean_data"] = 0
            feedback += "clean_data: clean_data function failed.\n"
        
        #try:
        #    df_student = student_submission.load_data(csv_path)
        #    student_submission.feature_engineering(df_student)
        #    total_score += section_scores["feature_engineering"]
        #    score_breakdown["feature_engineering"] = section_scores["feature_engineering"]
        #except:
        #    print("feature_engineering function failed.")
        #    score_breakdown["feature_engineering"] = 0
        #    feedback += "feature_engineering: feature_engineering function failed.\n"
        try:
            df_student = student_submission.load_data(csv_path)
            df_student = student_submission.clean_data(df_student)
            df_student = student_submission.feature_engineering(df_student)

            passed_all = True
            feature_errors = []

            # 1. temp_range
            if "temp_range" in df_student.columns:
                if not np.allclose(df_student["temp_range"], df_student["temp_max"] - df_student["temp_min"], atol=1e-2):
                    feature_errors.append("Incorrect values in 'temp_range'")
                    passed_all = False
            else:
                feature_errors.append("'temp_range' column missing")
                passed_all = False

            # 2. month
            if "month" in df_student.columns:
                try:
                    expected_months = pd.to_datetime(df_student["date"]).dt.month
                    if not df_student["month"].equals(expected_months):
                        feature_errors.append("Incorrect values in 'month' column")
                        passed_all = False
                except:
                    feature_errors.append("Failed to parse 'date' for month extraction")
                    passed_all = False
            else:
                feature_errors.append("'month' column missing")
                passed_all = False

            # 3. is_freezing
            if not "is_freezing" in df_student.columns:
                feature_errors.append("'is_freezing' column missing")
                passed_all = False

            # 4. weather_label
            if "weather_label" in df_student.columns:
                if not pd.api.types.is_integer_dtype(df_student["weather_label"]):
                    feature_errors.append("'weather_label' should be integers")
                    passed_all = False
            else:
                feature_errors.append("'weather_label' column missing")
                passed_all = False

            if passed_all:
                total_score += section_scores["feature_engineering"]
                score_breakdown["feature_engineering"] = section_scores["feature_engineering"]
            else:
                score_breakdown["feature_engineering"] = 0
                feedback += "feature_engineering issues:\n"
                for err in feature_errors:
                    feedback += f" - {err}\n"

        except Exception as e:
            print(f"feature_engineering function failed: {e}")
            score_breakdown["feature_engineering"] = 0
            feedback += f"feature_engineering: function failed with error: {e}\n"


        try:
            df_student = student_submission.load_data(csv_path)
            df_student = student_submission.clean_data(df_student)
            df_student = student_submission.feature_engineering(df_student)
            
            X_train_s, X_test_s, y_train_s, y_test_s = student_submission.split_data(df_student)
            expected_features = ['precipitation', 'temp_max', 'temp_min', 'wind', 'temp_range', 'month', 'is_freezing']
            if list(X_train_s.columns) == expected_features:
                total_score += section_scores["split_data"]
                score_breakdown["split_data"] = section_scores["split_data"]
            else:
                score_breakdown["split_data"] = 0
                feedback += f"split_data: Incorrect features used. Expected {expected_features}\n"

        except Exception as e:
            print(f"split_data function failed: {e}")
            score_breakdown["split_data"] = 0
            feedback += f"split_data: function failed with error: {e}\n"

        
        try:
            df_student = student_submission.load_data(csv_path)
            df_student = student_submission.clean_data(df_student)
            df_student = student_submission.feature_engineering(df_student)
            X_train_s, X_test_s, y_train_s, y_test_s = student_submission.split_data(df_student)

            student_submission.train_model_logreg(X_train_s, y_train_s)
            total_score += section_scores["train_model_logreg"]
            score_breakdown["train_model_logreg"] = section_scores["train_model_logreg"]
        except:
            print("train_model_logreg function failed.")
            score_breakdown["train_model_logreg"] = 0
            feedback += "train_model_logreg: train_model_logreg function failed.\n"
        
        try:
            df_student = student_submission.load_data(csv_path)
            df_student = student_submission.clean_data(df_student)
            df_student = student_submission.feature_engineering(df_student)
            X_train_s, X_test_s, y_train_s, y_test_s = student_submission.split_data(df_student)

            student_submission.train_model_rf(X_train_s, y_train_s)
            total_score += section_scores["train_model_rf"]
            score_breakdown["train_model_rf"] = section_scores["train_model_rf"]
        except:
            print("train_model_rf function failed.")
            score_breakdown["train_model_rf"] = 0
            feedback += "train_model_rf: train_model_rf function failed.\n"
        try:
          df_student = student_submission.load_data(csv_path)
          df_student = student_submission.clean_data(df_student)
          df_student = student_submission.feature_engineering(df_student)
          X_train_s, X_test_s, y_train_s, y_test_s = student_submission.split_data(df_student)

          # ensure is_freezing is int
      #    X_train_s['is_freezing'] = X_train_s['is_freezing'].astype(int)
      #    X_test_s['is_freezing'] = X_test_s['is_freezing'].astype(int)
          model_logreg_s = student_submission.train_model_logreg(X_train_s, y_train_s)
          model_rf_s = student_submission.train_model_rf(X_train_s, y_train_s)
          
          # Evaluate models
          preds_logreg_s = model_logreg_s.predict(X_test_s)
          preds_rf_s = model_rf_s.predict(X_test_s)
          
          acc_logreg = accuracy_score(y_test_s, preds_logreg_s)
          acc_rf = accuracy_score(y_test_s, preds_rf_s)
          
          if acc_logreg > 0.7 and acc_rf > 0.7:
              total_score += section_scores["accuracy"]
              score_breakdown["accuracy"] = section_scores["accuracy"]
          else:
              print("Accuracy below expected threshold.")
              score_breakdown["accuracy"] = 0
              feedback += "accuracy: Accuracy below expected threshold (>0.70) for some models.\n"
        except:
            print("Evaluation failed.")
            score_breakdown["accuracy"] = 0
            feedback += "accuracy: evaluation failed.\n"
            feedback += "-----\n\n"
        print(f"Final Grade: {total_score}/{max_score}")
        
        # Write results to file
        with open("/home/codio/workspace/grading_results.txt", "w") as f:
            f.write(f"Final Grade: {total_score}/{max_score}\n")
            f.write("Breakdown:\n")
            for key, value in score_breakdown.items():
                comment = f"{key}: {value}/{section_scores[key]}\n"
                f.write(comment)
                feedback += comment
        os.system('curl --retry 1 -s "$CODIO_AUTOGRADE_V2_URL" -d grade="'+str(total_score)+'" -d format=md -d feedback="'+feedback+'"') 

          
    except Exception as e:
        feedback = f"Error during execution: {e}"
        total_score = 0
        os.system('curl --retry 1 -s "$CODIO_AUTOGRADE_V2_URL" -d grade="'+str(total_score)+'" -d format=md -d feedback="'+feedback+'"') 
        print(feedback)
    finally:
        sys.path.pop(0)

if __name__ == "__main__":
    submission_path = "/home/codio/workspace/"  # Directory where student solution exists
    csv_path = submission_path + "/data/weather-data.csv"  # Path to the dataset
    grade_submission(submission_path, csv_path)
    