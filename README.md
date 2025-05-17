# CSM010-2025-APR_MCW
CSM010-2025-APR Midterm Course Work
=======
Your task is to develop a machine learning model that predicts the **weather type** based on meteorological data. You will implement **feature engineering, data preprocessing, and model training** to classify weather conditions such as drizzle, rain, snow, sun, and fog.  

You are required to complete the following steps:

### **Data preprocessing**
- Convert `date` to a datetime object for easier extraction of time-based features.
- Handle missing values by removing rows with missing data.

### **Feature engineering**
Enhance the dataset by creating new features:
- **`temp_range`** = `temp_max - temp_min` (difference between max and min temperature)
- **`month`** = Extracted from the `date` column (1–12)
- **`is_freezing`** = Create a boolean feature (`True` if `temp_min < 0`, else `False`). Then cast to integers.

### Encode the target variable
- The `weather` column is a categorical variable (e.g., 'drizzle', 'rain', 'sun'). Use **label encoding** to convert `weather` into a numeric format suitable for classification models. Example of label encoding:
    | weather  | weather_label |
    |----------|--------------|
    | drizzle  | 0            |
    | rain     | 1            |
    | sun      | 2            |
    | snow     | 3            |
    | fog      | 4            |

### **Model training**
- Split the data into training (80%) and test (20%) sets and train at least **two classifiers**:
  - Logistic Regression
  - Random Forest Classifier
  
### **Model evaluation**
- Evaluate the models using:
  - Accuracy Score
  - Classification Report (Precision, Recall, F1-score)
  - Confusion matrix

### **Deliverables**
You must provide the Python script (`weather_forecast.py`) containing the completed functionality for:
   - Data loading and preprocessing
   - Feature engineering
   - Label encoding for the target variable
   - Model training and evaluation

