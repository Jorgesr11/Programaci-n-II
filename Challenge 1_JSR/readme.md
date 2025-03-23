# 1. Dataset Extraction: Obtaining and Loading Data
## 1.1 Dataset Description

The dataset used is the "Breast Cancer Wisconsin" (diagnostic dataset), which contains information on cancerous cells obtained through biopsy. Each sample has 30 numerical attributes related to cell nucleus characteristics.

## 1.2 Preprocessing

Removing unnecessary columns (id, Unnamed: 32).

Handling missing values (df.dropna(inplace=True)).

Encoding the target variable (LabelEncoder).

Data normalization (StandardScaler).

Splitting into training and testing sets (train_test_split).

# 2. Model Construction: Model Explanation

A RandomForestClassifier is used, which is an ensemble of decision trees that improves accuracy and reduces overfitting.
GridSearchCV is used to optimize hyperparameters.

# 3. MLOps: MLflow Integration

MLflow is used to manage training, evaluation, and model storage.

# 4. Execution Guide: Usage Instructions

## 4.1 Installing Dependencies
Run the following command in the terminal: pip install pandas scikit-learn matplotlib seaborn mlflow

## 4.2 Running the Notebook

Ensure the dataset is in the correct path.

Run each cell of the Jupyter Notebook step by step.

Access MLflow to visualize executed experiments:mlflow ui

# 5.Interpreting Results
ROC Curve: Evaluate model performance.

Confusion Matrix: Analyze prediction accuracy.

Metrics in MLflow: Compare different experiments