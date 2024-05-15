import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# -------------------------------------------------------------------------------------------- #
# 1.  Import Dataset
data = pd.read_csv("diabetes.csv")
# print(data)
# -------------------------------------------------------------------------------------------- #
# 2.  Statistical Summary

# 2.1 Summary Statistics
summary_stats = data.describe()
print("Summary Statistics:")
print(summary_stats)

# 2.2 Correlation Matrix
correlation_matrix = data.corr()
print("Correlation Matrix:")
print(
    correlation_matrix
)  # No correlation value > 0.7 or < -0.7 => the dataset is not multicollinear (low correlation coefficients)


# 2.3 Missing Values
missing_values = data.isna().sum()
print("Missing Values:")
print(missing_values)  # No Missing values was founded across the dataset

# 2.4 Plotting histograms for all numeric features
data.hist(figsize=(10, 8))
plt.show()  # Having the first look at the data distribution

# 2.5 Checking Outliers in each column
outliers_count = {}
for column in data.columns[:-1]:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Count outliers
    outliers_count[column] = data[
        (data[column] < lower_bound) | (data[column] > upper_bound)
    ].shape[0]

print("Outliers count per column:")
print(
    outliers_count
)  # the number of outliers in each column is not significant (less than 5% of the total datas)

# Scatter plot between Pregnancies and SkinThickness
plt.figure(figsize=(8, 6))
plt.scatter(data["Pregnancies"], data["SkinThickness"], color="blue", alpha=0.5)
plt.title("Scatter Plot between Pregnancies and SkinThickness")
plt.xlabel("Pregnancies")
plt.ylabel("SkinThickness")
plt.grid(True)
plt.show()

# Scatter plot between BMI and Age
plt.figure(figsize=(8, 6))
plt.scatter(data["BMI"], data["Age"], color="green", alpha=0.5)
plt.title("Scatter Plot between BMI and Age")
plt.xlabel("BMI")
plt.ylabel("Age")
plt.grid(True)
plt.show()

# 2.6 Data Types
dtypes = data.dtypes
print(dtypes)

# Pregnancies                   int64
# Glucose                       int64
# BloodPressure                 int64
# SkinThickness                 int64
# Insulin                       int64
# BMI                         float64
# DiabetesPedigreeFunction    float64
# Age                           int64
# Outcome                       int64

# => All columns are numeric (7 integer and 2 float columns)

# -------------------------------------------------------------------------------------------- #
# 3.  Divide The Dataset

# 3.1 Vertical Split
target = "Outcome"
x = data.drop(target, axis=1)  # all feature columns except the target column
y = data[target]  # Outcome

# 3.2 Horizontal Split
# 60% train, 20% validation, 20% test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.25, random_state=42
)
print("Datasets - Shape:")
print(x_train.shape)  # (460, 8) - 614 rows and 8 columns
print(x_test.shape)  # (154, 8) - 614 rows and 8 columns
print(y_train.shape)  # (460,  ) - 614 rows and 1 columns
print(y_test.shape)  # (154,  ) - 614 rows and 1 columns

# -------------------------------------------------------------------------------------------- #
# 4.  Preprocessing Data

# 4.1 Standardization
# From the summary_stats, The data values in each column vary significantly, indicating that standardisation might be potentially needed.
# The Outliers are not significant, so it's possible to use the StandardScaler to standardize the data as the standard deviation and mean values won't be affected.

scaler = StandardScaler()
scaler.fit(
    x_train
)  # Calculate the mean and standard deviation for each column in the training set
x_train = scaler.fit_transform(x_train)  # convert data to standard normal distribution
x_test = scaler.transform(
    x_test
)  # apply the mean and standard deviation calculated on the training set to the test set to avoid data leakage
print("Datasets for Training and Testing after Standardizing:")
print(x_train)
print(x_test)


# -------------------------------------------------------------------------------------------- #
# 5.  Model Selection
# From the Correlation Matrix, there is no strong correlation between the features and the target column.
# Therefore, The Non-Linear models are more suitable for this dataset.

# 5.1 Support Vector Machine (SVM) - Classification
model_SVC = SVC()
model_SVC.fit(x_train, y_train)
# Evaluate the model
y_predict_SVC = model_SVC.predict(x_test)  # Test the trained model on the test set
# Showing the model's performance and actual Values
# for i, j in zip(y_predict_SVC, y_test.values):
#     print("Predicted: ", i, "Actual: ", j)
report_SVC = classification_report(y_test, y_predict_SVC)
print("Training Result - Support Vector Machine for Classification:")
print(report_SVC)

# #               precision    recall  f1-score   support
# #            0       0.75      0.83      0.79        99
# #            1       0.62      0.51      0.56        55
# #     accuracy                           0.71       154
# #    macro avg       0.69      0.67      0.67       154
# # weighted avg       0.71      0.71      0.71       154

# # In the context of the diabetes dataset, the recall value is more important than the precision value as identifying people having diabetes is a priority.
# # The Recall Value of model is 71% which is acceptable for the support vector machine model.

# -------------------------------------------------------------------------------------------- #
# 5.2 Logistic Regression Model
model_LR = LogisticRegression()
model_LR.fit(x_train, y_train)
y_predict_LR = model_LR.predict(x_test)
# Evaluate the model
report_LR = classification_report(y_test, y_predict_LR)
print("Training Result - Logistric Regression:")
print(report_LR)

# #               precision    recall  f1-score   support
# #            0       0.80      0.77      0.78        99
# #            1       0.61      0.65      0.63        55
# #     accuracy                           0.73       154
# #    macro avg       0.71      0.71      0.71       154
# # weighted avg       0.73      0.73      0.73       154

# # The Recall Value of model is 73% which is higher than Support Vector Machine Model.

# # -------------------------------------------------------------------------------------------- #
# # 5.3 Random forest :
model_RF = RandomForestClassifier(
    n_estimators=100, criterion="entropy", random_state=42
)
model_RF.fit(x_train, y_train)
y_predict_RF = model_RF.predict(x_test)

# Evaluate the model
report_RF = classification_report(y_test, y_predict_RF)
print("Training Result - Random forest:")
print(report_RF)

#               precision    recall  f1-score   support
#            0       0.79      0.79      0.79        99
#            1       0.62      0.62      0.62        55
#     accuracy                           0.73       154
#    macro avg       0.70      0.70      0.70       154
# weighted avg       0.73      0.73      0.73       154

# # The Recall Value of model is 74% which is higher than both Support Vector Machine Model and Logistic Regression.
# # -------------------------------------------------------------------------------------------- #
# # 6. Model Performance Enhancement:
# # Randome Forest comes with different hyper-parameter to tune the model Eg:
# # It takes times to tune the model and find the best hyper-parameters
# # The model can be enhanced by using the GridSearchCV to find the best hyper-parameters
params = {
    "n_estimators": [100, 200, 300],
    "criterion": ["gini", "entropy", "log_loss"],
}  # Candicate: 3 * 3 = 9

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=params,
    cv=5,
    verbose=2,
    scoring="recall",
)  # 5 folds * 9 candidates = 45 fits
grid_search.fit(x_train, y_train)
print(grid_search.best_estimator_)

print(
    grid_search.best_score_
)  # 0.7652173913043478 if the GridsearchCV is not assigned Score parameter, accuracy is assigned by default
print(grid_search.best_params_)  # {'criterion': 'gini', 'n_estimators': 200}

y_predict_RF_enhanced = grid_search.predict(x_test)
print("Training Result - Random forest - enhanced:")
print(classification_report(y_test, y_predict_RF_enhanced))

#               precision    recall  f1-score   support
#            0       0.80      0.80      0.80        99
#            1       0.64      0.64      0.64        55
#     accuracy                           0.74       154
#    macro avg       0.72      0.72      0.72       154
# weighted avg       0.74      0.74      0.74       154
