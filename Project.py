import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score ,confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv("C:/Users/LENOVO/Downloads/Project/Electric Vehicle Population Data.csv")
print(data.head())
print(data.info())
print(data.describe())
print("\nMissing Values:")
print(data.isnull().sum())

print("\nMissing Value Percentage:")
print((data.isnull().mean() * 100).round(2))
for col in data.select_dtypes(include=np.number).columns:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(data[col].median())
for col in data.select_dtypes(include="object").columns:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(data[col].mode()[0])

print("\nMissing Values After Handling:")
print(data.isnull().sum())
print("\nTotal Missing Values Remaining:",data.isnull().sum().sum())
print("\nUnique Values:")
print("Vehicle Types:", data["Electric Vehicle Type"].nunique())
print("Makes:", data["Make"].nunique())
print("Models:", data["Model"].nunique())
print("Counties:", data["County"].nunique())
print("Model Years:", data["Model Year"].nunique())
plt.figure()
sns.boxplot(x=data["Electric Range"].dropna())
plt.title("Box Plot of Electric Range")
plt.xlabel("Electric Range (miles)")
plt.tight_layout()
plt.show()
Q1 = data["Electric Range"].quantile(0.25)
Q3 = data["Electric Range"].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = data[
    (data["Electric Range"] < lower_limit) |
    (data["Electric Range"] > upper_limit)
]

print("\nLower Limit:", lower_limit)
print("Upper Limit:", upper_limit)
print("Potential Range Outliers:", len(outliers))
numeric_data = data[["Model Year", "Electric Range"]]
correlation = numeric_data.corr()
print("\nCorrelation Matrix:")
print(correlation.round(2))
plt.figure(figsize=(6, 4))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Model Year and Electric Range")
plt.tight_layout()
plt.show()

#---------------------------------------------------------
#Part 2: Data Preprocessing
#---------------------------------------------------------
print("\n--- INVALID VALUE CHECK ---")
print("Negative Model Years:",(data["Model Year"] < 0).sum())
print("Negative Electric Range:",(data["Electric Range"] < 0).sum())
print("Zero Electric Range:",(data["Electric Range"] == 0).sum())
print("\n--- DUPLICATE CHECK ---")
print("Duplicate Rows:", data.duplicated().sum())
data = data.drop_duplicates()
print("Shape after duplicate removal:", data.shape)
regression_data = data[data["Electric Range"].notna() &(data["Electric Range"] > 0)].copy()
print("\nRegression Dataset Shape:", regression_data.shape)
print("\nRegression Target Summary:")
print(regression_data["Electric Range"].describe())
classification_data = data[data["Electric Vehicle Type"].isin(["Battery Electric Vehicle (BEV)","Plug-in Hybrid Electric Vehicle (PHEV)"])].copy()
print("\nClassification Dataset Shape:",classification_data.shape)
print("\nClassification Target Distribution:")
print(classification_data["Electric Vehicle Type"].value_counts())
classification_data["EV_Type"] = (classification_data["Electric Vehicle Type"].map({"Battery Electric Vehicle (BEV)": 1,"Plug-in Hybrid Electric Vehicle (PHEV)": 0}))
print("\nEncoded Target Values:")
print(classification_data["EV_Type"].value_counts())
print("\nRegression Missing Values:")
print(regression_data.isnull().sum())
print("\nClassification Missing Values:")
print(classification_data.isnull().sum())
print("\nData Preprocessing Checks Completed!")


#----------------------------------------------
#SIMPLE LINEAR REGRESSION
#----------------------------------------------
print("\n--- SIMPLE LINEAR REGRESSION ---")
X = regression_data[["Model Year"]]
y = regression_data["Electric Range"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))
slr_model = LinearRegression()
slr_model.fit(X_train, y_train)
y_pred = slr_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("\nSimple Linear Regression Results")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R-squared:", round(r2, 4))
print("Slope:", round(slr_model.coef_[0], 2))
print("Intercept:", round(slr_model.intercept_, 2))
plt.figure(figsize=(8, 5))
plt.scatter(X_test["Model Year"], y_test,alpha=0.3, label="Actual Values")
sorted_indices = np.argsort(X_test["Model Year"].values)

plt.plot(X_test["Model Year"].values[sorted_indices],y_pred[sorted_indices],color="red", 
         linewidth=2, label="Regression Line")
plt.xlabel("Model Year")
plt.ylabel("Electric Range (miles)")
plt.title("Simple Linear Regression: Model Year vs Electric Range")
plt.legend()
plt.tight_layout()
plt.show()

#-------------------------------------------
#Multiple Linear Regression
#-------------------------------------------
print("\n--- MULTIPLE LINEAR REGRESSION ---")
regression_data = data[(data["Electric Range"] > 0) & data["Postal Code"].notna()].copy()
x = regression_data[["Model Year", "Postal Code"]]
y = regression_data[["Electric Range"]]
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
x = x_scaler.fit_transform(x)
y = y_scaler.fit_transform(y)
plt.scatter(x[:, 0], y, alpha=0.3)
plt.xlabel("Model Year (Normalized)")
plt.ylabel("Electric Range (Normalized)")
plt.title("MULTIPLE LINEAR REGRESSION: Model Year vs Electric Range")
plt.grid(True)
plt.show()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
input_data = pd.DataFrame({"Model Year": [2025],"Postal Code": [98101]})
input_scaled = x_scaler.transform(input_data)
result = model.predict(input_scaled)
print("\nPredicted Normalized Electric Range:", result[0][0])
y_pred = model.predict(x_test)
plt.scatter(y_test, y_pred, color="purple", alpha=0.4)
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("Actual Electric Range (Normalized)")
plt.ylabel("Predicted Electric Range (Normalized)")
plt.title("Actual vs Predicted Electric Range")
plt.grid(True)
plt.show()
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error (MSE): {mse:.4f}")
r2 = r2_score(y_test, y_pred)
print(f"R-Squared Score: {r2:.4f}")
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.4f}")

#---------------------------------------------------
#Logistic Regression
#---------------------------------------------------

print("\n--- LOGISTIC REGRESSION ---")
classification_data = data[data["Electric Vehicle Type"].isin(["Battery Electric Vehicle (BEV)",
                                                               "Plug-in Hybrid Electric Vehicle (PHEV)"])].copy()
classification_data["EV_Type"] = (classification_data["Electric Vehicle Type"].map({
    "Battery Electric Vehicle (BEV)": 1,"Plug-in Hybrid Electric Vehicle (PHEV)": 0}))
X = classification_data[["Model Year"]]
y = classification_data["EV_Type"]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=95,stratify=y)
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
X_plot = np.sort(X_test, axis=0)
y_prob = log_model.predict_proba(X_plot)[:, 1]
plt.scatter(X_test, y_test, c=y_test, alpha=0.4)
plt.plot(X_plot, y_prob, "r-")
plt.xlabel("Model Year (Normalized)")
plt.ylabel("Probability of BEV")
plt.title("Logistic Regression")
plt.grid(True)
plt.show()

#-------------------------------------------
#Decision Tree Classifier
#-------------------------------------------

print("\n--- DECISION TREE CLIASSIFIER ---")
# Feature for Decision Tree
X = classification_data[["Model Year"]]
y = classification_data["EV_Type"]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=95,stratify=y)
tree_model = DecisionTreeClassifier(max_depth=5,random_state=95)
tree_model.fit(X_train, y_train)
y_pred = tree_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nDecision Tree Classifier Results")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Vehicle Type")
plt.ylabel("Actual Vehicle Type")
plt.title("Decision Tree Confusion Matrix")
plt.show()

#-----------------------------------------------
#K-Nearest Neighbors (KNN)
#-----------------------------------------------

print("\n--- K-NEAREST NEIGHBORS ---")
X = classification_data[["Model Year"]]
y = classification_data["EV_Type"]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=95,stratify=y)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
print("\nKNN Classifier Results")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Vehicle Type")
plt.ylabel("Actual Vehicle Type")
plt.title("KNN Confusion Matrix")
plt.show()







