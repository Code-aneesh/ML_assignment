import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Dataset
file_path = 'household_power_consumption.csv'  # Replace with your dataset path
data = pd.read_csv(file_path, sep=',', low_memory=False)

# Step 2: Check for required columns and preprocess
# Check if 'Date' and 'Time' columns exist
if 'Date' in data.columns and 'Time' in data.columns:
    # Combine 'Date' and 'Time' into a single datetime column
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    data.drop(columns=['Date', 'Time'], inplace=True)
else:
    print("Error: 'Date' or 'Time' column is missing in the dataset!")
    print("Available columns:", data.columns)
    exit()

# Convert all non-numeric values to NaN and drop rows with missing values
data.replace('?', float('nan'), inplace=True)
data = data.dropna()

# Convert all numeric columns to float
numeric_columns = data.select_dtypes(include='object').columns
for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# Step 3: Create a target variable based on a threshold for consumption
threshold = 4.0  # Define the threshold for classification
data['Consumption_Level'] = (data['Global_active_power'] > threshold).astype(int)

# Step 4: Define features (X) and target (y)
X = data.drop(columns=['Consumption_Level', 'Datetime'])
y = data['Consumption_Level']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train Models
# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)

# Support Vector Machine Classifier
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
cm_svm = confusion_matrix(y_test, y_pred_svm)

# K-Nearest Neighbors Classifier
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
cm_knn = confusion_matrix(y_test, y_pred_knn)

# Step 6: Visualization of Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title(f"Random Forest (Accuracy: {acc_rf:.2f})")
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title(f"SVM (Accuracy: {acc_svm:.2f})")
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=axes[2])
axes[2].set_title(f"KNN (Accuracy: {acc_knn:.2f})")
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Step 7: Display Classification Reports
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn))
