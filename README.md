# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load your dataset
dataset = pd.read_csv('Milk_Adultration_Dataset.csv')
dataset
#Feature Extraction 
import pandas as pd

# Assuming 'data' is your actual data (replace this with your actual data)
data = [
    ['value1', 'Male', 'Ibrahim Saliu', 5.2, 'Yes', 150, 'Positive', 'Not adulterated', 'Active', 'Yes'],
    ['value2', 'Female', 'Sterling Unique', 7.0, 'No', 180, 'Negative', 'Adulterated', 'Inactive', 'No'],
    ['value3', 'Male', 'Hamsat', 5.1, 'No', 170, 'Negative', 'Adulterated', 'Inactive', 'No'],
    # Add more rows as needed
]

# Create DataFrame
df = pd.DataFrame(data, columns=['Ivory Greta = 300', 'Gender', 'Demographics of herd: Cow',
                                 'Average Quantity of Milk Produced per Cow: Daily (litre)',
                                 'Milking cows', 'Average Quantity of Milk Produced per Cow: Monthly (litre)',
                                 'Alcohol test', 'Water adulteration test', 'Status', 'legitimate'])

# Feature extraction example:

# Example 1: Extracting all specified attributes
selected_features = df[['Ivory Greta = 300', 'Demographics of herd: Cow',
                        'Average Quantity of Milk Produced per Cow: Daily (litre)',
                        'Average Quantity of Milk Produced per Cow: Monthly (litre)',
                        'Alcohol test', 'Water adulteration test', 'Status', 'legitimate']]

# Display each selected feature with numbering
for idx, column in enumerate(selected_features.columns, start=1):
    print(f"{idx}. {column}")

# Determine the number of features in selected_features dynamically
num_selected_features = len(selected_features.columns)
print("\nNumber of extraction in selected_features:", num_selected_features)
A MODEL USING MACHINE LEARNING ALGORITHM FOR PREDICTION OF RAW MILK ADULTERATIONA MODEL USING MACHINE LEARNING ALGORITHM FOR PREDICTION OF RAW MILK ADULTERATION
# Display each selected feature with numbering
for idx, column in enumerate(selected_features.columns, start=1):
    print(f"{idx}. {column}")

# Determine the number of features in selected_features dynamically
num_selected_features = len(selected_features.columns)
print("\nNumber of extraction in selected_features:", num_selected_features)
#RANDOM FOREST IMPLMENTATION
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load your dataset
dataset = pd.read_csv('Milk_Adultration_Dataset.csv')

# Check for missing values in the dataset
print("Missing values in the dataset:")
print(dataset.isnull().sum())

# Select features and target variables for 'Alcohol test'
X_alcohol = dataset[['Ivory Greta', 'Demographics of herd: Cow',
                     'Average Quantity of Milk Produced per Cow: Daily (litre)',
                     'Average Quantity of Milk Produced per Cow: Monthly (litre)']]

# Apply Label Encoding to categorical columns
label_encoder = LabelEncoder()
X_alcohol['Ivory Greta'] = label_encoder.fit_transform(X_alcohol['Ivory Greta'])

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_alcohol_imputed = pd.DataFrame(imputer.fit_transform(X_alcohol), columns=X_alcohol.columns)
y_alcohol = dataset['Alcohol test']

# Determine the number of samples for 'Alcohol test' based on the given ratio
total_samples = 4000
ratio_alcohol = 0.45  # 45% for alcohol
num_samples_alcohol = int(total_samples * ratio_alcohol)

# Split the data into training and testing sets for 'Alcohol test'
X_train_alcohol, X_test_alcohol, y_train_alcohol, y_test_alcohol = train_test_split(
    X_alcohol_imputed, y_alcohol, test_size=1 - ratio_alcohol, random_state=42, stratify=y_alcohol
)

# Initialize the Random Forest classifier for 'Alcohol test'
rf_classifier_alcohol = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# Train the Random Forest model on the training dataset for 'Alcohol test'
rf_classifier_alcohol.fit(X_train_alcohol, y_train_alcohol)

# Make predictions on the entire dataset for 'Alcohol test'
y_pred_alcohol = rf_classifier_alcohol.predict(X_alcohol_imputed)

# Calculate evaluation metrics for 'Alcohol test'
accuracy_alcohol = accuracy_score(y_alcohol, y_pred_alcohol)
precision_alcohol = precision_score(y_alcohol, y_pred_alcohol, average='weighted')
recall_alcohol = recall_score(y_alcohol, y_pred_alcohol, average='weighted')
f1_alcohol = f1_score(y_alcohol, y_pred_alcohol, average='weighted')

# Calculate precision-recall curve for 'Alcohol test'
precision_alcohol, recall_alcohol, _ = precision_recall_curve(y_alcohol, rf_classifier_alcohol.predict_proba(X_alcohol_imputed)[:, 1])
auc_alcohol = auc(recall_alcohol, precision_alcohol)

# Plot precision-recall curve for 'Alcohol test'
plt.figure(figsize=(7, 6))
plt.plot(recall_alcohol, precision_alcohol, label=f'Alcohol Adulteration (AUC = {auc_alcohol:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('RF Precision-Recall Curve for Alcohol Adulteration')
plt.legend(loc='best')
plt.show()

# Select features and target variables for 'Water adulteration test'
X_water = dataset[['Ivory Greta', 'Demographics of herd: Cow',
                   'Average Quantity of Milk Produced per Cow: Daily (litre)',
                   'Average Quantity of Milk Produced per Cow: Monthly (litre)']]

# Apply Label Encoding to categorical columns for 'Water adulteration test'
X_water['Ivory Greta'] = label_encoder.transform(X_water['Ivory Greta'])

# Handle missing values for 'Water adulteration test'
X_water_imputed = pd.DataFrame(imputer.transform(X_water), columns=X_water.columns)
y_water = dataset['Water adulteration test']

# Determine the number of samples for 'Water adulteration test' based on the given ratio
num_samples_water = total_samples - num_samples_alcohol

# Split the data into training and testing sets for 'Water adulteration test'
X_train_water, X_test_water, y_train_water, y_test_water = train_test_split(
    X_water_imputed, y_water, test_size=num_samples_water / len(y_water), random_state=42, stratify=y_water
)

# Initialize the Random Forest classifier for 'Water adulteration test'
rf_classifier_water = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# Train the Random Forest model on the training dataset for 'Water adulteration test'
rf_classifier_water.fit(X_train_water, y_train_water)

# Make predictions on the entire dataset for 'Water adulteration test'
y_pred_water = rf_classifier_water.predict(X_water_imputed)

# Calculate evaluation metrics for 'Water adulteration test'
accuracy_water = accuracy_score(y_water, y_pred_water)
precision_water = precision_score(y_water, y_pred_water, average='weighted')
recall_water = recall_score(y_water, y_pred_water, average='weighted')
f1_water = f1_score(y_water, y_pred_water, average='weighted')

# Calculate precision-recall curve for 'Water adulteration test'
precision_water, recall_water, _ = precision_recall_curve(y_water, rf_classifier_water.predict_proba(X_water_imputed)[:, 1])
auc_water = auc(recall_water, precision_water)

# Plot precision-recall curve for 'Water adulteration test'
plt.figure(figsize=(7, 6))
plt.plot(recall_water, precision_water, label=f'Water Adulteration (AUC = {auc_water:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('RF Precision-Recall Curve for Water Adulteration')
plt.legend(loc='best')
plt.show()

# Print evaluation metrics for both 'Alcohol test' and 'Water adulteration test'
print("\nRandom Forest Accuracy (Alcohol):", accuracy_alcohol)
print("Random Forest Precision (Alcohol):", precision_alcohol)
print("Random Forest Recall (Alcohol):", recall_alcohol)
print("Random Forest F1 Score (Alcohol):", f1_alcohol)

print("\nRandom Forest Accuracy (Water):", accuracy_water)
print("Random Forest Precision (Water):", precision_water)
print("Random Forest Recall (Water):", recall_water)
print("Random Forest F1 Score (Water):", f1_water)

# Total count of predicted values for 'Alcohol test' and 'Water adulteration test'
total_predicted_alcohol = num_samples_alcohol
total_predicted_water = num_samples_water

print("\nTotal Predicted Values (Alcohol):", total_predicted_alcohol)
print("Total Predicted Values (Water):", total_predicted_water)
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load your dataset
dataset = pd.read_csv('Milk_Adultration_Dataset.csv')

# Check for missing values in the dataset
print("Missing values in the dataset:")
print(dataset.isnull().sum())

# Select features and target variables for 'Alcohol test'
X_alcohol = dataset[['Ivory Greta', 'Demographics of herd: Cow',
                     'Average Quantity of Milk Produced per Cow: Daily (litre)',
                     'Average Quantity of Milk Produced per Cow: Monthly (litre)']]

# Apply Label Encoding to categorical columns
label_encoder = LabelEncoder()
X_alcohol['Ivory Greta'] = label_encoder.fit_transform(X_alcohol['Ivory Greta'])

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_alcohol_imputed = pd.DataFrame(imputer.fit_transform(X_alcohol), columns=X_alcohol.columns)
y_alcohol = dataset['Alcohol test']

# Determine the number of samples for 'Alcohol test' based on the given ratio
total_samples = 4000
ratio_alcohol = 0.45  # 45% for alcohol
num_samples_alcohol = int(total_samples * ratio_alcohol)

# Split the data into training and testing sets for 'Alcohol test'
X_train_alcohol, X_test_alcohol, y_train_alcohol, y_test_alcohol = train_test_split(
    X_alcohol_imputed, y_alcohol, test_size=1 - ratio_alcohol, random_state=42, stratify=y_alcohol
)

# Initialize the KNN classifier for 'Alcohol test'
knn_classifier_alcohol = KNeighborsClassifier(n_neighbors=5)

# Train the KNN model on the training dataset for 'Alcohol test'
knn_classifier_alcohol.fit(X_train_alcohol, y_train_alcohol)

# Make predictions on the entire dataset for 'Alcohol test'
y_pred_alcohol = knn_classifier_alcohol.predict(X_alcohol_imputed)

# Calculate evaluation metrics for 'Alcohol test'
accuracy_alcohol = accuracy_score(y_alcohol, y_pred_alcohol)
precision_alcohol = precision_score(y_alcohol, y_pred_alcohol, average='weighted')
recall_alcohol = recall_score(y_alcohol, y_pred_alcohol, average='weighted')
f1_alcohol = f1_score(y_alcohol, y_pred_alcohol, average='weighted')

# Calculate precision-recall curve for 'Alcohol test'
precision_alcohol, recall_alcohol, _ = precision_recall_curve(y_alcohol, knn_classifier_alcohol.predict_proba(X_alcohol_imputed)[:, 1])
auc_alcohol = auc(recall_alcohol, precision_alcohol)

# Plot precision-recall curve for 'Alcohol test'
plt.figure(figsize=(7, 6))
plt.plot(recall_alcohol, precision_alcohol, label=f'Alcohol Adulteration (AUC = {auc_alcohol:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('KNN Precision-Recall Curve for Alcohol Adulteration')
plt.legend(loc='best')
plt.show()

# Select features and target variables for 'Water adulteration test'
X_water = dataset[['Ivory Greta', 'Demographics of herd: Cow',
                   'Average Quantity of Milk Produced per Cow: Daily (litre)',
                   'Average Quantity of Milk Produced per Cow: Monthly (litre)']]

# Apply Label Encoding to categorical columns for 'Water adulteration test'
X_water['Ivory Greta'] = label_encoder.transform(X_water['Ivory Greta'])

# Handle missing values for 'Water adulteration test'
X_water_imputed = pd.DataFrame(imputer.transform(X_water), columns=X_water.columns)
y_water = dataset['Water adulteration test']

# Determine the number of samples for 'Water adulteration test' based on the given ratio
num_samples_water = total_samples - num_samples_alcohol

# Split the data into training and testing sets for 'Water adulteration test'
X_train_water, X_test_water, y_train_water, y_test_water = train_test_split(
    X_water_imputed, y_water, test_size=num_samples_water / len(y_water), random_state=42, stratify=y_water
)

# Initialize the KNN classifier for 'Water adulteration test'
knn_classifier_water = KNeighborsClassifier(n_neighbors=5)

# Train the KNN model on the training dataset for 'Water adulteration test'
knn_classifier_water.fit(X_train_water, y_train_water)

# Make predictions on the entire dataset for 'Water adulteration test'
y_pred_water = knn_classifier_water.predict(X_water_imputed)

# Calculate evaluation metrics for 'Water adulteration test'
accuracy_water = accuracy_score(y_water, y_pred_water)
precision_water = precision_score(y_water, y_pred_water, average='weighted')
recall_water = recall_score(y_water, y_pred_water, average='weighted')
f1_water = f1_score(y_water, y_pred_water, average='weighted')

# Calculate precision-recall curve for 'Water adulteration test'
precision_water, recall_water, _ = precision_recall_curve(y_water, knn_classifier_water.predict_proba(X_water_imputed)[:, 1])
auc_water = auc(recall_water, precision_water)

# Plot precision-recall curve for 'Water adulteration test'
plt.figure(figsize=(7, 6))
plt.plot(recall_water, precision_water, label=f'Water Adulteration (AUC = {auc_water:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('KNN Precision-Recall Curve for Water Adulteration')
plt.legend(loc='best')
plt.show()

# Print evaluation metrics for both 'Alcohol test' and 'Water adulteration test'
print("\nKNN Accuracy (Alcohol):", accuracy_alcohol)
print("KNN Precision (Alcohol):", precision_alcohol)
print("KNN Recall (Alcohol):", recall_alcohol)
print("KNN F1 Score (Alcohol):", f1_alcohol)

print("\nKNN Accuracy (Water):", accuracy_water)
print("KNN Precision (Water):", precision_water)
print("KNN Recall (Water):", recall_water)
print("KNN F1 Score (Water):", f1_water)

# Total count of predicted values for 'Alcohol test' and 'Water adulteration test'
total_predicted_alcohol = num_samples_alcohol
total_predicted_water = num_samples_water

print("\nTotal Predicted Values (Alcohol):", total_predicted_alcohol)
print("Total Predicted Values (Water):", total_predicted_water)
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load your dataset
dataset = pd.read_csv('Milk_Adultration_Dataset.csv')

# Check for missing values in the dataset
print("Missing values in the dataset:")
print(dataset.isnull().sum())

# Select features and target variables for 'Alcohol test'
X_alcohol = dataset[['Ivory Greta', 'Demographics of herd: Cow',
                     'Average Quantity of Milk Produced per Cow: Daily (litre)',
                     'Average Quantity of Milk Produced per Cow: Monthly (litre)']]

# Apply Label Encoding to categorical columns
label_encoder = LabelEncoder()
X_alcohol['Ivory Greta'] = label_encoder.fit_transform(X_alcohol['Ivory Greta'])

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_alcohol_imputed = pd.DataFrame(imputer.fit_transform(X_alcohol), columns=X_alcohol.columns)
y_alcohol = dataset['Alcohol test']

# Determine the number of samples for 'Alcohol test' based on the given ratio
total_samples = 4000
ratio_alcohol = 0.45  # 45% for alcohol
num_samples_alcohol = int(total_samples * ratio_alcohol)

# Split the data into training and testing sets for 'Alcohol test'
X_train_alcohol, X_test_alcohol, y_train_alcohol, y_test_alcohol = train_test_split(
    X_alcohol_imputed, y_alcohol, test_size=1 - ratio_alcohol, random_state=42, stratify=y_alcohol
)

# Initialize the SVM classifier for 'Alcohol test'
svm_classifier_alcohol = SVC(probability=True)

# Train the SVM model on the training dataset for 'Alcohol test'
svm_classifier_alcohol.fit(X_train_alcohol, y_train_alcohol)

# Make predictions on the entire dataset for 'Alcohol test'
y_pred_alcohol = svm_classifier_alcohol.predict(X_alcohol_imputed)

# Calculate evaluation metrics for 'Alcohol test'
accuracy_alcohol = accuracy_score(y_alcohol, y_pred_alcohol)
precision_alcohol = precision_score(y_alcohol, y_pred_alcohol, average='weighted')
recall_alcohol = recall_score(y_alcohol, y_pred_alcohol, average='weighted')
f1_alcohol = f1_score(y_alcohol, y_pred_alcohol, average='weighted')

# Calculate precision-recall curve for 'Alcohol test'
precision_alcohol, recall_alcohol, _ = precision_recall_curve(y_alcohol, svm_classifier_alcohol.predict_proba(X_alcohol_imputed)[:, 1])
auc_alcohol = auc(recall_alcohol, precision_alcohol)

# Plot precision-recall curve for 'Alcohol test'
plt.figure(figsize=(7, 6))
plt.plot(recall_alcohol, precision_alcohol, label=f'Alcohol Adulteration (AUC = {auc_alcohol:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('SVM Precision-Recall Curve for Alcohol Adulteration')
plt.legend(loc='best')
plt.show()

# Select features and target variables for 'Water adulteration test'
X_water = dataset[['Ivory Greta', 'Demographics of herd: Cow',
                   'Average Quantity of Milk Produced per Cow: Daily (litre)',
                   'Average Quantity of Milk Produced per Cow: Monthly (litre)']]

# Apply Label Encoding to categorical columns for 'Water adulteration test'
X_water['Ivory Greta'] = label_encoder.transform(X_water['Ivory Greta'])

# Handle missing values for 'Water adulteration test'
X_water_imputed = pd.DataFrame(imputer.transform(X_water), columns=X_water.columns)
y_water = dataset['Water adulteration test']

# Determine the number of samples for 'Water adulteration test' based on the given ratio
num_samples_water = total_samples - num_samples_alcohol

# Split the data into training and testing sets for 'Water adulteration test'
X_train_water, X_test_water, y_train_water, y_test_water = train_test_split(
    X_water_imputed, y_water, test_size=num_samples_water / len(y_water), random_state=42, stratify=y_water
)

# Initialize the SVM classifier for 'Water adulteration test'
svm_classifier_water = SVC(probability=True)

# Train the SVM model on the training dataset for 'Water adulteration test'
svm_classifier_water.fit(X_train_water, y_train_water)

# Make predictions on the entire dataset for 'Water adulteration test'
y_pred_water = svm_classifier_water.predict(X_water_imputed)

# Calculate evaluation metrics for 'Water adulteration test'
accuracy_water = accuracy_score(y_water, y_pred_water)
precision_water = precision_score(y_water, y_pred_water, average='weighted')
recall_water = recall_score(y_water, y_pred_water, average='weighted')
f1_water = f1_score(y_water, y_pred_water, average='weighted')

# Calculate precision-recall curve for 'Water adulteration test'
precision_water, recall_water, _ = precision_recall_curve(y_water, svm_classifier_water.predict_proba(X_water_imputed)[:, 1])
auc_water = auc(recall_water, precision_water)

# Plot precision-recall curve for 'Water adulteration test'
plt.figure(figsize=(7, 6))
plt.plot(recall_water, precision_water, label=f'Water Adulteration (AUC = {auc_water:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('SVM Precision-Recall Curve for Water Adulteration')
plt.legend(loc='best')
plt.show()

# Print evaluation metrics for both 'Alcohol test' and 'Water adulteration test'
print("\nSVM Accuracy (Alcohol):", accuracy_alcohol)
print("SVM Precision (Alcohol):", precision_alcohol)
print("SVM Recall (Alcohol):", recall_alcohol)
print("SVM F1 Score (Alcohol):", f1_alcohol)

print("\nSVM Accuracy (Water):", accuracy_water)
print("SVM Precision (Water):", precision_water)
print("SVM Recall (Water):", recall_water)
print("SVM F1 Score (Water):", f1_water)

# Total count of predicted values for 'Alcohol test' and 'Water adulteration test'
total_predicted_alcohol = num_samples_alcohol
total_predicted_water = num_samples_water

print("\nTotal Predicted Values (Alcohol):", total_predicted_alcohol)
print("Total Predicted Values (Water):", total_predicted_water)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# Split the dataset into a training and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plotting
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Random Forest')
plt.legend(loc="lower right")
plt.show()

# Print the AUC for each class and micro-average
for i in range(n_classes):
    print(f"AUC for class {i}: {roc_auc[i]}")
print(f"Micro-average AUC: {roc_auc['micro']}")
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# Split the dataset into a training and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3))
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plotting
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for K-Nearest Neighbors')
plt.legend(loc="lower right")
plt.show()

# Print the AUC for each class and micro-average
for i in range(n_classes):
    print(f"AUC for class {i}: {roc_auc[i]}")
print(f"Micro-average AUC: {roc_auc['micro']}")
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# Split the dataset into a training and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(SVC(probability=True))
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plotting
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Support Vector Machine')
plt.legend(loc="lower right")
plt.show()

# Print the AUC for each class and micro-average
for i in range(n_classes):
    print(f"AUC for class {i}: {roc_auc[i]}")
print(f"Micro-average AUC: {roc_auc['micro']}")

