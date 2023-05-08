#Bharathidasan Ilango - 201670038

# Libraries used
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score, cross_validate
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataset = pd.read_csv('dataset_assignment1.csv', header=0)

# Extract the features (X) and target variable (Y) from the dataset
X = dataset.drop('class', axis=1)
Y = dataset['class']

# Print the number of rows and columns in the data
print("Number of rows and columns in the data:",dataset.shape)

# Print some basic Information about the data
print("Basic Information about the data:")
print(dataset.info())

# Print the number of samples for each class
print("Number of samples for each class:/n",dataset['class'].value_counts())

# Ploting figures to visualize the dataset
plt.figure(figsize=(10, 6))
for feature in X.columns:
    count = X[feature].value_counts().sort_index()
    plt.plot(count.index, count.values, label=feature, linestyle='-')
plt.title('Line Chart of All Features')
plt.xlabel('Feature_Values')
plt.ylabel('Count')
plt.legend()
plt.xticks(range(0, len(count.index)+1, 1))
plt.show()

# Group the DataFrame by the class and compute the statistical description of each group
unique_elements = dataset["class"].unique()

for x in unique_elements:
  cl = dataset.loc[dataset["class"].isin([x])]
  print("\nFor class" ,x, "The statistical description of features:")
  print(cl.describe())

# Split the data into training and testing sets
X_train,X_test,Y_train, Y_test = train_test_split(X,Y, train_size=0.8, random_state = 24)

results = {'Classifier:':['Accuracy:','Precision:','Recall:','F1:']}

#Naive Bayes classifier
# Define a list of types of classifiers methods
classifiers = [GaussianNB(), MultinomialNB(), ComplementNB(), BernoulliNB()]

# Create a k-fold cross-validation object
kf = KFold(n_splits=5)

# Train each classifier and evaluate its accuracy on the test set
best_res = []
for clf in classifiers:

    # Use cross_val_score to perform cross-validation
    scores = cross_val_score(clf, X_train, Y_train, cv=kf)
    avg_score = scores.mean()  
    best_res.append({'Classifier': clf, 'Score': avg_score})
    
# Combine the results into a DataFrame and sort by score in descending order
best_res_df = sorted(best_res, key=lambda k: k['Score'], reverse=True)

#Taking best classifier method
b = best_res_df[0]['Classifier']

# Train the model using the training sets
b.fit(X_train, Y_train)

# Predict the response for the testing dataset
y_pred = b.predict(X_test)
# Create confusion matrix
cm = confusion_matrix(Y_test, y_pred)
print(cm)
# Plot confusion matrix as heatmap
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Test')
plt.title('NB Confusion Matrix')
plt.show()

# Calculate the accuracy of the model
accuracy = accuracy_score(Y_test, y_pred)*100
precision = precision_score(Y_test, y_pred)*100
recall = recall_score(Y_test, y_pred)*100
f1 = f1_score(Y_test, y_pred)*100
print("\nAccuracy of Naive Bayes best model:", accuracy)
print("Precision of Naive Bayes best model:", precision)
print("Recall of Naive Bayes best model:", recall)
print("F1 of Naive Bayes Gausbestsian model:", f1)

results['Naive Bayes'] = [accuracy,precision,recall,f1]

#KNN
# Define the parameter grid for hyperparameter tuning
param_grid = {'n_neighbors': np.arange(1, 10)}

# Define the K-Fold Cross Validation object
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Create the KNN model and perform hyperparameter tuning using GridSearchCV
knn = KNeighborsClassifier()
scoring = ['accuracy','precision','recall','f1']
grid = GridSearchCV(knn, param_grid, scoring = scoring, cv=kf, refit = 'f1')
grid.fit(X_train, Y_train)

# Print the best parameters and score
print("Best parameters:", grid.best_params_)
print("Best score:", grid.best_score_)

# Evaluate the model on the testing set using the best parameters
knn_best = KNeighborsClassifier(n_neighbors=grid.best_params_['n_neighbors'])
knn_best.fit(X_train, Y_train)
print("Testing score:", knn_best.score(X_test, Y_test))

# Predict the response for the testing dataset
y_pre = knn_best.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(Y_test, y_pre)
print(cm)
# Plot confusion matrix as heatmap#Kernel SVM
sns.heatmap(cm, annot=True, cmap='Reds')
plt.xlabel('Predicted')
plt.ylabel('Test')
plt.title('KNN Confusion Matrix')
plt.show()

# Calculate the accuracy of the model
accuracy = accuracy_score(Y_test, y_pre)*100
precision = precision_score(Y_test, y_pre)*100
recall = recall_score(Y_test, y_pre)*100
f1 = f1_score(Y_test, y_pre)*100
print("Accuracy of KNN model:", accuracy)
print("Precision of KNN model:", precision)
print("Recall of KNN model:", recall)
print("F1 of KNN model:", f1)
results['KNN'] = [accuracy,precision,recall,f1]

#Kernel SVM
# Create SVM classifier with kernel='rbf'
svm = SVC()
# Define the hyperparameters and their possible values
param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
scoring = ['accuracy','precision','recall','f1']
# Create the GridSearchCV object
grid = GridSearchCV(svm, param_grid, scoring = scoring, refit = 'f1')
# Fit the GridSearchCV object to the data
grid.fit(X_train, Y_train)

# Print the best parameters and score
print("Best kernel:", grid.best_params_['kernel'])

# Get the best hyperparameters
best_params = grid.best_params_
# Create a new classifier with the best hyperparameters
best_clf = SVC(**best_params)
# Fit the new classifier on the training data
best_clf.fit(X_train, Y_train)
# Make predictions on the test data
y_pre = best_clf.predict(X_test)


# Create confusion matrix
cm = confusion_matrix(Y_test, y_pre)
print(cm)
# Plot confusion matrix as heatmap
sns.heatmap(cm, annot=True, cmap='Greens')
plt.xlabel('Predicted')
plt.ylabel('Test')
plt.title('SVM Confusion Matrix')
plt.show()

# Calculate the accuracy of the model
accuracy = accuracy_score(Y_test, y_pre)*100
precision = precision_score(Y_test, y_pre)*100
recall = recall_score(Y_test, y_pre)*100
f1 = f1_score(Y_test, y_pre)*100    
print("Accuracy of SVM model with kernel:", accuracy)
print("Precission of SVM model with kernel:", precision)
print("Recall of SVM model with kernel:", recall)
print("F1 of SVM model with kernel:", f1)
results['SVM'] = [accuracy,precision,recall,f1]


#Results
results_df = pd.DataFrame.from_dict(results)
print("Classifer Performance table\n",results_df)
results_df.set_index('Classifier:', inplace=True)

# Create the plot
fig = plt.figure(figsize=(20, 12))
ax = results_df.plot(kind='bar', rot=0)

# Set the axis labels and title
ax.set_xlabel('Performance Metric')
ax.set_ylabel('Percentage')
ax.set_title('Classifier Performance')
# Show the plot
plt.show()