import re
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import csv
import warnings
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load training and testing data
training_data = pd.read_csv('Data/Training.csv')
testing_data = pd.read_csv('Data/Testing.csv')

# Extract feature columns and target column
feature_columns = training_data.columns[:-1]
X = training_data[feature_columns]
y = training_data['prognosis']

# Group data by prognosis and get the maximum value for each group
grouped_data = training_data.groupby(training_data['prognosis']).max()

# Encode target labels
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(y)
y_encoded = label_encoder.transform(y)

# Create a mapping of labels to their encoded values
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Plot the mapping
plt.figure(figsize=(10, 6))
plt.bar(label_mapping.keys(), label_mapping.values(), color='skyblue')
plt.xlabel('Prognosis')
plt.ylabel('Encoded Value')
plt.title('Label Encoding of Prognosis')
plt.xticks(rotation=90)
plt.show()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.33, random_state=42)
test_X = testing_data[feature_columns]
test_y = label_encoder.transform(testing_data['prognosis'])

# Train Decision Tree Classifier
decision_tree_clf = DecisionTreeClassifier()
decision_tree_clf.fit(X_train, y_train)
# Cross-validation score
cross_val_scores = cross_val_score(decision_tree_clf, X_test, y_test, cv=3)
print("Decision Tree Classifier Cross-Validation Score:", cross_val_scores.mean())

# Train Support Vector Machine Classifier
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
print("SVM Classifier Score:", svm_clf.score(X_test, y_test))

# Feature importances
feature_importances = decision_tree_clf.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

# Dictionaries to store severity, description, and precautions
severity_dict = {}
description_dict = {}
precaution_dict = {}

# Dictionary to map symptoms to indices
symptoms_index_dict = {symptom: index for index, symptom in enumerate(X)}

# Calculate condition severity
def calculate_condition(severity_list, days):
    total_severity = sum(severity_dict[item] for item in severity_list)
    if (total_severity * days) / (len(severity_list) + 1) > 13:
        print("You should take the consultation from doctor.")
    else:
        print("It might not be that bad but you should take precautions.")

# Load symptom descriptions
def load_descriptions():
    global description_dict
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            description_dict[row[0]] = row[1]

# Load symptom severity
def load_severity():
    global severity_dict
    with open('MasterData/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            severity_dict[row[0]] = int(row[1])

# Load symptom precautions
def load_precautions():
    global precaution_dict
    with open('MasterData/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            precaution_dict[row[0]] = row[1:]

# Get user information
def get_user_info():
    print("-----------------------------------HealthCare ChatBot-----------------------------------")
    name = input("Your Name? -> ")
    print("Hello,", name)

# Check pattern in disease list
def check_pattern(disease_list, input_symptom):
    input_symptom = input_symptom.replace(' ', '_')
    pattern = re.compile(f"{input_symptom}")
    matched_list = [item for item in disease_list if pattern.search(item)]
    return (1, matched_list) if matched_list else (0, [])

# Secondary prediction
def secondary_prediction(symptoms_list):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    decision_tree_clf = DecisionTreeClassifier()
    decision_tree_clf.fit(X_train, y_train)

    symptoms_index_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_index_dict))
    for symptom in symptoms_list:
        input_vector[symptoms_index_dict[symptom]] = 1

    return decision_tree_clf.predict([input_vector])

# Print disease
def print_disease(node):
    node = node[0]
    disease_indices = node.nonzero()
    diseases = label_encoder.inverse_transform(disease_indices[0])
    return [disease.strip() for disease in diseases]

# Convert decision tree to code
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    symptoms_present = []

    while True:
        input_symptom = input("Enter the symptom you are experiencing -> ")
        found, matched_symptoms = check_pattern(feature_names, input_symptom)
        if found:
            print("Searches related to input:")
            for num, symptom in enumerate(matched_symptoms):
                print(f"{num}) {symptom}")
            selected_symptom = int(input(f"Select the one you meant (0 - {num}): "))
            input_symptom = matched_symptoms[selected_symptom]
            break
        else:
            print("Enter valid symptom.")

    while True:
        try:
            num_days = int(input("Okay. From how many days? "))
            break
        except ValueError:
            print("Enter valid input.")

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == input_symptom:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            symptoms_given = grouped_data.columns[grouped_data.loc[present_disease].values[0].nonzero()]
            print("Are you experiencing any of the following symptoms?")
            symptoms_experienced = []
            for symptom in symptoms_given:
                response = input(f"{symptom}? (yes/no): ").strip().lower()
                if response == "yes":
                    symptoms_experienced.append(symptom)

            second_prediction = secondary_prediction(symptoms_experienced)
            calculate_condition(symptoms_experienced, num_days)
            if present_disease[0] == second_prediction[0]:
                print(f"You may have {present_disease[0]}")
                print(description_dict[present_disease[0]])
            else:
                print(f"You may have {present_disease[0]} or {second_prediction[0]}")
                print(description_dict[present_disease[0]])
                print(description_dict[second_prediction[0]])

            precautions = precaution_dict[present_disease[0]]
            print("Take the following measures:")
            for i, precaution in enumerate(precautions):
                print(f"{i + 1}) {precaution}")

    recurse(0, 1)

# Load data and start chatbot
load_severity()
load_descriptions()
load_precautions()
get_user_info()
tree_to_code(decision_tree_clf, feature_columns)
print("----------------------------------------------------------------------------------------")
