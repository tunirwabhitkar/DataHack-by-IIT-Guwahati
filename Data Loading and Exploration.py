import pandas as pd

# Correct file paths
submission_format_path = 'D:/Hack-a-thon/submission_format.csv'
training_features_path = 'D:/Hack-a-thon/training_set_features.csv'
training_labels_path = 'D:/Hack-a-thon/training_set_labels.csv'
test_features_path = 'D:/Hack-a-thon/test_set_features.csv'

# Load the datasets
submission_format = pd.read_csv(submission_format_path)
train_features = pd.read_csv(training_features_path)
train_labels = pd.read_csv(training_labels_path)
test_features = pd.read_csv(test_features_path)

# Display first few rows of each dataframe to understand the structure
print("Submission Format:")
print(submission_format.head())

print("\nTraining Features:")
print(train_features.head())

print("\nTraining Labels:")
print(train_labels.head())

print("\nTest Features:")
print(test_features.head())

# Display basic statistics for numerical features
print("\nTraining Features Description:")
print(train_features.describe())

print("\nTraining Labels Distribution:")
print(train_labels.describe())

# Check for missing values in training features and labels
print("\nMissing Values in Training Features:")
print(train_features.isnull().sum())

print("\nMissing Values in Training Labels:")
print(train_labels.isnull().sum())
