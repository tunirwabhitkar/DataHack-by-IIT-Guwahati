import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Load the datasets
train_features = pd.read_csv('D:/Hack-a-thon/training_set_features.csv')
train_labels = pd.read_csv('D:/Hack-a-thon/training_set_labels.csv')

# Combine train features and labels
train_data = pd.merge(train_features, train_labels, on='respondent_id')

# Separate features and targets
X = train_data.drop(['respondent_id', 'xyz_vaccine', 'seasonal_vaccine'], axis=1)
y = train_data[['xyz_vaccine', 'seasonal_vaccine']]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing pipelines for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Define the model
model = LGBMClassifier()

# Create a pipeline that includes preprocessing and the classifier
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', MultiOutputClassifier(model, n_jobs=-1))])

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict_proba(X_val)

# Calculate ROC AUC score for each label and average
roc_auc_xyz = roc_auc_score(y_val['xyz_vaccine'], [prob[1] for prob in y_pred[0]])
roc_auc_seasonal = roc_auc_score(y_val['seasonal_vaccine'], [prob[1] for prob in y_pred[1]])
mean_roc_auc = (roc_auc_xyz + roc_auc_seasonal) / 2

print(f'ROC AUC Score for xyz_vaccine: {roc_auc_xyz}')
print(f'ROC AUC Score for seasonal_vaccine: {roc_auc_seasonal}')
print(f'Mean ROC AUC Score: {mean_roc_auc}')
