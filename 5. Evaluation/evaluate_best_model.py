import numpy as np
import os
import joblib
from sklearn.metrics import (precision_score, recall_score, 
                             f1_score, confusion_matrix, cohen_kappa_score, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from tools.basic import get_data
from tools.embeddings import get_embeddings
from tools.encoding import encode_feature
from tools.preprocessing import lemmatize

# Step 0: Load data from SQLite3
# Fetch data from the database and store it in a DataFrame
df = get_data()

# Step 1: Preprocess textual data
# Apply lemmatization to reduce words to their base forms for clean and consistent text
df['lemma_homepage_text'] = lemmatize(df['homepage_text'])
df['lemma_homepage_keywords'] = lemmatize(df['homepage_keywords'])

# Merge textual data from multiple columns to create a unified representation
df['text_merged'] = (
    df['homepage_text'] + " " +
    df['homepage_keywords'] + " " +
    df['meta_description']
)

# Generate embeddings for text features using different techniques
text_features = [
    get_embeddings(df, 'lemma_homepage_text', 'word2vec'),  # Word2Vec embeddings
    get_embeddings(df, 'lemma_homepage_keywords', 'tfidf'),  # TF-IDF embeddings
    get_embeddings(df, 'meta_description', 'bert')  # BERT embeddings
]
# Step 2: Encode categorical features
# Transform categorical columns into numerical format using one-hot encoding
categorical_features = [
    encode_feature('one_hot_encoding', df, 'industry', 'category'),
    encode_feature('one_hot_encoding', df, 'country', 'category'),
    encode_feature('one_hot_encoding', df, 'employee_range', 'category')
]

# Step 3: Scale numerical features
# Normalize the 'year_founded' column to the range [0, 1] for uniformity
scaler = MinMaxScaler()
year_scaled = scaler.fit_transform(df[['year_founded']])
other_features = year_scaled

# Step 4: Combine all feature types
# Concatenate text, categorical, and numerical features into a single feature matrix
features = text_features + categorical_features + [other_features]
X = np.hstack([f for f in features])  # Final feature matrix
y = df['category']  # Target variable

# Step 5: Split the data into training and testing sets
# Reserve 20% of the data for testing to evaluate model performance on unseen data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a classifier using predefined best parameters
# Initialize the Logistic Regression model with optimal parameters
best_params = {'C': 1, 'class_weight': None, 'solver': 'liblinear'}
clf = LogisticRegression(**best_params)

# Fit the model on the training data
clf.fit(X_train, y_train)

# Step 7: Make predictions
# Predict the target variable for the test set
y_pred = clf.predict(X_test)

# Step 8: Evaluate the model's performance
# Calculate accuracy
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Calculate precision
precision = precision_score(y_test, y_pred, average='weighted')
print(f"Precision: {precision:.2f}")

# Calculate recall
recall = recall_score(y_test, y_pred, average='weighted')
print(f"Recall: {recall:.2f}")

# Calculate F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1 Score: {f1:.2f}")

# Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate Cohen's Kappa statistic
kappa = cohen_kappa_score(y_test, y_pred)
print(f"Cohen's Kappa: {kappa:.2f}")

# Generate and display the classification report
# Provides a detailed breakdown of precision, recall, and F1 score for each class
print(classification_report(y_test, y_pred))

# Save the model
best_model_file_path = "../models/best_model.joblib"
# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(best_model_file_path), exist_ok=True)
joblib.dump(clf, best_model_file_path)