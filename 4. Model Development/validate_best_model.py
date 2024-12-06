from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, cohen_kappa_score)
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tools.basic import get_data
from tools.embeddings import get_embeddings
from tools.encoding import encode_feature
from tools.preprocessing import lemmatize

# 0. Load data from SQLite3
# Fetch data from the database and store it in a DataFrame
df = get_data()

# 1. Preprocess textual data
# Apply lemmatization to key textual columns for cleaner and more uniform embeddings
df['lemma_homepage_text'] = lemmatize(df['homepage_text'])
df['lemma_homepage_keywords'] = lemmatize(df['homepage_keywords'])

# 2. Generate text-based features
# Combine Word2Vec, TF-IDF, and BERT embeddings from different columns for comprehensive representation
text_features = [
    get_embeddings(df, 'lemma_homepage_text', 'word2vec'),
    get_embeddings(df, 'lemma_homepage_keywords', 'tfidf'),
    get_embeddings(df, 'meta_description', 'bert')
]

# 3. Encode categorical features
# Define the encoding method to be used for categorical features
encoding_method = 'one_hot_encoding'
categorical_features = [
    encode_feature(encoding_method, df, 'industry', 'category'),
    encode_feature(encoding_method, df, 'country', 'category'),
    encode_feature(encoding_method, df, 'employee_range', 'category')
]

# 4. Scale numerical features
# Normalize 'year_founded' column using MinMaxScaler for uniform scaling
scaler = MinMaxScaler()
year_scaled = scaler.fit_transform(df[['year_founded']])
other_features = year_scaled

# 5. Combine all features into a single array
# Concatenate text, categorical, and numerical features horizontally
features = text_features + categorical_features + [other_features]
X = np.hstack([f for f in features])  # Feature matrix
y = df['category']  # Target variable

# 6. K-Fold Cross-Validation setup
# Define the number of folds and initialize KFold
n_splits = 5  # Number of folds
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize Logistic Regression model
model = LogisticRegression()

# Store evaluation metrics for each fold
results = []

# 7. Perform K-Fold Cross-Validation
for train_index, test_index in kf.split(X):
    # Split data into training and testing sets for the current fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Predict the target variable on the test data
    y_pred = model.predict(X_test)
    
    # Compute evaluation metrics for the current fold
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
        "cohen_kappa": cohen_kappa_score(y_test, y_pred),
    }
    # Append metrics to the results list
    results.append(metrics)

# 8. Calculate average metrics across all folds
# Compute the mean of each metric over all folds
avg_metrics = {
    metric: np.mean([fold[metric] for fold in results]) for metric in results[0]
}

# 9. Print averaged results
# Display the average evaluation metrics over all folds
print("Average metrics over", n_splits, "folds:")
for metric, value in avg_metrics.items():
    print(f"{metric}: {value:.4f}")
