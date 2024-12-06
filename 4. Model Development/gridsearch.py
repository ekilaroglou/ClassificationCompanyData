import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, cohen_kappa_score)
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from tools.basic import get_data
from tools.embeddings import get_embeddings
from tools.encoding import encode_feature
from tools.preprocessing import lemmatize

# Step 0: Load data from SQLite3
# Retrieve the dataset from the database into a DataFrame
df = get_data()

# Step 1: Preprocess textual data
# Lemmatize key textual columns for cleaner and standardized text representation
df['lemma_homepage_text'] = lemmatize(df['homepage_text'])
df['lemma_homepage_keywords'] = lemmatize(df['homepage_keywords'])

# Merge various text fields for combined text representation
df['text_merged'] = (
    df['homepage_text'] + " " +
    df['homepage_keywords'] + " " +
    df['meta_description']
)

# Generate embeddings for text columns using different techniques
text_features = [
    get_embeddings(df, 'lemma_homepage_text', 'word2vec'),  # Word2Vec embeddings
    get_embeddings(df, 'lemma_homepage_keywords', 'tfidf'),  # TF-IDF embeddings
    get_embeddings(df, 'meta_description', 'bert')  # BERT embeddings
]

# Step 2: Encode categorical features
# Apply one-hot encoding to categorical columns for numerical representation
categorical_features = [
    encode_feature('one_hot_encoding', df, 'industry', 'category'),
    encode_feature('one_hot_encoding', df, 'country', 'category'),
    encode_feature('one_hot_encoding', df, 'employee_range', 'category')
]

# Step 3: Scale numerical features
# Normalize the 'year_founded' column to a [0, 1] range using MinMaxScaler
scaler = MinMaxScaler()
year_scaled = scaler.fit_transform(df[['year_founded']])
other_features = year_scaled

# Step 4: Combine all feature types into a single matrix
# Horizontally stack text, categorical, and numerical features into the final feature set
features = text_features + categorical_features + [other_features]
X = np.hstack([f for f in features])  # Feature matrix
y = df['category']  # Target variable

# Step 5: Define model and grid search parameters
# Specify hyperparameter grid for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'solver': ['liblinear', 'lbfgs', 'saga'],  # Optimization algorithms
    'class_weight': [None, 'balanced']  # Adjust weights for imbalanced classes
}

# Initialize GridSearchCV
# Perform hyperparameter tuning using cross-validation
grid_search = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid=param_grid,
    scoring='accuracy',  # Evaluation metric
    cv=5,  # Inner CV folds
    n_jobs=-1,  # Use all available CPU cores for parallel computation
)

# Step 6: Run GridSearchCV to find the best parameters
grid_search.fit(X, y)

# Retrieve the best model and its parameters from grid search
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Print the optimal hyperparameters and the best cross-validated score
print("Best Parameters:", best_params)
print("Best Cross-Validated Score:", grid_search.best_score_)

# Step 7: Perform k-fold cross-validation with the best model
# Outer CV to validate model performance on unseen data
n_splits = 5  # Number of folds for outer CV
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store metrics for each fold
results = []

print("Starting K-Fold Cross-Validation with Best Model...")
for train_index, test_index in kf.split(X):
    # Split data into training and testing sets for the current fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the best model on the training set
    best_model.fit(X_train, y_train)
    
    # Predict the target variable on the testing set
    y_pred = best_model.predict(X_test)
    
    # Compute evaluation metrics for the current fold
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),  # Accuracy metric
        "precision": precision_score(y_test, y_pred, average='weighted'),  # Weighted precision
        "recall": recall_score(y_test, y_pred, average='weighted'),  # Weighted recall
        "f1_score": f1_score(y_test, y_pred, average='weighted'),  # Weighted F1-score
        "cohen_kappa": cohen_kappa_score(y_test, y_pred),  # Cohen's Kappa statistic
    }
    results.append(metrics)

# Step 8: Aggregate results across all folds
# Compute the average value for each metric over all CV folds
avg_metrics = {
    metric: np.mean([fold[metric] for fold in results]) for metric in results[0]
}

# Print the average performance metrics
print("\nAverage metrics over", n_splits, "folds:")
for metric, value in avg_metrics.items():
    print(f"{metric}: {value:.4f}")
