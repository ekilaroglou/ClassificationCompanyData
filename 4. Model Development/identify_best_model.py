import joblib
import numpy as np
import json
import os
from sklearn.metrics import (precision_score, recall_score, 
                             f1_score, cohen_kappa_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tools.basic import get_data
from tools.embeddings import get_embeddings
from tools.encoding import encode_feature
from tools.preprocessing import lemmatize

# Paths for saving models and results
best_model_path = "../models/"
results_path = "../results/"

# 0. Load data from SQLite3
df = get_data()

# Ensure all necessary columns are present
required_columns = ['homepage_text', 'homepage_keywords', 'meta_description',
                    'industry', 'country', 'employee_range', 'year_founded', 'category']
assert all(col in df.columns for col in required_columns), "Missing required columns in DataFrame."

# 1. Preprocess textual data
# Lemmatize textual columns for cleaner embeddings
df['lemma_homepage_text'] = lemmatize(df['homepage_text'])
df['lemma_homepage_keywords'] = lemmatize(df['homepage_keywords'])

# Merge multiple text-based columns into a single feature for certain embeddings
df['text_merged'] = (
    df['homepage_text'] + " " +
    df['homepage_keywords'] + " " +
    df['meta_description']
)

# 2. Define experiment configurations
# Text representation methods to evaluate
text_representations = ['tfidf_word2vec_bert', 'tfidf_tfidf_bert', 'all_bert']
# Encoding methods for categorical features
encoding_methods = ['one_hot_encoding', 'target_encoding']
# Feature sets to use: all features or only text-based features
features_used = ['all', 'text_only']

# Define classifiers with default parameters
classifiers = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(),
    "k-NN": KNeighborsClassifier()
}

# Initialize variables to track the best model
overall_best_model = None
overall_best_score = 0
overall_best_name = ""

# Results dictionary to store metrics for all configurations
results = {}

# 3. Iterate over text representations
for text_representation in text_representations:

    # Generate text features based on the selected representation
    if text_representation == 'all_bert':
        # Use BERT embeddings for all merged text
        text_features = [get_embeddings(df, 'text_merged', 'bert')]
    elif text_representation == 'tfidf_word2vec_bert':
        # Combine Word2Vec, TF-IDF, and BERT embeddings for different columns
        text_features = [
            get_embeddings(df, 'lemma_homepage_text', 'word2vec'),
            get_embeddings(df, 'lemma_homepage_keywords', 'tfidf'),
            get_embeddings(df, 'meta_description', 'bert')
        ]
    elif text_representation == 'tfidf_tfidf_bert':
        # Combine TF-IDF for text and keywords, and BERT for meta description
        text_features = [
            get_embeddings(df, 'lemma_homepage_text', 'tfidf'),
            get_embeddings(df, 'lemma_homepage_keywords', 'tfidf'),
            get_embeddings(df, 'meta_description', 'bert')
        ]

    # 4. Iterate over encoding methods
    for encoding_method in encoding_methods:

        # Encode categorical features based on the selected method
        categorical_features = [
            encode_feature(encoding_method, df, 'industry', 'category'),
            encode_feature(encoding_method, df, 'country', 'category'),
            encode_feature(encoding_method, df, 'employee_range', 'category')
        ]

        # Scale numerical feature: year_founded
        scaler = MinMaxScaler()
        year_scaled = scaler.fit_transform(df[['year_founded']])
        other_features = year_scaled

        # 5. Iterate over feature configurations
        for features_used_configuration in features_used:

            # Select features based on the configuration
            if features_used_configuration == 'all':
                features = text_features + categorical_features + [other_features]
            else:
                features = text_features

            # Stack all feature arrays horizontally
            X = np.hstack([f for f in features])
            y = df['category']

            # Split dataset into training and testing subsets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

            # 6. Iterate over classifiers
            for name, clf in classifiers.items():
                # Train the model
                clf.fit(X_train, y_train)
                
                # Predict on test set
                y_pred = clf.predict(X_test)
                
                # Calculate evaluation metrics
                accuracy = clf.score(X_test, y_test)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                kappa = cohen_kappa_score(y_test, y_pred)
                
                # Store results for this configuration
                config_key = f"{text_representation}_{encoding_method}_{features_used_configuration}_{name}"
                results[config_key] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "cohen_kappa": kappa,
                }
                
                # Update overall best model if this one is better
                if accuracy > overall_best_score:
                    overall_best_score = accuracy
                    overall_best_model = clf
                    overall_best_name = config_key

# 7. Save the best model
if overall_best_model:
    best_model_name = f"{overall_best_name}.joblib"
    best_model_file_path = best_model_path + best_model_name
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(best_model_file_path), exist_ok=True)
    joblib.dump(overall_best_model, best_model_file_path)

# 8. Save all results to a JSON file
json_file_path = results_path + "results.json"
# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
with open(json_file_path, "w") as json_file:
    json.dump(results, json_file, indent=4)
