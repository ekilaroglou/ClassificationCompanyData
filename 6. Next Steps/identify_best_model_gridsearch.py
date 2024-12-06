import joblib
import numpy as np
import time
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from tools.basic import get_data
from tools.embeddings import get_embeddings
from tools.encoding import encode_feature
from tools.preprocessing import lemmatize

# 0. Load data from SQLite3
df = get_data()

# Ensure all necessary columns are present
required_columns = ['homepage_text', 'homepage_keywords', 'meta_description', 
                    'industry', 'country', 'employee_range', 'year_founded', 'category']
assert all(col in df.columns for col in required_columns), "Missing required columns in DataFrame."

# Preprocess textual data
df['lemma_homepage_text'] = lemmatize(df['homepage_text'])
df['lemma_homepage_keywords'] = lemmatize(df['homepage_keywords'])
df['text_merged'] = (
    df['homepage_text'] + " " +
    df['homepage_keywords'] + " " +
    df['meta_description']
)

# Define configurations
text_representations = ['tfidf_word2vec_bert', 'tfidf_tfidf_bert', 'all_bert']
encoding_methods = ['one_hot_encoding', 'target_encoding']
features_used = ['all', 'text_only']

# Classifier dictionary
classifiers = {
    "Random Forest": (RandomForestClassifier(random_state=42), {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }),
    "Gradient Boosting (GBM)": (GradientBoostingClassifier(random_state=42), {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 10],
        'subsample': [0.8, 1.0]
    }),
    "Support Vector Machine (SVM)": (SVC(random_state=42, probability=True), {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }),
    "Logistic Regression": (LogisticRegression(max_iter=500, random_state=42), {
        'penalty': ['l2', 'none'],
        'C': [0.1, 1, 10],
        'solver': ['lbfgs', 'saga']
    }),
    "k-NN": (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }),
    "MLP (Neural Network)": (MLPClassifier(random_state=42), {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [500]
    })
}


# Define StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Track overall best model
overall_best_model = None
overall_best_score = 0
overall_best_name = ""

# Create a dictionary to store the best models for each configuration
best_models = {}

# Iterate over text representations
for text_representation in text_representations:
    print(f"\nStarting GridSearch for text representation: {text_representation}")
    start_time = time.time()  # Start timer for this section

    # Generate text features
    if text_representation == 'all_bert':
        text_features = [get_embeddings(df, 'text_merged', 'bert')]
    elif text_representation == 'tfidf_word2vec_bert':
        text_features = [
            get_embeddings(df, 'lemma_homepage_text', 'word2vec'),
            get_embeddings(df, 'lemma_homepage_keywords', 'tfidf'),
            get_embeddings(df, 'meta_description', 'bert')
        ]
    elif text_representation == 'tfidf_tfidf_bert':
        text_features = [
            get_embeddings(df, 'lemma_homepage_text', 'tfidf'),
            get_embeddings(df, 'lemma_homepage_keywords', 'tfidf'),
            get_embeddings(df, 'meta_description', 'bert')
        ]

    elapsed_time = time.time() - start_time  # End timer
    print(f"Text representation {text_representation} completed in {elapsed_time:.2f} seconds.")

    # Iterate over encoding methods
    for encoding_method in encoding_methods:
        print(f"\nStarting GridSearch for encoding method: {encoding_method}")
        start_time = time.time()

        # Encode categorical features
        categorical_features = [
            encode_feature(encoding_method, df, 'industry', 'category'),
            encode_feature(encoding_method, df, 'country', 'category'),
            encode_feature(encoding_method, df, 'employee_range', 'category')
        ]

        # Scale year_founded
        scaler = MinMaxScaler()
        year_scaled = scaler.fit_transform(df[['year_founded']])
        other_features = year_scaled  # No need to wrap in a list

        elapsed_time = time.time() - start_time
        print(f"Encoding method {encoding_method} completed in {elapsed_time:.2f} seconds.")

        for features_used_configuration in features_used:
            print(f"\nStarting GridSearch for features used configuration: {features_used_configuration}")
            start_time = time.time()

            if features_used_configuration == 'all':
                # Combine all features (flattened)
                features = text_features + categorical_features + [other_features]
            else:
                features = text_features

            # Flatten the list of features and concatenate them
            X = np.hstack([f for f in features])  # Use hstack to concatenate arrays

            # Define target variable
            y = df['category']  # Replace 'category' with the actual target column name

            # Track the best model for this combination of configurations
            best_model_for_config = None
            best_score_for_config = 0

            # Perform GridSearchCV for each classifier
            for name, (clf, param_grid) in classifiers.items():
                print(f"\nRunning GridSearch for {name} with features={features_used_configuration}")
                grid_start_time = time.time()

                grid_search = GridSearchCV(
                    estimator=clf,
                    param_grid=param_grid,
                    scoring='accuracy',
                    cv=skf,
                    verbose=2,
                    n_jobs=-1
                )
                grid_search.fit(X, y)

                grid_elapsed_time = time.time() - grid_start_time
                print(f"GridSearch for {name} completed in {grid_elapsed_time:.2f} seconds.")

                # Save the best model for the current configuration
                if grid_search.best_score_ > best_score_for_config:
                    best_score_for_config = grid_search.best_score_
                    best_model_for_config = grid_search.best_estimator_
                    print(f"New Best Model for this config: {best_score_for_config}")

                # Optionally, print the best parameters and accuracy
                print(f"Best Parameters for {name}: {grid_search.best_params_}")
                print(f"Best Cross-Validation Accuracy for {name}: {grid_search.best_score_}")

            # Save the best model for this combination (text_representation, encoding_method, features_used_configuration)
            if best_model_for_config:
                config_name = f"{text_representation}_{encoding_method}_{features_used_configuration}"
                joblib.dump(best_model_for_config, f"best_model_{config_name}.joblib")
                best_models[config_name] = best_model_for_config
                print(f"Best model for {config_name} saved.")

            # Track the overall best model
            if best_score_for_config > overall_best_score:
                overall_best_score = best_score_for_config
                overall_best_model = best_model_for_config
                overall_best_name = f"{text_representation}_{encoding_method}_{features_used_configuration}"
                print(f"New Overall Best Model: {overall_best_name} with Score: {overall_best_score}")

            elapsed_time = time.time() - start_time
            print(f"Features used configuration {features_used_configuration} completed in {elapsed_time:.2f} seconds.")

# Save the overall best model
if overall_best_model:
    joblib.dump(overall_best_model, f"best_model_overall_{overall_best_name}.joblib")
    print(f"Overall best model saved as best_model_overall_{overall_best_name}.joblib")

# Optionally, evaluate the overall best model on the entire dataset or a hold-out test set
y_pred = overall_best_model.predict(X)
print(f"Overall Accuracy on the dataset: {accuracy_score(y, y_pred)}")
print(f"Classification Report for Overall Best Model:\n{classification_report(y, y_pred)}")