import json
import pandas as pd


results_path = "../results/"
results_file_name = "results.json"
xlsx_file_name = "results.xlsx"

# Define configurations and their abbreviations
text_representation_map = {
    'tfidf_word2vec_bert': 'TWB',
    'tfidf_tfidf_bert': 'TTB',
    'all_bert': 'B'
}

encoding_method_map = {
    'one_hot_encoding': 'OH',
    'target_encoding': 'T'
}

# check text_only first since 'all" is also in 'all_bert'
feature_used_map = {
    'text_only': 'text',
    'all': 'all'
}

classifier_map = {
    'Random Forest': 'RF',
    'Logistic Regression': 'LR',
    'k-NN': 'kNN'
}

# Read the results from the JSON file
json_file_path = results_path + results_file_name
with open(json_file_path, 'r') as f:
    results = json.load(f)

# Prepare the list to hold structured results
structured_results = []

# Iterate over the dictionary to extract the metrics and configurations
for key, metrics in results.items():
    # Initialize empty variables to hold the matched configurations
    text_representation = None
    encoding_method = None
    feature_used = None
    classifier = None
    
    # Match the text representation
    for tr in text_representation_map.keys():
        if tr in key:
            text_representation = text_representation_map[tr]
            break
    
    # Match the encoding method
    for em in encoding_method_map.keys():
        if em in key:
            encoding_method = encoding_method_map[em]
            break
    
    # Match the feature used
    for fu in feature_used_map.keys():
        if fu in key:
            feature_used = feature_used_map[fu]
            break
    
    # Match the classifier
    for cl in classifier_map.keys():
        if cl in key:
            classifier = classifier_map[cl]
            break
    
    # Format the metrics to 3 decimal places
    formatted_metrics = {metric: f"{value:.3f}" for metric, value in metrics.items()}
    
    # Append the result with relevant metrics
    structured_results.append({
        "text_representation": text_representation,
        "encoding_method": encoding_method,
        "feature_used": feature_used,
        "classifier": classifier,
        "accuracy": formatted_metrics.get('accuracy', None),
        "precision": formatted_metrics.get('precision', None),
        "recall": formatted_metrics.get('recall', None),
        "f1_score": formatted_metrics.get('f1_score', None),
        "cohen_kappa": formatted_metrics.get('cohen_kappa', None)
    })

# Convert the structured results to a pandas DataFrame
df_results = pd.DataFrame(structured_results)

# Export to Excel
output_path = results_path + xlsx_file_name
df_results.to_excel(output_path, index=False)

print(f"Excel file saved at {output_path}")
