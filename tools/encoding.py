from sklearn.preprocessing import OneHotEncoder

def encode_feature(method, df, feature, target = None):
    if method == 'one_hot_encoding':
        one_hot_encoder = OneHotEncoder(sparse=False)
        X_encoded = one_hot_encoder.fit_transform(df[[feature]])
    elif method == 'target_encoding':
        # Step 1: Encode the target variable as numeric values
        target_mapping = {label: idx for idx, label in enumerate(df[target].unique())}
        df[f'{target}_encoded'] = df[target].map(target_mapping)
        
        # Step 2: Calculate the mean target value for each target value in the feature
        mean_target_per_feature = df.groupby(feature)[f'{target}_encoded'].mean()
        
        # Step 3: Replace 'industry' with the calculated target encoding
        df[f'{feature}_encoded'] = df[feature].map(mean_target_per_feature)
        
        # Step 4: Apply smoothing to prevent overfitting
        global_mean = df[f'{target}_encoded'].mean()
        counts = df.groupby(feature).size()
        
        smooth_factor = 10 
        smoothed_target_encoded = (mean_target_per_feature * counts + global_mean * smooth_factor) / (counts + smooth_factor)
        
        X_encoded = df[feature].map(smoothed_target_encoded).to_numpy().reshape(-1, 1)
    return X_encoded