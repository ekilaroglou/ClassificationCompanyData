import sqlite3
import pandas as pd
import numpy as np
from tools.preprocessing import preprocess_text_keywords, preprocess_text_bert

# Path to your SQLite database file
db_path = "../db/combined_data.db"

# Connect to the SQLite database
connection = sqlite3.connect(db_path)

# Load the datasets into pandas DataFrames
df = pd.read_sql_query("SELECT * FROM CompanyData", connection)

# Close the connection
connection.close()

##
# Drop columns
columns_to_drop = ['Unnamed: 0', 'website', 'linkedin_url']
df.drop(columns=columns_to_drop, inplace=True)

##
# Ensure dtypes are correct
# Convert to integer, keeping NaN as NaN
df['year_founded'] = pd.to_numeric(df['year_founded'], errors='coerce').astype('Int64')

##
# Define the bin edges and labels
bins = [0, 3, 10, 20, 50, 200, 500, 1000, 5000, float('inf')]
labels = ['1-3', '4-10', '11-20', '21-50', '51-200', '201-500', '501-1000', '1001-5000', '5001+']
# Create a new column with the binned categories
df['employee_range'] = pd.cut(df['total_employee_estimate'], bins=bins, labels=labels, right=True)


##
# Fill missing year_founded values with distribution based imputation 
# Step 1: Create a dictionary to store the distribution of year_founded for each category
category_year_distribution = {}

# Group by 'category' and calculate the distribution of 'year_founded' within each category
for category in df['category'].unique():
    category_data = df[df['category'] == category]
    year_founded_counts = category_data['year_founded'].dropna().value_counts(normalize=True)  # get normalized frequencies
    category_year_distribution[category] = year_founded_counts

# Step 2: Impute missing 'year_founded' values based on the category distribution
def impute_year_founded(row, category_year_distribution):
    if pd.isna(row['year_founded']):
        category = row['category']
        distribution = category_year_distribution[category]
        return np.random.choice(distribution.index, p=distribution.values)  # Randomly pick a year based on the distribution
    return row['year_founded']

# Apply the imputation function
df['year_founded'] = df.apply(impute_year_founded, axis=1, category_year_distribution=category_year_distribution)



##
# merge h1, h2, h3
df['headings'] = df[['h1', 'h2', 'h3']].apply(
    lambda row: ' '.join(row.dropna().astype(str)) if row.notna().any() else np.nan,
    axis=1
)

##    
# Merge 'headings', 'meta_keywords', and 'nav_link_text' into 'homepage_keywords'
df['homepage_keywords'] = df[['headings', 'meta_keywords', 'nav_link_text']].apply(
    lambda row: ' '.join(row.dropna().astype(str)) if row.notna().any() else np.nan,
    axis=1
)



##
# create column to count the text nan values
df['text_nan_count'] = df[['homepage_text', 
                       'homepage_keywords', 
                       'meta_description']] \
                            .isnull().sum(axis=1)
                            
# keep only columns with less than 2 null count 
df = df[df['text_nan_count'] < 2]
df.reset_index(inplace=True, drop=True)


extra_stop_words = ['contact', 'services', 'home']
df['homepage_keywords'] = df['homepage_keywords'].apply(
    lambda x: preprocess_text_keywords(
        x, lemmatize=False, extra_stop_words=extra_stop_words) if pd.notnull(x) else '')
df['homepage_text'] = df['homepage_text'].apply(
    lambda x: preprocess_text_keywords(
        x, lemmatize=False, extra_stop_words=extra_stop_words) if pd.notnull(x) else '')
df['meta_description'] = df['meta_description'].apply(
    lambda x: preprocess_text_bert(x) if pd.notnull(x) else '')

# keep only wanted columns
columns = [
    'company_name',
    'category',
    'year_founded',
    'industry',
    'employee_range',
    'country',
    'homepage_text',
    'homepage_keywords',
    'meta_description'
    ]
df = df[columns]

# Create a SQLite3 connection
conn = sqlite3.connect(db_path)  # Creates a file-based database

# Save DataFrame to a table named 'company_table'
df.to_sql('CompanyDataProcessed', conn, if_exists='replace', index=False)

# Close the connection
conn.close()


