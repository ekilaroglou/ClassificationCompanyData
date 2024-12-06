import sqlite3
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from preprocessing import preprocess_text_keywords, preprocess_text_bert
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
plt.style.use('ggplot')


# Path to your SQLite database file
db_path = "../db/combined_data.db"

# Connect to the SQLite database
connection = sqlite3.connect(db_path)

# Load the datasets into pandas DataFrames
df = pd.read_sql_query("SELECT * FROM CompanyData", connection)

# Close the connection
connection.close()

### DATA UNDERSTANDING ###

# Shape
df.shape

# Columns
df.columns

# DTypes
df.dtypes

# Describe
df.describe()

### DATA PREPARATION

# Drop columns
columns_to_drop = ['Unnamed: 0', 'website', 'linkedin_url']
df.drop(columns=columns_to_drop, inplace=True)

# Ensure dtypes are correct
# Convert to integer, keeping NaN as NaN
df['year_founded'] = pd.to_numeric(df['year_founded'], errors='coerce').astype('Int64')

# Check duplicates
duplicate_companies = df[df.duplicated(subset='company_name', keep=False)]

# Check for NaN
df.isna().sum()



### NUMERIC COLUMNS ###

### Total employee estimate

##
# Correlation
column_names = ['current_employee_estimate',
                'total_employee_estimate']
df_corr = df[column_names].dropna().corr()
sns.heatmap(df_corr, annot=True)

##
# Count of top values
ax =  df['total_employee_estimate'].value_counts() \
    .head(15) \
    .plot(kind='bar')
                        
ax.set_xlabel('Total Employee Estimate')
ax.set_ylabel('Count')

##
# Histogram
df['total_employee_estimate'].plot(kind='hist')

##
# Top values
ax =  df['total_employee_estimate'] \
    .sort_values(ascending=False) \
    .head(25) \
    .plot(kind='bar')
                        
ax.set_xlabel('Index')
ax.set_ylabel('Total Employee Estimate')

##
# Check employees based on threshold values
# Define the thresholds
thresholds = [25000, 10000, 5000, 1000, 500, 300, 200, 100, 50, 40, 30, 20, 10, 9, 8,7, 6, 5, 4, 3, 2, 1]

# Calculate the counts for each threshold
count_dict = {threshold: (df['total_employee_estimate'] > threshold).sum() for threshold in thresholds}

# Create a DataFrame from the counts
count_table = pd.DataFrame(list(count_dict.items()), columns=['Threshold', 'Count'])

# Calculate the percentage of total rows
total_rows = len(df)
count_table['Percentage'] = (count_table['Count'] / total_rows * 100).round(2)

# Print the table
print(count_table)

##
# size_range based on category
# Plot the count plot using seaborn
sns.countplot(x='size_range', hue='category', data=df)
plt.show()

# excluding top values to depict better the lower values
# Filter the DataFrame to exclude the specified ranges
df_filtered = df[~df['size_range'].isin(['1 - 10', '11 - 50', '51 - 200'])]

# Plot the countplot using the filtered DataFrame
sns.countplot(x='size_range', hue='category', data=df_filtered)
plt.show()

##
# Total employees based on each category
df.plot(kind='scatter', x='total_employee_estimate', y='category')
plt.show()

##
# Change bins
# Define the bin edges and labels
bins = [0, 1, 3, 5, 10, 20, 50, 200, 500, 1000, 5000, float('inf')]
labels = ['1', '2-3', '4-5', '6-10', '11-20', '21-50', '51-200', '201-500', '501-1000', '1001-5000', '5001+']
# Create a new column with the binned categories
df['employee_range'] = pd.cut(df['total_employee_estimate'], bins=bins, labels=labels, right=True)

# Plot the countplot using the filtered DataFrame
sns.countplot(x='employee_range', hue='category', data=df)
plt.show()

##
# Final bins
# Define the bin edges and labels
bins = [0, 3, 10, 20, 50, 200, 500, 1000, 5000, float('inf')]
labels = ['1-3', '4-10', '11-20', '21-50', '51-200', '201-500', '501-1000', '1001-5000', '5001+']
# Create a new column with the binned categories
df['employee_range'] = pd.cut(df['total_employee_estimate'], bins=bins, labels=labels, right=True)

# Plot the countplot using the filtered DataFrame
sns.countplot(x='employee_range', hue='category', data=df)
plt.show()

### Year founded
##
# Histogram
df['year_founded'].plot(kind='hist', bins=40)

##
# Top year founded counts
ax =  df['year_founded'].value_counts() \
    .head(20) \
    .plot(kind='bar',
          title='Top Years founded')
                        
ax.set_xlabel('Year Founded')
ax.set_ylabel('Count')

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

### CATEGORY COLUMNS ###

### Category
# check value counts for each category
ax =  df['category'].value_counts() \
    .plot(kind='bar')
                        
ax.set_xlabel('Category')
ax.set_ylabel('Count')

### Industry
# check value counts for top industries
ax =  df['industry'].value_counts() \
    .head(10) \
    .plot(kind='bar')
                        
ax.set_xlabel('Industry')
ax.set_ylabel('Count')

##
# scatter between category an industry
df.plot(kind='scatter', x='industry', y='category')
plt.xticks(rotation=90)
plt.show()

### Country
# check value counts
ax =  df['country'].value_counts() \
    .plot(kind='bar')
                        
ax.set_xlabel('Country')
ax.set_ylabel('Count')

##
# Check how many times each category occurs in each country
# Define a palette for the categories
palette = sns.color_palette("Set2", len(df['category'].unique()))
category_order = df['category'].unique()

# Create the FacetGrid
g = sns.FacetGrid(df, col='country', col_wrap=4, sharex=False, sharey=False, height=4)

# Map barplot with hue for categories
def category_counts(data, color):
    sns.barplot(
        x='category',
        y='counts',
        hue='category',
        data=data.groupby('category').size().reset_index(name='counts'),
        dodge=False,
        palette=palette,
        hue_order=category_order,
        legend=False
    )

# Apply the function to the FacetGrid
g.map_dataframe(category_counts)

# Remove x-axis labels
for ax in g.axes.flat:
    ax.set_xticklabels([])

# Create a centralized legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color=palette[i], label=cat, linestyle='', markersize=10)
    for i, cat in enumerate(category_order)
]
g.fig.legend(handles=legend_elements, title="Category", bbox_to_anchor=(1.05, 0.5), loc='center left')

# Adjust layout
plt.tight_layout()
plt.show()

### Locality
# check value counts
ax =  df['locality'].value_counts() \
    .plot(kind='bar')
                        
ax.set_xlabel('Location')
ax.set_ylabel('Count')


### TEXT COLUMNS

##
## Combination of h1,h2,h3
# Find the total missing values
has_missing = df[['h1', 'h2', 'h3']].isna().all(axis=1).sum()
print(f"Combination h1,h2,h3 has {has_missing} rows with missing values.")

##
# Merge 'h1', 'h2', 'h3' into 'headings'
df['headings'] = df[['h1', 'h2', 'h3']].apply(
    lambda row: ' '.join(row.dropna().astype(str)) if row.notna().any() else np.nan,
    axis=1
)

# check nan values
df.isna().sum()

##
# process columns
columns = ['headings', 'nav_link_text', 'meta_keywords', 'homepage_text']

for c in columns:
    c_processed = f'{c}_processed'
    df[c_processed] = df[c].apply(lambda x: preprocess_text_keywords(x) if pd.notnull(x) else '')

##
# calculate word frequency
def calculate_average_word_length(df, columns):
    avg_word_lengths = {}
    for column in columns:
        avg_word_lengths[column] = df[column].apply(lambda x: len(x.split())).mean()
    return avg_word_lengths

# Calculate average word counts
average_word_lengths = calculate_average_word_length(df, columns)

# Display results
for column, avg_length in average_word_lengths.items():
    print(f"Average word count in '{column}': {avg_length:.2f} words")
    
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
                            
# all of them nan
len(df[df['text_nan_count']>2])

# at least 2 o them nan
len(df[df['text_nan_count']>1])

# keep only columns with less than 2 null count 
df = df[df['text_nan_count'] < 2]

# reset index without adding the old index as a new column
df.reset_index(inplace=True, drop=True)

## process columns again
df['homepage_keywords_processed'] = df['homepage_keywords'].apply(lambda x: preprocess_text_keywords(x) if pd.notnull(x) else '')
df['meta_description_processed'] = df['meta_description'].apply(lambda x: preprocess_text_bert(x) if pd.notnull(x) else '')

## Term frequency analysis
def term_frequency_analysis(text_data, top_n=20):
    vectorizer = CountVectorizer(stop_words='english')
    term_matrix = vectorizer.fit_transform(text_data)
    term_frequencies = term_matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    term_freq_df = pd.DataFrame({"term": terms, "frequency": term_frequencies})
    term_freq_df = term_freq_df.sort_values(by="frequency", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="frequency", y="term", data=term_freq_df, hue="term", dodge=False, legend=False)
    plt.title("Top Terms by Frequency", fontsize=16)
    plt.xlabel("Frequency")
    plt.ylabel("Term")
    plt.show()
    
term_frequency_analysis(df['homepage_text_processed'], top_n=20)
term_frequency_analysis(df['homepage_keywords_processed'], top_n=20)
term_frequency_analysis(df['meta_description_processed'], top_n=20)


## Top terms by category

def compare_top_words_grouped_bar(df, text_column, target_column, top_n=10):
    
    vectorizer = CountVectorizer(stop_words='english')
    unique_targets = df[target_column].unique()
    all_term_frequencies = []

    for target in unique_targets:
        target_data = df[df[target_column] == target]
        term_matrix = vectorizer.fit_transform(target_data[text_column])
        term_frequencies = term_matrix.sum(axis=0).A1
        terms = vectorizer.get_feature_names_out()

        term_freq_df = pd.DataFrame({"term": terms, "frequency": term_frequencies})
        term_freq_df = term_freq_df.sort_values(by="frequency", ascending=False).head(top_n)
        term_freq_df["target"] = target
        all_term_frequencies.append(term_freq_df)

    combined_df = pd.concat(all_term_frequencies)

    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=combined_df,
        x="frequency",
        y="term",
        hue="target",
        dodge=True,
    )
    plt.title(f"Top {top_n} Words by Target", fontsize=16)
    plt.xlabel("Frequency")
    plt.ylabel("Term")
    plt.legend(title="Target")
    plt.tight_layout()
    plt.show()


compare_top_words_grouped_bar(df, text_column='homepage_text_processed', target_column='category', top_n=5)
compare_top_words_grouped_bar(df, text_column='homepage_keywords_processed', target_column='category', top_n=5)
compare_top_words_grouped_bar(df, text_column='meta_description_processed', target_column='category', top_n=5)

def analyze_text_length_with_bar(df, text_column, target_column):
    # Calculate text length
    df["text_length"] = df[text_column].apply(len)
    
    plt.figure(figsize=(10, 6))
    
    # Create a bar plot with updated Seaborn parameters
    sns.barplot(
        x=target_column,
        y="text_length",
        hue=target_column, 
        data=df,
        errorbar='sd', 
        palette="Set2",
        dodge=False 
    )
    
    plt.title("Average Text Length vs Target Variable", fontsize=16)
    plt.xlabel("Target", fontsize=14)
    plt.ylabel("Average Text Length", fontsize=14)
    plt.xticks(fontsize=12, rotation=90)
    plt.yticks(fontsize=12)
    plt.legend([], [], frameon=False)  # Hide legend since hue is the same as x
    
    plt.show()
    
analyze_text_length_with_bar(df, 'homepage_text_processed', 'category')
analyze_text_length_with_bar(df, 'homepage_keywords_processed', 'category')
analyze_text_length_with_bar(df, 'meta_description_processed', 'category')

## word clouds

def generate_word_cloud(text_data):
    text = " ".join(text_data)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud", fontsize=16)
    plt.show()
    
generate_word_cloud(df['homepage_text'])
generate_word_cloud(df['homepage_keywords'])
generate_word_cloud(df['meta_description'])


### Final modifying
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

# write to database
# Create a SQLite3 connection
conn = sqlite3.connect(db_path)  # Creates a file-based database

# Save DataFrame to a table named 'company_table'
df.to_sql('CompanyDataProcessed', conn, if_exists='replace', index=False)

# Close the connection
conn.close()
