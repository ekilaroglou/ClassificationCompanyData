import sqlite3
import pandas as pd


### 1. IMPORT THE DATA ###

# Path to your SQLite database file
db_path = "../db/combined_data.db"

# Connect to the SQLite database
connection = sqlite3.connect(db_path)

# Load the datasets into pandas DataFrames
cd = pd.read_sql_query("SELECT * FROM CompanyDataset", connection)
cc = pd.read_sql_query("SELECT * FROM CompanyClassification", connection)

# Close the connection
connection.close()



### 2. REMOVE DUPLICATES ###

# CompanyClassification
cc = cc.drop_duplicates(subset='CompanyName', keep='first')

# CompanyDataset
# Step 1: Filter cd to only include rows with CompanyName in cc
cd = cd[cd['CompanyName'].isin(cc['CompanyName'])]

# Step 2: Identify duplicates in cd
duplicates_cd = cd[cd.duplicated(subset=['CompanyName'], keep=False)] \
                    .sort_values(by='CompanyName')

# Step 3: Process duplicates
result = []
other = []
for company, group in duplicates_cd.groupby('CompanyName'):
    # Get the CompanyClassification website for this CompanyName
    cc_website = cc.loc[cc['CompanyName'] == company, 'Website'].iloc[0] \
                    if company in cc['CompanyName'].values else None
                        
            
    if cc_website:
        # Check if cc_website matches or is a substring in any of group['Website']
        matches = group['Website'].str.contains(cc_website, na=False)

        if matches.sum() == 1:  
            # Only one duplicate match
            result.append(group[matches])
        else:  
            # All duplicates match or none of them match
            other.append(group)
    else:
        # No corresponding Website in cc, keep all duplicates
        other.append(group)
    
# Create a dataframe containing only the values that we want from the duplicates
filtered_duplicates = pd.concat(result).sort_index()

# Step 4: Final deduplicated DataFrame
cd = pd.concat([cd[~cd.index.isin(duplicates_cd.index)], 
                      filtered_duplicates]).sort_index()

### 3. INNER JOIN ###

# Step 1: Inner join on CompanyName
merged = pd.merge(cd, cc, on='CompanyName', how='inner', suffixes=('_cd', '_cc'))

# Step 2: Resolve Website column
def resolve_website(row):
    # if CompanyDataset.Website exists and includes CompanyClassification.Website
    if pd.notna(row['Website_cd']) and row['Website_cc'] in row['Website_cd']:
        return row['Website_cd']
    return row['Website_cc']

merged['Resolved_Website'] = merged.apply(resolve_website, axis=1)

# Drop the original Website columns if not needed
merged = merged.drop(columns=['Website_cd', 'Website_cc'])


### 4. RENAME COLUMNS WITH SNAKE CASE ###

rename_dictionary = {
    'CompanyName': 'company_name',
    'Category': 'category',
    'Resolved_Website': 'website',
    'year founded': 'year_founded',
    'size range': 'size_range',
    'linkedin url': 'linkedin_url',
    'current employee estimate': 'current_employee_estimate',
    'total employee estimate': 'total_employee_estimate',
    }

# Rename columns using the rename dictionary
merged = merged.rename(columns=rename_dictionary)

### 5. Column re-ordering
# Define the desired column order
desired_order = ['company_name', 'category', 'website']

# Reorder columns: put the desired columns first, followed by the rest
merged = merged[desired_order + \
        [col for col in merged.columns if col not in desired_order]]

### 6. SAVE TO SQLITE3 DATABASE ###
# Create a SQLite3 connection
conn = sqlite3.connect(db_path)  # Creates a file-based database

# Save DataFrame to a table named 'company_table'
merged.to_sql('CompanyData', conn, if_exists='replace', index=False)

# Close the connection
conn.close()
