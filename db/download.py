import kagglehub
import os
import sqlite3
import pandas as pd
import shutil

# Get the current working directory
current_dir = os.getcwd()

# Download and extract the datasets
datasets = {
    "CompanyDataset": "peopledatalabssf/free-7-million-company-dataset",
    "CompanyClassification": "charanpuvvala/company-classification"
}

files = {}

for name, kaggle_path in datasets.items():
    # Download the dataset
    temp_path = kagglehub.dataset_download(kaggle_path)
    
    # Get file path
    file_name = os.listdir(temp_path)[0]
    file_path = os.path.join(temp_path, file_name)
    
    # Move to directory
    shutil.move(file_path, current_dir)
    shutil.rmtree(temp_path)
    
    # Store info about the file
    files[name] = file_name
    

# SQLite database file in the current directory
sqlite_file = os.path.join(current_dir, "combined_data.db")
conn = sqlite3.connect(sqlite_file)

for name, file_name in files.items():
    
    # Load CSV into a pandas DataFrame
    df = pd.read_csv(file_name)

    # Rename columns based on the dataset
    if name == "CompanyDataset":
        df.rename(columns={"name": "CompanyName", "domain": "Website"}, inplace=True)
    elif name == "CompanyClassification":
        df.rename(columns={"website": "Website", "company_name": "CompanyName"}, inplace=True)

    # Write DataFrame to SQLite
    df.to_sql(name, conn, if_exists="replace", index=False)
    # Delete file
    os.remove(file_name)

# Close the database connection
conn.close()
print(f"All data has been stored in {sqlite_file}")
