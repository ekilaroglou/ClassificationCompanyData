import sqlite3
import pandas as pd

def get_data():
    # Path to your SQLite database file
    db_path = "../db/combined_data.db"

    # Connect to the SQLite database
    connection = sqlite3.connect(db_path)

    # Load the datasets into pandas DataFrames
    df = pd.read_sql_query("SELECT * FROM CompanyDataProcessed", connection)

    # Close the connection
    connection.close()
    
    return df


       






