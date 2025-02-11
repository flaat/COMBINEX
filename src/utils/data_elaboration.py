import os
import pandas as pd

def read_csv_files(folder_path):


    # List all files in the given folder
    files = os.listdir(folder_path)

    # Filter out files that are not CSV
    csv_files = [file for file in files if file.endswith('.csv')]

    # Dictionary to hold dataframes
    dataframes = {}

    # Loop through the CSV files and read them into pandas dataframes
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        dataframes[file] = pd.read_csv(file_path)

    return dataframes

# Usage
folder_path = 'path_to_your_folder'  # Replace with the path to your folder
csv_data = read_csv_files(folder_path)

# Now you can access each dataframe using csv_data['filename.csv']
