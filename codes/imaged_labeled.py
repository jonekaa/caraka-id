import os
import shutil
import pandas as pd

root_dir = '../data/'


def organize_files(csv_path, source_dir, target_dir):
    # read csv
    df = pd.read_csv(csv_path)

    # Iterate through each row in the csv
    for index, row in df.iterrows():
        filename = row['filename']
        label = row.drop('filename').idxmax()

        # Check if the file exists before moving
        source_path = os.path.join(source_dir, filename)
        if os.path.exists(source_path):
            # Create destination folder
            destination_folder = os.path.join(target_dir, label)
            os.makedirs(destination_folder, exist_ok=True)

            # Move the file to the destination folder
            destination_path = os.path.join(destination_folder, filename)
            shutil.move(source_path, destination_path)
        else:
            print(f'File {filename} not found in {source_dir}')
