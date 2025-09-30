# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training and test datasets.
"""

import argparse
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow
import logging # Added logging import

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--data", type=str, help="Path to raw data")
    parser.add_argument("--train_data", type=str, help="Path to train dataset directory") # Clarified as directory
    parser.add_argument("--test_data", type=str, help="Path to test dataset directory") # Clarified as directory
    parser.add_argument("--test_train_ratio", type=float, default=0.2, help="Test-train ratio")
    args = parser.parse_args()
    return args

def main(args):
    '''Read, preprocess, split, and save datasets'''

    # Reading Data
    if not args.data:
        logging.error("Error: --data argument is required.")
        return
    try:
        df = pd.read_csv(args.data)
        logging.info(f"Successfully read data. Initial shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {args.data}")
        return
    except Exception as e:
        logging.error(f"Error reading data: {e}")
        return

    # Encode categorical feature
    if 'Segment' in df.columns:
        le = LabelEncoder()
        df['Segment'] = le.fit_transform(df['Segment'])
        logging.info("Successfully applied Label Encoding to 'Segment' column.") # Corrected logging message
    else:
        logging.warning("The 'Segment' column was not found in the dataset for encoding.")

    # Split Data into train and test datasets
    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)

    # Save train and test data
    # No need to call os.makedirs if the path is provided by Azure ML outputs
    # as the directory should already exist.
    # If it doesn't, this line is fine, but the problem is with the path content. 
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)


    # Use pathlib for cleaner and more robust path handling
    train_path = Path(args.train_data) / "data.csv"
    test_path = Path(args.test_data) / "data.csv"

    # Save the files directly to the constructed paths
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logging.info(f"Saved train data to {train_path}")
    logging.info(f"Saved test data to {test_path}")

    train_df.to_csv(os.path.join(args.train_data, "data.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_data, "data.csv"), index=False)
    logging.info(f"Saved train data to {os.path.join(args.train_data, 'data.csv')}")
    logging.info(f"Saved test data to {os.path.join(args.test_data, 'data.csv')}")

    # Log the metrics
    mlflow.log_metric('train size', train_df.shape[0])
    mlflow.log_metric('test size', test_df.shape[0])

if __name__ == "__main__":
    mlflow.start_run()
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Configured logging
    
    lines = [
        f"Raw data path: {args.data}",
        f"Train dataset output path: {args.train_data}",
        f"Test dataset path: {args.test_data}",
        f"Test-train ratio: {args.test_train_ratio}",
    ]

    for line in lines:
        print(line)
    
    main(args)
    mlflow.end_run()