# Data Download Module  
# First: test with a small dataset, then scale up to the full dataset
# Second: Replace the sample dataset URL with the actual dataset URL containing earthquake tweets.

import os  
import pandas as pd
from datasets import load_dataset


# Sample dataset to check whether dataset donwloading works.   
# DATA_URL_ADDRESS = "https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json"   

SAVE_DATA_PATH = "data/raw/humaid.csv" 


# Definition of the function to download the data   
def download_data():
    os.makedirs("data/raw", exist_ok=True)     
    
    print("Downloading HumAID dataset...")
    
    # Sample dataset: It would be replaced with the actual dataset with earthquake tweets.    
    dataset = load_dataset("Firoj/HumAID", split="train", trust_remote_code=True)

    df = pd.DataFrame(dataset)  
    
    #Keep only the relevant columns
    df = df[["tweet_text", "class_label"]]
    
    # converting json format into csv.   
    df.to_csv(SAVE_DATA_PATH, index=False)
    
    print(f"Dataset downloaded and saved to {SAVE_DATA_PATH}")  
    print(df.head())  # Print the first few rows of the dataset to verify   
    
    
    
    
if __name__ == "__main__":
    download_data()    