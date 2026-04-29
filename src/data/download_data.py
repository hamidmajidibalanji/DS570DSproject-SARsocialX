# Data Download Module  
# First: test with a small dataset, then scale up to the full dataset

import os  
import pandas as pd

DATA_URL_ADDRESS = "https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json"   

SAVE_DATA_PATH = "data/raw/data.csv" 


# Definition of the function to download the data   
def download_data():
    os.makedirs("data/raw", exist_ok=True)     
    
    print("Downloading dataset...")
    
    # Sample dataset: It would be replaced with the actual dataset with earthquake tweets.    
    df = pd.read_jason(DATA_URL_ADDRESS)
    
    # converting json format into csv.   
    df.to_csv(SAVE_DATA_PATH, index=False)
    
    print(f"Dataset downloaded and saved to {SAVE_DATA_PATH}")   
    
    
    
    
if __name__ == "__main__":
    download_data()    