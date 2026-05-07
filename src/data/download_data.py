# Data Download Module  
# First: test with a small dataset, then scale up to the full dataset
# Second: Replace the sample dataset URL with the actual dataset URL containing earthquake tweets.

import os  
import pandas as pd



# Sample dataset to check whether dataset donwloading works.   
# DATA_URL_ADDRESS = "https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json"   

SAVE_DATA_PATH = "data/raw/disaster_tweets.csv" 
DATA_URL_ADDRESS = (
    "https://raw.githubusercontent.com/"
    "thepanacealab/covid19_twitter/master/dailies/2020_07_08/2020_07_08_clean-dataset.csv"
    )  


# Definition of the function to download the data   
def download_data():
    os.makedirs("data/raw", exist_ok=True)     
    
    print("Downloading HumAID dataset...")
    
    # Sample dataset: It would be replaced with the actual dataset with earthquake tweets.    
    #dataset = load_dataset("Firoj/HumAID", split="train", trust_remote_code=True)

    df = pd.read_csv(DATA_URL_ADDRESS)  
    
    # Keep only text column
    df = df[["text"]]
    
    # Rename column
    df.rename(columns={"text": "tweet_text"}, inplace=True)
    
    print("Dataset size:", len(df))   
    
    #Keep only the relevant columns
    df.to_csv(SAVE_DATA_PATH, index=False)

    print(f"Saved dataset to: {SAVE_DATA_PATH}")
    
    
    
    
if __name__ == "__main__":
    download_data()    