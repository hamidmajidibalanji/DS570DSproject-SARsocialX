# Dataset preprocessing and manipulation   
import pandas as pd   
import re   
import os


INPUT_FILE_PATH = 'data/raw/train.csv'
OUTPUT_FILE_PATH = 'data/processed/processed_disaster_tweets.csv'     



def clean_text(text):    
    text = str(text).lower()    
      
    text = re.sub(r"http\S+", "", text)     # remove URLs    
    text = re.sub(r"@\w+", "", text)        # remove mentions   
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove special chars   
    
    return text.strip()    





def preprocess_data():
    os.makedirs("data/processed", exist_ok=True)
    
    df = pd.read_csv(INPUT_FILE_PATH)    
    
    print("Original Dataset shape:", df.shape)   
    
    # clean tweets
    df["clean_text"] = df["text"].apply(clean_text)   
    
    
    # Real labels
    df["label"] = df["target"]

    df = df[["clean_text", "label"]]
    
    # Remove empty tweets
    df = df[df["clean_text"].str.len() > 5]

    print("Processed shape:", df.shape)

    df.to_csv(OUTPUT_FILE_PATH, index=False)

    print(f"Saved cleaned dataset to: {OUTPUT_FILE_PATH}")
    
    
if __name__ == "__main__":
    preprocess_data()    
    


