# Dataset preprocessing and manipulation   
import pandas as pd   
import re   


INPUT_FILE_PATH = 'data/raw/humid.csv'
OUTPUT_FILE_PATH = 'data/processed/humid_cleaned.csv'     



def clean_data(text):
    text = text.lower()      
    text = re.sub(r"http\S+", "", text)     # remove URLs    
    text = re.sub(r"@\w+", "", text)        # remove mentions    
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove special chars   
    return text.strip()    





def preprocess_data():
    df = pd.read_csv(INPUT_FILE_PATH)    
    
    print("Original Dataset shape:", df.shape)   
    
    # clean
    df["cleaned_text"] = df["tweet_text"].astype(str).apply(clean_data)    
    
    # Binary label (SAR vs Non_SAR)  
    df["label"] = df["class_label"].apply(
        lambda x: 1 if x in ["requests_or_urgent_needs", "missing_or_found_people"] else 0
    )
    
    
    # Drop empty rows   
    df = df[df["cleaned_text"].str.len() > 5]
    
    print("Processed Dataset shape:", df.shape)   
    
    # Exporrt the cleaned dataset as csv file and save it in the output path    
    df.to_csv(OUTPUT_FILE_PATH, index=False)   
    
    
if __name__ == "__main__":
    preprocess_data()    
    


