import os  
import pandas as pd  


def save_metrics(
    model_name: str,
    accuracy: float,
    precision: float,
    recall: float,  
    f1: float,
):
    
    os.makedirs("results", exist_ok=True)   
    
    metrics_file = "results/model_comparison.csv"  
    
    
    new_row = pd.DataFrame({
        "Model": [model_name],
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1": [f1]
    })
    
    
    if os.path.exists(metrics_file):
        
        existing = pd.read_csv(metrics_file)  
        
        
        existing = existing[
            existing["Model"] != model_name
        ]
        
        updated = pd.concar(
            [existing, new_row],
            ignore_index=True
        )
        
        updated.to_csv(
            metrics_file,
            index=False
        )
    else:
        
        new_row.to_csv(
            metrics_file,
            index=False
        )
        
        
    print(
        f"Metrics saved for {model_name}"
    )  