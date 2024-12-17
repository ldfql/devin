from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

def initialize_finbert():
    """Download and save FinBERT model and tokenizer."""
    model_name = "ProsusAI/finbert"
    save_path = "app/services/web_scraping/models/finbert"
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    print("Downloading FinBERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    print(f"Saving model and tokenizer to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("FinBERT initialization complete!")

if __name__ == "__main__":
    initialize_finbert()
