import nltk

def download_nltk_data():
    """Download required NLTK data for text processing."""
    resources = [
        'punkt',
        'averaged_perceptron_tagger',
        'wordnet',
        'vader_lexicon'
    ]
    
    for resource in resources:
        print(f"Downloading {resource}...")
        nltk.download(resource)
        print(f"Successfully downloaded {resource}")

if __name__ == "__main__":
    download_nltk_data()
