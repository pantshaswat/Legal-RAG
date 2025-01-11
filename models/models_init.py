
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer

def init_models():
    # Load the model and tokenizer
    model_name = "sentence-transformers/all-mpnet-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=10000
    )
    
    return model, tokenizer, tfidf_vectorizer
