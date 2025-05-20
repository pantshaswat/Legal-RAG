from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
def init_models():
    # Load the model and tokenizer
    model = SentenceTransformer('universalml/Nepali_Embedding_Model')

    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        # stop_words='english',
        stop_words=None,
        ngram_range=(1, 2),
        max_features=10000
    )
    
    return model, tfidf_vectorizer


# XLM-R (XLM-RoBERTa)