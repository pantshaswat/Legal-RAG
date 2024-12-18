import os
from typing import Dict
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

class VectorizerManager:
    def __init__(self, vectorizer_dir='vectorizers'):
        # Create directory to store vectorizers if it doesn't exist
        os.makedirs(vectorizer_dir, exist_ok=True)
        self.vectorizer_dir = vectorizer_dir
        self.vectorizers: Dict[str, TfidfVectorizer] = {}

    def create_vectorizer(self, collection_name: str, documents: list):
        """
        Create and save a TF-IDF vectorizer for a specific collection
        
        :param collection_name: Unique identifier for the collection
        :param documents: List of documents to fit the vectorizer
        """
        # Create vectorizer
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Save vectorizer
        vectorizer_path = os.path.join(self.vectorizer_dir, f'{collection_name}_vectorizer.pkl')
        joblib.dump(vectorizer, vectorizer_path)
        
        # Store in memory for quick access
        self.vectorizers[collection_name] = vectorizer
        
        return tfidf_matrix

    def get_vectorizer(self, collection_name: str):
        """
        Retrieve a vectorizer for a specific collection
        
        :param collection_name: Unique identifier for the collection
        :return: TF-IDF Vectorizer
        """
        # Check if vectorizer is already loaded in memory
        if collection_name in self.vectorizers:
            return self.vectorizers[collection_name]
        
        # Try to load from file
        vectorizer_path = os.path.join(self.vectorizer_dir, f'{collection_name}_vectorizer.pkl')
        
        if os.path.exists(vectorizer_path):
            vectorizer = joblib.load(vectorizer_path)
            self.vectorizers[collection_name] = vectorizer
            return vectorizer
        
        raise ValueError(f"No vectorizer found for collection: {collection_name}")