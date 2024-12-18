import numpy as np
import torch
from typing import List, Dict
from app.services.embeddings import create_bert_embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.services.vectorizer_manager import VectorizerManager
from app.services.llm_prompt import get_query_expansion
def hybrid_retriever(
    client,
    collection_name,
    query,
    model,
    tokenizer,
    tfidf_vectorizer,
    top_k: int = 5,
    bert_weight: float = 0.7
) -> List[Dict]:
    query = get_query_expansion(query)
    print('expanded query', query)
    # Convert query embedding to list of floats
    query_embedding = create_bert_embeddings([query], model, tokenizer)
    print('shape of query embedding', query_embedding.shape)
    
    # Convert NumPy array to list
    query_embedding_list = query_embedding[0].tolist()

    vectorizer_manager = VectorizerManager()
    vectorizer = vectorizer_manager.get_vectorizer(collection_name)
    query_tfidf = vectorizer.transform([query])

    points_count = client.count(collection_name=collection_name)
    print(f"Number of points in collection: {points_count}")

    # Pass list of floats to Qdrant
    qdrant_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding_list,
        limit=top_k
    )

    results = []

    for result in qdrant_results:
        # Ensure TF-IDF vector is properly converted
        tfidf_vector = np.array(result.payload['tfidf_vector']).reshape(1, -1)
        print('shape of tfidf vector', tfidf_vector.shape)
        
        # Cosine similarity returns a scalar, not an array
        tfidf_similarity = cosine_similarity(query_tfidf, tfidf_vector)[0][0]

        # Weighted combination
        combined_similarity = bert_weight * result.score + (1 - bert_weight) * tfidf_similarity

        results.append({
            'section_num': result.payload.get('section_num'),
            'content': result.payload.get('content'),
            'bert_similarity': float(result.score),  # Ensure float
            'tfidf_similarity': float(tfidf_similarity),  # Ensure float
            'combined_similarity': float(combined_similarity)  # Ensure float
        })

    # Sort results by combined similarity
    return sorted(results, key=lambda x: x['combined_similarity'], reverse=True)